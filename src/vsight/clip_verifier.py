"""Trainable CLIP candidate verifier used by the E2 experiment.

This module intentionally depends on the ML environment. Dependency-free
decision and metric utilities remain in :mod:`vsight.e2_verifier`.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Mapping, Sequence

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .e2_verifier import candidate_geometry


GEOMETRY_DIM = 18


def read_selector_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _clip_box(
    box: Sequence[float], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = max(0, min(image_width - 1, int(round(x1))))
    y1 = max(0, min(image_height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(image_width, int(round(x2))))
    y2 = max(y1 + 1, min(image_height, int(round(y2))))
    return x1, y1, x2, y2


def candidate_views(
    image: Image.Image, box: Sequence[float]
) -> tuple[Image.Image, Image.Image]:
    """Return an object crop and a full scene with the candidate marked."""

    image = image.convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = _clip_box(box, width, height)
    pad_x = max(2, int(round((x2 - x1) * 0.08)))
    pad_y = max(2, int(round((y2 - y1) * 0.08)))
    crop_box = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )
    crop = image.crop(crop_box)

    marked = image.copy()
    draw = ImageDraw.Draw(marked, "RGBA")
    shade = (0, 0, 0, 42)
    draw.rectangle((0, 0, width, y1), fill=shade)
    draw.rectangle((0, y2, width, height), fill=shade)
    draw.rectangle((0, y1, x1, y2), fill=shade)
    draw.rectangle((x2, y1, width, y2), fill=shade)
    line_width = max(3, min(width, height) // 120)
    draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(230, 32, 32, 255), width=line_width)
    return crop, marked


class E2SelectorDataset(Dataset):
    """Image/query examples with exactly two eligible generated boxes."""

    def __init__(
        self,
        rows: Sequence[Mapping],
        image_root: Path,
        *,
        training: bool,
    ) -> None:
        self.rows = [dict(row) for row in rows if row.get("selector_eligible")]
        self.image_root = Path(image_root)
        self.training = training
        if not self.rows:
            raise ValueError("selector dataset has no eligible rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = self.image_root / Path(str(row["image_filename"])).name
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        boxes = [
            [float(value) for value in row["baseline_bbox_xyxy"]],
            [float(value) for value in row["challenger_bbox_xyxy"]],
        ]
        ious = [float(row["baseline_iou"]), float(row["challenger_iou"])]
        label = 1 if str(row["selector_action"]) == "switch" else 0

        # Random source order makes accidental positional/source shortcuts
        # impossible even though the scorer itself is permutation equivariant.
        if self.training and random.random() < 0.5:
            boxes.reverse()
            ious.reverse()
            label = 1 - label

        views = [candidate_views(image, box) for box in boxes]
        geometry = [
            candidate_geometry(boxes[0], boxes[1], image.width, image.height),
            candidate_geometry(boxes[1], boxes[0], image.width, image.height),
        ]
        return {
            "query_id": str(row["query_id"]),
            "query": str(row["query"]),
            "views": views,
            "geometry": geometry,
            "label": label,
            "ious": ious,
        }


class E2BatchCollator:
    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, samples: Sequence[Mapping]) -> dict:
        images = []
        for sample in samples:
            for crop, marked in sample["views"]:
                images.extend((crop, marked))
        encoded = self.processor(
            text=[str(sample["query"]) for sample in samples],
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {
            "pixel_values": encoded["pixel_values"],
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "geometry": torch.tensor(
                [sample["geometry"] for sample in samples], dtype=torch.float32
            ),
            "labels": torch.tensor(
                [sample["label"] for sample in samples], dtype=torch.long
            ),
            "ious": torch.tensor(
                [sample["ious"] for sample in samples], dtype=torch.float32
            ),
            "query_ids": [str(sample["query_id"]) for sample in samples],
        }


class ClipCandidateVerifier(nn.Module):
    """A shared, permutation-equivariant scorer for exactly two candidates."""

    def __init__(
        self,
        clip_model: nn.Module,
        hidden_dim: int = 256,
        geometry_dim: int = GEOMETRY_DIM,
    ) -> None:
        super().__init__()
        self.clip = clip_model
        projection_dim = int(self.clip.config.projection_dim)
        candidate_input_dim = projection_dim * 8
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(candidate_input_dim),
            nn.Linear(candidate_input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim),
            nn.GELU(),
        )
        self.geometry_dim = int(geometry_dim)
        comparison_dim = hidden_dim * 3 + self.geometry_dim * 3
        self.shared_score = nn.Sequential(
            nn.LayerNorm(comparison_dim),
            nn.Linear(comparison_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.shared_quality = nn.Sequential(
            nn.LayerNorm(comparison_dim),
            nn.Linear(comparison_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def configure_adaptation(self, mode: str) -> None:
        if mode not in {"frozen", "last_block"}:
            raise ValueError(f"unknown adaptation mode: {mode}")
        for parameter in self.clip.parameters():
            parameter.requires_grad = False
        if mode == "last_block":
            modules = [
                self.clip.vision_model.encoder.layers[-1],
                self.clip.vision_model.post_layernorm,
                self.clip.text_model.encoder.layers[-1],
                self.clip.text_model.final_layer_norm,
                self.clip.visual_projection,
                self.clip.text_projection,
            ]
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def _clip_features(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clip_trainable = any(parameter.requires_grad for parameter in self.clip.parameters())
        context = torch.enable_grad() if clip_trainable and self.training else torch.no_grad()
        with context:
            image_features = self.clip.get_image_features(pixel_values=pixel_values)
            text_features = self.clip.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
        # transformers 5 returns BaseModelOutputWithPooling here, while 4.x
        # returned the projected tensor directly.
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        return F.normalize(image_features, dim=-1), F.normalize(text_features, dim=-1)

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        geometry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = input_ids.shape[0]
        if pixel_values.shape[0] != batch_size * 4:
            raise ValueError("expected crop and marked-scene views for two candidates")
        image_features, text_features = self._clip_features(
            pixel_values, input_ids, attention_mask
        )
        image_features = image_features.reshape(batch_size, 2, 2, -1)
        crop = image_features[:, :, 0]
        scene = image_features[:, :, 1]
        text = text_features[:, None, :].expand(-1, 2, -1)
        raw = torch.cat(
            (
                crop,
                scene,
                text,
                crop * text,
                scene * text,
                torch.abs(crop - text),
                torch.abs(scene - text),
                crop * scene,
            ),
            dim=-1,
        )
        candidate = self.candidate_encoder(raw)
        candidate_mean = candidate.mean(dim=1, keepdim=True).expand(-1, 2, -1)
        geometry_mean = geometry.mean(dim=1, keepdim=True).expand(-1, 2, -1)
        comparison = torch.cat(
            (
                candidate,
                candidate_mean,
                candidate - candidate_mean,
                geometry,
                geometry_mean,
                geometry - geometry_mean,
            ),
            dim=-1,
        )
        scores = self.shared_score(comparison).squeeze(-1)
        quality_logits = self.shared_quality(comparison).squeeze(-1)
        return scores, quality_logits


def verifier_loss(
    scores: torch.Tensor,
    quality_logits: torch.Tensor,
    labels: torch.Tensor,
    ious: torch.Tensor,
    *,
    quality_weight: float = 0.5,
    delta_weight: float = 0.25,
    safe_weight: float = 1.0,
    safe_margin: float = 0.75,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    listwise = F.cross_entropy(scores, labels)
    quality = F.smooth_l1_loss(torch.sigmoid(quality_logits), ious)
    predicted_delta = torch.tanh(scores[:, 1] - scores[:, 0])
    target_delta = ious[:, 1] - ious[:, 0]
    delta = F.smooth_l1_loss(predicted_delta, target_delta)

    good = ious.argmax(dim=1)
    bad = 1 - good
    batch = torch.arange(scores.shape[0], device=scores.device)
    severe = (ious.max(dim=1).values >= 0.5) & (ious.min(dim=1).values <= 1e-12)
    margins = scores[batch, good] - scores[batch, bad]
    safe = (
        F.relu(safe_margin - margins[severe]).mean()
        if severe.any()
        else scores.sum() * 0.0
    )
    total = listwise + quality_weight * quality + delta_weight * delta + safe_weight * safe
    return total, {
        "listwise": listwise.detach(),
        "quality": quality.detach(),
        "delta": delta.detach(),
        "safe": safe.detach(),
    }
