"""Permutation-equivariant candidate scorer over reference proposal sets."""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .relation_supervision import (
    RELATION_PATTERNS,
    candidate_reference_features,
    normalized_box_features,
)


RELATION_NAMES = tuple(name for name, _ in RELATION_PATTERNS)
RELATION_TO_INDEX = {name: index for index, name in enumerate(RELATION_NAMES)}
CANDIDATE_DIM = 10
PAIR_DIM = 32


def read_e2b_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class E2bRelationDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping], *, training: bool, max_proposals: int = 5):
        self.rows = [dict(row) for row in rows if row.get("relation_selector_eligible")]
        self.training = training
        self.max_proposals = max_proposals
        if not self.rows:
            raise ValueError("relation dataset has no eligible rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        boxes = [row["baseline_bbox_xyxy"], row["challenger_bbox_xyxy"]]
        ious = [float(row["baseline_iou"]), float(row["challenger_iou"])]
        label = 1 if row["selector_action"] == "switch" else 0
        if self.training and random.random() < 0.5:
            boxes.reverse()
            ious.reverse()
            label = 1 - label
        width, height = int(row["image_width"]), int(row["image_height"])
        proposals = list(row["reference_proposals"])[: self.max_proposals]
        candidate = [normalized_box_features(box, width, height) for box in boxes]
        pair = [
            [
                candidate_reference_features(
                    box,
                    proposal["bbox_xyxy"],
                    float(proposal["score"]),
                    width,
                    height,
                )
                for proposal in proposals
            ]
            for box in boxes
        ]
        mask = [True] * len(proposals)
        for values in pair:
            values.extend([[0.0] * PAIR_DIM for _ in range(self.max_proposals - len(values))])
        mask.extend([False] * (self.max_proposals - len(mask)))
        best_reference = row.get("reference_best_index")
        if best_reference is None or int(best_reference) >= self.max_proposals:
            best_reference = -1
        return {
            "query_id": str(row["query_id"]),
            "task": str(row["task"]),
            "relation_index": torch.tensor(
                RELATION_TO_INDEX[str(row["relation"])], dtype=torch.long
            ),
            "candidate_features": torch.tensor(candidate, dtype=torch.float32),
            "pair_features": torch.tensor(pair, dtype=torch.float32),
            "reference_mask": torch.tensor(mask, dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.long),
            "ious": torch.tensor(ious, dtype=torch.float32),
            "reference_best_index": torch.tensor(int(best_reference), dtype=torch.long),
        }


class RelationSetVerifier(nn.Module):
    def __init__(self, hidden_dim: int = 64, relation_dim: int = 16) -> None:
        super().__init__()
        self.relation_embedding = nn.Embedding(len(RELATION_NAMES), relation_dim)
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(CANDIDATE_DIM),
            nn.Linear(CANDIDATE_DIM, 32),
            nn.GELU(),
        )
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(PAIR_DIM + relation_dim),
            nn.Linear(PAIR_DIM + relation_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.reference_attention = nn.Linear(hidden_dim, 1)
        comparison_dim = 32 + relation_dim + hidden_dim * 2
        self.shared_score = nn.Sequential(
            nn.LayerNorm(comparison_dim),
            nn.Linear(comparison_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.shared_quality = nn.Sequential(
            nn.LayerNorm(comparison_dim),
            nn.Linear(comparison_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        relation_index: torch.Tensor,
        candidate_features: torch.Tensor,
        pair_features: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        relation = self.relation_embedding(relation_index)
        relation_pairs = relation[:, None, None, :].expand(
            -1, 2, pair_features.shape[2], -1
        )
        pair_hidden = self.pair_encoder(torch.cat((pair_features, relation_pairs), dim=-1))
        reference_logits = self.reference_attention(pair_hidden).squeeze(-1)
        mask = reference_mask[:, None, :].expand(-1, 2, -1)
        reference_logits = reference_logits.masked_fill(~mask, -1e4)
        attention = F.softmax(reference_logits, dim=-1)
        pooled = (attention[..., None] * pair_hidden).sum(dim=2)
        maximum = pair_hidden.masked_fill(~mask[..., None], -1e4).max(dim=2).values
        candidate = self.candidate_encoder(candidate_features)
        relation_candidates = relation[:, None, :].expand(-1, 2, -1)
        comparison = torch.cat((candidate, relation_candidates, pooled, maximum), dim=-1)
        scores = self.shared_score(comparison).squeeze(-1)
        quality = self.shared_quality(comparison).squeeze(-1)
        return scores, quality, reference_logits


def relation_verifier_loss(
    scores: torch.Tensor,
    quality: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
    ious: torch.Tensor,
    reference_best_index: torch.Tensor,
    *,
    reference_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    listwise = F.cross_entropy(scores, labels)
    score_quality = F.smooth_l1_loss(torch.sigmoid(scores), ious)
    auxiliary_quality = F.smooth_l1_loss(torch.sigmoid(quality), ious)
    predicted_delta = torch.tanh(scores[:, 1] - scores[:, 0])
    target_delta = ious[:, 1] - ious[:, 0]
    delta = F.smooth_l1_loss(predicted_delta, target_delta)
    good = ious.argmax(dim=1)
    bad = 1 - good
    batch_indices = torch.arange(scores.shape[0], device=scores.device)
    severe = (ious.max(dim=1).values >= 0.5) & (ious.min(dim=1).values <= 1e-12)
    margins = scores[batch_indices, good] - scores[batch_indices, bad]
    safe = (
        F.relu(0.75 - margins[severe]).mean()
        if severe.any()
        else scores.sum() * 0.0
    )
    total = (
        0.2 * listwise
        + 2.0 * score_quality
        + 0.5 * auxiliary_quality
        + 0.5 * delta
        + safe
    )
    pieces = {
        "listwise": listwise.detach(),
        "score_quality": score_quality.detach(),
        "quality": auxiliary_quality.detach(),
        "delta": delta.detach(),
        "safe": safe.detach(),
    }
    valid = reference_best_index >= 0
    if valid.any():
        batch = torch.arange(scores.shape[0], device=scores.device)
        target_candidate_logits = reference_logits[batch, labels]
        reference_loss = F.cross_entropy(
            target_candidate_logits[valid], reference_best_index[valid]
        )
    else:
        reference_loss = scores.sum() * 0.0
    total = total + reference_weight * reference_loss
    pieces["reference"] = reference_loss.detach()
    return total, pieces
