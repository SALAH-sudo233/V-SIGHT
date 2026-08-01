"""CLIP dataset that augments each candidate with explicit relation context."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image
from torch.utils.data import Dataset

from .clip_verifier import candidate_views
from .e2_verifier import candidate_geometry
from .relation_supervision import relation_context_features


E2B_GEOMETRY_DIM = 18 + 85


class E2bClipRelationDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping], image_root: Path, *, training: bool):
        self.rows = [dict(row) for row in rows if row.get("relation_selector_eligible")]
        self.image_root = Path(image_root)
        self.training = training
        if not self.rows:
            raise ValueError("E2b CLIP relation dataset has no eligible rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with Image.open(self.image_root / Path(row["image_filename"]).name) as opened:
            image = opened.convert("RGB")
        boxes = [row["baseline_bbox_xyxy"], row["challenger_bbox_xyxy"]]
        ious = [float(row["baseline_iou"]), float(row["challenger_iou"])]
        label = 1 if row["selector_action"] == "switch" else 0
        if self.training and random.random() < 0.5:
            boxes.reverse()
            ious.reverse()
            label = 1 - label
        proposals = row["reference_proposals"]
        geometry = []
        for candidate, other in ((boxes[0], boxes[1]), (boxes[1], boxes[0])):
            geometry.append(
                [
                    *candidate_geometry(candidate, other, image.width, image.height),
                    *relation_context_features(
                        candidate,
                        proposals,
                        str(row["relation"]),
                        image.width,
                        image.height,
                    ),
                ]
            )
        return {
            "query_id": str(row["query_id"]),
            "query": str(row["query"]),
            "views": [candidate_views(image, box) for box in boxes],
            "geometry": geometry,
            "label": label,
            "ious": ious,
        }
