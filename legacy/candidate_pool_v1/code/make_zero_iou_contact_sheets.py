#!/usr/bin/env python3
"""Render baseline/result/GT overlays for zero-IoU audit categories."""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"GT": "#27ae60", "BASE": "#e74c3c", "CAND": "#2980b9"}


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def parse_box(value: str) -> list[float] | None:
    parsed = json.loads(value)
    return [float(item) for item in parsed] if parsed else None


def render_cell(row: dict[str, str], image_dir: Path, width: int = 500, height: int = 390) -> Image.Image:
    source = Image.open(image_dir / row["image_filename"]).convert("RGB")
    image_height = 270
    scale = min((width - 20) / source.width, image_height / source.height)
    resized = source.resize((round(source.width * scale), round(source.height * scale)))
    cell = Image.new("RGB", (width, height), "white")
    offset_x = (width - resized.width) // 2
    offset_y = 8
    cell.paste(resized, (offset_x, offset_y))
    draw = ImageDraw.Draw(cell)

    for label, field in (("BASE", "baseline_box"), ("CAND", "result_box"), ("GT", "gt_box")):
        box = parse_box(row[field])
        if box is None:
            continue
        scaled = [
            offset_x + box[0] * scale,
            offset_y + box[1] * scale,
            offset_x + box[2] * scale,
            offset_y + box[3] * scale,
        ]
        draw.rectangle(scaled, outline=COLORS[label], width=4)
        draw.text((scaled[0] + 3, scaled[1] + 2), label, fill=COLORS[label], font=font(13))

    text_y = image_height + 16
    title = f"{row['base_sample_id']}  IoU {float(row['baseline_iou']):.3f} -> {float(row['result_iou']):.3f}"
    draw.text((8, text_y), title, fill="black", font=font(14))
    text_y += 20
    for line in textwrap.wrap(row["query"], width=58)[:2]:
        draw.text((8, text_y), line, fill="black", font=font(14))
        text_y += 18
    detail = (
        f"base={row['baseline_zero_box_class'] or '-'} "
        f"match={row['baseline_zero_match_category'] or '-'} "
        f"same-class distractors={row['same_category_distractors']}"
    )
    draw.text((8, text_y + 2), detail, fill="#333333", font=font(12))
    return cell


def sort_rows(rows: list[dict[str, str]], transition: str) -> list[dict[str, str]]:
    if transition == "valid_zero_recovered":
        return sorted(rows, key=lambda row: float(row["result_iou"]), reverse=True)
    if transition == "nonzero_regressed_to_zero":
        return sorted(rows, key=lambda row: float(row["baseline_iou"]), reverse=True)
    return sorted(rows, key=lambda row: (row["baseline_zero_box_class"], row["base_sample_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=ROOT / "analysis_outputs/zero_iou/samples.csv")
    parser.add_argument("--image-dir", type=Path, default=Path("/home/u2025141034/benchmark/benchmark_images"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis_outputs/zero_iou/contact_sheets")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    with args.samples.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = ("t2_vqa_grounding", "t4_caption_grounding")
    transitions = ("valid_zero_recovered", "valid_zero_unresolved", "nonzero_regressed_to_zero")
    for task in tasks:
        for transition in transitions:
            selected = [row for row in rows if row["task"] == task and row["transition"] == transition]
            selected = sort_rows(selected, transition)[: args.limit]
            if not selected:
                continue
            columns = 3
            cell_width, cell_height = 500, 390
            sheet = Image.new(
                "RGB",
                (cell_width * columns, cell_height * ((len(selected) + columns - 1) // columns)),
                "#dddddd",
            )
            for index, row in enumerate(selected):
                cell = render_cell(row, args.image_dir, cell_width, cell_height)
                sheet.paste(cell, ((index % columns) * cell_width, (index // columns) * cell_height))
            output = args.output_dir / f"{task}_{transition}.jpg"
            sheet.save(output, quality=92)
            print(output)


if __name__ == "__main__":
    main()
