#!/usr/bin/env python3
"""
Build refcocog_500_base.json — 500 unique RefCOCOg images with full annotation format.

Sources:
- 100 existing images from refcocog_100_expanded.json (original entries)
- 314 new RefCOCOg images from benchmark.json (full format, not in expanded set)
- 86 new RefCOCOg images from mixed_benchmark.json (mapped to full format)

Output: refcocog_500_base.json (500 entries, one per unique image)
"""

import json
import os
import re
from collections import OrderedDict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EXPANDED_FILE = os.path.join(_SCRIPT_DIR, "refcocog_100_expanded.json")
BENCHMARK_FILE = os.path.join(_SCRIPT_DIR, "benchmark.json")
MIXED_FILE = os.path.join(_SCRIPT_DIR, "mixed_benchmark.json")
IMAGE_DIR = os.path.join(_SCRIPT_DIR, "benchmark_images")
OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "refcocog_500_base.json")

# ── Step 1: Load existing 100 images' original entries ─────────────────────

with open(EXPANDED_FILE, 'r', encoding='utf-8') as f:
    expanded = json.load(f)

existing_originals = [s for s in expanded if s.get("expansion_method") == "original_reclassified"]
existing_images = {s["image_filename"] for s in existing_originals}

print(f"[1] Existing originals: {len(existing_originals)} entries, {len(existing_images)} unique images")

# ── Step 2: Load new entries from benchmark.json ───────────────────────────

with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
    benchmark = json.load(f)

benchmark_refcoco = [s for s in benchmark if s.get("source") == "RefCOCOg"]
benchmark_new = [s for s in benchmark_refcoco if s["image_filename"] not in existing_images]
# Deduplicate by image_filename (keep first entry per image)
benchmark_new_dedup = {}
for s in benchmark_new:
    fn = s["image_filename"]
    if fn not in benchmark_new_dedup:
        benchmark_new_dedup[fn] = s
benchmark_new_list = list(benchmark_new_dedup.values())
benchmark_new_images = {s["image_filename"] for s in benchmark_new_list}

print(f"[2] benchmark.json RefCOCOg: {len(benchmark_refcoco)} entries, "
      f"{len(benchmark_new_list)} new unique images (not in expanded)")

# ── Step 3: Load additional entries from mixed_benchmark.json ──────────────

with open(MIXED_FILE, 'r', encoding='utf-8') as f:
    mixed = json.load(f)

mixed_refcoco = [s for s in mixed if s.get("source") == "RefCOCOg"]
# Filter out images already in expanded or benchmark_new sets
already_used = existing_images | benchmark_new_images
mixed_new = [s for s in mixed_refcoco if s["image_filename"] not in already_used]
# Deduplicate by image
mixed_new_dedup = {}
for s in mixed_new:
    fn = s["image_filename"]
    if fn not in mixed_new_dedup:
        mixed_new_dedup[fn] = s
mixed_new_list = list(mixed_new_dedup.values())

print(f"[3] mixed_benchmark.json RefCOCOg: {len(mixed_refcoco)} source entries, "
      f"{len(mixed_new_list)} new unique images")

# ── Step 4: Select exactly 400 new images (314 bench + 86 mixed) ──────────

NEED_NEW = 500 - len(existing_originals)
print(f"\nNeed {NEED_NEW} additional images to reach 500")

# Take all benchmark_new (up to NEED_NEW)
selected_bench = benchmark_new_list[:min(len(benchmark_new_list), NEED_NEW)]
remaining = NEED_NEW - len(selected_bench)

# Fill remaining from mixed
selected_mixed = mixed_new_list[:min(len(mixed_new_list), remaining)]
remaining_after = remaining - len(selected_mixed)

print(f"  - from benchmark.json: {len(selected_bench)}")
print(f"  - from mixed_benchmark.json: {len(selected_mixed)}")
print(f"  - remaining unfilled: {remaining_after}")

if remaining_after > 0:
    print(f"  WARNING: Could only get {500 - remaining_after} images!")
    remaining_after = 0

# ── Step 5: Build full-format entries ──────────────────────────────────────

def build_from_benchmark_entry(src: dict) -> dict:
    """benchmark.json entries already have the full format — use as-is."""
    return dict(src)

def build_from_mixed_entry(src: dict) -> dict:
    """
    mixed_benchmark.json entries have a simpler format.
    Map fields to match benchmark.json format.
    """
    fn = src["image_filename"]
    # Generate sample_id: extract COCO ID from filename
    coco_id_match = re.search(r'COCO_train2014_(\d+)', fn)
    if coco_id_match:
        coco_num = int(coco_id_match.group(1))
        sample_id = f"hallu_{coco_num:06d}_{fn.replace('.jpg', '')}"
    else:
        # Fallback: use hash of filename
        import hashlib
        h = hashlib.md5(fn.encode()).hexdigest()[:6]
        sample_id = f"hallu_{h}_{fn.replace('.jpg', '')}"

    bbox = src.get("bbox", [0, 0, 0, 0])
    chosen = src.get("chosen", "")
    rejected = src.get("rejected", "")
    hallucinated_entity = src.get("hallucinated_entity", "")

    entry = {
        "source": "RefCOCOg",
        "image_filename": fn,
        "chosen": chosen,
        "rejected": rejected,
        "sample_id": sample_id,
        "hallucination_type": "object",
        "positive_text": chosen,
        "negative_text": rejected,
        "positive_bbox": bbox,
        "positive_bbox_format": "xyxy",
        "gt_bbox_xyxy": bbox,
        "chosen_bbox_xyxy": bbox,
        "classification_confidence": None,
        "expansion_origin": "base",
        "expansion_method": "original_reclassified",
        "image_description": "",
        "chair_annotation": {
            "version": "chair_v1",
            "annotation_source": "llm_text_or_vlm_image_text",
            "image_visible_objects": [],
            "image_visible_relations": [],
            "free_caption_eval_units": {
                "coverage_objects": [],
                "allowed_objects": [],
                "allowed_aliases": {},
                "allowed_relations": [],
                "uncertain_objects": []
            },
            "target_eval_units": {
                "eval_type": "object",
                "positive_head_object": hallucinated_entity,
                "positive_related_objects": [hallucinated_entity],
                "positive_attributes": [],
                "positive_relations": [],
                "negative_head_object": hallucinated_entity,
                "negative_related_objects": [hallucinated_entity],
                "negative_attributes": [],
                "negative_relations": [],
                "shared_objects": [hallucinated_entity],
                "target_coverage_units": [hallucinated_entity],
                "hallucination_units": [{
                    "type": "object",
                    "object": hallucinated_entity,
                    "attribute": "",
                    "subject": "",
                    "relation": "",
                    "target_object": "",
                    "count_as_hallucination_if_mentioned": True,
                    "note": f"Mentioning '{hallucinated_entity}' counts as object hallucination."
                }]
            },
            "quality_flags": {
                "description_is_sparse": True,
                "needs_human_review": False,
                "reason": ""
            },
            "annotator_model": "",
            "annotated_at": "",
            "used_image": True
        }
    }
    return entry

# Convert all selected entries to full format
final_entries = []

# 1. Existing originals (keep as-is)
for s in existing_originals:
    # Ensure required fields exist
    entry = dict(s)
    # Make sure hallucination_type is 'object' (or keep original)
    if "difficulty_tier" in entry and "hallucination_type" not in entry:
        entry["hallucination_type"] = entry["difficulty_tier"]
    final_entries.append(entry)

# 2. New from benchmark.json
for s in selected_bench:
    final_entries.append(build_from_benchmark_entry(s))

# 3. New from mixed_benchmark.json
for s in selected_mixed:
    final_entries.append(build_from_mixed_entry(s))

# ── Step 6: Final validation ───────────────────────────────────────────────

# Check unique images
final_images = set(s["image_filename"] for s in final_entries)
dupes = len(final_entries) - len(final_images)

# Check all images exist
missing_imgs = [fn for fn in final_images if not os.path.exists(os.path.join(IMAGE_DIR, fn))]

print(f"\n── Final Summary ──")
print(f"Total entries: {len(final_entries)}")
print(f"Unique images: {len(final_images)}")
print(f"Duplicate images: {dupes}")
print(f"Missing image files: {len(missing_imgs)}")
if missing_imgs:
    print(f"  Sample missing: {missing_imgs[:5]}")

# Check hallucination_type distribution
from collections import Counter
ht = Counter(s.get("hallucination_type", "unknown") for s in final_entries)
print(f"Hallucination types: {dict(ht)}")

# ── Step 7: Save ───────────────────────────────────────────────────────────

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(final_entries, f, ensure_ascii=False, indent=2)

print(f"\nSaved to: {OUTPUT_FILE}")
print(f"File size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")
