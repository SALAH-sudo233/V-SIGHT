#!/usr/bin/env python3
"""
Build refcocog_2000_expanded.json — expand from 500 to 1500 images using RefCOCOg.
Data source: refs(google).p (RefCOCOg) + instances.json (COCO bbox)
For each new image: keep 1 human-annotated chosen + VLM generates rejected + 3 hallucination types.
"""
import json, os, sys, re, time, base64, pickle, random
from datetime import datetime
from collections import defaultdict
from openai import OpenAI

API_KEY = os.environ.get("ROH_VCD_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
BASE_URL = os.environ.get("ROH_VCD_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = "qwen3.7-plus"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXISTING_FILE = os.path.join(_SCRIPT_DIR, "refcocog_500_expanded.json")
OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "refcocog_2000_expanded.json")
REFCOCOG_PKL = "/home/u2025141034/models/LENS/data/refcoco/refs(google).p"
INSTANCES = "/home/u2025141034/models/LENS/data/refcoco/instances.json"
TRAIN_DIR = "/home/u2025141034/models/LENS/data/refcoco/train2014"
BENCH_IMG_DIR = os.path.join(_SCRIPT_DIR, "benchmark_images")

random.seed(42)
if not API_KEY:
    raise RuntimeError("ROH_VCD_API_KEY or DASHSCOPE_API_KEY is required")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ── Step 1: Build bbox lookup ──
print("[1/5] Loading bbox data from instances.json ...")
with open(INSTANCES) as f:
    coco_data = json.load(f)
ann_to_bbox = {}
for ann in coco_data['annotations']:
    x, y, w, h = ann['bbox']
    ann_to_bbox[ann['id']] = [x, y, x+w, y+h]

# ── Step 2: Load RefCOCOg and select new images ──
print("[2/5] Loading RefCOCOg references ...")
with open(REFCOCOG_PKL, 'rb') as f:
    refs_google = pickle.load(f)

# Build image → refs map (train split only)
img_refs = defaultdict(list)
for ref in refs_google:
    if ref['split'] != 'train': continue
    fn = re.sub(r'_\d+\.jpg$', '.jpg', ref.get('file_name', ''))
    if fn: img_refs[fn].append(ref)

print(f"  RefCOCOg train images: {len(img_refs)}")

# Find existing images
with open(EXISTING_FILE) as f:
    existing = json.load(f)
used_imgs = set(s['image_filename'] for s in existing)
print(f"  Already used: {len(used_imgs)}")

# New images with bbox and file available
new_candidates = []
for fn, refs in img_refs.items():
    if fn in used_imgs: continue
    # Check image exists
    img_path = os.path.join(TRAIN_DIR, fn)
    if not os.path.exists(img_path): continue
    # Check at least one ref has bbox
    has_bbox = False
    best_ref = None
    for ref in refs:
        ann_id = ref.get('ann_id')
        if ann_id in ann_to_bbox:
            has_bbox = True
            best_ref = ref
            break
    if has_bbox:
        new_candidates.append((fn, best_ref, ann_to_bbox[best_ref['ann_id']]))

random.shuffle(new_candidates)
TARGET = 1000  # 500 existing + 1000 new = 1500
selected = new_candidates[:TARGET]
print(f"  Selected: {len(selected)} new images")

# ── Step 3: Build base entries ──
print("[3/5] Building base entries ...")
counter = 500000  # start counter after existing entries

base_entries = []
for fn, ref, bbox_xyxy in selected:
    # Pick best sentence as chosen
    sentences = ref.get('sentences', [])
    chosen = sentences[0]['sent'] if sentences else f"object in image"
    
    base_entries.append({
        "source": "RefCOCOg",
        "image_filename": fn,
        "chosen": chosen,
        "rejected": "",  # To be filled by VLM
        "sample_id": f"hallu_{counter:06d}_{fn.replace('.jpg', '')}",
        "hallucination_type": "object",
        "positive_text": chosen,
        "negative_text": "",  # To be filled by VLM
        "positive_bbox": bbox_xyxy,
        "positive_bbox_format": "xyxy",
        "gt_bbox_xyxy": bbox_xyxy,
        "chosen_bbox_xyxy": bbox_xyxy,
        "hallucination_subtype": "other",
        "classification_confidence": None,
        "expansion_origin": "RefCOCOg",
        "expansion_method": "original_reclassified",
        "image_description": "",  # To be filled by VLM
        "chair_annotation": {},
    })
    counter += 1

print(f"  Base entries: {len(base_entries)}")

# ── Step 4: VLM generation ──
print("[4/5] Calling qwen3.7-plus for VLM generation ...")

SYSTEM_PROMPT = """You are a visual grounding annotation assistant. Analyze the image and generate:

1. A rejected description: an object NOT in the image (object hallucination)
2. Three hallucination negatives: co-occurrence, attribute, relation

Return exactly this JSON (no markdown):
{
  "visible_objects": [{"canonical": "obj", "aliases": ["a1"], "attributes": ["attr1"]}],
  "visible_relations": [{"subject": "s", "relation": "r", "object": "o"}],
  "rejected_text": "the NOUN in the image",
  "image_description": "One sentence describing the image.",
  "co_occurrence": {
    "negative_text": "...", "positive_text": "...",
    "hallucination_object": "...", "shared_object": "...", "note": "..."
  },
  "attribute": {
    "negative_text": "...", "positive_text": "...",
    "hallucination_object": "...", "wrong_attribute": "...", "correct_attribute": "...", "note": "..."
  },
  "relation": {
    "negative_text": "...", "positive_text": "...",
    "subject": "...", "object": "...", "wrong_relation": "...", "correct_relation": "...", "note": "..."
  }
}

IMPORTANT:
- rejected must be an object NOT in the image
- co_occurrence: object NOT in image but commonly co-occurs with visible objects
- attribute: visible object with WRONG attribute
- relation: visible objects with WRONG spatial relation
- Return pure JSON only, no markdown."""

def encode_image(img_path):
    with open(img_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def call_vlm(img_path):
    b64 = encode_image(img_path)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": "Analyze this image and generate outputs as specified. Return pure JSON."}
                    ]}
                ],
                max_tokens=4096, temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"): raw = raw.split("\n",1)[1]; raw = raw[:-3].strip() if raw.endswith("```") else raw
            return json.loads(raw)
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(3)
    raise RuntimeError("VLM call failed after 3 attempts")

# Initialize expanded with existing entries
expanded = list(existing)
existing_base_ids = {s['sample_id'] for s in expanded if s.get('expansion_method') == 'original_reclassified'}

# Load existing output if resuming
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        expanded = json.load(f)
    processed_imgs = set(s['image_filename'] for s in expanded if s.get('expansion_method','').startswith('vlm_generated'))
    print(f"  Resuming from {len(expanded)} records, {len(processed_imgs)} images done")

processed_base = set()
for i, entry in enumerate(base_entries):
    fn = entry['image_filename']
    img_path = os.path.join(TRAIN_DIR, fn)
    
    if fn in processed_base:
        continue
    
    base_id = entry['sample_id']
    if base_id in existing_base_ids:
        continue
    
    print(f"  [{i+1}/{len(base_entries)}] {fn}")
    
    try:
        vlm_out = call_vlm(img_path)
        
        # Update base entry with VLM-generated rejected and description
        entry['rejected'] = vlm_out.get('rejected_text', '')
        entry['negative_text'] = vlm_out.get('rejected_text', '')
        entry['image_description'] = vlm_out.get('image_description', '')
        entry['chair_annotation'] = {
            "version": "chair_v1",
            "annotation_source": "vlm_image",
            "image_visible_objects": [
                {"canonical": o['canonical'], "aliases": o.get('aliases',[]), "attributes": o.get('attributes',[]),
                 "source": "vlm_image", "confidence": "high"}
                for o in vlm_out.get('visible_objects', [])
            ],
            "image_visible_relations": [
                {"subject": r['subject'], "relation": r['relation'], "object": r['object'], "confidence": "high"}
                for r in vlm_out.get('visible_relations', [])
            ],
            "annotator_model": MODEL,
            "annotated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "needs_human_review": True,  # VLM-generated annotations
        }
        
        # Add original
        if base_id not in {s['sample_id'] for s in expanded}:
            expanded.append(entry)
        
        # Add 3 hallucination types
        for htype in ['co_occurrence', 'attribute', 'relation']:
            try:
                hdata = vlm_out[htype]
                suffix = {'co_occurrence': 'cooc', 'attribute': 'attr', 'relation': 'rel'}[htype]
                new_s = {
                    'source': 'RefCOCOg',
                    'image_filename': fn,
                    'chosen': hdata['positive_text'],
                    'rejected': hdata['negative_text'],
                    'sample_id': f"{base_id}_{suffix}",
                    'hallucination_type': htype,
                    'positive_text': hdata['positive_text'],
                    'negative_text': hdata['negative_text'],
                    'positive_bbox': entry['positive_bbox'],
                    'positive_bbox_format': 'xyxy',
                    'gt_bbox_xyxy': entry['gt_bbox_xyxy'],
                    'chosen_bbox_xyxy': entry['chosen_bbox_xyxy'],
                    'hallucination_subtype': 'other',
                    'classification_confidence': None,
                    'expansion_origin': 'RefCOCOg',
                    'expansion_method': f'vlm_generated_{htype}',
                    'image_description': entry['image_description'],
                    'chair_annotation': entry.get('chair_annotation', {}),
                }
                expanded.append(new_s)
                print(f"    + {htype}")
            except KeyError as e:
                print(f"    x {htype}: missing {e}")
        
        processed_base.add(fn)
        
        # Save incrementally
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(expanded, f, ensure_ascii=False, indent=2)
        print(f"    -> {len(expanded)} total records")
        
        time.sleep(1)  # rate limit
        
    except Exception as e:
        print(f"  ERROR: {e}")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(expanded, f, ensure_ascii=False, indent=2)
        time.sleep(3)

print(f"\n[5/5] Done! {len(expanded)} records in {OUTPUT_FILE}")

# Final stats
total_imgs = len(set(s['image_filename'] for s in expanded))
print(f"  Unique images: {total_imgs}")
sources = defaultdict(int)
for s in expanded:
    sources[s.get('hallucination_type','?')] += 1
print(f"  Types: {dict(sources)}")
