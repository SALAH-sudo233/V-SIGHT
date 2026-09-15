import json
from pathlib import Path

RUN = Path(r"C:/Users/30796/Desktop/V-SIGHT-assets/data_quality_audit/auto_repair/runs/2000held_finegrained_v2")
src = json.loads((RUN / "source_2000held.json").read_text(encoding="utf-8"))
repaired = json.loads((RUN / "corrected_no_labels.json").read_text(encoding="utf-8"))
assert isinstance(repaired, list)
by_id = {x["content_id"]: x for x in repaired}
def cid(x):
    import hashlib
    payload = {k: x[k] for k in ("set", "sid", "ht", "img", "bbox", "pos", "neg")}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
output = []
changed = 0
for row in src:
    key = cid(row)
    if key in by_id:
        r = by_id[key]
        output.append({"set": r["set"], "sid": r["sid"], "ht": r["ht"], "img": r["img"], "bbox": r["bbox"], "pos": r["pos"], "neg": r["neg"], "content_id": key, "repair_version": "2000held_finegrained_v2"})
        changed += 1
    else:
        output.append({**row, "content_id": key, "repair_version": "2000held_finegrained_v2_original_unchanged"})
assert len(output) == 8000
assert len({x["content_id"] for x in output}) == 8000
assert changed == len(repaired) == 219
assert not any(x["set"] == "500dev" for x in output)
out = RUN / "refcocog_train2000.corrected_v2.json"
out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"path": str(out), "rows": len(output), "unique_content_id": len({x["content_id"] for x in output}), "repaired_rows": changed, "unchanged_rows": len(output)-changed, "excluded_500dev": True}, ensure_ascii=False))
