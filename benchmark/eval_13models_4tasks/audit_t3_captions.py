#!/usr/bin/env python3
"""Audit T3 captions for validity before trusting any T3 metric.

A near-zero caption hallucination rate can mean two very different things:
  - the model really does not assert the annotated false claim, or
  - the "caption" is not a caption at all (coordinates, empty string, raw CoT),
    in which case no object word can ever match and every rate collapses to 0.

Flags per record:
  coord_like : mostly digits/parens/commas -> a box, not a sentence
  too_short  : fewer than 4 alphabetic tokens
  has_tags   : leftover <think>/<answer> markup
  empty      : blank after extraction
"""
import json, os, re, collections

R = os.path.expanduser("~/benchmark/refcocog_eval_13models_4tasks_500/run_20260918_125802")

word_re = re.compile(r"[A-Za-z]{2,}")

def classify(cap):
    c = str(cap or "").strip()
    if not c:
        return "empty"
    if re.search(r"</?(think|answer)>", c, re.I):
        return "has_tags"
    words = word_re.findall(c)
    non_ws = re.sub(r"\s", "", c)
    digitish = sum(ch.isdigit() or ch in "(),.[]-" for ch in non_ws)
    if non_ws and digitish / len(non_ws) > 0.6:
        return "coord_like"
    if len(words) < 4:
        return "too_short"
    return "ok"

rows = {}
for mk in sorted(os.listdir(R)):
    d = os.path.join(R, mk)
    if not os.path.isdir(d):
        continue
    p = os.path.join(d, "records.jsonl")
    if not os.path.exists(p):
        continue
    c = collections.Counter()
    sample_bad = {}
    n = 0
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("task") != "t3_pure_caption" or r.get("error"):
            continue
        n += 1
        k = classify(r.get("caption"))
        c[k] += 1
        if k != "ok" and k not in sample_bad:
            sample_bad[k] = str(r.get("caption"))[:90]
    if n:
        rows[mk] = {"n": n, "counts": dict(c), "ok_rate": c["ok"] / n, "samples": sample_bad}

print(f"{'model':22s} {'n':>4s} {'ok%':>6s}  breakdown")
for mk, v in sorted(rows.items(), key=lambda kv: kv[1]["ok_rate"]):
    bad = {k: n for k, n in v["counts"].items() if k != "ok"}
    print(f"{mk:22s} {v['n']:4d} {v['ok_rate']*100:5.1f}%  {bad or '-'}")
    for k, s in v["samples"].items():
        print(f"{'':22s}      [{k}] {s!r}")

out = os.path.expanduser("~/t3_caption_audit.json")
json.dump(rows, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\nwrote", out)
