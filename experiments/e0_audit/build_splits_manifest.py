#!/usr/bin/env python3
"""
E0 deliverable: build the frozen split manifest (binding_v2_splits.json) and
assert zero-overlap isolation across every data role.

Roadmap refs: sec 0.1 P0 ("真正独立的训练/校准/开发/最终测试边界"), sec 4.1 E0
("划分: 训练/校准/测试交集为零"), sec 5.1 (data-role table), sec 2.2 (fit/calib
must be decoupled, not just independent of test).

Design decisions grounded in what we actually verified on the servers:
  * pilot_300  (goldenapple human gold, vsight_human_review_event_v1): 300 coco imgs.
  * dev500     (11-model eval, run_500_semantic_strict):              500 coco imgs.
  * bench1996  (refcocog_1996_heldout.manual_v2, MEASUREMENT-ONLY):  1996 coco imgs.
  Verified pairwise-disjoint: pilot∩bench=0, pilot∩dev500=0 (this session).

We keep the IMAGE (coco id) as the isolation unit. Everything derived from one
coco image (all expressions, all model outputs, all counterfactual suffixes,
near-dup variants) travels together and lands in exactly one split.

The dev500 pool is where B1a/B1c currently fit fusion weights AND pick thresholds
on the SAME calib images (sec 2.2 problem). We split dev500 deterministically into
fusion-fit / calibration / development so weight-fit and threshold-selection no
longer share rows. Assignment is a stable hash of the coco id -> reproducible,
seed-free, order-independent.

Roles that require NEW annotation (train-seed, feedback R1/R2) are declared in the
manifest as PLANNED (empty id lists) so downstream code can assert against them
without them existing yet.
"""
import json, hashlib, os, datetime, sys

HERE = os.path.dirname(os.path.abspath(__file__))

def load_ids(fn):
    p = os.path.join(HERE, fn)
    return sorted({int(x) for x in open(p).read().split() if x.strip()})

def stable_bucket(coco_id, salt, n_buckets):
    """Deterministic bucket in [0, n_buckets) from a salted hash of the id."""
    h = hashlib.sha256(f"{salt}:{coco_id}".encode()).hexdigest()
    return int(h, 16) % n_buckets

def split_dev500(ids):
    """
    Deterministically partition the 500 dev image groups into three
    mutually independent roles. Proportions chosen per roadmap sec 5.1
    (fusion-fit 100 / calibration 150 / development 150 style ratios,
    here scaled to the 500 pool we actually have):
        fusion-fit  : 40%  (~200) fit fusion / small router weights
        calibration : 30%  (~150) pick thresholds & action-stop params ONLY
        development : 30%  (~150) scheme & checkpoint selection
    Assignment via stable hash so it is reproducible and reviewer-checkable.
    """
    fit, cal, dev = [], [], []
    for cid in ids:
        b = stable_bucket(cid, salt="vsight_binding_v2_dev500", n_buckets=10)
        if b < 4:      fit.append(cid)
        elif b < 7:    cal.append(cid)
        else:          dev.append(cid)
    return sorted(fit), sorted(cal), sorted(dev)

def main():
    pilot     = load_ids("pilot_image_ids.txt") if os.path.exists(os.path.join(HERE,"pilot_image_ids.txt")) else load_ids("../../golden_review/pilot_image_ids.txt")
    dev500    = load_ids("dev500_ids.txt")
    bench1996 = load_ids("bench1996_ids.txt")

    # ---- hard isolation assertions (fail loud) ----
    problems = []
    def disjoint(a, b, na, nb):
        inter = set(a) & set(b)
        if inter:
            problems.append(f"OVERLAP {na}∩{nb} = {len(inter)}: {sorted(inter)[:10]}")
    disjoint_pairs = [
        (pilot, dev500, "pilot", "dev500"),
        (pilot, bench1996, "pilot", "bench1996"),
        (dev500, bench1996, "dev500", "bench1996"),
    ]
    for a, b, na, nb in disjoint_pairs:
        disjoint(a, b, na, nb)

    fit, cal, dev = split_dev500(dev500)
    # within-dev500 partitions must be a clean, complete, disjoint cover
    if sorted(fit + cal + dev) != dev500:
        problems.append("dev500 sub-splits do not cover the pool exactly")
    for (a, na), (b, nb) in [((fit,"fusion_fit"),(cal,"calibration")),
                             ((fit,"fusion_fit"),(dev,"development")),
                             ((cal,"calibration"),(dev,"development"))]:
        if set(a) & set(b):
            problems.append(f"dev500 sub-split OVERLAP {na}∩{nb}")

    manifest = {
        "schema_version": "vsight_binding_v2_splits_v1",
        "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "isolation_unit": "coco_image_id",
        "note": "All expressions/model-outputs/counterfactual-suffixes/near-dups of one "
                "coco image belong to exactly one role. dev500 sub-split by stable "
                "sha256 hash (salt=vsight_binding_v2_dev500), seed-free & reproducible.",
        "roles": {
            # AVAILABLE NOW (real ids)
            "pilot_human_dev": {
                "status": "available",
                "source": "goldenapple .worktrees/chinese-review/outputs/review/pilot_300 (human, vsight_human_review_event_v1)",
                "purpose": "E2 human evidence-oracle dev; diagnostics; protocol tuning",
                "count": len(pilot), "coco_ids": pilot,
                "forbid": "do NOT promote to train or final test; do NOT tune final thresholds here",
            },
            "dev500_fusion_fit": {
                "status": "available",
                "source": "dev500 pool, stable-hash bucket<4",
                "purpose": "fit score-fusion / small router weights (B1a/B1c)",
                "count": len(fit), "coco_ids": fit,
                "forbid": "no threshold selection here; no training reflow",
            },
            "dev500_calibration": {
                "status": "available",
                "source": "dev500 pool, stable-hash bucket 4..6",
                "purpose": "pick decision thresholds & action-stop params ONLY (frozen after)",
                "count": len(cal), "coco_ids": cal,
                "forbid": "no weight fitting here; no training reflow",
            },
            "dev500_development": {
                "status": "available",
                "source": "dev500 pool, stable-hash bucket 7..9",
                "purpose": "scheme & checkpoint selection",
                "count": len(dev), "coco_ids": dev,
                "forbid": "not a held-out test; do not repeatedly reselect on final test",
            },
            "final_test_bench1996": {
                "status": "sealed",
                "source": "refcocog_1996_heldout.manual_v2.json (MEASUREMENT-ONLY)",
                "purpose": "final confirmation only, one-shot after freeze",
                "count": len(bench1996), "coco_ids": bench1996,
                "forbid": "never read labels early, never fit/select on it, never mine examples",
            },
            # PLANNED (need new annotation; declared empty so downstream can assert)
            "train_seed": {
                "status": "planned",
                "source": "NEW annotation via review_app_zh.py; 300 independent image groups",
                "purpose": "R0 relation supervision",
                "count": 0, "coco_ids": [],
                "forbid": "must be disjoint from ALL above; no protected images",
            },
            "train_feedback_R1": {
                "status": "planned", "purpose": "flywheel R1; per-strategy independent selection",
                "count": 0, "coco_ids": [], "forbid": "never mine from eval errors",
            },
            "train_feedback_R2": {
                "status": "planned", "purpose": "flywheel R2",
                "count": 0, "coco_ids": [], "forbid": "never mine from eval errors",
            },
        },
        "verified_disjoint_pairs": [
            "pilot_human_dev ∩ final_test_bench1996 = 0",
            "pilot_human_dev ∩ dev500(all) = 0",
            "dev500(all) ∩ final_test_bench1996 = 0",
            "dev500_fusion_fit ∩ dev500_calibration ∩ dev500_development = pairwise 0, exact cover",
        ],
    }

    out = os.path.join(HERE, "manifests", "binding_v2_splits.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False)
    with open(out, "w", encoding="utf-8") as f:
        f.write(payload)
    manifest_sha = hashlib.sha256(payload.encode()).hexdigest()

    print("=== E0 SPLIT MANIFEST ===")
    for role, d in manifest["roles"].items():
        print(f"  {role:24s} status={d['status']:10s} n={d['count']}")
    print(f"\nmanifest sha256: {manifest_sha}")
    print(f"written: {out}")
    if problems:
        print("\n!!! ISOLATION ASSERTIONS FAILED !!!")
        for p in problems:
            print("   ", p)
        sys.exit(1)
    print("\nAll isolation assertions PASSED (all splits pairwise disjoint on coco image id).")

if __name__ == "__main__":
    main()
