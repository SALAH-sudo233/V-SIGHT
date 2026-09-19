"""T3 (pure caption) record builder and metrics for the V-SIGHT 4-task benchmark.

Reads the SAME annotation units the repaired benchmark already carries:
    item["chair_annotation"]["target_eval_units"]        -> annotated false claim
    item["chair_annotation"]["free_caption_eval_units"]  -> caption whitelist

T3 asks ONLY for a free-form caption: no existence question, no bounding box.
Three things are measured per query:
  1. caption_target_hallucination - the caption spontaneously asserts the
     annotated false claim of the negative query (intrinsic, target-specific).
  2. target_covered - the caption mentions the positive target (alias-aware).
  3. extrinsic_objects - objects named that are not in the allowed whitelist.

Records whose annotation units are missing are marked units_available=False and
scored as None. They stay in the denominator and are reported separately, so a
missing annotation can never be silently counted as a clean caption.
"""
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

BOH_TYPES = {"object", "co_occurrence"}
ROH_TYPES = {"attribute", "relation"}


def tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(text or "").lower()) if t]


def contains_phrase(caption_tokens: set, phrase: str) -> bool:
    toks = tokens(phrase)
    return bool(toks) and all(t in caption_tokens for t in toks)


def _units(item: Dict[str, Any]) -> tuple:
    ca = (item or {}).get("chair_annotation") or {}
    if not isinstance(ca, dict):
        return {}, {}
    t = ca.get("target_eval_units") or {}
    f = ca.get("free_caption_eval_units") or {}
    return (t if isinstance(t, dict) else {}), (f if isinstance(f, dict) else {})


def target_hallucination(caption: str, target_units: Dict[str, Any]) -> Dict[str, Any]:
    """Does the caption assert the annotated false claim?

    Only units flagged count_as_hallucination_if_mentioned are scored. Blank
    unit fields fall back to positive_head_object (the real schema leaves
    `object` empty for attribute/relation units).
    """
    ct = set(tokens(caption))
    head = str(target_units.get("positive_head_object") or "")
    matched = []
    for unit in target_units.get("hallucination_units") or []:
        if not unit.get("count_as_hallucination_if_mentioned", True):
            continue
        utype = str(unit.get("type") or target_units.get("eval_type") or "")
        if utype in BOH_TYPES:
            hit = contains_phrase(ct, str(unit.get("object") or ""))
        elif utype == "attribute":
            attr = str(unit.get("attribute") or "")
            obj = str(unit.get("object") or head or "")
            hit = contains_phrase(ct, attr) and (not obj or contains_phrase(ct, obj))
        elif utype == "relation":
            subj = str(unit.get("subject") or head or "")
            hit = all(contains_phrase(ct, p) for p in
                      (subj, str(unit.get("relation") or ""), str(unit.get("target_object") or "")))
        else:
            hit = False
        if hit:
            matched.append(unit)
    return {"caption_target_hallucination": bool(matched),
            "caption_target_hallucination_unit_count": len(matched),
            "caption_target_hallucination_units": matched}


def target_coverage(caption: str, target_units: Dict[str, Any],
                    free_units: Dict[str, Any]) -> bool:
    """Alias-aware: 'armchair' counts as covering 'chair'."""
    ct = set(tokens(caption))
    aliases = free_units.get("allowed_aliases") or {}
    wanted = [str(x) for x in (target_units.get("target_coverage_units") or [])]
    if not wanted:
        head = str(target_units.get("positive_head_object") or "")
        wanted = [head] if head else []
    if not wanted:
        return False
    for name in wanted:
        names = [name] + [str(a) for a in (aliases.get(name) or [])]
        # annotators list one lexical choice; captions freely use a near synonym
        # ("couch" for "chair"), so soft synonyms count as coverage too.
        names += sorted(SOFT_SYNONYMS.get(str(name).lower(), set()))
        if any(contains_phrase(ct, n) for n in names):
            return True
    return False


# Conservative closed vocabulary of object nouns.
# Words that double as colours/materials/adjectives are deliberately EXCLUDED
# (orange, rose, peach, silver, ...): "an orange shirt" is not a fruit.
KNOWN_OBJECT_WORDS = {
    "giraffe", "zebra", "elephant", "dog", "cat", "horse", "sheep", "cow", "bear",
    "bird", "car", "truck", "bus", "train", "boat", "airplane", "bicycle",
    "motorcycle", "table", "chair", "couch", "sofa", "bed", "toilet", "sink",
    "television", "laptop", "keyboard", "phone", "book", "clock",
    "vase", "scissors", "umbrella", "handbag", "suitcase", "bottle", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "pizza",
    "cake", "donut", "sandwich", "broccoli", "carrot", "ottoman", "fireplace",
    "lamp", "rug", "carpet", "curtain", "cabinet", "cushion", "pillow",
    "controller", "remote", "picture", "painting", "frame", "woman", "man",
    "person", "child", "boy", "girl", "hearth", "kite", "surfboard", "skateboard",
    "mouse",
}

# Only flagged when used as a bare noun; as a modifier they are colours or
# materials, so they are dropped unless nothing follows them.
COLOUR_OR_MATERIAL_WORDS = {"orange", "rose", "peach", "silver", "gold", "cream",
                            "olive", "plum", "salmon", "chocolate", "lime"}

# Tokens that cannot be the head noun of a colour modifier, so a colour word
# directly before one of these is being used as a noun itself.
FUNCTION_WORDS = {
    "and", "or", "but", "with", "without", "of", "in", "on", "at", "to", "for",
    "from", "by", "near", "next", "behind", "above", "below", "under", "over",
    "beside", "between", "is", "are", "was", "were", "sits", "sit", "sitting",
    "stands", "standing", "lies", "lying", "rests", "resting", "placed",
    "the", "a", "an", "there", "while", "as", "that", "which", "into", "onto",
}

# A modifier that changes the referent: "spray bottle" is not a drink bottle,
# "computer mouse" is not an animal. Presence of the modifier suppresses the head.
COMPOUND_MODIFIERS = {
    "bottle": {"spray", "water", "squeeze", "soap", "detergent", "perfume"},
    "mouse": {"computer", "wireless", "usb"},
    "frame": {"picture", "photo", "window", "door", "bed"},
    "remote": {"tv", "television", "game"},
    "table": {"coffee", "side", "dining", "end", "pool", "picnic"},
}

# Bidirectional soft synonyms: annotators list one, captions often use another.
SOFT_SYNONYMS = {
    "chair": {"couch", "sofa", "armchair", "seat", "recliner", "stool", "bench"},
    "couch": {"chair", "sofa", "armchair", "seat", "settee"},
    "sofa": {"chair", "couch", "armchair", "seat", "settee"},
    "television": {"tv", "monitor", "screen"},
    "tv": {"television", "monitor", "screen"},
    "monitor": {"television", "tv", "screen"},
    "screen": {"television", "tv", "monitor"},
    "phone": {"cellphone", "smartphone", "mobile"},
    "cellphone": {"phone", "smartphone", "mobile"},
    "rug": {"carpet", "mat"},
    "carpet": {"rug", "mat"},
    "fireplace": {"hearth", "fire"},
    "hearth": {"fireplace", "fire"},
    "seat": {"chair", "couch", "sofa", "armchair"},
    "armchair": {"chair", "couch", "sofa", "seat"},
    "picture": {"painting", "frame", "artwork", "photo"},
    "painting": {"picture", "frame", "artwork"},
    "cushion": {"pillow"},
    "pillow": {"cushion"},
    "lamp": {"light"},
    "desk": {"table"},
    "table": {"desk"},
}


def _expand_soft(names: set) -> set:
    out = set(names)
    for n in list(names):
        out |= SOFT_SYNONYMS.get(n, set())
    return out


def _allowed_vocab(free_units: Dict[str, Any]) -> set:
    """Whitelist = allowed_objects + aliases + uncertain_objects + soft synonyms.

    uncertain_objects are whitelisted on purpose: the annotator could not
    confirm them either way, so naming one is not evidence of a hallucination.
    """
    allowed = set()
    aliases = free_units.get("allowed_aliases") or {}
    for o in (free_units.get("allowed_objects") or []):
        allowed.update(tokens(o))
        for a in (aliases.get(str(o)) or []):
            allowed.update(tokens(a))
    for group in aliases.values():
        for a in group or []:
            allowed.update(tokens(a))
    for u in (free_units.get("uncertain_objects") or []):
        allowed.update(tokens(u))
    for rel in (free_units.get("allowed_relations") or []):
        if isinstance(rel, dict):
            allowed.update(tokens(rel.get("subject")))
            allowed.update(tokens(rel.get("object")))
    return _expand_soft(allowed)


def unlisted_objects(caption: str, free_units: Dict[str, Any],
                     reference_text: str = "") -> List[str]:
    """Object nouns named in the caption that no evidence source mentions.

    EXPLORATORY ONLY. An allowed_objects list enumerates what the annotator
    chose to record, not everything visible in the image, so an unlisted object
    is NOT proof of hallucination. Reported separately from the oracle-validated
    caption_target_hallucination metric and never mixed into it.

    reference_text (the benchmark's image_description) acts as a second,
    independent description: anything it mentions is suppressed.
    """
    allowed = _allowed_vocab(free_units)
    if not allowed:
        return []
    allowed |= _expand_soft(set(tokens(reference_text)))
    toks = tokens(caption)
    tokset = set(toks)
    hits = set()
    for i, t in enumerate(toks):
        if t not in KNOWN_OBJECT_WORDS or t in allowed:
            continue
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        # Colour/material word acting as a modifier ("orange shirt") is not an
        # object. It only counts as a modifier when a NOUN follows: in
        # "an orange and a banana" the next token is a function word, so
        # `orange` is the fruit.
        if t in COLOUR_OR_MATERIAL_WORDS and nxt and nxt not in FUNCTION_WORDS:
            continue
        # compound head whose modifier changes the referent ("spray bottle")
        prev = toks[i - 1] if i else ""
        if prev in COMPOUND_MODIFIERS.get(t, set()):
            continue
        hits.add(t)
    return sorted(hits)


def _cos(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    da = sum(v * v for v in a.values()) ** 0.5
    db = sum(v * v for v in b.values()) ** 0.5
    return (num / (da * db)) if da and db else 0.0


def amber_cosine_score(pred: str, ref: str, positive_text: str = "",
                       negative_text: str = "") -> Dict[str, Any]:
    p = Counter(tokens(pred))
    base = _cos(p, Counter(tokens(ref))) if ref else 0.0
    pos = _cos(p, Counter(tokens(positive_text))) if positive_text else 0.0
    neg = _cos(p, Counter(tokens(negative_text))) if negative_text else 0.0
    return {"amber_cosine": base,
            "amber_cosine_penalized": base * (1.0 - max(0.0, neg - pos)),
            "pos_sim": pos, "neg_sim": neg}


def build_t3_record(caption: str, item: Dict[str, Any], latency: float, raw: str,
                    parse_failure: bool = False) -> Dict[str, Any]:
    """One T3 record. Has no pred_bbox_xyxy / pred_exists by construction."""
    t_units, f_units = _units(item)
    htype = str((item or {}).get("hallucination_type") or "")
    available = bool(t_units.get("hallucination_units"))
    rec: Dict[str, Any] = {
        "task": "t3_pure_caption",
        "sample_id": (item or {}).get("sample_id", ""),
        "base_sample_id": (item or {}).get("base_sample_id", ""),
        "hallucination_type": htype,
        "hallucination_group": "BOH" if htype in BOH_TYPES else ("ROH" if htype in ROH_TYPES else "UNKNOWN"),
        "caption": caption,
        "raw_output": raw,
        "latency_sec": latency,
        "parse_failure": bool(parse_failure),
        "units_available": available,
    }
    if available:
        rec.update(target_hallucination(caption, t_units))
        rec["target_covered"] = target_coverage(caption, t_units, f_units)
        rec["unlisted_objects"] = unlisted_objects(
            caption, f_units, (item or {}).get("image_description") or "")
    else:
        rec.update({"caption_target_hallucination": None,
                    "caption_target_hallucination_unit_count": None,
                    "caption_target_hallucination_units": [],
                    "target_covered": None, "unlisted_objects": []})
    rec.update(amber_cosine_score(caption,
                                  (item or {}).get("image_description") or (item or {}).get("positive_text") or "",
                                  (item or {}).get("positive_text") or "",
                                  (item or {}).get("negative_text") or ""))
    return rec


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    v = [x for x in xs if isinstance(x, (int, float))]
    return sum(v) / len(v) if v else None


def _rate(recs: List[Dict[str, Any]], key: str) -> Optional[float]:
    scored = [r for r in recs if r.get(key) is not None]
    return (sum(1 for r in scored if r[key]) / len(scored)) if scored else None


def compute_t3_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Full-denominator T3 metrics.

    num_queries counts every record. Rates are over records with usable
    annotation units (n_scored), which is reported alongside so the two
    denominators are never conflated.
    """
    recs = [r for r in records if r.get("task") == "t3_pure_caption"]
    n = len(recs)
    by_type: Dict[str, Dict[str, Any]] = {}
    for ht in ["object", "co_occurrence", "attribute", "relation"]:
        sub = [r for r in recs if r.get("hallucination_type") == ht]
        scored = [r for r in sub if r.get("caption_target_hallucination") is not None]
        cnt = sum(1 for r in scored if r["caption_target_hallucination"])
        by_type[ht] = {"n": len(sub), "n_scored": len(scored), "count": cnt,
                       "rate": (cnt / len(scored)) if scored else None}

    def grp(types: set) -> Optional[float]:
        return _rate([r for r in recs if r.get("hallucination_type") in types],
                     "caption_target_hallucination")

    boh, roh = grp(BOH_TYPES), grp(ROH_TYPES)
    rates = [v["rate"] for v in by_type.values() if v["rate"] is not None]
    scored_unlisted = [r for r in recs if r.get("units_available")]
    return {
        "num_queries": n,
        "units_unavailable": sum(1 for r in recs if not r.get("units_available")),
        "caption_hallucination_rate": _rate(recs, "caption_target_hallucination"),
        "hallucination_rate_by_type": by_type,
        "macro_caption_hallucination_rate": (sum(rates) / len(rates)) if rates else None,
        "boh_caption_hallucination_rate": boh,
        "roh_caption_hallucination_rate": roh,
        "roh_minus_boh_error_gap": (roh - boh) if (boh is not None and roh is not None) else None,
        "target_coverage": _rate(recs, "target_covered"),
        # EXPLORATORY: allowed_objects is what the annotator recorded, not an
        # exhaustive inventory of the image, so this is not a hallucination rate.
        "unlisted_object_rate_exploratory": (sum(1 for r in scored_unlisted if r.get("unlisted_objects")) / len(scored_unlisted)) if scored_unlisted else None,
        "avg_amber_cosine": _mean([r.get("amber_cosine") for r in recs]),
        "avg_amber_cosine_penalized": _mean([r.get("amber_cosine_penalized") for r in recs]),
        "parse_failures": sum(1 for r in recs if r.get("parse_failure")),
        "avg_latency_sec": _mean([r.get("latency_sec") for r in recs]),
    }
