"""T3 (pure caption) prompts, per chat_template_style family.

T3 asks for a caption ONLY. It must not mention any referring expression, must
not ask an existence question, and must not request a box. That is what makes
T3 the unprompted-hallucination probe: whatever false claim appears comes from
the model itself, not from a question that planted it.

Because the prompt carries no expression, one image needs exactly ONE caption.
The caption is then scored against all 4 hallucination types of that image.
"""

# Plain instruct models (qwen2.5-vl, Qwen3-VL, Qwen3.5, llava-ov, InternVL3.5)
T3_PLAIN = (
    "Describe this image in two or three sentences.\n"
    "Mention only what is clearly visible. Do not guess or infer.\n"
    "Output the description only."
)

# think/answer RL-tuned families (LENS, Seg-zero, VisionReasoner, TreeVGR, ...)
T3_THINK_ANSWER = (
    "Describe this image in two or three sentences.\n"
    "Mention only what is clearly visible. Do not guess or infer.\n"
    "Output your thinking process in <think></think> and the final description "
    "in <answer></answer>."
)

# visual-rft uses a bare Question: form with no system prompt
T3_QUESTION = (
    "Question: Describe this image in two or three sentences.\n"
    "Mention only what is clearly visible. Do not guess or infer.\n"
    "Output the description only."
)

STYLE_TO_T3 = {
    "standard": T3_PLAIN,
    "lens": T3_THINK_ANSWER,
    "seg_zero": T3_THINK_ANSWER,
    "vision_reasoner": T3_THINK_ANSWER,
    "seg_r1": T3_THINK_ANSWER,
    "simple_seg": T3_PLAIN,
    "question": T3_QUESTION,
    "visual_rft": T3_QUESTION,
}


def t3_prompt_for_style(style: str) -> str:
    return STYLE_TO_T3.get(str(style or "standard"), T3_PLAIN)


# Qwen3.5 emits chain-of-thought with NO <think> tags unless thinking is
# disabled, so the reasoning arrives where the caption should be. Detect it:
# a silently-accepted reasoning dump corrupts metrics worse than a hard error.
_LEAK_PATTERNS = [
    r"\bthe user (?:wants|asks|is asking|requests|wanted)\b",
    # NOTE: patterns are matched against a lowercased string, so they must be
    # written in lowercase. Using "\bI " here silently never matches.
    r"\bi (?:need|should|will|must|have|am going) to\b",
    r"\bi(?:'ll| will| can) (?:describe|analyze|focus|avoid|start)\b",
    r"\blet me (?:analyze|look|examine|think|describe|start)\b",
    r"\*\*[^*]{0,40}(?:analysis|breakdown|observation|step|reasoning)[^*]{0,40}\*\*",
    r"\bstep[- ]by[- ]step\b",
    r"\bthe (?:description|caption|answer) (?:needs|should|must)\b",
    r"^\s*\d+\.\s*\*\*",
    r"\bfirst,? I\b",
    r"\bokay,? (?:so|let)\b",
]


def detect_thinking_leak(text: str) -> bool:
    """True when the text looks like reasoning about the task, not a caption."""
    import re
    t = str(text or "")
    low = t.lower()
    for pat in _LEAK_PATTERNS:
        if re.search(pat, low, re.M):
            return True
    return False


# model_type -> chat_template kwargs needed to get a plain caption.
_CHAT_KWARGS = {
    "qwen3_5": {"enable_thinking": False},
}


def chat_template_kwargs(model_type: str) -> dict:
    """Extra apply_chat_template kwargs for a model family.

    Qwen3.5 needs enable_thinking=False; every other family here needs nothing.
    Returning {} for unknown types keeps new models safe by default.

    IMPORTANT: splat these DIRECTLY into apply_chat_template(**kwargs). Wrapping
    them as chat_template_kwargs={...} is silently ignored by
    Qwen3VLProcessor.apply_chat_template — verified on Qwen3.5-9B, where only
    the direct form produces the empty '<think>\\n\\n</think>' prefix that turns
    thinking off.
    """
    return dict(_CHAT_KWARGS.get(str(model_type or "").lower(), {}))


def extract_caption(raw_text: str) -> dict:
    """Pull the caption out of a raw generation.

    Handles <answer>...</answer> wrappers and <think> blocks used by the
    RL-tuned families. Returns parse_failure=True only when nothing usable is
    left, so an empty caption is never silently scored as a clean caption.
    """
    import re
    t = str(raw_text or "").strip()
    m = re.search(r"<answer>(.*?)</answer>", t, re.S | re.I)
    if m:
        t = m.group(1).strip()
    else:
        # drop a leading think block if the model never closed an answer tag
        m2 = re.search(r"</think>(.*)$", t, re.S | re.I)
        if m2:
            t = m2.group(1).strip()
    t = re.sub(r"</?think>|</?answer>", " ", t, flags=re.I)
    t = re.sub(r"^\s*(Description|Caption|Answer)\s*[:：]\s*", "", t, flags=re.I)
    # a JSON-ish reply still carrying a description field
    if t.startswith("{"):
        m3 = re.search(r'"description"\s*:\s*"([^"]+)"', t)
        if m3:
            t = m3.group(1).strip()
    t = " ".join(t.split())
    # A tagged <think> block was already stripped above; a leak detected here is
    # untagged reasoning sitting where the caption belongs, so it is a parse
    # failure rather than a usable caption.
    leak = detect_thinking_leak(t)
    return {"caption": "" if leak else t,
            "parse_failure": leak or len(t) == 0,
            "thinking_leak": leak}
