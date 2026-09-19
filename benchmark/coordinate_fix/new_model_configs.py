"""MODEL_CONFIGS entries for the three models added in the 14-model benchmark.

Prompt wording is copied verbatim from the existing "qwen2.5-vl-7b" entry in
eval_11models_refcocog_500_run.py so the new models are asked exactly what the
original 11 were asked. Only model_path / loader hints differ.

Sizes chosen to match the 7B-8B band of the existing suite:
    llava-ov-7b      7B   LlavaOnevisionForConditionalGeneration (native)
    InternVL3.5-8B   8B   AutoModel + .chat()  (needs internvl_adapter)
    Qwen3.5-9B       9B   native, REQUIRES enable_thinking=False
"""
import pathlib

HOME = pathlib.Path.home()

T1 = ('Task: Decide whether the referring expression strictly matches a visible target in the image.\n'
      'Referring expression: "{expr}"\n'
      "The object identity, number, attributes, colors, and spatial relations must all match.\n"
      "Answer with exactly one word: yes or no.")

T2 = ('Task: Decide whether the referring expression strictly matches a visible target in the image, then localize it if it exists.\n'
      'Referring expression: "{expr}"\n'
      "If the target exists, return exactly one bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
      "If the target does not exist, return exactly: not found.\n"
      "Do not output explanations.")

# T3 carries NO {expr}: pure captioning is the unprompted-hallucination probe.
T3 = ("Describe this image in two or three sentences.\n"
      "Mention only what is clearly visible. Do not guess or infer.\n"
      "Output the description only.")

T4 = ('Task: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
      'Step 1 — Describe the image concisely.\n'
      'Step 2 — Is there "{expr}" in the image?\n'
      'The object identity, number, attributes, colors, and spatial relations must all match.\n'
      'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
      'If no, output "not found".\n'
      'Return JSON only:\n'
      '{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}\n'
      '{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}\n'
      'Do not output explanations.')

SYSTEM = ("You are a careful visual grounding assistant. "
          "Use only visible image evidence. Follow the requested output format exactly.")

NEW_MODEL_CONFIGS = {
    "llava-ov-7b": {
        "name": "llava-onevision-qwen2-7b-ov-hf",
        # HF-converted weights: the lmms-lab original needs the llava package
        # (LlavaQwenForCausalLM is not in transformers, auto_map is null).
        "model_path": str(HOME / ".cache/huggingface/hub/models--llava-hf--llava-onevision-qwen2-7b-ov-hf/snapshots/0d50680527681998e456c7b78950205bedd8a068"),
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": SYSTEM,
        "t1_prompt": T1, "t2_prompt": T2, "t3_prompt": T3, "t4_prompt": T4,
        "chat_template_style": "standard",
        "param_band": "7B",
        # llava outputs [0,1] float normalized coords; parse_bbox_output already
        # handles them (auto-multiplies when max(abs(x)) <= 1.5). Explicit declaration
        # kept for consistency.
        "bbox_coordinate_system": "norm_01",
    },
    "InternVL3.5-8B": {
        "name": "InternVL3_5-8B",
        # Materialized directory of REAL files, not the HF snapshot symlinks:
        # transformers resolves remote-code relative imports next to the
        # resolved file, and the blob cache uses hashed filenames.
        "model_path": str(HOME / "models/InternVL3_5-8B"),
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": SYSTEM,
        "t1_prompt": T1, "t2_prompt": T2, "t3_prompt": T3, "t4_prompt": T4,
        "chat_template_style": "internvl",   # -> InternVLAdapter, not generate_single
        "loader": "internvl_adapter",
        "param_band": "8B",
        # InternVL outputs [0,1000]×[0,1000] normalized coords (confirmed via 40-sample
        # diagnosis: mean IoU 0.039 raw vs 0.566 after norm_1000 rescale, 24/40 ≥0.5).
        "bbox_coordinate_system": "norm_1000",
    },
    "Qwen3.5-9B": {
        "name": "Qwen3.5-9B",
        "model_path": str(HOME / ".cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": SYSTEM,
        "t1_prompt": T1, "t2_prompt": T2, "t3_prompt": T3, "t4_prompt": T4,
        "chat_template_style": "standard",
        # Splat DIRECTLY into apply_chat_template. As chat_template_kwargs={...}
        # it is silently ignored and the model emits raw reasoning instead.
        "chat_template_extra_kwargs": {"enable_thinking": False},
        "param_band": "9B",
        # Qwen3.5 outputs [0,1000]×[0,1000] normalized coords, same as Qwen3-VL
        # (confirmed via diagnosis: mean IoU 0.035 raw vs 0.809 after rescale, 6/7 ≥0.5).
        "bbox_coordinate_system": "norm_1000",
    },
}

# Qwen3.5-9B is EXCLUDED. Its checkpoint carries 333 model.visual.* weight keys,
# but transformers 5.12.1 builds Qwen3_5ForConditionalGeneration with no vision
# tower (vision attrs absent, generate() rejects pixel_values/image_grid_thw), so
# every image is silently dropped and any T2/T4 number would be text-only
# guessing dressed up as grounding. Re-test if a later transformers wires the
# vision path; until then it must not enter the table.
NEW_MODEL_KEYS = ["llava-ov-7b", "InternVL3.5-8B"]
