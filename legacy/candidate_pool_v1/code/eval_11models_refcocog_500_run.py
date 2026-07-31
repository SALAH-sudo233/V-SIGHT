#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_6models_refcocog.py — Evaluate 6 models on refcocog_100_expanded.json (400 samples).
Tasks: T1 (discriminative VQA), T2 (VQA + Grounding), T4 (Caption + Grounding).

Each model uses its own prompt style as defined in prompt.md.
All 6 models are Qwen2.5-VL-7B based; loading is handled via trust_remote_code.

Usage:
  CUDA_VISIBLE_DEVICES=0 python eval_6models_refcocog.py --model all
  CUDA_VISIBLE_DEVICES=0 python eval_6models_refcocog.py --model qwen2.5-vl-7b
"""

from __future__ import annotations

import sys; sys.path.insert(0, "/home/u2025141034/models/LENS/src")
import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import time
import traceback
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_PATH = "/home/u2025141034/benchmark/repaired/refcocog_500_dev.semantic_strict.json"
IMAGE_DIR = "/home/u2025141034/benchmark/benchmark_images"
OUTPUT_ROOT = "/home/u2025141034/benchmark/refcocog_eval_11models_500_repaired"
LEGACY_BENCHMARK_PATH = "/home/u2025141034/benchmark/refcocog_500_expanded.json"
LEGACY_RECORDS_ROOT = "/home/u2025141034/benchmark/refcocog_eval_6models_500/run_500_sdpa"

QWEN25_VL_PATH = "/home/u2025141034/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"

# ---------------------------------------------------------------------------
# Model configs: path, prompt templates for T1/T2/T4
# ---------------------------------------------------------------------------

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # -----------------------------------------------------------------------
    "qwen2.5-vl-7b": {
        "name": "Qwen2.5-VL-7B-Instruct",
        "model_path": QWEN25_VL_PATH,
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": (
            "You are a careful visual grounding assistant. "
            "Use only visible image evidence. Follow the requested output format exactly."
        ),
        "t1_prompt": (
            'Task: Decide whether the referring expression strictly matches a visible target in the image.\n'
            'Referring expression: "{expr}"\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "Answer with exactly one word: yes or no."
        ),
        "t2_prompt": (
            'Task: Decide whether the referring expression strictly matches a visible target in the image, then localize it if it exists.\n'
            'Referring expression: "{expr}"\n'
            "If the target exists, return exactly one bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If the target does not exist, return exactly: not found.\n"
            "Do not output explanations."
        ),
        "t4_prompt": (
            'Task: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Return JSON only:\n'
            '{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}\n'
            '{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}\n'
            'Do not output explanations.'
        ),
        "chat_template_style": "standard",  # uses processor.apply_chat_template
    },
    # -----------------------------------------------------------------------
    "LENS": {
        "name": "LENS (qwen2p5_refcoco)",
        "model_path": "/home/u2025141034/LENS/pretrained/qwen2p5_refcoco",
        "processor_path": QWEN25_VL_PATH,  # LENS uses base Qwen2.5 processor
        "extra_pythonpath": "/home/u2025141034/LENS/src",
        "system_prompt": "You are a helpful assistant.",
        # T1: yes/no discrimination in LENS think/answer style
        "t1_prompt": (
            'Task: Decide whether the referring expression strictly matches a visible target in the image.\n'
            'Referring expression: "{expr}"\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "Output your thinking process in <think></think> and final answer in <answer></answer>.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding with bbox, LENS style — explicit existence check first
        "t2_prompt": (
            'Task: First determine whether "{expr}" strictly matches a visible target in the image.\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "If the target exists, localize it with a bounding box. If not, say not found.\n"
            "Please:\n"
            "1. Check all objects in the image against \"{expr}\"\n"
            "2. Decide whether a strict match exists (matching identity, attributes, and spatial relations)\n"
            "3. If a match exists, select the closest one and provide precise bounding box coordinates\n"
            "4. If no match exists, explicitly reject with \"not found\"\n"
            "Format your response as:\n"
            "<think>\n"
            "[Your analysis: which objects are present, whether they match or don't match]\n"
            "</think><answer>[x1, y1, x2, y2]</answer>\n"
            "If no matching object exists: <answer>not found</answer>"
        ),
        # T4: caption + existence check + bbox in LENS think/answer style
        "t4_prompt": (
            'Task: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Output your thinking process in <think></think> and final answer in <answer></answer>.\n'
            'If target exists: <answer>{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}</answer>\n'
            'If not: <answer>{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}</answer>'
        ),
        "chat_template_style": "lens",
    },
    # -----------------------------------------------------------------------
    "visual-rft": {
        "name": "visual-rft-7b",
        "model_path": "/home/u2025141034/visual-rft-7b",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": None,  # visual-rft doesn't use system prompt (Question: format)
        # T1: yes/no in visual-rft style
        "t1_prompt": (
            'Question: Decide whether the referring expression strictly matches a visible target in the image.\n'
            'Referring expression: "{expr}"\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "Output the thinking process in <think></think> and your answer in <answer></answer>.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding, visual-rft style — existence check FIRST, standard bbox format
        "t2_prompt": (
            'Question: First check whether "{expr}" strictly matches a visible target in the image.\n'
            "If it exists, output its bounding box. If it does NOT exist, say not found.\n"
            'Output the thinking process in <think></think> and your answer in <answer></answer>.\n'
            'If the target exists: <answer>[x1, y1, x2, y2]</answer> (absolute pixel coordinates)\n'
            'If not: <answer>not found</answer>'
        ),
        # T4: caption + existence check + bbox in visual-rft think/answer style
        "t4_prompt": (
            'Question: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Output the thinking process in <think></think> and your answer in <answer></answer>.\n'
            'If target exists: <answer>{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}</answer>\n'
            'If not: <answer>{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}</answer>'
        ),
        "chat_template_style": "visual_rft",
    },
    # -----------------------------------------------------------------------
    "Seg-R1": {
        "name": "Seg-R1-7B",
        "model_path": "/home/u2025141034/seg-r1-7b",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": None,
        # T1: simple yes/no in Seg-R1 style
        "t1_prompt": (
            'Segment: Decide whether "{expr}" strictly matches a visible target in the image.\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            'Answer with exactly "yes" or "no".'
        ),
        # T2: simple grounding, Seg-R1 style from prompt.md
        "t2_prompt": (
            'Segment the main object: "{expr}".\n'
            "Provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If no matching object exists, output exactly: not found."
        ),
        # T4: caption + existence check + bbox in Seg-R1 style
        "t4_prompt": (
            'Task: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Return JSON:\n'
            '{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}\n'
            '{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}'
        ),
        "chat_template_style": "seg_r1",
    },
    # -----------------------------------------------------------------------
    "Seg-zero": {
        "name": "Seg-Zero-7B",
        "model_path": "/home/u2025141034/Seg-Zero-7B",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": None,
        # T1: yes/no in Seg-zero think/answer style
        "t1_prompt": (
            "Please decide if '{expr}' matches a visible target in the image.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding, Seg-zero style from prompt.md
        "t2_prompt": (
            "Please find '{expr}' with bbox and points.\n"
            "Compare the difference between objects and find the most closely matched one.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            "Output the one bbox and center points of two largest inscribed circles inside the interested object in JSON format.\n"
            "i.e., <think> thinking process here </think><answer>{{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}}</answer>\n"
            "If no matching object exists, output <answer>not found</answer>."
        ),
        # T4: caption + existence check + bbox in Seg-zero think/answer style
        "t4_prompt": (
            "Please first describe the image in one concise sentence, then check whether '{expr}' strictly matches a visible target.\n"
            "Step 1 — Describe the image concisely.\n"
            "Step 2 — Is there '{expr}' in the image?\n"
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If no, output \"not found\".\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            "If target exists: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"yes\", \"bbox\":[x1,y1,x2,y2]}}</answer>\n"
            "If not: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"no\", \"bbox\":\"not found\"}}</answer>"
        ),
        "chat_template_style": "seg_zero",
    },
    # -----------------------------------------------------------------------
    "VisionReasoner": {
        "name": "VisionReasoner-7B",
        "model_path": "/home/u2025141034/VisionReasoner-7B",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": None,
        # T1: yes/no in VisionReasoner think/answer style
        "t1_prompt": (
            "Please decide if '{expr}' matches a visible target in the image.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding, VisionReasoner style from prompt.md
        "t2_prompt": (
            "Please find '{expr}' with bboxs and points.\n"
            "Compare the difference between object(s) and find the most closely matched object(s).\n"
            "Output the thinking process in <think> </think>\n"
            "and final answer in <answer> </answer> tags.\n"
            "Output the bbox(es) and point(s) inside the interested object(s) in JSON format.\n"
            "i.e.\n"
            "<think>\n"
            "thinking process here\n"
            "</think>\n"
            '<answer>[{{"bbox_2d": [x1,y1,x2,y2], "point_2d": [x,y]}}]</answer>\n'
            "If no matching object exists, output <answer>not found</answer>."
        ),
        # T4: caption + existence check + bbox in VisionReasoner think/answer style
        "t4_prompt": (
            "Please first describe the image in one concise sentence, then check whether '{expr}' strictly matches a visible target.\n"
            "Step 1 — Describe the image concisely.\n"
            "Step 2 — Is there '{expr}' in the image?\n"
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If no, output \"not found\".\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            "If target exists: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"yes\", \"bbox\":[x1,y1,x2,y2]}}</answer>\n"
            "If not: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"no\", \"bbox\":\"not found\"}}</answer>"
        ),
        "chat_template_style": "vision_reasoner",
    },
    # -----------------------------------------------------------------------
    "SimpleSeg": {
        "name": "SimpleSeg-Qwen2.5-VL-7B",
        "model_path": "/home/u2025141034/models/SimpleSeg-Qwen2.5-VL-7B",
        "processor_path": QWEN25_VL_PATH,  # SimpleSeg uses opencua tokenizer which needs Qwen2.5-VL processor
        "extra_pythonpath": None,
        "system_prompt": (
            "You are a careful visual grounding assistant. " 
            "Use only visible image evidence. Follow the requested output format exactly." 
        ),
        "t1_prompt": (
            "Task: Decide whether the referring expression strictly matches a visible target in the image.\n" 
            "Referring expression: \"{expr}\"\n" 
            "The object identity, number, attributes, colors, and spatial relations must all match.\n" 
            "Answer with exactly one word: yes or no." 
        ),
        "t2_prompt": (
            "Task: Decide whether the referring expression strictly matches a visible target in the image, then localize it if it exists.\n" 
            "Referring expression: \"{expr}\"\n" 
            "If the target exists, return exactly one bounding box in absolute image pixels as [x1, y1, x2, y2].\n" 
            "If the target does not exist, return exactly: not found.\n" 
            "Do not output explanations." 
        ),
        "t4_prompt": (
            "Task: First describe the image in one concise sentence. Then check whether \"{expr}\" strictly matches a visible target.\n" 
            "If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n" 
            "If no, output \"not found\".\n" 
            "Do not output explanations." 
        ),
        "chat_template_style": "standard",
    },
    # -----------------------------------------------------------------------
    "TreeVGR": {
        "name": "TreeVGR-7B",
        "model_path": "/home/u2025141034/models/TreeVGR-7B",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": (
            "You are a careful visual grounding assistant. " 
            "Use only visible image evidence. Follow the requested output format exactly." 
        ),
        "t1_prompt": (
            "Please decide if \"{expr}\" matches a visible target in the image.\n" 
            "The object identity, number, attributes, colors, and spatial relations must all match.\n" 
            "Output the thinking process in «think» and final answer in <answer></answer> tags.\n" 
            "The answer should be exactly \"yes\" or \"no\"." 
        ),
        "t2_prompt": (
            "Please find \"{expr}\" in the image.\n" 
            "If the target exists, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n" 
            "If no matching object exists, output not found.\n" 
            "Output the thinking process in «think» and final answer in <answer></answer> tags.\n" 
            "If target exists: <answer>[x1,y1,x2,y2]</answer>\n" 
            "If not: <answer>not found</answer>" 
        ),
        "t4_prompt": (
            "Please first describe the image in one concise sentence, then check whether \"{expr}\" strictly matches a visible target.\n" 
            "Output the thinking process in «think» and final answer in <answer></answer> tags.\n" 
            "If target exists: <answer>[x1,y1,x2,y2]</answer>\n" 
            "If not: <answer>not found</answer>" 
        ),
        "chat_template_style": "vision_reasoner",
    },
"Orsta-7B": {
        "name": "Orsta-7B (Qwen2.5-VL-7B + V-Triune RL)",
        "model_path": "/home/u2025141034/models/Orsta-7B",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": (
            "You are a careful visual grounding assistant. "
            "Use only visible image evidence. Follow the requested output format exactly."
        ),
        "t1_prompt": (
            'Task: Decide whether the referring expression strictly matches a visible target in the image.\n'
            'Referring expression: "{expr}"\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "Answer with exactly one word: yes or no."
        ),
        "t2_prompt": (
            'Task: Decide whether the referring expression strictly matches a visible target in the image, then localize it if it exists.\n'
            'Referring expression: "{expr}"\n'
            "If the target exists, return exactly one bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If the target does not exist, return exactly: not found.\n"
            "Do not output explanations."
        ),
        "t4_prompt": (
            'Task: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Return JSON only:\n'
            '{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}\n'
            '{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}\n'
            'Do not output explanations.'
        ),
        "chat_template_style": "standard",  # uses processor.apply_chat_template
    },
    # -----------------------------------------------------------------------
    "LENS": {
        "name": "LENS (qwen2p5_refcoco)",
        "model_path": "/home/u2025141034/LENS/pretrained/qwen2p5_refcoco",
        "processor_path": QWEN25_VL_PATH,  # LENS uses base Qwen2.5 processor
        "extra_pythonpath": "/home/u2025141034/LENS/src",
        "system_prompt": "You are a helpful assistant.",
        # T1: yes/no discrimination in LENS think/answer style
        "t1_prompt": (
            'Task: Decide whether the referring expression strictly matches a visible target in the image.\n'
            'Referring expression: "{expr}"\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "Output your thinking process in <think></think> and final answer in <answer></answer>.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding with bbox, LENS style — explicit existence check first
        "t2_prompt": (
            'Task: First determine whether "{expr}" strictly matches a visible target in the image.\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "If the target exists, localize it with a bounding box. If not, say not found.\n"
            "Please:\n"
            "1. Check all objects in the image against \"{expr}\"\n"
            "2. Decide whether a strict match exists (matching identity, attributes, and spatial relations)\n"
            "3. If a match exists, select the closest one and provide precise bounding box coordinates\n"
            "4. If no match exists, explicitly reject with \"not found\"\n"
            "Format your response as:\n"
            "<think>\n"
            "[Your analysis: which objects are present, whether they match or don't match]\n"
            "</think><answer>[x1, y1, x2, y2]</answer>\n"
            "If no matching object exists: <answer>not found</answer>"
        ),
        # T4: caption + existence check + bbox in LENS think/answer style
        "t4_prompt": (
            'Task: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Output your thinking process in <think></think> and final answer in <answer></answer>.\n'
            'If target exists: <answer>{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}</answer>\n'
            'If not: <answer>{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}</answer>'
        ),
        "chat_template_style": "lens",
    },
    # -----------------------------------------------------------------------
    "visual-rft": {
        "name": "visual-rft-7b",
        "model_path": "/home/u2025141034/visual-rft-7b",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": None,  # visual-rft doesn't use system prompt (Question: format)
        # T1: yes/no in visual-rft style
        "t1_prompt": (
            'Question: Decide whether the referring expression strictly matches a visible target in the image.\n'
            'Referring expression: "{expr}"\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "Output the thinking process in <think></think> and your answer in <answer></answer>.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding, visual-rft style — existence check FIRST, standard bbox format
        "t2_prompt": (
            'Question: First check whether "{expr}" strictly matches a visible target in the image.\n'
            "If it exists, output its bounding box. If it does NOT exist, say not found.\n"
            'Output the thinking process in <think></think> and your answer in <answer></answer>.\n'
            'If the target exists: <answer>[x1, y1, x2, y2]</answer> (absolute pixel coordinates)\n'
            'If not: <answer>not found</answer>'
        ),
        # T4: caption + existence check + bbox in visual-rft think/answer style
        "t4_prompt": (
            'Question: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Output the thinking process in <think></think> and your answer in <answer></answer>.\n'
            'If target exists: <answer>{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}</answer>\n'
            'If not: <answer>{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}</answer>'
        ),
        "chat_template_style": "visual_rft",
    },
    # -----------------------------------------------------------------------
    "Seg-R1": {
        "name": "Seg-R1-7B",
        "model_path": "/home/u2025141034/seg-r1-7b",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": None,
        # T1: simple yes/no in Seg-R1 style
        "t1_prompt": (
            'Segment: Decide whether "{expr}" strictly matches a visible target in the image.\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            'Answer with exactly "yes" or "no".'
        ),
        # T2: simple grounding, Seg-R1 style from prompt.md
        "t2_prompt": (
            'Segment the main object: "{expr}".\n'
            "Provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If no matching object exists, output exactly: not found."
        ),
        # T4: caption + existence check + bbox in Seg-R1 style
        "t4_prompt": (
            'Task: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Return JSON:\n'
            '{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}\n'
            '{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}'
        ),
        "chat_template_style": "seg_r1",
    },
    # -----------------------------------------------------------------------
    "Seg-zero": {
        "name": "Seg-Zero-7B",
        "model_path": "/home/u2025141034/Seg-Zero-7B",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": None,
        # T1: yes/no in Seg-zero think/answer style
        "t1_prompt": (
            "Please decide if '{expr}' matches a visible target in the image.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding, Seg-zero style from prompt.md
        "t2_prompt": (
            "Please find '{expr}' with bbox and points.\n"
            "Compare the difference between objects and find the most closely matched one.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            "Output the one bbox and center points of two largest inscribed circles inside the interested object in JSON format.\n"
            "i.e., <think> thinking process here </think><answer>{{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}}</answer>\n"
            "If no matching object exists, output <answer>not found</answer>."
        ),
        # T4: caption + existence check + bbox in Seg-zero think/answer style
        "t4_prompt": (
            "Please first describe the image in one concise sentence, then check whether '{expr}' strictly matches a visible target.\n"
            "Step 1 — Describe the image concisely.\n"
            "Step 2 — Is there '{expr}' in the image?\n"
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If no, output \"not found\".\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            "If target exists: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"yes\", \"bbox\":[x1,y1,x2,y2]}}</answer>\n"
            "If not: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"no\", \"bbox\":\"not found\"}}</answer>"
        ),
        "chat_template_style": "seg_zero",
    },
    # -----------------------------------------------------------------------
    "VisionReasoner": {
        "name": "VisionReasoner-7B",
        "model_path": "/home/u2025141034/VisionReasoner-7B",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": None,
        # T1: yes/no in VisionReasoner think/answer style
        "t1_prompt": (
            "Please decide if '{expr}' matches a visible target in the image.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding, VisionReasoner style from prompt.md
        "t2_prompt": (
            "Please find '{expr}' with bboxs and points.\n"
            "Compare the difference between object(s) and find the most closely matched object(s).\n"
            "Output the thinking process in <think> </think>\n"
            "and final answer in <answer> </answer> tags.\n"
            "Output the bbox(es) and point(s) inside the interested object(s) in JSON format.\n"
            "i.e.\n"
            "<think>\n"
            "thinking process here\n"
            "</think>\n"
            '<answer>[{{"bbox_2d": [x1,y1,x2,y2], "point_2d": [x,y]}}]</answer>\n'
            "If no matching object exists, output <answer>not found</answer>."
        ),
        # T4: caption + existence check + bbox in VisionReasoner think/answer style
        "t4_prompt": (
            "Please first describe the image in one concise sentence, then check whether '{expr}' strictly matches a visible target.\n"
            "Step 1 — Describe the image concisely.\n"
            "Step 2 — Is there '{expr}' in the image?\n"
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If no, output \"not found\".\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            "If target exists: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"yes\", \"bbox\":[x1,y1,x2,y2]}}</answer>\n"
            "If not: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"no\", \"bbox\":\"not found\"}}</answer>"
        ),
        "chat_template_style": "vision_reasoner",
    },
    # -----------------------------------------------------------------------
    "SimpleSeg": {
        "name": "SimpleSeg-Qwen2.5-VL-7B",
        "model_path": "/home/u2025141034/models/SimpleSeg-Qwen2.5-VL-7B",
        "processor_path": QWEN25_VL_PATH,  # SimpleSeg uses opencua tokenizer which needs Qwen2.5-VL processor
        "extra_pythonpath": None,
        "system_prompt": (
            "You are a careful visual grounding assistant. " 
            "Use only visible image evidence. Follow the requested output format exactly." 
        ),
        "t1_prompt": (
            "Task: Decide whether the referring expression strictly matches a visible target in the image.\n" 
            "Referring expression: \"{expr}\"\n" 
            "The object identity, number, attributes, colors, and spatial relations must all match.\n" 
            "Answer with exactly one word: yes or no." 
        ),
        "t2_prompt": (
            "Task: Decide whether the referring expression strictly matches a visible target in the image, then localize it if it exists.\n" 
            "Referring expression: \"{expr}\"\n" 
            "If the target exists, return exactly one bounding box in absolute image pixels as [x1, y1, x2, y2].\n" 
            "If the target does not exist, return exactly: not found.\n" 
            "Do not output explanations." 
        ),
        "t4_prompt": (
            "Task: First describe the image in one concise sentence. Then check whether \"{expr}\" strictly matches a visible target.\n" 
            "If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n" 
            "If no, output \"not found\".\n" 
            "Do not output explanations." 
        ),
        "chat_template_style": "standard",
        "attn_override": "eager",
        "device_map_override": "auto",
    },
"Vision-R1": {
        "name": "Qwen2.5-VL-7B-Instruct-Vision-R1",
        "model_path": "/home/u2025141034/models/Qwen2.5-VL-7B-Instruct-Vision-R1",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": (
            "You are a careful visual grounding assistant. "
            "Use only visible image evidence. Follow the requested output format exactly."
        ),
        "t1_prompt": (
            'Task: Decide whether the referring expression strictly matches a visible target in the image.\n'
            'Referring expression: "{expr}"\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "Answer with exactly one word: yes or no."
        ),
        "t2_prompt": (
            'Task: Decide whether the referring expression strictly matches a visible target in the image, then localize it if it exists.\n'
            'Referring expression: "{expr}"\n'
            "If the target exists, return exactly one bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If the target does not exist, return exactly: not found.\n"
            "Do not output explanations."
        ),
        "t4_prompt": (
            'Task: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Return JSON only:\n'
            '{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}\n'
            '{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}\n'
            'Do not output explanations.'
        ),
        "chat_template_style": "standard",  # uses processor.apply_chat_template
    },
    # -----------------------------------------------------------------------
    "LENS": {
        "name": "LENS (qwen2p5_refcoco)",
        "model_path": "/home/u2025141034/LENS/pretrained/qwen2p5_refcoco",
        "processor_path": QWEN25_VL_PATH,  # LENS uses base Qwen2.5 processor
        "extra_pythonpath": "/home/u2025141034/LENS/src",
        "system_prompt": "You are a helpful assistant.",
        # T1: yes/no discrimination in LENS think/answer style
        "t1_prompt": (
            'Task: Decide whether the referring expression strictly matches a visible target in the image.\n'
            'Referring expression: "{expr}"\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "Output your thinking process in <think></think> and final answer in <answer></answer>.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding with bbox, LENS style — explicit existence check first
        "t2_prompt": (
            'Task: First determine whether "{expr}" strictly matches a visible target in the image.\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "If the target exists, localize it with a bounding box. If not, say not found.\n"
            "Please:\n"
            "1. Check all objects in the image against \"{expr}\"\n"
            "2. Decide whether a strict match exists (matching identity, attributes, and spatial relations)\n"
            "3. If a match exists, select the closest one and provide precise bounding box coordinates\n"
            "4. If no match exists, explicitly reject with \"not found\"\n"
            "Format your response as:\n"
            "<think>\n"
            "[Your analysis: which objects are present, whether they match or don't match]\n"
            "</think><answer>[x1, y1, x2, y2]</answer>\n"
            "If no matching object exists: <answer>not found</answer>"
        ),
        # T4: caption + existence check + bbox in LENS think/answer style
        "t4_prompt": (
            'Task: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Output your thinking process in <think></think> and final answer in <answer></answer>.\n'
            'If target exists: <answer>{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}</answer>\n'
            'If not: <answer>{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}</answer>'
        ),
        "chat_template_style": "lens",
    },
    # -----------------------------------------------------------------------
    "visual-rft": {
        "name": "visual-rft-7b",
        "model_path": "/home/u2025141034/visual-rft-7b",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": None,  # visual-rft doesn't use system prompt (Question: format)
        # T1: yes/no in visual-rft style
        "t1_prompt": (
            'Question: Decide whether the referring expression strictly matches a visible target in the image.\n'
            'Referring expression: "{expr}"\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "Output the thinking process in <think></think> and your answer in <answer></answer>.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding, visual-rft style — existence check FIRST, standard bbox format
        "t2_prompt": (
            'Question: First check whether "{expr}" strictly matches a visible target in the image.\n'
            "If it exists, output its bounding box. If it does NOT exist, say not found.\n"
            'Output the thinking process in <think></think> and your answer in <answer></answer>.\n'
            'If the target exists: <answer>[x1, y1, x2, y2]</answer> (absolute pixel coordinates)\n'
            'If not: <answer>not found</answer>'
        ),
        # T4: caption + existence check + bbox in visual-rft think/answer style
        "t4_prompt": (
            'Question: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Output the thinking process in <think></think> and your answer in <answer></answer>.\n'
            'If target exists: <answer>{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}</answer>\n'
            'If not: <answer>{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}</answer>'
        ),
        "chat_template_style": "visual_rft",
    },
    # -----------------------------------------------------------------------
    "Seg-R1": {
        "name": "Seg-R1-7B",
        "model_path": "/home/u2025141034/seg-r1-7b",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": None,
        # T1: simple yes/no in Seg-R1 style
        "t1_prompt": (
            'Segment: Decide whether "{expr}" strictly matches a visible target in the image.\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            'Answer with exactly "yes" or "no".'
        ),
        # T2: simple grounding, Seg-R1 style from prompt.md
        "t2_prompt": (
            'Segment the main object: "{expr}".\n'
            "Provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If no matching object exists, output exactly: not found."
        ),
        # T4: caption + existence check + bbox in Seg-R1 style
        "t4_prompt": (
            'Task: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Return JSON:\n'
            '{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}\n'
            '{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}'
        ),
        "chat_template_style": "seg_r1",
    },
    # -----------------------------------------------------------------------
    "Seg-zero": {
        "name": "Seg-Zero-7B",
        "model_path": "/home/u2025141034/Seg-Zero-7B",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": None,
        # T1: yes/no in Seg-zero think/answer style
        "t1_prompt": (
            "Please decide if '{expr}' matches a visible target in the image.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding, Seg-zero style from prompt.md
        "t2_prompt": (
            "Please find '{expr}' with bbox and points.\n"
            "Compare the difference between objects and find the most closely matched one.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            "Output the one bbox and center points of two largest inscribed circles inside the interested object in JSON format.\n"
            "i.e., <think> thinking process here </think><answer>{{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}}</answer>\n"
            "If no matching object exists, output <answer>not found</answer>."
        ),
        # T4: caption + existence check + bbox in Seg-zero think/answer style
        "t4_prompt": (
            "Please first describe the image in one concise sentence, then check whether '{expr}' strictly matches a visible target.\n"
            "Step 1 — Describe the image concisely.\n"
            "Step 2 — Is there '{expr}' in the image?\n"
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If no, output \"not found\".\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            "If target exists: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"yes\", \"bbox\":[x1,y1,x2,y2]}}</answer>\n"
            "If not: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"no\", \"bbox\":\"not found\"}}</answer>"
        ),
        "chat_template_style": "seg_zero",
    },
    # -----------------------------------------------------------------------
    "VisionReasoner": {
        "name": "VisionReasoner-7B",
        "model_path": "/home/u2025141034/VisionReasoner-7B",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": None,
        # T1: yes/no in VisionReasoner think/answer style
        "t1_prompt": (
            "Please decide if '{expr}' matches a visible target in the image.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding, VisionReasoner style from prompt.md
        "t2_prompt": (
            "Please find '{expr}' with bboxs and points.\n"
            "Compare the difference between object(s) and find the most closely matched object(s).\n"
            "Output the thinking process in <think> </think>\n"
            "and final answer in <answer> </answer> tags.\n"
            "Output the bbox(es) and point(s) inside the interested object(s) in JSON format.\n"
            "i.e.\n"
            "<think>\n"
            "thinking process here\n"
            "</think>\n"
            '<answer>[{{"bbox_2d": [x1,y1,x2,y2], "point_2d": [x,y]}}]</answer>\n'
            "If no matching object exists, output <answer>not found</answer>."
        ),
        # T4: caption + existence check + bbox in VisionReasoner think/answer style
        "t4_prompt": (
            "Please first describe the image in one concise sentence, then check whether '{expr}' strictly matches a visible target.\n"
            "Step 1 — Describe the image concisely.\n"
            "Step 2 — Is there '{expr}' in the image?\n"
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If no, output \"not found\".\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            "If target exists: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"yes\", \"bbox\":[x1,y1,x2,y2]}}</answer>\n"
            "If not: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"no\", \"bbox\":\"not found\"}}</answer>"
        ),
        "chat_template_style": "vision_reasoner",
    },
    # -----------------------------------------------------------------------
    "SimpleSeg": {
        "name": "SimpleSeg-Qwen2.5-VL-7B",
        "model_path": "/home/u2025141034/models/SimpleSeg-Qwen2.5-VL-7B",
        "processor_path": QWEN25_VL_PATH,  # SimpleSeg uses opencua tokenizer which needs Qwen2.5-VL processor
        "extra_pythonpath": None,
        "system_prompt": (
            "You are a careful visual grounding assistant. " 
            "Use only visible image evidence. Follow the requested output format exactly." 
        ),
        "t1_prompt": (
            "Task: Decide whether the referring expression strictly matches a visible target in the image.\n" 
            "Referring expression: \"{expr}\"\n" 
            "The object identity, number, attributes, colors, and spatial relations must all match.\n" 
            "Answer with exactly one word: yes or no." 
        ),
        "t2_prompt": (
            "Task: Decide whether the referring expression strictly matches a visible target in the image, then localize it if it exists.\n" 
            "Referring expression: \"{expr}\"\n" 
            "If the target exists, return exactly one bounding box in absolute image pixels as [x1, y1, x2, y2].\n" 
            "If the target does not exist, return exactly: not found.\n" 
            "Do not output explanations." 
        ),
        "t4_prompt": (
            "Task: First describe the image in one concise sentence. Then check whether \"{expr}\" strictly matches a visible target.\n" 
            "If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n" 
            "If no, output \"not found\".\n" 
            "Do not output explanations." 
        ),
        "chat_template_style": "standard",
    },
"omnex-vl-7b": {
        "name": "Omnex-VL-7B",
        "model_path": "/home/u2025141034/omnex-vl-7b",
        "processor_path": QWEN25_VL_PATH,
        "extra_pythonpath": None,
        "system_prompt": (
            "You are a careful visual grounding assistant. "
            "Use only visible image evidence. Follow the requested output format exactly."
        ),
        "t1_prompt": (
            'Task: Decide whether the referring expression strictly matches a visible target in the image.\n'
            'Referring expression: "{expr}"\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "Answer with exactly one word: yes or no."
        ),
        "t2_prompt": (
            'Task: Decide whether the referring expression strictly matches a visible target in the image, then localize it if it exists.\n'
            'Referring expression: "{expr}"\n'
            "If the target exists, return exactly one bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If the target does not exist, return exactly: not found.\n"
            "Do not output explanations."
        ),
        "t4_prompt": (
            'Task: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Return JSON only:\n'
            '{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}\n'
            '{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}\n'
            'Do not output explanations.'
        ),
        "chat_template_style": "standard",  # uses processor.apply_chat_template
    },
    # -----------------------------------------------------------------------
    "LENS": {
        "name": "LENS (qwen2p5_refcoco)",
        "model_path": "/home/u2025141034/LENS/pretrained/qwen2p5_refcoco",
        "processor_path": QWEN25_VL_PATH,  # LENS uses base Qwen2.5 processor
        "extra_pythonpath": "/home/u2025141034/LENS/src",
        "system_prompt": "You are a helpful assistant.",
        # T1: yes/no discrimination in LENS think/answer style
        "t1_prompt": (
            'Task: Decide whether the referring expression strictly matches a visible target in the image.\n'
            'Referring expression: "{expr}"\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "Output your thinking process in <think></think> and final answer in <answer></answer>.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding with bbox, LENS style — explicit existence check first
        "t2_prompt": (
            'Task: First determine whether "{expr}" strictly matches a visible target in the image.\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "If the target exists, localize it with a bounding box. If not, say not found.\n"
            "Please:\n"
            "1. Check all objects in the image against \"{expr}\"\n"
            "2. Decide whether a strict match exists (matching identity, attributes, and spatial relations)\n"
            "3. If a match exists, select the closest one and provide precise bounding box coordinates\n"
            "4. If no match exists, explicitly reject with \"not found\"\n"
            "Format your response as:\n"
            "<think>\n"
            "[Your analysis: which objects are present, whether they match or don't match]\n"
            "</think><answer>[x1, y1, x2, y2]</answer>\n"
            "If no matching object exists: <answer>not found</answer>"
        ),
        # T4: caption + existence check + bbox in LENS think/answer style
        "t4_prompt": (
            'Task: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Output your thinking process in <think></think> and final answer in <answer></answer>.\n'
            'If target exists: <answer>{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}</answer>\n'
            'If not: <answer>{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}</answer>'
        ),
        "chat_template_style": "lens",
    },
    # -----------------------------------------------------------------------
    "visual-rft": {
        "name": "visual-rft-7b",
        "model_path": "/home/u2025141034/visual-rft-7b",
        "processor_path": QWEN25_VL_PATH,
        "extra_pythonpath": None,
        "system_prompt": None,  # visual-rft doesn't use system prompt (Question: format)
        # T1: yes/no in visual-rft style
        "t1_prompt": (
            'Question: Decide whether the referring expression strictly matches a visible target in the image.\n'
            'Referring expression: "{expr}"\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "Output the thinking process in <think></think> and your answer in <answer></answer>.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding, visual-rft style — existence check FIRST, standard bbox format
        "t2_prompt": (
            'Question: First check whether "{expr}" strictly matches a visible target in the image.\n'
            "If it exists, output its bounding box. If it does NOT exist, say not found.\n"
            'Output the thinking process in <think></think> and your answer in <answer></answer>.\n'
            'If the target exists: <answer>[x1, y1, x2, y2]</answer> (absolute pixel coordinates)\n'
            'If not: <answer>not found</answer>'
        ),
        # T4: caption + existence check + bbox in visual-rft think/answer style
        "t4_prompt": (
            'Question: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Output the thinking process in <think></think> and your answer in <answer></answer>.\n'
            'If target exists: <answer>{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}</answer>\n'
            'If not: <answer>{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}</answer>'
        ),
        "chat_template_style": "visual_rft",
    },
    # -----------------------------------------------------------------------
    "Seg-R1": {
        "name": "Seg-R1-7B",
        "model_path": "/home/u2025141034/seg-r1-7b",
        "processor_path": QWEN25_VL_PATH,
        "extra_pythonpath": None,
        "system_prompt": None,
        # T1: simple yes/no in Seg-R1 style
        "t1_prompt": (
            'Segment: Decide whether "{expr}" strictly matches a visible target in the image.\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            'Answer with exactly "yes" or "no".'
        ),
        # T2: simple grounding, Seg-R1 style from prompt.md
        "t2_prompt": (
            'Segment the main object: "{expr}".\n'
            "Provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If no matching object exists, output exactly: not found."
        ),
        # T4: caption + existence check + bbox in Seg-R1 style
        "t4_prompt": (
            'Task: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Return JSON:\n'
            '{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}\n'
            '{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}'
        ),
        "chat_template_style": "seg_r1",
    },
    # -----------------------------------------------------------------------
    "Seg-zero": {
        "name": "Seg-Zero-7B",
        "model_path": "/home/u2025141034/Seg-Zero-7B",
        "processor_path": QWEN25_VL_PATH,
        "extra_pythonpath": None,
        "system_prompt": None,
        # T1: yes/no in Seg-zero think/answer style
        "t1_prompt": (
            "Please decide if '{expr}' matches a visible target in the image.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding, Seg-zero style from prompt.md
        "t2_prompt": (
            "Please find '{expr}' with bbox and points.\n"
            "Compare the difference between objects and find the most closely matched one.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            "Output the one bbox and center points of two largest inscribed circles inside the interested object in JSON format.\n"
            "i.e., <think> thinking process here </think><answer>{{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}}</answer>\n"
            "If no matching object exists, output <answer>not found</answer>."
        ),
        # T4: caption + existence check + bbox in Seg-zero think/answer style
        "t4_prompt": (
            "Please first describe the image in one concise sentence, then check whether '{expr}' strictly matches a visible target.\n"
            "Step 1 — Describe the image concisely.\n"
            "Step 2 — Is there '{expr}' in the image?\n"
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If no, output \"not found\".\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            "If target exists: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"yes\", \"bbox\":[x1,y1,x2,y2]}}</answer>\n"
            "If not: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"no\", \"bbox\":\"not found\"}}</answer>"
        ),
        "chat_template_style": "seg_zero",
    },
    # -----------------------------------------------------------------------
    "VisionReasoner": {
        "name": "VisionReasoner-7B",
        "model_path": "/home/u2025141034/VisionReasoner-7B",
        "processor_path": QWEN25_VL_PATH,
        "extra_pythonpath": None,
        "system_prompt": None,
        # T1: yes/no in VisionReasoner think/answer style
        "t1_prompt": (
            "Please decide if '{expr}' matches a visible target in the image.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding, VisionReasoner style from prompt.md
        "t2_prompt": (
            "Please find '{expr}' with bboxs and points.\n"
            "Compare the difference between object(s) and find the most closely matched object(s).\n"
            "Output the thinking process in <think> </think>\n"
            "and final answer in <answer> </answer> tags.\n"
            "Output the bbox(es) and point(s) inside the interested object(s) in JSON format.\n"
            "i.e.\n"
            "<think>\n"
            "thinking process here\n"
            "</think>\n"
            '<answer>[{{"bbox_2d": [x1,y1,x2,y2], "point_2d": [x,y]}}]</answer>\n'
            "If no matching object exists, output <answer>not found</answer>."
        ),
        # T4: caption + existence check + bbox in VisionReasoner think/answer style
        "t4_prompt": (
            "Please first describe the image in one concise sentence, then check whether '{expr}' strictly matches a visible target.\n"
            "Step 1 — Describe the image concisely.\n"
            "Step 2 — Is there '{expr}' in the image?\n"
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If no, output \"not found\".\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            "If target exists: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"yes\", \"bbox\":[x1,y1,x2,y2]}}</answer>\n"
            "If not: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"no\", \"bbox\":\"not found\"}}</answer>"
        ),
        "chat_template_style": "vision_reasoner",
    },
    # -----------------------------------------------------------------------
    "SimpleSeg": {
        "name": "SimpleSeg-Qwen2.5-VL-7B",
        "model_path": "/home/u2025141034/models/SimpleSeg-Qwen2.5-VL-7B",
        "processor_path": QWEN25_VL_PATH,  # SimpleSeg uses opencua tokenizer which needs Qwen2.5-VL processor
        "extra_pythonpath": None,
        "system_prompt": (
            "You are a careful visual grounding assistant. " 
            "Use only visible image evidence. Follow the requested output format exactly." 
        ),
        "t1_prompt": (
            "Task: Decide whether the referring expression strictly matches a visible target in the image.\n" 
            "Referring expression: \"{expr}\"\n" 
            "The object identity, number, attributes, colors, and spatial relations must all match.\n" 
            "Answer with exactly one word: yes or no." 
        ),
        "t2_prompt": (
            "Task: Decide whether the referring expression strictly matches a visible target in the image, then localize it if it exists.\n" 
            "Referring expression: \"{expr}\"\n" 
            "If the target exists, return exactly one bounding box in absolute image pixels as [x1, y1, x2, y2].\n" 
            "If the target does not exist, return exactly: not found.\n" 
            "Do not output explanations." 
        ),
        "t4_prompt": (
            "Task: First describe the image in one concise sentence. Then check whether \"{expr}\" strictly matches a visible target.\n" 
            "If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n" 
            "If no, output \"not found\".\n" 
            "Do not output explanations." 
        ),
        "chat_template_style": "standard",
        "use_cache_patch": True,
    },
"Qwen3-VL-8B": {
        "name": "Qwen3-VL-8B-Instruct",
        "model_path": "/home/u2025141034/models/Qwen3-VL-8B-Instruct",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": (
            "You are a careful visual grounding assistant. "
            "Use only visible image evidence. Follow the requested output format exactly."
        ),
        "t1_prompt": (
            'Task: Decide whether the referring expression strictly matches a visible target in the image.\n'
            'Referring expression: "{expr}"\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "Answer with exactly one word: yes or no."
        ),
        "t2_prompt": (
            'Task: Decide whether the referring expression strictly matches a visible target in the image, then localize it if it exists.\n'
            'Referring expression: "{expr}"\n'
            "If the target exists, return exactly one bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If the target does not exist, return exactly: not found.\n"
            "Do not output explanations."
        ),
        "t4_prompt": (
            'Task: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Return JSON only:\n'
            '{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}\n'
            '{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}\n'
            'Do not output explanations.'
        ),
        "chat_template_style": "standard",  # uses processor.apply_chat_template
    },
    # -----------------------------------------------------------------------
    "LENS": {
        "name": "LENS (qwen2p5_refcoco)",
        "model_path": "/home/u2025141034/LENS/pretrained/qwen2p5_refcoco",
        "processor_path": QWEN25_VL_PATH,  # LENS uses base Qwen2.5 processor
        "extra_pythonpath": "/home/u2025141034/LENS/src",
        "system_prompt": "You are a helpful assistant.",
        # T1: yes/no discrimination in LENS think/answer style
        "t1_prompt": (
            'Task: Decide whether the referring expression strictly matches a visible target in the image.\n'
            'Referring expression: "{expr}"\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "Output your thinking process in <think></think> and final answer in <answer></answer>.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding with bbox, LENS style — explicit existence check first
        "t2_prompt": (
            'Task: First determine whether "{expr}" strictly matches a visible target in the image.\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "If the target exists, localize it with a bounding box. If not, say not found.\n"
            "Please:\n"
            "1. Check all objects in the image against \"{expr}\"\n"
            "2. Decide whether a strict match exists (matching identity, attributes, and spatial relations)\n"
            "3. If a match exists, select the closest one and provide precise bounding box coordinates\n"
            "4. If no match exists, explicitly reject with \"not found\"\n"
            "Format your response as:\n"
            "<think>\n"
            "[Your analysis: which objects are present, whether they match or don't match]\n"
            "</think><answer>[x1, y1, x2, y2]</answer>\n"
            "If no matching object exists: <answer>not found</answer>"
        ),
        # T4: caption + existence check + bbox in LENS think/answer style
        "t4_prompt": (
            'Task: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Output your thinking process in <think></think> and final answer in <answer></answer>.\n'
            'If target exists: <answer>{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}</answer>\n'
            'If not: <answer>{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}</answer>'
        ),
        "chat_template_style": "lens",
    },
    # -----------------------------------------------------------------------
    "visual-rft": {
        "name": "visual-rft-7b",
        "model_path": "/home/u2025141034/visual-rft-7b",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": None,  # visual-rft doesn't use system prompt (Question: format)
        # T1: yes/no in visual-rft style
        "t1_prompt": (
            'Question: Decide whether the referring expression strictly matches a visible target in the image.\n'
            'Referring expression: "{expr}"\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "Output the thinking process in <think></think> and your answer in <answer></answer>.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding, visual-rft style — existence check FIRST, standard bbox format
        "t2_prompt": (
            'Question: First check whether "{expr}" strictly matches a visible target in the image.\n'
            "If it exists, output its bounding box. If it does NOT exist, say not found.\n"
            'Output the thinking process in <think></think> and your answer in <answer></answer>.\n'
            'If the target exists: <answer>[x1, y1, x2, y2]</answer> (absolute pixel coordinates)\n'
            'If not: <answer>not found</answer>'
        ),
        # T4: caption + existence check + bbox in visual-rft think/answer style
        "t4_prompt": (
            'Question: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Output the thinking process in <think></think> and your answer in <answer></answer>.\n'
            'If target exists: <answer>{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}</answer>\n'
            'If not: <answer>{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}</answer>'
        ),
        "chat_template_style": "visual_rft",
    },
    # -----------------------------------------------------------------------
    "Seg-R1": {
        "name": "Seg-R1-7B",
        "model_path": "/home/u2025141034/seg-r1-7b",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": None,
        # T1: simple yes/no in Seg-R1 style
        "t1_prompt": (
            'Segment: Decide whether "{expr}" strictly matches a visible target in the image.\n'
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            'Answer with exactly "yes" or "no".'
        ),
        # T2: simple grounding, Seg-R1 style from prompt.md
        "t2_prompt": (
            'Segment the main object: "{expr}".\n'
            "Provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If no matching object exists, output exactly: not found."
        ),
        # T4: caption + existence check + bbox in Seg-R1 style
        "t4_prompt": (
            'Task: First describe the image in one concise sentence. Then check whether "{expr}" strictly matches a visible target.\n'
            'Step 1 — Describe the image concisely.\n'
            'Step 2 — Is there "{expr}" in the image?\n'
            'The object identity, number, attributes, colors, and spatial relations must all match.\n'
            'If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n'
            'If no, output "not found".\n'
            'Return JSON:\n'
            '{{"description":"one concise sentence", "exists":"yes", "bbox":[x1,y1,x2,y2]}}\n'
            '{{"description":"one concise sentence", "exists":"no", "bbox":"not found"}}'
        ),
        "chat_template_style": "seg_r1",
    },
    # -----------------------------------------------------------------------
    "Seg-zero": {
        "name": "Seg-Zero-7B",
        "model_path": "/home/u2025141034/Seg-Zero-7B",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": None,
        # T1: yes/no in Seg-zero think/answer style
        "t1_prompt": (
            "Please decide if '{expr}' matches a visible target in the image.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding, Seg-zero style from prompt.md
        "t2_prompt": (
            "Please find '{expr}' with bbox and points.\n"
            "Compare the difference between objects and find the most closely matched one.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            "Output the one bbox and center points of two largest inscribed circles inside the interested object in JSON format.\n"
            "i.e., <think> thinking process here </think><answer>{{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}}</answer>\n"
            "If no matching object exists, output <answer>not found</answer>."
        ),
        # T4: caption + existence check + bbox in Seg-zero think/answer style
        "t4_prompt": (
            "Please first describe the image in one concise sentence, then check whether '{expr}' strictly matches a visible target.\n"
            "Step 1 — Describe the image concisely.\n"
            "Step 2 — Is there '{expr}' in the image?\n"
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If no, output \"not found\".\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            "If target exists: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"yes\", \"bbox\":[x1,y1,x2,y2]}}</answer>\n"
            "If not: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"no\", \"bbox\":\"not found\"}}</answer>"
        ),
        "chat_template_style": "seg_zero",
    },
    # -----------------------------------------------------------------------
    "VisionReasoner": {
        "name": "VisionReasoner-7B",
        "model_path": "/home/u2025141034/VisionReasoner-7B",
        "processor_path": None,
        "extra_pythonpath": None,
        "system_prompt": None,
        # T1: yes/no in VisionReasoner think/answer style
        "t1_prompt": (
            "Please decide if '{expr}' matches a visible target in the image.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            'The answer should be exactly "yes" or "no".'
        ),
        # T2: grounding, VisionReasoner style from prompt.md
        "t2_prompt": (
            "Please find '{expr}' with bboxs and points.\n"
            "Compare the difference between object(s) and find the most closely matched object(s).\n"
            "Output the thinking process in <think> </think>\n"
            "and final answer in <answer> </answer> tags.\n"
            "Output the bbox(es) and point(s) inside the interested object(s) in JSON format.\n"
            "i.e.\n"
            "<think>\n"
            "thinking process here\n"
            "</think>\n"
            '<answer>[{{"bbox_2d": [x1,y1,x2,y2], "point_2d": [x,y]}}]</answer>\n'
            "If no matching object exists, output <answer>not found</answer>."
        ),
        # T4: caption + existence check + bbox in VisionReasoner think/answer style
        "t4_prompt": (
            "Please first describe the image in one concise sentence, then check whether '{expr}' strictly matches a visible target.\n"
            "Step 1 — Describe the image concisely.\n"
            "Step 2 — Is there '{expr}' in the image?\n"
            "The object identity, number, attributes, colors, and spatial relations must all match.\n"
            "If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n"
            "If no, output \"not found\".\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
            "If target exists: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"yes\", \"bbox\":[x1,y1,x2,y2]}}</answer>\n"
            "If not: <answer>{{\"description\":\"one concise sentence\", \"exists\":\"no\", \"bbox\":\"not found\"}}</answer>"
        ),
        "chat_template_style": "vision_reasoner",
    },
    # -----------------------------------------------------------------------
    "SimpleSeg": {
        "name": "SimpleSeg-Qwen2.5-VL-7B",
        "model_path": "/home/u2025141034/models/SimpleSeg-Qwen2.5-VL-7B",
        "processor_path": QWEN25_VL_PATH,  # SimpleSeg uses opencua tokenizer which needs Qwen2.5-VL processor
        "extra_pythonpath": None,
        "system_prompt": (
            "You are a helpful assistant." 
        ),
        "t1_prompt": (
            "Task: Decide whether the referring expression strictly matches a visible target in the image.\n" 
            "Referring expression: \"{expr}\"\n" 
            "The object identity, number, attributes, colors, and spatial relations must all match.\n" 
            "Answer with exactly one word: yes or no." 
        ),
        "t2_prompt": (
            "Task: Decide whether the referring expression strictly matches a visible target in the image, then localize it if it exists.\n" 
            "Referring expression: \"{expr}\"\n" 
            "If the target exists, return exactly one bounding box in absolute image pixels as [x1, y1, x2, y2].\n" 
            "If the target does not exist, return exactly: not found.\n" 
            "Do not output explanations." 
        ),
        "t4_prompt": (
            "Task: First describe the image in one concise sentence. Then check whether \"{expr}\" strictly matches a visible target.\n" 
            "If yes, provide the bounding box in absolute image pixels as [x1, y1, x2, y2].\n" 
            "If no, output \"not found\".\n" 
            "Do not output explanations." 
        ),
        "chat_template_style": "standard",
    },

}

# The paper comparison uses exactly these 11 models. SimpleSeg is incompatible
# with this evaluator, while omnex-vl-7b was not part of the original report.
MODEL_CONFIGS["UniVG-R1"] = {
    "name": "UniVG-R1 (Qwen2-VL + UniVG)",
    "model_path": "/home/u2025141034/models/UniVG-R1",
    "processor_path": None,
    "extra_pythonpath": None,
    "system_prompt": "You are a helpful assistant.",
    "t1_prompt": (
        'Please decide if "{expr}" strictly matches a visible target in the image. '
        'Output the thinking process in <think></think> and final answer in <answer></answer> tags. '
        'The answer should be exactly "yes" or "no".'
    ),
    "t2_prompt": (
        'Please find "{expr}" in the image. '
        'Output the thinking process in <think></think> and final answer in <answer></answer> tags. '
        'If a matching object exists, output bbox as [x1,y1,x2,y2]. '
        'If no matching object exists, output not found.'
    ),
    "t4_prompt": (
        'Please first describe the image in one concise sentence, then check whether "{expr}" strictly matches a visible target. '
        'Output the thinking process in <think></think> and final answer in <answer></answer> tags. '
        'If target exists: <answer>[x1,y1,x2,y2]</answer>. If not: <answer>not found</answer>.'
    ),
    "chat_template_style": "lens",
}

DEFAULT_MODEL_KEYS = [
    "qwen2.5-vl-7b",
    "LENS",
    "visual-rft",
    "Seg-R1",
    "Seg-zero",
    "VisionReasoner",
    "TreeVGR",
    "Orsta-7B",
    "Vision-R1",
    "UniVG-R1",
    "Qwen3-VL-8B",
]

_MODEL_PATH_OVERRIDES = {
    "LENS": "/home/u2025141034/models/LENS/pretrained/qwen2p5_refcoco",
    "visual-rft": "/home/u2025141034/models/visual-rft-7b",
    "Seg-R1": "/home/u2025141034/models/seg-r1-7b",
    "Seg-zero": "/home/u2025141034/models/Seg-Zero-7B",
    "VisionReasoner": "/home/u2025141034/models/VisionReasoner-7B",
}
for _model_key, _model_path in _MODEL_PATH_OVERRIDES.items():
    MODEL_CONFIGS[_model_key]["model_path"] = _model_path
MODEL_CONFIGS["LENS"]["extra_pythonpath"] = "/home/u2025141034/models/LENS/src"
MODEL_CONFIGS = {key: MODEL_CONFIGS[key] for key in DEFAULT_MODEL_KEYS}
# ---------------------------------------------------------------------------
# Utility functions (mostly from eval_hallu4tasks_attention.py)
# ---------------------------------------------------------------------------

def safe_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", safe_text(text))
    for a, b in {
        "'": "'", "'": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", " ": " ",
    }.items():
        text = text.replace(a, b)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def output_cache_key(
    task: str, image_filename: str, query: str, label_exists: bool
) -> Tuple[str, str, str, bool]:
    return (
        task,
        Path(image_filename).name,
        normalize_text(query),
        bool(label_exists),
    )


def current_record_key(
    task: str, role: str, base_sample_id: str, pair_id: Optional[str]
) -> Tuple[str, str, str, str]:
    return task, role, base_sample_id, safe_text(pair_id)


def load_legacy_output_cache(
    model_key: str, records_root: Optional[str], legacy_benchmark: Optional[str]
) -> Tuple[Dict[Tuple[str, str, str, bool], Dict[str, Any]], Dict[str, Any]]:
    """Index deterministic old outputs only when their full model input matches."""
    stats = {"loaded": 0, "ambiguous": 0, "records_path": None}
    if not records_root or not legacy_benchmark:
        return {}, stats

    records_path = Path(records_root) / model_key / "records.jsonl"
    benchmark_path = Path(legacy_benchmark)
    stats["records_path"] = str(records_path)
    if not records_path.exists() or not benchmark_path.exists():
        return {}, stats

    legacy_rows = json.load(open(benchmark_path, encoding="utf-8"))
    if isinstance(legacy_rows, dict):
        legacy_rows = next(
            (legacy_rows[k] for k in ["samples", "data", "items", "examples", "annotations"]
             if isinstance(legacy_rows.get(k), list)),
            [],
        )
    sample_images = {
        safe_text(row.get("sample_id") or row.get("id")): safe_text(
            row.get("image_filename") or row.get("image") or row.get("file_name")
        )
        for row in legacy_rows
        if isinstance(row, dict)
    }

    candidates: Dict[Tuple[str, str, str, bool], List[Dict[str, Any]]] = defaultdict(list)
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            image_filename = sample_images.get(safe_text(record.get("sample_id")))
            if not image_filename or "raw_output_text" not in record:
                continue
            key = output_cache_key(
                safe_text(record.get("task")),
                image_filename,
                safe_text(record.get("query")),
                bool(record.get("label_exists")),
            )
            candidates[key].append(record)

    cache = {}
    for key, records in candidates.items():
        raw_outputs = {safe_text(record.get("raw_output_text")) for record in records}
        if len(raw_outputs) != 1:
            stats["ambiguous"] += 1
            continue
        cache[key] = records[0]
    stats["loaded"] = len(cache)
    return cache, stats


def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "run"


def dump_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_bbox(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            nums = re.findall(r"-?\d+(?:\.\d+)?", value)
            if len(nums) >= 4:
                value = [float(x) for x in nums[:4]]
            else:
                return None
    if isinstance(value, list) and len(value) == 4:
        try:
            x1, y1, x2, y2 = [float(x) for x in value]
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        except Exception:
            return None
    return None


def clamp_bbox(box: Sequence[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    x1 = max(0.0, min(float(w), x1))
    x2 = max(0.0, min(float(w), x2))
    y1 = max(0.0, min(float(h), y1))
    y2 = max(0.0, min(float(h), y2))
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def bbox_area(box: Optional[Sequence[float]]) -> float:
    if box is None:
        return 0.0
    x1, y1, x2, y2 = [float(x) for x in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return 0.0 if union <= 0 else inter / union


def robust_number_list(text: str) -> Optional[List[float]]:
    if not text:
        return None
    m = re.search(
        r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]",
        text,
    )
    if m:
        return [float(m.group(i)) for i in range(1, 5)]
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(nums) >= 4:
        return [float(nums[i]) for i in range(4)]
    return None


# ---------------------------------------------------------------------------
# Output parsing (per-task)
# ---------------------------------------------------------------------------

def extract_answer_content(raw_text: str) -> str:
    """Extract content from <answer>...</answer> tags, or return raw text."""
    m = re.search(r"<answer>(.*?)</answer>", raw_text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    return raw_text.strip()


def parse_yes_no(raw_text: str) -> Dict[str, Any]:
    """Parse yes/no from model output. Handles think/answer tags."""
    answer_text = extract_answer_content(raw_text)
    low = normalize_text(answer_text).lower()

    if "not found" in low:
        return {"pred_exists": False, "parse_valid": True, "parse_method": "not_found", "cleaned_text": answer_text}

    m = re.match(r'^[\s"\']*(yes|no)\b', low)
    if m:
        return {"pred_exists": m.group(1) == "yes", "parse_valid": True, "parse_method": "prefix", "cleaned_text": answer_text}

    toks = re.findall(r"\b(yes|no)\b", low)
    if len(toks) == 1:
        return {"pred_exists": toks[0] == "yes", "parse_valid": True, "parse_method": "single_token", "cleaned_text": answer_text}

    if low.strip() in {"true", "found", "exists", "present"}:
        return {"pred_exists": True, "parse_valid": True, "parse_method": "alias_yes", "cleaned_text": answer_text}
    if low.strip() in {"false", "none", "absent", "null"}:
        return {"pred_exists": False, "parse_valid": True, "parse_method": "alias_no", "cleaned_text": answer_text}

    return {"pred_exists": False, "parse_valid": False, "parse_method": "parse_fail_default_no", "cleaned_text": answer_text}


def parse_bbox_output(raw_text: str, image_size: Tuple[int, int]) -> Dict[str, Any]:
    """Parse bbox from model output. Supports multiple formats."""
    answer_text = extract_answer_content(raw_text)
    low = normalize_text(answer_text).lower()
    w, h = image_size

    # "not found" check
    if "not found" in low or low.strip() in {"no", "none", "null", "[]", "{}"}:
        return {"pred_bbox_xyxy": None, "pred_found": False, "parse_valid": True, "parse_method": "not_found",
                "cleaned_text": answer_text, "raw_numbers": None}

    # Try JSON first (Seg-zero style: {"bbox": [...], "points_1": [...], ...})
    json_match = re.search(r'\{[^{}]*"bbox_2d"\s*:\s*\[[^\]]+\][^{}]*\}', answer_text)
    if not json_match:
        json_match = re.search(r'\{[^{}]*"bbox"\s*:\s*\[[^\]]+\][^{}]*\}', answer_text)
    if not json_match:
        # VisionReasoner array format: [{"bbox_2d": [x1,y1,x2,y2], ...}]
        json_match = re.search(r'\[\s*\{[^{}]*"bbox_2d"\s*:\s*\[[^\]]+\][^{}]*\}\s*\]', answer_text)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            if isinstance(obj, list):
                obj = obj[0]
            bbox_val = obj.get("bbox_2d") or obj.get("bbox") or obj.get("box")
            if bbox_val and isinstance(bbox_val, list) and len(bbox_val) == 4:
                box = [float(x) for x in bbox_val]
                if max(abs(x) for x in box) <= 1.5:
                    box = [box[0]*w, box[1]*h, box[2]*w, box[3]*h]
                return {"pred_bbox_xyxy": clamp_bbox(box, w, h), "pred_found": True,
                        "parse_valid": True, "parse_method": "json_bbox",
                        "cleaned_text": answer_text, "raw_numbers": box}
        except Exception:
            pass

    # visual-rft format: (x1,y1),(x2,y2)
    paren_match = re.findall(r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)', answer_text)
    if len(paren_match) >= 2:
        try:
            x1, y1 = float(paren_match[0][0]), float(paren_match[0][1])
            x2, y2 = float(paren_match[1][0]), float(paren_match[1][1])
            box = [x1, y1, x2, y2]
            return {"pred_bbox_xyxy": clamp_bbox(box, w, h), "pred_found": True,
                    "parse_valid": True, "parse_method": "paren_bbox",
                    "cleaned_text": answer_text, "raw_numbers": box}
        except Exception:
            pass

    # Standard [x1,y1,x2,y2] format
    nums = robust_number_list(answer_text)
    if nums is not None:
        vals = [float(x) for x in nums]
        if max(abs(x) for x in vals) <= 1.5:
            vals = [vals[0]*w, vals[1]*h, vals[2]*w, vals[3]*h]
        if vals[2] <= vals[0] or vals[3] <= vals[1]:
            x, y, bw, bh = vals
            box = [x, y, x + max(0.0, bw), y + max(0.0, bh)]
            method = "auto_xywh"
        else:
            box = vals
            method = "auto_xyxy"
        return {"pred_bbox_xyxy": clamp_bbox(box, w, h), "pred_found": True,
                "parse_valid": True, "parse_method": method,
                "cleaned_text": answer_text, "raw_numbers": vals}

    # Could not parse bbox, but model didn't say "not found" either
    return {"pred_bbox_xyxy": None, "pred_found": False, "parse_valid": False,
            "parse_method": "no_bbox", "cleaned_text": answer_text, "raw_numbers": None}


def parse_caption_ground_output(raw_text: str, image_size: Tuple[int, int]) -> Dict[str, Any]:
    """Parse caption + existence check + bbox from model output.

    New T4 prompt asks model to output JSON with "exists" field ("yes"/"no")
    and "bbox" (either [x1,y1,x2,y2] or "not found").
    """
    answer_text = extract_answer_content(raw_text)
    w, h = image_size
    desc = ""
    pred_exists = None  # True/False/None (None = could not parse)

    # Try JSON
    json_match = re.search(r'\{[^{}]*"description"\s*:\s*"[^"]*"[^{}]*"exists"\s*:\s*"[^"]*"[^{}]*"bbox"\s*:\s*[^}]*\}', answer_text, flags=re.S)
    if not json_match:
        json_match = re.search(r'\{[^{}]*"description"\s*:\s*"[^"]*"[^{}]*"bbox"\s*:\s*[^}]*"exists"\s*:\s*"[^"]*"[^{}]*\}', answer_text, flags=re.S)
    if not json_match:
        json_match = re.search(r'\{[^{}]*"exists"\s*:\s*"[^"]*"[^{}]*"description"\s*:\s*"[^"]*"[^{}]*"bbox"\s*:\s*[^}]*\}', answer_text, flags=re.S)
    if not json_match:
        json_match = re.search(r'\{[^{}]*"exists"\s*:\s*"[^"]*"[^{}]*"bbox"\s*:\s*[^}]*"description"\s*:\s*"[^"]*"[^{}]*\}', answer_text, flags=re.S)
    if not json_match:
        json_match = re.search(r'\{[^{}]*"bbox"\s*:\s*[^}]*"exists"\s*:\s*"[^"]*"[^{}]*"description"\s*:\s*"[^"]*"[^{}]*\}', answer_text, flags=re.S)
    if not json_match:
        json_match = re.search(r'\{[^{}]*"bbox"\s*:\s*[^}]*"description"\s*:\s*"[^"]*"[^{}]*"exists"\s*:\s*"[^"]*"[^{}]*\}', answer_text, flags=re.S)
    # Fallback: old-style JSON without "exists" field
    if not json_match:
        json_match = re.search(r'\{[^{}]*"description"\s*:\s*"[^"]*"[^{}]*"bbox"\s*:\s*\[[^\]]*\][^{}]*\}', answer_text, flags=re.S)
    if not json_match:
        json_match = re.search(r'\{[^{}]*"bbox"\s*:\s*\[[^\]]*\][^{}]*"description"\s*:\s*"[^"]*"[^{}]*\}', answer_text, flags=re.S)

    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            desc = safe_text(obj.get("description") or obj.get("caption") or "")
            # Parse "exists" field
            exists_val = safe_text(obj.get("exists", "")).lower()
            if exists_val in ("yes", "true", "1"):
                pred_exists = True
            elif exists_val in ("no", "false", "0", "not found", "none"):
                pred_exists = False
            bbox_val = obj.get("bbox") or obj.get("box") or obj.get("bbox_xyxy")
            # If bbox is string "not found", treat as no bbox
            if isinstance(bbox_val, str) and "not found" in bbox_val.lower():
                bbox_val = None
                if pred_exists is None:
                    pred_exists = False
            bbox_parsed = parse_bbox_output(json.dumps(bbox_val) if bbox_val is not None else "not found", image_size)
            return {
                "cleaned_text": answer_text,
                "generated_description": desc,
                "pred_exists": pred_exists,
                "pred_bbox_xyxy": bbox_parsed.get("pred_bbox_xyxy"),
                "pred_found": bbox_parsed.get("pred_found", False),
                "bbox_parse_valid": bbox_parsed.get("parse_valid", False),
                "bbox_parse_method": bbox_parsed.get("parse_method", "no_json"),
                "raw_numbers": bbox_parsed.get("raw_numbers"),
            }
        except Exception:
            pass

    # Fallback: extract description text (remove bbox-like parts)
    desc = re.sub(r"\[[^\]]*\]", "", answer_text)
    desc = re.sub(r"\([^)]*,[^)]*\)", "", desc)
    desc = re.sub(r"\s+", " ", desc).strip()
    desc = re.sub(r"^(description|caption)\s*[:=]\s*", "", desc, flags=re.I).strip()

    bbox_parsed = parse_bbox_output(answer_text, image_size)
    return {
        "cleaned_text": answer_text,
        "generated_description": desc,
        "pred_exists": pred_exists,
        "pred_bbox_xyxy": bbox_parsed.get("pred_bbox_xyxy"),
        "pred_found": bbox_parsed.get("pred_found", False),
        "bbox_parse_valid": bbox_parsed.get("parse_valid", False),
        "bbox_parse_method": bbox_parsed.get("parse_method", "no_json"),
        "raw_numbers": bbox_parsed.get("raw_numbers"),
    }


# ---------------------------------------------------------------------------
# AMBER-style lexical cosine (for T4 caption scoring)
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "is", "are", "was", "were",
    "in", "on", "at", "of", "to", "with", "and", "or", "for", "from", "by", "as",
    "it", "its", "there", "here", "image", "picture", "photo", "scene", "shows",
    "showing", "visible", "contains", "containing", "red", "box", "bounding",
}

SIMPLE_SYNONYMS = {
    "man": "person", "woman": "person", "boy": "person", "girl": "person", "people": "person",
    "men": "person", "women": "person", "child": "person", "children": "person",
    "couch": "sofa", "bike": "bicycle", "motorbike": "motorcycle",
    "cellphone": "phone", "mobile": "phone", "tv": "television",
}


def tokenize_for_cosine(text: str) -> List[str]:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    toks = re.findall(r"[a-z0-9]+", text)
    out = []
    for t in toks:
        if t in STOPWORDS:
            continue
        if len(t) <= 1:
            continue
        if t.endswith("s") and len(t) > 3:
            t = t[:-1]
        t = SIMPLE_SYNONYMS.get(t, t)
        out.append(t)
    return out


def cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def amber_cosine_score(pred: str, ref: str, positive_text: str = "", negative_text: str = "") -> Dict[str, Any]:
    pred_toks = tokenize_for_cosine(pred)
    ref_toks = tokenize_for_cosine(ref)
    pos_toks = tokenize_for_cosine(positive_text)
    neg_toks = tokenize_for_cosine(negative_text)
    cos_val = cosine_counter(Counter(pred_toks), Counter(ref_toks))
    pos_overlap = len(set(pred_toks) & set(pos_toks)) / max(1, len(set(pos_toks)))
    neg_overlap = len(set(pred_toks) & set(neg_toks)) / max(1, len(set(neg_toks)))
    penalized = max(0.0, cos_val * (1.0 - 0.5 * neg_overlap))
    return {
        "amber_cosine": cos_val,
        "amber_cosine_penalized": penalized,
        "positive_token_cover": pos_overlap,
        "negative_token_overlap": neg_overlap,
        "pred_tokens": pred_toks,
        "ref_tokens": ref_toks,
    }


def atomic_phrase_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", normalize_text(text).lower())
    return [SIMPLE_SYNONYMS.get(token, token) for token in tokens if token not in {"a", "an", "the"}]


def contains_atomic_phrase(caption_tokens: set, phrase: str) -> bool:
    phrase_tokens = atomic_phrase_tokens(phrase)
    return bool(phrase_tokens) and set(phrase_tokens).issubset(caption_tokens)


def caption_target_hallucination(caption: str, annotation: Dict[str, Any]) -> Dict[str, Any]:
    """Detect whether a caption states the negative query's annotated false claim."""
    caption_tokens = set(atomic_phrase_tokens(caption))
    target_eval = (annotation or {}).get("target_eval_units") or {}
    matched = []
    for unit in target_eval.get("hallucination_units") or []:
        hallu_type = str(unit.get("type") or target_eval.get("eval_type") or "")
        if hallu_type in {"object", "co_occurrence"}:
            is_match = contains_atomic_phrase(caption_tokens, str(unit.get("object") or ""))
        elif hallu_type == "attribute":
            attribute_match = contains_atomic_phrase(caption_tokens, str(unit.get("attribute") or ""))
            object_name = str(unit.get("object") or target_eval.get("positive_head_object") or "")
            object_match = not object_name or contains_atomic_phrase(caption_tokens, object_name)
            is_match = attribute_match and object_match
        elif hallu_type == "relation":
            is_match = all(
                contains_atomic_phrase(caption_tokens, str(unit.get(key) or ""))
                for key in ("subject", "relation", "target_object")
            )
        else:
            is_match = False
        if is_match:
            matched.append(unit)
    return {
        "caption_target_hallucination": bool(matched),
        "caption_target_hallucination_unit_count": len(matched),
        "caption_target_hallucination_units": matched,
    }


# ---------------------------------------------------------------------------
# CLIP-based bbox-caption similarity scorer
# ---------------------------------------------------------------------------

class BBoxCLIPScorer:
    """Compute CLIP similarity between cropped bbox regions and captions.

    Core metric: clip_score_pred_bbox — cosine similarity between the CLIP
    image embedding of the predicted bbox crop and the CLIP text embedding
    of the generated caption. This directly measures "does the caption
    describe what's inside the predicted bbox?"

    Diagnostic suite (5 metrics from one forward pass per sample):
      clip_score_pred_bbox   — caption ↔ pred bbox crop (THE core metric)
      clip_score_gt_bbox     — caption ↔ gt bbox crop (sanity: should be high)
      clip_score_full_img    — caption ↔ full image (baseline: generic captions score high)
      clip_score_target_expr — pred bbox crop ↔ target expression (bbox relevance)
      clip_localization_gain — pred_bbox − full_img (positive = caption is region-specific)
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cpu"):
        """Load CLIP model. Uses CPU by default to avoid competing with VLM for GPU VRAM.
        CLIP inference is lightweight (~150MB model) and CPU encoding is fast enough
        for the ~400 T4 evaluations per model run.
        """
        from transformers import CLIPProcessor, CLIPModel

        self.device = device
        print(f"  [CLIP] Loading {model_name} ...")
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self._image_size = (
            getattr(self.model.config, "image_size", None)
            or getattr(self.model.config.vision_config, "image_size", 224)
        )
        print(f"  [CLIP] Loaded. image_size={self._image_size}")

    @torch.no_grad()
    def score(self, image: Image.Image, bbox: Optional[List[float]],
              caption: str, target_expr: str = "",
              gt_bbox: Optional[List[float]] = None) -> Dict[str, Optional[float]]:
        """Compute all CLIP-based scores for one (image, bbox, caption) triplet.

        Args:
            image: PIL full image
            bbox: predicted [x1,y1,x2,y2] in absolute pixels, or None
            caption: generated caption text
            target_expr: the referring expression used as the query
        Returns:
            dict with all clip_score_* values (None where inputs are invalid)
        """
        result: Dict[str, Optional[float]] = {
            "clip_score_pred_bbox": None,
            "clip_score_gt_bbox": None,
            "clip_score_full_img": None,
            "clip_score_target_expr": None,
            "clip_localization_gain": None,
        }

        img_w, img_h = image.size

        # --- Prepare image crops ---
        crops: Dict[str, Optional[Image.Image]] = {
            "full": image.copy(),
            "pred_bbox": None,
            "gt_bbox": None,
        }

        if bbox is not None:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            # Clamp to image bounds
            x1 = max(0, min(img_w, x1))
            y1 = max(0, min(img_h, y1))
            x2 = max(x1 + 1, min(img_w, x2))  # ensure non-zero width
            y2 = max(y1 + 1, min(img_h, y2))  # ensure non-zero height
            try:
                crops["pred_bbox"] = image.crop((int(x1), int(y1), int(x2), int(y2)))
            except Exception:
                crops["pred_bbox"] = None
        if gt_bbox is not None:
            gx1, gy1, gx2, gy2 = [float(v) for v in gt_bbox]
            gx1 = max(0, min(img_w, gx1))
            gy1 = max(0, min(img_h, gy1))
            gx2 = max(gx1 + 1, min(img_w, gx2))
            gy2 = max(gy1 + 1, min(img_h, gy2))
            try:
                crops["gt_bbox"] = image.crop((int(gx1), int(gy1), int(gx2), int(gy2)))
            except Exception:
                crops["gt_bbox"] = None

        # --- Compute text embedding for caption (once, reused) ---
        caption_emb = None
        if caption.strip():
            try:
                text_inputs = self.processor(
                    text=[caption], return_tensors="pt", padding=True, truncation=True
                )
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                caption_emb = self.model.get_text_features(**text_inputs).pooler_output
                caption_emb = caption_emb / caption_emb.norm(dim=-1, keepdim=True)
            except Exception:
                pass

        # --- Compute text embedding for target expression ---
        target_emb = None
        if target_expr.strip():
            try:
                text_inputs = self.processor(
                    text=[target_expr], return_tensors="pt", padding=True, truncation=True
                )
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                target_emb = self.model.get_text_features(**text_inputs).pooler_output
                target_emb = target_emb / target_emb.norm(dim=-1, keepdim=True)
            except Exception:
                pass

        # --- Compute image embeddings and similarities ---
        # Helper: get normalized image embedding for a crop
        def _get_img_emb(pil_img: Image.Image):
            img_inputs = self.processor(images=[pil_img], return_tensors="pt")
            img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
            emb = self.model.get_image_features(**img_inputs).pooler_output
            return emb / emb.norm(dim=-1, keepdim=True)

        # Pred bbox crop vs caption
        if crops["pred_bbox"] is not None and caption_emb is not None:
            try:
                img_emb = _get_img_emb(crops["pred_bbox"])
                result["clip_score_pred_bbox"] = float(
                    (caption_emb @ img_emb.T).squeeze().cpu()
                )
            except Exception:
                pass

        # GT bbox crop vs caption
        if crops["gt_bbox"] is not None and caption_emb is not None:
            try:
                img_emb = _get_img_emb(crops["gt_bbox"])
                result["clip_score_gt_bbox"] = float(
                    (caption_emb @ img_emb.T).squeeze().cpu()
                )
            except Exception:
                pass

        # GT bbox crop vs target expression
        if crops["gt_bbox"] is not None and target_emb is not None:
            try:
                img_emb = _get_img_emb(crops["gt_bbox"])
                result["clip_score_gt_bbox_target_expr"] = float(
                    (target_emb @ img_emb.T).squeeze().cpu()
                )
            except Exception:
                pass

        # Pred bbox crop vs target expression
        if crops["pred_bbox"] is not None and target_emb is not None:
            try:
                img_emb = _get_img_emb(crops["pred_bbox"])
                result["clip_score_target_expr"] = float(
                    (target_emb @ img_emb.T).squeeze().cpu()
                )
            except Exception:
                pass

        # Full image vs caption
        if caption_emb is not None:
            try:
                img_emb = _get_img_emb(crops["full"])
                result["clip_score_full_img"] = float(
                    (caption_emb @ img_emb.T).squeeze().cpu()
                )
            except Exception:
                pass

        # Localization gain: how much more relevant the caption is to the bbox region
        # than to the full image. Positive = caption describes the specific region.
        if result["clip_score_pred_bbox"] is not None and result["clip_score_full_img"] is not None:
            result["clip_localization_gain"] = (
                result["clip_score_pred_bbox"] - result["clip_score_full_img"]
            )

        # Clean up crops
        for c in crops.values():
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass

        return result

    def cleanup(self):
        """Free CLIP model memory."""
        del self.model
        del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Singleton holder (loaded once per evaluation run)
_clip_scorer: Optional[BBoxCLIPScorer] = None


def get_clip_scorer(device: str, model_name: str = "openai/clip-vit-base-patch32") -> BBoxCLIPScorer:
    global _clip_scorer
    if _clip_scorer is None:
        _clip_scorer = BBoxCLIPScorer(model_name=model_name, device=device)
    return _clip_scorer


def release_clip_scorer():
    global _clip_scorer
    if _clip_scorer is not None:
        _clip_scorer.cleanup()
        _clip_scorer = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(model_key: str, device: torch.device, dtype: torch.dtype, attn_impl: str = "flash_attention_2"):
    """Load a model and its processor, handling various architectures."""
    from transformers import AutoConfig, AutoProcessor, AutoModel

    cfg = MODEL_CONFIGS[model_key]
    model_path = cfg["model_path"]
    processor_path = cfg["processor_path"] or model_path

    # Add extra PYTHONPATH if needed (e.g., LENS)
    if cfg.get("extra_pythonpath"):
        sys.path.insert(0, cfg["extra_pythonpath"])

    print(f"  Loading processor from: {processor_path}")
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True, use_fast=False)

    # Set pad_token if missing
    if getattr(processor, "pad_token_id", None) is None:
        eos = getattr(processor, "eos_token_id", None)
        if eos is not None:
            processor.pad_token_id = eos
    if hasattr(processor, "tokenizer") and getattr(processor.tokenizer, "pad_token_id", None) is None:
        eos = getattr(processor.tokenizer, "eos_token_id", None)
        if eos is not None:
            processor.tokenizer.pad_token_id = eos

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # --- Patch config for models with use_cache=None ---
    cfg_dict = MODEL_CONFIGS[model_key]
    if cfg_dict.get("use_cache_patch") or getattr(config, "use_cache", None) is None:
        if hasattr(config, "use_cache"):
            config.use_cache = True
        if hasattr(config, "text_config") and getattr(config.text_config, "use_cache", None) is None:
            config.text_config.use_cache = True

    # Compute attn_override BEFORE building load_kwargs and patching config
    model_attn_override_pre = cfg_dict.get("attn_override")
    effective_attn = model_attn_override_pre or attn_impl

    # Also patch _attn_implementation on the config object
    # Some model configs have attn_implementation set in generation_config,
    # which overrides load_kwargs. Force override via config object.
    if effective_attn:
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = effective_attn
        if hasattr(config, "_attn_implementation_internal"):
            config._attn_implementation_internal = effective_attn

    mt = str(getattr(config, "model_type", "")).lower()
    arch = str(getattr(config, "architectures", [])).lower()
    print(f"  model_type={mt}, architectures={arch}")

    # Device map: use "auto" for models that need multi-GPU, specific device otherwise


    device_map_val = cfg_dict.get("device_map_override", device)


    load_kwargs: Dict[str, Any] = {


        "torch_dtype": dtype,


        "trust_remote_code": True,


        "low_cpu_mem_usage": True,


        "device_map": device_map_val,


    }

    # Handle opencua model type (SimpleSeg) - not registered for AutoModelForImageTextToText
    if mt == "opencua":
        print("  Loading as AutoModel (opencua architecture)")
        model = AutoModel.from_pretrained(model_path, **load_kwargs)
        print(f"  Loaded model: {type(model).__name__}")
        model.eval()
        return processor, model

    # 4-bit / 8-bit quantization for memory-constrained models
    if cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs.pop("torch_dtype", None)
        print("  Using 4-bit quantization (bnb-nf4)")
    elif cfg.get("load_in_8bit"):
        load_kwargs["load_in_8bit"] = True
        load_kwargs.pop("torch_dtype", None)
        print("  Using 8-bit quantization")
    # attn_override already computed above as effective_attn
    if effective_attn and effective_attn != "none":
        load_kwargs["attn_implementation"] = effective_attn

    # Build class list based on model_type
    class_names = []
    if "qwen2_5_vl" in mt or "qwen2.5" in mt.lower():
        class_names = ["Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration"]
    elif "qwen3_vl" in mt or "qwen3" in mt.lower():
        class_names = ["Qwen3VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration"]
    elif "qwen2_vl" in mt or "qwen2" in mt.lower():
        class_names = ["Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration"]
    else:
        class_names = []

    # Add architecture-specific class name
    arch_name = config.architectures[0] if config.architectures else None
    if arch_name and arch_name not in class_names:
        class_names.insert(0, arch_name)

    model = None
    last_error = None

    for name in class_names:
        try:
            print(f"  Trying class: {name}")
            cls = getattr(__import__("transformers"), name, None)
            if cls is None:
                # Try AutoModel for custom architectures
                from transformers import AutoModelForImageTextToText
                print(f"  {name} not in transformers, trying AutoModel")
                model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
                print(f"  Success with AutoModel -> {type(model).__name__}")
                break
            model = cls.from_pretrained(model_path, **load_kwargs)
            print(f"  Success with: {name}")
            break
        except Exception as e:
            last_error = e
            continue

    if model is None:
        try:
            from transformers import AutoModelForImageTextToText
            print("  Fallback: AutoModel")
            model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
            print(f"  Success with AutoModel -> {type(model).__name__}")
        except Exception as e2:
            raise RuntimeError(
                f"Could not load model from {model_path}. Last class error: {last_error}. Fallback error: {e2}"
            )

    model.eval()
    # With device_map="auto", model is already on the correct device(s)
    # Only call .to() if model is still on CPU
    try:
        first_param_device = next(model.parameters()).device
        if first_param_device.type == "cpu":
            model.to(device)
    except Exception:
        model.to(device)
    return processor, model


# ---------------------------------------------------------------------------
# Prompt rendering by chat_template_style
# ---------------------------------------------------------------------------

def build_messages(cfg: Dict[str, Any], user_prompt: str) -> List[Dict[str, Any]]:
    """Build messages list for a given model config style."""
    style = cfg.get("chat_template_style", "standard")
    sys_prompt = cfg.get("system_prompt")

    if style == "visual_rft":
        # visual-rft uses "Question:" prefix, no system prompt
        return [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
        ]
    elif style == "seg_r1":
        # Seg-R1 uses simple user message, no system prompt
        return [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
        ]
    elif style in ("lens", "seg_zero", "vision_reasoner"):
        # These use system prompt + image + user prompt
        msgs = []
        if sys_prompt:
            msgs.append({"role": "system", "content": sys_prompt})
        msgs.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]})
        return msgs
    else:
        # standard: system prompt + image + user prompt
        msgs = []
        if sys_prompt:
            msgs.append({"role": "system", "content": sys_prompt})
        msgs.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]})
        return msgs


def render_prompt(processor: Any, cfg: Dict[str, Any], user_prompt: str) -> str:
    """Render a chat template prompt string for a given model config."""
    messages = build_messages(cfg, user_prompt)
    # Replace {"type": "image"} with actual image path handling
    # We render the template without the image (processor handles images separately)
    try:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback: manually build
        parts = []
        if cfg.get("system_prompt"):
            parts.append(f"<|im_start|>system\n{cfg['system_prompt']}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{user_prompt}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)


# ---------------------------------------------------------------------------
# Generation wrapper
# ---------------------------------------------------------------------------

def generate_single(
    model: Any,
    processor: Any,
    cfg: Dict[str, Any],
    image: Image.Image,
    user_prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """Run a single forward generation and return output text + latency."""
    messages = build_messages(cfg, user_prompt)

    # Build the prompt text (rendered)
    try:
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        if cfg.get("system_prompt"):
            parts.append(f"<|im_start|>system\n{cfg['system_prompt']}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{user_prompt}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        prompt_text = "".join(parts)

    # Process image
    try:
        inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
    except Exception:
        # Fallback: try without text processing
        tok = getattr(processor, "tokenizer", processor)
        text_inputs = tok([prompt_text], padding=True, return_tensors="pt")
        img_inputs = processor.image_processor(image, return_tensors="pt")
        inputs = {**text_inputs, **img_inputs}

    # Move to device
    inputs = {k: v.to(device=device, dtype=dtype) if v.dtype.is_floating_point else v.to(device=device)
              for k, v in inputs.items() if torch.is_tensor(v)}

    prompt_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.pad_token_id if hasattr(processor, "pad_token_id") else None,
        )
    t1 = time.time()

    new_ids = generated[0][prompt_len:].detach().cpu().tolist()
    try:
        raw_text = processor.batch_decode([new_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    except Exception:
        tok = getattr(processor, "tokenizer", processor)
        raw_text = tok.decode(new_ids, skip_special_tokens=True)

    del inputs, generated
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "raw_output_text": raw_text,
        "latency_sec": t1 - t0,
        "prompt_token_count": prompt_len,
        "generated_token_count": len(new_ids),
    }


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------

def resolve_image_path(image_dir: str, image_filename: str) -> Optional[str]:
    base = Path(image_dir)
    p = Path(image_filename)
    cands = [base / p.name, base / image_filename]
    if p.is_absolute():
        cands.insert(0, p)
    for c in cands:
        if c.exists():
            return str(c)
    return None


def benchmark_base_id(item: Dict[str, Any], fallback_sample_id: str) -> str:
    explicit = safe_text(item.get("base_sample_id"))
    if explicit:
        return explicit
    if item.get("expansion_method") == "original_reclassified":
        return fallback_sample_id
    return re.sub(r"_(?:cooc|attr|rel)$", "", fallback_sample_id)


def load_benchmark(
    path: str, image_dir: str, require_strict_groups: bool = True
) -> List[Dict[str, Any]]:
    """Load benchmark and validate shared positive/bbox group invariants."""
    raw = json.load(open(path, encoding="utf-8"))
    if isinstance(raw, dict):
        for k in ["samples", "data", "items", "examples", "annotations"]:
            if isinstance(raw.get(k), list):
                raw = raw[k]
                break
    if not isinstance(raw, list):
        raise TypeError("benchmark must be a list or dict with a list field")

    samples = []
    missing_img = 0
    missing_bbox = 0
    missing_desc = 0

    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue

        img_fn = safe_text(item.get("image_filename") or item.get("image") or item.get("file_name"))
        img_path = resolve_image_path(image_dir, img_fn)
        if img_path is None:
            missing_img += 1
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        bbox = (
            parse_bbox(item.get("gt_bbox_xyxy"))
            or parse_bbox(item.get("positive_bbox"))
            or parse_bbox(item.get("chosen_bbox_xyxy"))
        )
        if bbox is None:
            missing_bbox += 1
            continue
        bbox = clamp_bbox(bbox, w, h)

        pos_text = normalize_text(item.get("positive_text") or item.get("chosen"))
        neg_text = normalize_text(item.get("negative_text") or item.get("rejected"))
        desc = normalize_text(item.get("image_description"))

        sid = safe_text(item.get("sample_id") or item.get("id") or f"sample_{i:06d}")
        group_id = benchmark_base_id(item, sid)
        htype = safe_text(item.get("hallucination_type") or "unknown")
        source = safe_text(item.get("source") or "unknown")

        samples.append({
            "row_index": i,
            "sample_id": sid,
            "base_sample_id": group_id,
            "pair_id": safe_text(item.get("pair_id") or f"{group_id}::{htype}"),
            "source": source,
            "hallucination_type": htype,
            "image_filename": img_fn,
            "image_path": img_path,
            "image_size": (w, h),
            "positive_text": pos_text,
            "negative_text": neg_text,
            "gt_bbox_xyxy": bbox,
            "image_description": desc,
            "chair_annotation": item.get("chair_annotation") or {},
        })

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[sample["base_sample_id"]].append(sample)

    group_errors = []
    required_types = {"object", "co_occurrence", "attribute", "relation"}
    for group_id, group in groups.items():
        positive_values = {normalize_text(row["positive_text"]) for row in group}
        image_values = {row["image_filename"] for row in group}
        bbox_values = {tuple(round(float(v), 6) for v in row["gt_bbox_xyxy"]) for row in group}
        type_counts = Counter(row["hallucination_type"] for row in group)
        if len(positive_values) != 1:
            group_errors.append(f"{group_id}: multiple positive expressions")
        if len(image_values) != 1:
            group_errors.append(f"{group_id}: multiple images")
        if len(bbox_values) != 1:
            group_errors.append(f"{group_id}: multiple positive bboxes")
        if require_strict_groups and set(type_counts) != required_types:
            group_errors.append(
                f"{group_id}: types={sorted(type_counts)} expected={sorted(required_types)}"
            )
        if require_strict_groups and any(count != 1 for count in type_counts.values()):
            group_errors.append(f"{group_id}: duplicate typed negatives {dict(type_counts)}")
    if group_errors:
        preview = "\n  ".join(group_errors[:10])
        raise ValueError(
            f"Benchmark pair validation failed for {len(group_errors)} group checks.\n  {preview}\n"
            "Use a repaired strict-pair file. --allow-legacy-benchmark is diagnostic only."
        )

    print(
        f"  Loaded {len(samples)} negative pairs in {len(groups)} positive groups "
        f"(missing_img={missing_img}, missing_bbox={missing_bbox})"
    )
    return samples


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    ys = [float(x) for x in xs if x is not None]
    return None if not ys else sum(ys) / len(ys)


def div(a: float, b: float) -> Optional[float]:
    return None if b == 0 else a / b


# ---------------------------------------------------------------------------
# Confidence interval utilities
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, trials: int, alpha: float = 0.05) -> Optional[Tuple[float, float]]:
    """Wilson score interval for a binomial proportion. Returns (lower, upper) or None."""
    if trials <= 0:
        return None
    z = 1.96 if abs(alpha - 0.05) < 1e-6 else 1.96
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denom
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return (lo, hi)


def bootstrap_ci(
    values: List[float],
    statistic=np.mean,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Optional[Tuple[float, float]]:
    """Bootstrap percentile CI for a statistic over `values`. Returns (lower, upper) or None."""
    if len(values) < 2:
        return None
    arr = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(arr)
    stats = []
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        stats.append(float(statistic(sample)))
    stats.sort()
    lo_idx = int(n_resamples * alpha / 2)
    hi_idx = int(n_resamples * (1 - alpha / 2)) - 1
    return (stats[max(0, lo_idx)], stats[min(len(stats) - 1, hi_idx)])


def _ci_dict(estimate: Optional[float], ci: Optional[Tuple[float, float]]) -> Dict[str, Any]:
    """Pack estimate + CI into a dict for JSON output."""
    out: Dict[str, Any] = {"value": estimate}
    if ci is not None:
        out["ci_lower"] = ci[0]
        out["ci_upper"] = ci[1]
        out["ci_level"] = 0.95
    return out


def typed_negative_rates(
    records: List[Dict[str, Any]], predicate
) -> Tuple[Dict[str, Dict[str, Any]], Optional[float], Optional[float], Optional[float]]:
    hallu_types = ("object", "co_occurrence", "attribute", "relation")
    per_type: Dict[str, Dict[str, Any]] = {}
    for hallu_type in hallu_types:
        subset = [r for r in records if r.get("hallucination_type") == hallu_type]
        count = sum(bool(predicate(r)) for r in subset)
        per_type[hallu_type] = {"n": len(subset), "count": count, "rate": div(count, len(subset))}
    macro = mean([per_type[kind]["rate"] for kind in hallu_types])
    boh = mean([per_type[kind]["rate"] for kind in ("object", "co_occurrence")])
    roh = mean([per_type[kind]["rate"] for kind in ("attribute", "relation")])
    return per_type, macro, boh, roh


def compute_metrics(records: List[Dict[str, Any]], iou_thr: float = 0.5) -> Dict[str, Any]:
    """Compute metrics for T1, T2, T4 from records."""

    # --- T1: Discriminative VQA ---
    t1_recs = [r for r in records if r["task"] == "t1_discriminative_vqa"]
    tp = fn = fp = tn = 0
    t1_parse_fail = 0
    t1_lat = []
    for r in t1_recs:
        gt = bool(r.get("label_exists"))
        pred = bool(r.get("pred_exists"))
        if not r.get("parse_valid", True):
            t1_parse_fail += 1
        if r.get("latency_sec") is not None:
            t1_lat.append(float(r["latency_sec"]))
        if gt and pred:
            tp += 1
        elif gt and not pred:
            fn += 1
        elif not gt and pred:
            fp += 1
        else:
            tn += 1
    total_pos = tp + fn
    total_neg = fp + tn
    total = total_pos + total_neg

    # T1 point estimates
    t1_accuracy = div(tp + tn, total) if total > 0 else None
    t1_precision = div(tp, tp + fp) if (tp + fp) > 0 else None
    t1_recall = div(tp, tp + fn) if (tp + fn) > 0 else None
    t1_f1 = div(2 * tp, 2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else None
    t1_hr = div(fp, total_neg) if total_neg > 0 else None
    t1_fnr = div(fn, total_pos) if total_pos > 0 else None
    t1_neg = [r for r in t1_recs if r.get("label_exists") is False]
    t1_hr_types, t1_macro_hr, t1_boh_hr, t1_roh_hr = typed_negative_rates(
        t1_neg, lambda r: r.get("pred_exists") is True
    )

    # T1 bootstrap CIs for rates
    t1_preds = [(bool(r.get("label_exists")), bool(r.get("pred_exists"))) for r in t1_recs]
    t1_correct = [1.0 if gt == pred else 0.0 for gt, pred in t1_preds]
    t1_accuracy_ci = bootstrap_ci(t1_correct) if len(t1_correct) >= 2 else None
    t1_f1_ci = None
    if len(t1_preds) >= 2:
        # Use index-based bootstrap to avoid np.choice on list-of-tuples
        def _t1_f1_from_indices(indices):
            tp = sum(1 for idx in indices if t1_preds[idx][0] and t1_preds[idx][1])
            fp = sum(1 for idx in indices if (not t1_preds[idx][0]) and t1_preds[idx][1])
            fn = sum(1 for idx in indices if t1_preds[idx][0] and (not t1_preds[idx][1]))
            return (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        rng = np.random.RandomState(42)
        n = len(t1_preds)
        f1_stats = []
        for _ in range(2000):
            indices = rng.randint(0, n, size=n)
            f1_stats.append(_t1_f1_from_indices(indices))
        f1_stats.sort()
        lo, hi = f1_stats[50], f1_stats[1949]
        t1_f1_ci = (lo, hi)

    t1_metrics = {
        "num_queries": total,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": _ci_dict(t1_accuracy, t1_accuracy_ci),
        "precision": t1_precision,
        "recall": t1_recall,
        "f1": _ci_dict(t1_f1, t1_f1_ci),
        "hallucination_rate (HR)": _ci_dict(t1_hr, wilson_ci(fp, total_neg) if total_neg > 0 else None),
        "over_refusal_rate (FNR)": _ci_dict(t1_fnr, wilson_ci(fn, total_pos) if total_pos > 0 else None),
        "hallucination_rate_by_type": t1_hr_types,
        "macro_hallucination_rate": t1_macro_hr,
        "boh_hallucination_rate": t1_boh_hr,
        "roh_hallucination_rate": t1_roh_hr,
        "roh_minus_boh_error_gap": None if t1_boh_hr is None or t1_roh_hr is None else t1_roh_hr - t1_boh_hr,
        "balanced_accuracy_macro": mean([t1_recall, None if t1_macro_hr is None else 1.0 - t1_macro_hr]),
        "avg_latency_sec": mean(t1_lat),
        "parse_failures": t1_parse_fail,
    }

    # --- T2: VQA + Grounding ---
    t2_recs = [r for r in records if r["task"] == "t2_vqa_grounding"]
    t2_pos = [r for r in t2_recs if r.get("label_exists") is True]
    t2_neg = [r for r in t2_recs if r.get("label_exists") is False]
    ious = [float(r.get("iou") or 0.0) for r in t2_pos]
    valid = [r for r in t2_pos if r.get("pred_bbox_xyxy") is not None]
    neg_fp = [r for r in t2_neg if r.get("pred_found") is True]
    t2_tp = sum(1 for r in t2_pos if r.get("pred_found") is True)
    t2_fn = sum(1 for r in t2_pos if r.get("pred_found") is False)
    t2_fp = len(neg_fp)
    t2_tn = sum(1 for r in t2_neg if r.get("pred_found") is False)
    t2_total = t2_tp + t2_fn + t2_fp + t2_tn

    # T2 point estimates
    t2_mean_iou = mean(ious)
    t2_acc_iou = div(sum(1 for x in ious if x >= iou_thr), len(t2_pos)) if t2_pos else None
    t2_fg_neg = div(len(neg_fp), len(t2_neg)) if t2_neg else None
    t2_vqa_acc = div(t2_tp + t2_tn, t2_total) if t2_total > 0 else None
    t2_fg_types, t2_macro_fg, t2_boh_fg, t2_roh_fg = typed_negative_rates(
        t2_neg, lambda r: r.get("pred_found") is True
    )

    # T2 CIs
    t2_iou_ci = bootstrap_ci(ious) if len(ious) >= 2 else None
    t2_acc_iou_ci = wilson_ci(sum(1 for x in ious if x >= iou_thr), len(t2_pos)) if t2_pos else None
    t2_fg_neg_ci = wilson_ci(len(neg_fp), len(t2_neg)) if t2_neg else None
    t2_vqa_acc_ci = wilson_ci(t2_tp + t2_tn, t2_total) if t2_total > 0 else None

    t2_metrics = {
        "num_positive": len(t2_pos),
        "num_negative": len(t2_neg),
        "mean_iou_all": _ci_dict(t2_mean_iou, t2_iou_ci),
        "mean_iou_valid_only": mean([float(r.get("iou") or 0.0) for r in valid]) if valid else None,
        "positive_iou_zero_count": sum(1 for value in ious if value <= 1e-12),
        "positive_iou_zero_rate": div(sum(1 for value in ious if value <= 1e-12), len(t2_pos)) if t2_pos else None,
        "positive_false_rejection_count": t2_fn,
        "positive_false_rejection_rate": div(t2_fn, len(t2_pos)) if t2_pos else None,
        "positive_valid_bbox_zero_iou_count": sum(
            1 for record in valid if float(record.get("iou") or 0.0) <= 1e-12
        ),
        "positive_valid_bbox_zero_iou_rate": div(
            sum(1 for record in valid if float(record.get("iou") or 0.0) <= 1e-12),
            len(t2_pos),
        ) if t2_pos else None,
        f"acc@IoU_{iou_thr}": _ci_dict(t2_acc_iou, t2_acc_iou_ci),
        "valid_bbox_rate": div(len(valid), len(t2_pos)) if t2_pos else None,
        "false_grounding_on_neg": _ci_dict(t2_fg_neg, t2_fg_neg_ci),
        "false_grounding_by_type": t2_fg_types,
        "macro_false_grounding": t2_macro_fg,
        "boh_false_grounding": t2_boh_fg,
        "roh_false_grounding": t2_roh_fg,
        "roh_minus_boh_error_gap": None if t2_boh_fg is None or t2_roh_fg is None else t2_roh_fg - t2_boh_fg,
        "decision_balanced_accuracy_macro": mean([
            div(t2_tp, len(t2_pos)),
            None if t2_macro_fg is None else 1.0 - t2_macro_fg,
        ]),
        "vqa_accuracy": _ci_dict(t2_vqa_acc, t2_vqa_acc_ci),
        "vqa_precision": div(t2_tp, t2_tp + t2_fp) if (t2_tp + t2_fp) > 0 else None,
        "vqa_recall": div(t2_tp, t2_tp + t2_fn) if (t2_tp + t2_fn) > 0 else None,
        "vqa_f1": div(2 * t2_tp, 2 * t2_tp + t2_fp + t2_fn) if (2 * t2_tp + t2_fp + t2_fn) > 0 else None,
        "parse_failures": sum(1 for r in t2_recs if not r.get("parse_valid", True)),
        "avg_latency_sec": mean([r.get("latency_sec") for r in t2_recs]),
    }

    # --- T4: Caption + Grounding ---
    t4_recs = [r for r in records if r["task"] == "t4_caption_grounding"]
    t4_pos = [r for r in t4_recs if r.get("query_role") == "positive"]
    t4_neg = [r for r in t4_recs if r.get("query_role") == "negative"]
    # Overall T4 mIoU must count rejected / missing positive boxes as zero.
    # Keep the emitted-box mean separately so over-refusal cannot inflate mIoU.
    t4_valid_pos = [r for r in t4_pos if r.get("pred_bbox_xyxy") is not None]
    t4_pos_ious = [float(r.get("iou") or 0.0) for r in t4_pos]
    t4_valid_ious = [float(r.get("iou") or 0.0) for r in t4_valid_pos]
    t4_target_coverage_vals = [r.get("target_coverage") for r in t4_pos if r.get("target_coverage") is not None]
    t4_amber_vals = [r.get("amber_cosine") for r in t4_recs if r.get("amber_cosine") is not None]
    t4_clip_bbox_vals = [r.get("clip_clip_score_pred_bbox") for r in t4_recs if r.get("clip_clip_score_pred_bbox") is not None]
    t4_clip_gain_vals = [r.get("clip_clip_localization_gain") for r in t4_recs if r.get("clip_clip_localization_gain") is not None]
    t4_clip_iou_vals = [(r.get("clip_clip_score_pred_bbox") or 0.0) * (r.get("iou") or 0.0) for r in t4_pos]

    # --- NEW CLIPxIoU enhancement metrics ---
    # IoU x GT-CLIP composite metric (localization + semantic quality benchmark)
    t4_iou_gt_clip_vals = [(r.get('iou') or 0.0) * (r.get('clip_clip_score_gt_bbox') or 0.0) for r in t4_pos if r.get('clip_clip_score_gt_bbox') is not None]
    # CLIP delta: pred bbox CLIP minus GT bbox CLIP (semantic quality decay from localization error)
    t4_clip_delta_vals = [(r.get('clip_clip_score_pred_bbox') or 0.0) - (r.get('clip_clip_score_gt_bbox') or 0.0) for r in t4_pos if r.get('clip_clip_score_pred_bbox') is not None and r.get('clip_clip_score_gt_bbox') is not None]
    # IoU-weighted CLIP delta: weighted difference between GT and pred CLIP scores
    t4_iou_w_clip_delta_vals = [(r.get('iou') or 0.0) * ((r.get('clip_clip_score_gt_bbox') or 0.0) - (r.get('clip_clip_score_pred_bbox') or 0.0)) for r in t4_pos if r.get('clip_clip_score_gt_bbox') is not None and r.get('clip_clip_score_pred_bbox') is not None]


    # T4 point estimates (grounding)
    t4_mean_iou = mean(t4_pos_ious)
    t4_acc_iou = div(sum(1 for x in t4_pos_ious if x >= iou_thr), len(t4_pos)) if t4_pos else None

    # T4 VQA / existence check
    t4_tp = sum(1 for r in t4_pos if r.get("pred_exists") is True)
    t4_fn = sum(1 for r in t4_pos if r.get("pred_exists") is False)
    t4_fp = sum(1 for r in t4_neg if r.get("pred_exists") is True)
    t4_tn = sum(1 for r in t4_neg if r.get("pred_exists") is False)
    t4_total = t4_tp + t4_fn + t4_fp + t4_tn
    t4_vqa_acc = div(t4_tp + t4_tn, t4_total) if t4_total > 0 else None
    t4_hr = div(t4_fp, len(t4_neg)) if t4_neg else None       # hallucination rate on negatives
    t4_fnr = div(t4_fn, len(t4_pos)) if t4_pos else None       # over-refusal on positives
    t4_hr_types, t4_macro_hr, t4_boh_hr, t4_roh_hr = typed_negative_rates(
        t4_neg, lambda r: r.get("pred_exists") is True
    )
    t4_caption_hallu = [r for r in t4_neg if r.get("caption_target_hallucination") is True]
    t4_caption_clean = [r for r in t4_neg if r.get("caption_target_hallucination") is False]
    fg_given_caption_hallu = div(
        sum(r.get("pred_exists") is True for r in t4_caption_hallu), len(t4_caption_hallu)
    )
    fg_given_caption_clean = div(
        sum(r.get("pred_exists") is True for r in t4_caption_clean), len(t4_caption_clean)
    )
    caption_grounding_coupling_gap = (
        None
        if fg_given_caption_hallu is None or fg_given_caption_clean is None
        else fg_given_caption_hallu - fg_given_caption_clean
    )
    joint_caption_grounding_hallucination = div(
        sum(
            r.get("caption_target_hallucination") is True and r.get("pred_exists") is True
            for r in t4_neg
        ),
        len(t4_neg),
    )

    # T4 CIs
    t4_iou_ci = bootstrap_ci(t4_pos_ious) if len(t4_pos_ious) >= 2 else None
    t4_acc_iou_ci = wilson_ci(sum(1 for x in t4_pos_ious if x >= iou_thr), len(t4_pos)) if t4_pos else None
    t4_amber_ci = bootstrap_ci(t4_amber_vals) if len(t4_amber_vals) >= 2 else None
    t4_clip_bbox_ci = bootstrap_ci(t4_clip_bbox_vals) if len(t4_clip_bbox_vals) >= 2 else None
    t4_clip_iou_ci = bootstrap_ci(t4_clip_iou_vals) if len(t4_clip_iou_vals) >= 2 else None
    t4_iou_gt_clip_ci = bootstrap_ci(t4_iou_gt_clip_vals) if len(t4_iou_gt_clip_vals) >= 2 else None
    t4_clip_delta_ci = bootstrap_ci(t4_clip_delta_vals) if len(t4_clip_delta_vals) >= 2 else None
    t4_iou_w_clip_delta_ci = bootstrap_ci(t4_iou_w_clip_delta_vals) if len(t4_iou_w_clip_delta_vals) >= 2 else None
    t4_vqa_acc_ci = wilson_ci(t4_tp + t4_tn, t4_total) if t4_total > 0 else None
    t4_hr_ci = wilson_ci(t4_fp, len(t4_neg)) if t4_neg else None
    t4_fnr_ci = wilson_ci(t4_fn, len(t4_pos)) if t4_pos else None

    t4_metrics = {
        "num_positive": len(t4_pos),
        "num_negative": len(t4_neg),
        "target_coverage": mean(t4_target_coverage_vals),
        "mean_iou": _ci_dict(t4_mean_iou, t4_iou_ci),
        "mean_iou_valid_only": mean(t4_valid_ious),
        "positive_iou_zero_count": sum(1 for value in t4_pos_ious if float(value) <= 1e-12),
        "positive_iou_zero_rate": div(
            sum(1 for value in t4_pos_ious if float(value) <= 1e-12), len(t4_pos)
        ) if t4_pos else None,
        "positive_false_rejection_count": t4_fn,
        "positive_false_rejection_rate": div(t4_fn, len(t4_pos)) if t4_pos else None,
        "positive_valid_bbox_zero_iou_count": sum(
            1 for record in t4_valid_pos if float(record.get("iou") or 0.0) <= 1e-12
        ),
        "positive_valid_bbox_zero_iou_rate": div(
            sum(1 for record in t4_valid_pos if float(record.get("iou") or 0.0) <= 1e-12),
            len(t4_pos),
        ) if t4_pos else None,
        f"acc@IoU_{iou_thr}": _ci_dict(t4_acc_iou, t4_acc_iou_ci),
        "valid_bbox_rate": div(len(t4_valid_pos), len(t4_pos)) if t4_pos else None,
        "avg_amber_cosine": _ci_dict(mean(t4_amber_vals), t4_amber_ci),
        "avg_amber_cosine_penalized": mean([r.get("amber_cosine_penalized") for r in t4_recs]),
        # CLIP-based metrics
        "avg_clip_score_pred_bbox": _ci_dict(mean(t4_clip_bbox_vals), t4_clip_bbox_ci),
        "avg_clip_score_full_img": mean([r.get("clip_clip_score_full_img") for r in t4_recs]),
        "avg_clip_score_target_expr": mean([r.get("clip_clip_score_target_expr") for r in t4_recs]),
        "avg_clip_localization_gain": mean(t4_clip_gain_vals),
        "avg_clip_iou_weighted": _ci_dict(mean(t4_clip_iou_vals), t4_clip_iou_ci),
        "avg_iou_times_gt_clip": _ci_dict(mean(t4_iou_gt_clip_vals), t4_iou_gt_clip_ci),
        "avg_clip_delta": _ci_dict(mean(t4_clip_delta_vals), t4_clip_delta_ci),
        "avg_iou_weighted_clip_delta": _ci_dict(mean(t4_iou_w_clip_delta_vals), t4_iou_w_clip_delta_ci),
        # VQA / existence-check metrics
        "vqa_accuracy": _ci_dict(t4_vqa_acc, t4_vqa_acc_ci),
        "vqa_tp": t4_tp, "vqa_fn": t4_fn, "vqa_fp": t4_fp, "vqa_tn": t4_tn,
        "vqa_precision": div(t4_tp, t4_tp + t4_fp) if (t4_tp + t4_fp) > 0 else None,
        "vqa_recall": div(t4_tp, t4_tp + t4_fn) if (t4_tp + t4_fn) > 0 else None,
        "vqa_f1": div(2 * t4_tp, 2 * t4_tp + t4_fp + t4_fn) if (2 * t4_tp + t4_fp + t4_fn) > 0 else None,
        "hallucination_rate (HR)": _ci_dict(t4_hr, t4_hr_ci),
        "hallucination_rate_by_type": t4_hr_types,
        "macro_hallucination_rate": t4_macro_hr,
        "boh_hallucination_rate": t4_boh_hr,
        "roh_hallucination_rate": t4_roh_hr,
        "roh_minus_boh_error_gap": None if t4_boh_hr is None or t4_roh_hr is None else t4_roh_hr - t4_boh_hr,
        "decision_balanced_accuracy_macro": mean([
            div(t4_tp, len(t4_pos)),
            None if t4_macro_hr is None else 1.0 - t4_macro_hr,
        ]),
        "caption_grounding_coupling": {
            "caption_hallucination_n": len(t4_caption_hallu),
            "caption_clean_n": len(t4_caption_clean),
            "false_grounding_given_caption_hallucination": fg_given_caption_hallu,
            "false_grounding_given_caption_clean": fg_given_caption_clean,
            "coupling_gap": caption_grounding_coupling_gap,
            "joint_caption_and_grounding_hallucination_rate": joint_caption_grounding_hallucination,
        },
        "over_refusal_rate (FNR)": _ci_dict(t4_fnr, t4_fnr_ci),
        "avg_latency_sec": mean([r.get("latency_sec") for r in t4_recs]),
    }

    return {
        "t1_discriminative_vqa": t1_metrics,
        "t2_vqa_grounding": t2_metrics,
        "t4_caption_grounding": t4_metrics,
    }


# ---------------------------------------------------------------------------
# Main evaluation driver
# ---------------------------------------------------------------------------

def evaluate_model(
    model_key: str,
    samples: List[Dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
    output_dir: str,
    args: argparse.Namespace,
    tasks: Optional[set] = None,
) -> Dict[str, Any]:
    """Run T1, T2, T4 evaluation for one model.
    If `tasks` is provided, only those tasks are run (e.g. {"t1", "t2"}).
    """
    if tasks is None:
        tasks = {"t1", "t2", "t4"}
    cfg = MODEL_CONFIGS[model_key]
    print(f"\n{'='*70}")
    print(f"Evaluating: {cfg['name']} ({model_key})")
    print(f"Model path: {cfg['model_path']}")
    print(f"{'='*70}")

    legacy_cache, legacy_cache_stats = load_legacy_output_cache(
        model_key,
        getattr(args, "reuse_records_root", None),
        getattr(args, "legacy_benchmark", None),
    )
    reuse_counts = Counter()
    print(
        f"Legacy output cache: {legacy_cache_stats['loaded']} unique inputs "
        f"({legacy_cache_stats['ambiguous']} ambiguous inputs excluded)"
    )

    # Load model
    print("Loading model...")
    processor, model = load_model_and_processor(model_key, device, dtype, args.attn_implementation)
    print("Model loaded.\n")

    def infer_or_reuse(
        task_name: str,
        sample: Dict[str, Any],
        expr: str,
        label_exists: bool,
        prompt: str,
        image: Image.Image,
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        key = output_cache_key(
            task_name, sample["image_filename"], expr, label_exists
        )
        cached = legacy_cache.get(key)
        if cached is not None:
            reuse_counts["reused"] += 1
            return {
                "raw_output_text": cached["raw_output_text"],
                "latency_sec": cached.get("latency_sec"),
                "prompt_token_count": cached.get("prompt_token_count"),
                "generated_token_count": cached.get("generated_token_count"),
                "inference_reused": True,
                "reuse_source": legacy_cache_stats["records_path"],
            }
        reuse_counts["generated"] += 1
        output = generate_single(
            model, processor, cfg, image, prompt, device, dtype,
            max_new_tokens=max_new_tokens,
        )
        output["inference_reused"] = False
        output["reuse_source"] = None
        return output

    records_path = os.path.join(output_dir, f"records_{args.records_tag}.jsonl" if getattr(args, 'records_tag', None) else "records.jsonl")

    # If running a subset of tasks, keep existing records for other tasks
    run_t1 = "t1" in tasks
    run_t2 = "t2" in tasks
    run_t4 = "t4" in tasks
    is_full_run = run_t1 and run_t2 and run_t4

    existing_records = []
    resumed_keys = set()
    if os.path.exists(records_path) and getattr(args, "resume", True):
        with open(records_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if "error" in r:
                        continue
                    task = r.get("task", "")
                    role = safe_text(r.get("query_role"))
                    base_id = safe_text(r.get("base_sample_id") or r.get("sample_id"))
                    pair_id = r.get("pair_id")
                    if role == "negative" and not pair_id:
                        pair_id = r.get("sample_id")
                    key = current_record_key(task, role, base_id, pair_id)
                    resumed_keys.add(key)
                    existing_records.append(r)
        print(f"  Resuming from {len(existing_records)} successful current-run records")
    elif not is_full_run and os.path.exists(records_path):
        # Preserve records for tasks that were not requested in this invocation.
        with open(records_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    task = r.get("task", "")
                    if (task == "t1_discriminative_vqa" and not run_t1) or \
                       (task == "t2_vqa_grounding" and not run_t2) or \
                       (task == "t4_caption_grounding" and not run_t4):
                        existing_records.append(r)

    if os.path.exists(records_path):
        os.remove(records_path)
    # Re-write preserved records
    for rec in existing_records:
        append_jsonl(records_path, rec)

    # Run evaluation
    total_samples = len(samples)
    total_groups = len({sample["base_sample_id"] for sample in samples})
    queries_per_task = total_samples + total_groups
    total_queries = queries_per_task * sum((run_t1, run_t2, run_t4))

    count = 0
    seen_positive_groups = set()
    for i, s in enumerate(samples):
        image = Image.open(s["image_path"]).convert("RGB")
        w, h = s["image_size"]
        sid = s["sample_id"]
        base_sid = s["base_sample_id"]
        is_group_representative = base_sid not in seen_positive_groups
        seen_positive_groups.add(base_sid)
        pos_text = s["positive_text"]
        neg_text = s["negative_text"]
        role_queries = [("negative", neg_text, False)]
        if is_group_representative:
            role_queries.insert(0, ("positive", pos_text, True))

        # ---- T1: Discriminative VQA (positive → yes, negative → no) ----
        if run_t1:
            for role, expr, label_exists in role_queries:
                record_key = current_record_key(
                    "t1_discriminative_vqa", role, base_sid,
                    s["pair_id"] if role == "negative" else None,
                )
                if record_key in resumed_keys:
                    reuse_counts["resumed"] += 1
                    continue
                count += 1
                prompt = cfg["t1_prompt"].format(expr=expr)
                try:
                    out = infer_or_reuse(
                        "t1_discriminative_vqa", s, expr, label_exists, prompt,
                        image, args.max_new_tokens_t1,
                    )
                    parsed = parse_yes_no(out["raw_output_text"])
                    rec = {
                        "model": model_key,
                        "task": "t1_discriminative_vqa",
                        "sample_id": base_sid if role == "positive" else sid,
                        "base_sample_id": base_sid,
                        "pair_id": s["pair_id"] if role == "negative" else None,
                        "query_role": role,
                        "query": expr,
                        "label_exists": label_exists,
                        "hallucination_type": s["hallucination_type"] if role == "negative" else "positive",
                        "source": s["source"],
                        "pred_exists": bool(parsed["pred_exists"]),
                        "parse_valid": bool(parsed["parse_valid"]),
                        "parse_method": parsed["parse_method"],
                        "raw_output_text": out["raw_output_text"],
                        "cleaned_text": parsed.get("cleaned_text", ""),
                        "latency_sec": out["latency_sec"],
                        "prompt_token_count": out.get("prompt_token_count"),
                        "generated_token_count": out.get("generated_token_count"),
                        "inference_reused": out["inference_reused"],
                        "reuse_source": out["reuse_source"],
                    }
                    append_jsonl(records_path, rec)
                except Exception as e:
                    print(f"  ERR T1 {sid} {role}: {e}")
                    append_jsonl(records_path, {
                        "model": model_key, "task": "t1_discriminative_vqa",
                        "sample_id": base_sid if role == "positive" else sid,
                        "base_sample_id": base_sid, "query_role": role, "query": expr,
                        "label_exists": label_exists, "pred_exists": None, "error": repr(e),
                    })

        # ---- T2: VQA + Grounding (positive → bbox, negative → not found) ----
        if run_t2:
            for role, expr, label_exists in role_queries:
                record_key = current_record_key(
                    "t2_vqa_grounding", role, base_sid,
                    s["pair_id"] if role == "negative" else None,
                )
                if record_key in resumed_keys:
                    reuse_counts["resumed"] += 1
                    continue
                count += 1
                prompt = cfg["t2_prompt"].format(expr=expr)
                try:
                    out = infer_or_reuse(
                        "t2_vqa_grounding", s, expr, label_exists, prompt,
                        image, args.max_new_tokens_t2,
                    )
                    parsed = parse_bbox_output(out["raw_output_text"], (w, h))
                    pred_box = parsed["pred_bbox_xyxy"]
                    if label_exists:
                        iou = iou_xyxy(pred_box, s["gt_bbox_xyxy"])
                        pred_found = pred_box is not None
                    else:
                        iou = None
                        pred_found = parsed.get("pred_found", pred_box is not None)
                    rec = {
                        "model": model_key,
                        "task": "t2_vqa_grounding",
                        "sample_id": base_sid if role == "positive" else sid,
                        "base_sample_id": base_sid,
                        "pair_id": s["pair_id"] if role == "negative" else None,
                        "query_role": role,
                        "query": expr,
                        "label_exists": label_exists,
                        "hallucination_type": s["hallucination_type"] if role == "negative" else "positive",
                        "source": s["source"],
                        "gt_bbox_xyxy": s["gt_bbox_xyxy"],
                        "pred_found": pred_found,
                        "pred_bbox_xyxy": pred_box,
                        "iou": iou,
                        "parse_valid": bool(parsed["parse_valid"]),
                        "parse_method": parsed["parse_method"],
                        "raw_output_text": out["raw_output_text"],
                        "cleaned_text": parsed.get("cleaned_text", ""),
                        "latency_sec": out["latency_sec"],
                        "prompt_token_count": out.get("prompt_token_count"),
                        "generated_token_count": out.get("generated_token_count"),
                        "inference_reused": out["inference_reused"],
                        "reuse_source": out["reuse_source"],
                    }
                    append_jsonl(records_path, rec)
                except Exception as e:
                    print(f"  ERR T2 {sid} {role}: {e}")
                    append_jsonl(records_path, {
                        "model": model_key, "task": "t2_vqa_grounding",
                        "sample_id": base_sid if role == "positive" else sid,
                        "base_sample_id": base_sid, "query_role": role, "query": expr,
                        "label_exists": label_exists, "error": repr(e),
                    })

        # ---- T4: Caption + Existence Check + Grounding (positive → exists, negative → not found) ----
        if run_t4:
            for role, expr, label_exists in role_queries:
                record_key = current_record_key(
                    "t4_caption_grounding", role, base_sid,
                    s["pair_id"] if role == "negative" else None,
                )
                if record_key in resumed_keys:
                    reuse_counts["resumed"] += 1
                    continue
                count += 1
                prompt = cfg["t4_prompt"].format(expr=expr)
                try:
                    out = infer_or_reuse(
                        "t4_caption_grounding", s, expr, label_exists, prompt,
                        image, args.max_new_tokens_t4,
                    )
                    parsed = parse_caption_ground_output(out["raw_output_text"], (w, h))
                    pred_desc = parsed.get("generated_description") or ""
                    pred_box = parsed.get("pred_bbox_xyxy")

                    # Use model's explicit "exists" decision; fall back to bbox presence
                    pred_exists = parsed.get("pred_exists")
                    if pred_exists is None:
                        pred_exists = pred_box is not None

                    if label_exists:
                        iou = iou_xyxy(pred_box, s["gt_bbox_xyxy"])
                    else:
                        iou = None

                    # ── AMBER score (caption vs reference description) ──
                    score_negative_text = neg_text if role == "negative" else ""
                    score = amber_cosine_score(pred_desc, s["image_description"], pos_text, score_negative_text) if pred_desc else {
                        "amber_cosine": 0.0, "amber_cosine_penalized": 0.0,
                        "positive_token_cover": 0.0, "negative_token_overlap": 0.0,
                        "pred_tokens": [], "ref_tokens": [],
                    }

                    # ── CLIP-based bbox-caption similarity ──
                    clip_scores: Dict[str, Optional[float]] = {}
                    if args.enable_clip_score:
                        try:
                            clip = get_clip_scorer(str(device), args.clip_model_name)
                            clip_scores = clip.score(
                                image=image,
                                bbox=pred_box,
                                caption=pred_desc,
                                target_expr=expr,
                                gt_bbox=s.get("gt_bbox_xyxy") if role == "positive" else None,
                            )
                        except Exception as e:
                            clip_scores = {"clip_error": repr(e)}

                    rec = {
                        "model": model_key,
                        "task": "t4_caption_grounding",
                        "sample_id": base_sid if role == "positive" else sid,
                        "base_sample_id": base_sid,
                        "pair_id": s["pair_id"] if role == "negative" else None,
                        "query_role": role,
                        "query": expr,
                        "label_exists": label_exists,
                        "hallucination_type": s["hallucination_type"] if role == "negative" else "positive",
                        "source": s["source"],
                        "gt_bbox_xyxy": s["gt_bbox_xyxy"],
                        "pred_exists": pred_exists,
                        "pred_found": parsed.get("pred_found", pred_box is not None),
                        "pred_bbox_xyxy": pred_box,
                        "iou": iou,
                        "generated_description": pred_desc,
                        "reference_description": s["image_description"],
                        "amber_cosine": score["amber_cosine"],
                        "amber_cosine_penalized": score["amber_cosine_penalized"],
                        "positive_token_cover": score["positive_token_cover"],
                        "negative_token_overlap": score["negative_token_overlap"],
                        "target_coverage": score["positive_token_cover"],
                        **(
                            caption_target_hallucination(pred_desc, s["chair_annotation"])
                            if role == "negative"
                            else {
                                "caption_target_hallucination": None,
                                "caption_target_hallucination_unit_count": 0,
                                "caption_target_hallucination_units": [],
                            }
                        ),
                        "bbox_parse_valid": parsed.get("bbox_parse_valid", False),
                        "bbox_parse_method": parsed.get("bbox_parse_method", ""),
                        # CLIP scores
                        **{f"clip_{k}": v for k, v in clip_scores.items()},
                        "raw_output_text": out["raw_output_text"],
                        "cleaned_text": parsed.get("cleaned_text", ""),
                        "latency_sec": out["latency_sec"],
                        "prompt_token_count": out.get("prompt_token_count"),
                        "generated_token_count": out.get("generated_token_count"),
                        "inference_reused": out["inference_reused"],
                        "reuse_source": out["reuse_source"],
                    }
                    append_jsonl(records_path, rec)
                except Exception as e:
                    print(f"  ERR T4 {sid} {role}: {e}")
                    append_jsonl(records_path, {
                        "model": model_key, "task": "t4_caption_grounding",
                        "sample_id": base_sid if role == "positive" else sid,
                        "base_sample_id": base_sid, "query_role": role, "query": expr,
                        "label_exists": label_exists, "error": repr(e),
                    })

        image.close()

        if (i + 1) % 50 == 0:
            print(f"  [{model_key}] {i+1}/{total_samples} samples ({count}/{total_queries} queries)")

    # Clean up
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    release_clip_scorer()  # Free CLIP model between models to save VRAM

    # Load records and compute metrics
    records = []
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    metrics = compute_metrics(records, args.iou_threshold)

    summary = {
        "model_key": model_key,
        "model_name": cfg["name"],
        "model_path": cfg["model_path"],
        "num_samples": total_samples,
        "num_positive_groups": total_groups,
        "num_records": len(records),
        "metrics": metrics,
        "record_count_by_task": dict(Counter(r.get("task") for r in records)),
        "inference_reuse": dict(reuse_counts),
        "legacy_cache": legacy_cache_stats,
    }
    # Save the canonical summary only for the untagged master run.
    if not getattr(args, 'records_tag', None):
        dump_json(summary, os.path.join(output_dir, "summary.json"))

        # Save metrics CSV ...

    # Save metrics CSV — flatten CI dicts
    csv_rows = []
    all_fieldnames = set()
    for task, task_metrics in metrics.items():
        row = {"task": task}
        for k, v in task_metrics.items():
            if isinstance(v, dict) and "value" in v:
                row[k] = v["value"]
                if "ci_lower" in v:
                    row[f"{k}_ci_lower"] = v["ci_lower"]
                    row[f"{k}_ci_upper"] = v["ci_upper"]
                    all_fieldnames.update([f"{k}_ci_lower", f"{k}_ci_upper"])
            elif not isinstance(v, (list, dict)):
                row[k] = v
        all_fieldnames.update(row.keys())
        csv_rows.append(row)
    all_fieldnames = ["task"] + sorted(all_fieldnames - {"task"})
    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    # Print metrics
    print(f"\n--- {cfg['name']} Results ---")
    for task, task_metrics in metrics.items():
        print(f"  [{task}]")
        for k, v in task_metrics.items():
            if isinstance(v, dict) and "value" in v:
                ci_str = ""
                if v.get("ci_lower") is not None:
                    ci_str = f"  [95% CI: {v['ci_lower']:.4f}, {v['ci_upper']:.4f}]"
                val_str = f"{v['value']:.4f}" if v['value'] is not None else "N/A"
                print(f"    {k}: {val_str}{ci_str}")
            elif isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate 11 models on the repaired RefCOCOg-500 benchmark")
    p.add_argument("--model", default="all",
                   help="Model key to evaluate, or 'all' for all 6 models")
    p.add_argument("--models", nargs="+", default=None,
                   help="List of model keys to evaluate")
    p.add_argument("--benchmark", default=BENCHMARK_PATH)
    p.add_argument("--image-dir", default=IMAGE_DIR)
    p.add_argument("--output-root", default=OUTPUT_ROOT)
    p.add_argument(
        "--reuse-records-root", default=LEGACY_RECORDS_ROOT,
        help="Old run directory whose deterministic raw outputs may be reused; pass an empty string to disable.",
    )
    p.add_argument(
        "--legacy-benchmark", default=LEGACY_BENCHMARK_PATH,
        help="Benchmark used by the reusable records, needed to recover their image identity.",
    )
    p.add_argument("--run-name", default=None,
                   help="Custom run directory name (overrides timestamp). "
                        "Use when running multiple models in parallel to share a run dir.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn-implementation", default="flash_attention_2",
                   help="Attention implementation for generation (flash_attention_2 is faster)")
    p.add_argument("--max-new-tokens-t1", type=int, default=128)
    p.add_argument("--max-new-tokens-t2", type=int, default=256)
    p.add_argument("--max-new-tokens-t4", type=int, default=256)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--tasks", default="t1,t2,t4",
                   help="Comma-separated tasks to run: t1,t2,t4 (default all)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit positive groups for quick testing")
    p.add_argument("--sample-offset", type=int, default=0,
                   help="Start from this positive-group index (for parallel sharding across GPUs)")
    p.add_argument("--allow-legacy-benchmark", action="store_true",
                   help="Allow incomplete type groups, but still reject positive/bbox mismatch")
    p.add_argument("--records-tag", default=None,
                   help="Suffix for records.jsonl (e.g. 'gpu0' -> records_gpu0.jsonl). Used for parallel sharding.")
    p.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Keep successful records in the current output and regenerate only missing/error records.",
    )
    p.add_argument("--enable-clip-score", action="store_true", default=True,
                   help="Enable CLIP-based bbox-caption similarity scoring (default: on)")
    p.add_argument("--no-clip-score", action="store_false", dest="enable_clip_score",
                   help="Disable CLIP scoring")
    p.add_argument("--clip-model-name", default="openai/clip-vit-base-patch32",
                   help="CLIP model to use for bbox-caption similarity "
                        "(openai/clip-vit-base-patch32, openai/clip-vit-large-patch14, "
                        "google/siglip-so400m-patch14-384)")
    return p


def main():
    args = build_parser().parse_args()

    # Determine model list
    if args.models:
        model_keys = args.models
    elif args.model == "all":
        model_keys = list(MODEL_CONFIGS.keys())
    else:
        model_keys = [args.model]

    # Validate model keys
    for mk in model_keys:
        if mk not in MODEL_CONFIGS:
            print(f"Unknown model key: {mk}. Available: {list(MODEL_CONFIGS.keys())}")
            sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Device: {device}, dtype: {args.dtype}")
    print(f"Models to evaluate: {model_keys}")

    # Parse tasks
    task_set = set(t.strip() for t in args.tasks.split(","))
    valid_tasks = {"t1", "t2", "t4"}
    task_set = task_set & valid_tasks
    if not task_set:
        print(f"No valid tasks specified (got '{args.tasks}'). Using all tasks.")
        task_set = valid_tasks
    print(f"Tasks to run: {sorted(task_set)}")

    # Load benchmark once
    print(f"\nLoading benchmark: {args.benchmark}")
    samples = load_benchmark(
        args.benchmark,
        args.image_dir,
        require_strict_groups=not args.allow_legacy_benchmark,
    )
    ordered_group_ids = list(dict.fromkeys(sample["base_sample_id"] for sample in samples))
    selected_group_ids = ordered_group_ids[args.sample_offset:]
    if args.sample_offset:
        print(f"  Offset to positive group {args.sample_offset}")
    if args.max_samples:
        selected_group_ids = selected_group_ids[:args.max_samples]
        print(f"  Limited to {args.max_samples} positive groups")
    selected_group_ids = set(selected_group_ids)
    samples = [sample for sample in samples if sample["base_sample_id"] in selected_group_ids]

    if args.run_name:
        run_root = os.path.join(args.output_root, args.run_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = os.path.join(args.output_root, f"run_{timestamp}")
    os.makedirs(run_root, exist_ok=True)

    # Save run config (only if it doesn't exist yet — avoid race in parallel runs)
    config_path = os.path.join(run_root, "run_config.json")
    if not os.path.exists(config_path):
        dump_json({
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "models": model_keys,
            "benchmark": args.benchmark,
            "num_negative_pairs": len(samples),
            "num_positive_groups": len({sample['base_sample_id'] for sample in samples}),
            "args": vars(args),
        }, config_path)
    else:
        # Append this model to the existing config
        try:
            existing = json.load(open(config_path))
        except (json.JSONDecodeError, ValueError):
            existing = {}
        existing.setdefault("models", []).extend(model_keys)
        existing["models"] = sorted(set(existing["models"]))
        existing["last_model"] = model_keys[0] if len(model_keys) == 1 else model_keys
        existing["last_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_json(existing, config_path)

    # Evaluate each model
    all_summaries = {}
    for mk in model_keys:
        model_out_dir = os.path.join(run_root, mk)
        os.makedirs(model_out_dir, exist_ok=True)
        try:
            summary = evaluate_model(mk, samples, device, dtype, model_out_dir, args, tasks=task_set)
            all_summaries[mk] = summary
        except Exception as e:
            print(f"\n[FATAL] Failed to evaluate {mk}: {e}")
            traceback.print_exc()
            dump_json({"error": repr(e), "traceback": traceback.format_exc()},
                      os.path.join(model_out_dir, "fatal_error.json"))
            gc.collect()
            torch.cuda.empty_cache()

    # Cross-model comparison table
    print(f"\n{'='*90}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*90}")

    for task, task_label in [("t1_discriminative_vqa", "T1: Discriminative VQA"),
                              ("t2_vqa_grounding", "T2: VQA + Grounding"),
                              ("t4_caption_grounding", "T4: Caption + Grounding")]:
        print(f"\n{task_label}:")
        print(f"{'Model':<25} {'Metric':<30} {'Value':>10}")
        print("-" * 65)

        # Collect all metric keys for this task
        all_keys = set()
        for mk in model_keys:
            if mk in all_summaries:
                m = all_summaries[mk].get("metrics", {}).get(task, {})
                all_keys.update(m.keys())

        # Print key metrics first, then the rest
        priority_keys = ["accuracy", "vqa_accuracy", "f1", "vqa_f1",
                         "hallucination_rate (HR)", "false_grounding_on_neg",
                         "mean_iou_all", "mean_iou", "acc@IoU_0.5",
                         "valid_bbox_rate", "recall", "vqa_recall",
                         "target_coverage", "avg_amber_cosine",
                         "avg_clip_score_pred_bbox", "avg_clip_localization_gain",
                         "avg_clip_score_target_expr", "avg_clip_iou_weighted"]
        ordered_keys = [k for k in priority_keys if k in all_keys]
        ordered_keys += sorted(all_keys - set(ordered_keys))

        for metric_key in ordered_keys:
            for mk in model_keys:
                if mk not in all_summaries:
                    continue
                val = all_summaries[mk].get("metrics", {}).get(task, {}).get(metric_key)
                if val is not None:
                    if isinstance(val, dict) and "value" in val:
                        ci_str = ""
                        if val.get("ci_lower") is not None:
                            ci_str = f"  [{val['ci_lower']:.4f}, {val['ci_upper']:.4f}]"
                        v_str = f"{val['value']:>10.4f}" if val['value'] is not None else f"{'N/A':>10}"
                        print(f"{mk:<25} {metric_key:<30} {v_str}{ci_str}")
                    elif isinstance(val, float):
                        print(f"{mk:<25} {metric_key:<30} {val:>10.4f}")
                    else:
                        print(f"{mk:<25} {metric_key:<30} {str(val):>10}")

    # Save/merge cross-model summary (supports parallel runs appending to same file)
    cm_path = os.path.join(run_root, "cross_model_summary.json")
    new_entries = {mk: {"name": MODEL_CONFIGS[mk]["name"],
                         "metrics": all_summaries[mk].get("metrics", {})}
                   for mk in model_keys if mk in all_summaries}
    if os.path.exists(cm_path):
        existing = json.load(open(cm_path))
        existing["summaries"].update(new_entries)
        existing["models_evaluated"] = sorted(set(
            existing.get("models_evaluated", []) + list(new_entries.keys())
        ))
        existing["last_updated"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_json(existing, cm_path)
    else:
        dump_json({
            "timestamp": args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"),
            "models_evaluated": list(new_entries.keys()),
            "num_samples": len(samples),
            "summaries": new_entries,
        }, cm_path)

    print(f"\n[DONE] Results saved to: {run_root}")


if __name__ == "__main__":
    main()
