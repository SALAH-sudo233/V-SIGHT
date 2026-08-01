"""Conservative relation/reference extraction for E2b supervision."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from .e2_verifier import box_iou


RELATION_PATTERNS = (
    ("in_front_of", r"\bin front of\b"),
    ("next_to", r"\bnext to\b"),
    ("to_the_left_of", r"\b(?:to the )?left of\b"),
    ("to_the_right_of", r"\b(?:to the )?right of\b"),
    ("sitting_on", r"\b(?:sitting|seated|lying) on\b"),
    ("standing_on", r"\bstanding on\b"),
    ("held_by", r"\bheld by\b"),
    ("holding", r"\b(?:holding|holds|carrying|carries)\b"),
    ("wearing", r"\b(?:wearing|wears|dressed in)\b"),
    ("riding", r"\b(?:riding|rides)\b"),
    ("between", r"\bbetween\b"),
    ("behind", r"\bbehind\b"),
    ("above", r"\b(?:above|over)\b"),
    ("below", r"\b(?:below|under|beneath)\b"),
    ("beside", r"\b(?:beside|near|close to)\b"),
    ("on", r"\b(?:on top of|on)\b"),
    ("with", r"\bwith\b"),
    ("by", r"\bby\b"),
)


CATEGORY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "person": (
        "person", "people", "man", "men", "woman", "women", "boy", "boys",
        "girl", "girls", "child", "children", "lady", "guy", "player", "rider",
        "skier", "surfer", "batter", "catcher",
    ),
    "bicycle": ("bicycle", "bicycles", "bike", "bikes"),
    "car": ("car", "cars", "automobile", "vehicle"),
    "motorcycle": ("motorcycle", "motorcycles", "motorbike", "motorbikes"),
    "airplane": ("airplane", "airplanes", "plane", "planes", "aircraft"),
    "bus": ("bus", "buses"),
    "train": ("train", "trains", "locomotive"),
    "truck": ("truck", "trucks"),
    "boat": ("boat", "boats", "ship", "ships"),
    "traffic light": ("traffic light", "traffic lights", "stoplight"),
    "fire hydrant": ("fire hydrant", "fire hydrants", "hydrant"),
    "stop sign": ("stop sign", "stop signs"),
    "parking meter": ("parking meter", "parking meters", "meter"),
    "bench": ("bench", "benches"),
    "bird": ("bird", "birds"),
    "cat": ("cat", "cats", "kitten"),
    "dog": ("dog", "dogs", "puppy"),
    "horse": ("horse", "horses", "pony"),
    "sheep": ("sheep",),
    "cow": ("cow", "cows", "cattle"),
    "elephant": ("elephant", "elephants"),
    "bear": ("bear", "bears"),
    "zebra": ("zebra", "zebras"),
    "giraffe": ("giraffe", "giraffes"),
    "backpack": ("backpack", "backpacks", "rucksack"),
    "umbrella": ("umbrella", "umbrellas"),
    "handbag": ("handbag", "handbags", "purse", "purses", "bag"),
    "tie": ("tie", "ties", "necktie"),
    "suitcase": ("suitcase", "suitcases", "luggage"),
    "frisbee": ("frisbee", "frisbees", "disc"),
    "skis": ("ski", "skis"),
    "snowboard": ("snowboard", "snowboards"),
    "sports ball": ("ball", "balls", "soccer ball", "basketball", "baseball"),
    "kite": ("kite", "kites"),
    "baseball bat": ("baseball bat", "bat", "bats"),
    "baseball glove": ("baseball glove", "glove", "gloves", "mitt"),
    "skateboard": ("skateboard", "skateboards"),
    "surfboard": ("surfboard", "surfboards"),
    "tennis racket": ("tennis racket", "tennis rackets", "racket", "racquet"),
    "bottle": ("bottle", "bottles"),
    "wine glass": ("wine glass", "wine glasses", "goblet"),
    "cup": ("cup", "cups", "mug", "mugs"),
    "fork": ("fork", "forks"),
    "knife": ("knife", "knives"),
    "spoon": ("spoon", "spoons"),
    "bowl": ("bowl", "bowls"),
    "banana": ("banana", "bananas"),
    "apple": ("apple", "apples"),
    "sandwich": ("sandwich", "sandwiches"),
    "orange": ("orange", "oranges"),
    "broccoli": ("broccoli",),
    "carrot": ("carrot", "carrots"),
    "hot dog": ("hot dog", "hot dogs"),
    "pizza": ("pizza", "pizzas"),
    "donut": ("donut", "donuts", "doughnut", "doughnuts"),
    "cake": ("cake", "cakes"),
    "chair": ("chair", "chairs", "seat", "seats"),
    "couch": ("couch", "couches", "sofa", "sofas"),
    "potted plant": ("potted plant", "potted plants", "plant", "plants"),
    "bed": ("bed", "beds"),
    "dining table": ("dining table", "table", "tables", "desk", "desks"),
    "toilet": ("toilet", "toilets"),
    "tv": ("tv", "television", "televisions", "monitor", "screen"),
    "laptop": ("laptop", "laptops", "computer", "computers"),
    "mouse": ("mouse", "computer mouse"),
    "remote": ("remote", "remotes", "controller", "controllers"),
    "keyboard": ("keyboard", "keyboards"),
    "cell phone": ("cell phone", "cell phones", "phone", "phones", "mobile phone"),
    "microwave": ("microwave", "microwaves"),
    "oven": ("oven", "ovens"),
    "toaster": ("toaster", "toasters"),
    "sink": ("sink", "sinks"),
    "refrigerator": ("refrigerator", "refrigerators", "fridge", "fridges"),
    "book": ("book", "books"),
    "clock": ("clock", "clocks"),
    "vase": ("vase", "vases"),
    "scissors": ("scissors",),
    "teddy bear": ("teddy bear", "teddy bears", "teddy"),
    "hair drier": ("hair drier", "hair dryer"),
    "toothbrush": ("toothbrush", "toothbrushes"),
}


@dataclass(frozen=True)
class CategoryMention:
    category: str
    alias: str
    start: int
    end: int


@dataclass(frozen=True)
class RelationParse:
    relation: str | None
    target_category: str
    reference_categories: tuple[str, ...]
    mentions: tuple[CategoryMention, ...]


def category_mentions(query: str, available_categories: Sequence[str]) -> tuple[CategoryMention, ...]:
    text = query.casefold()
    mentions: list[CategoryMention] = []
    for category in available_categories:
        aliases = CATEGORY_ALIASES.get(category, (category, category + "s"))
        for alias in sorted(set(aliases), key=len, reverse=True):
            for match in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", text):
                mentions.append(CategoryMention(category, alias, match.start(), match.end()))
    # Prefer the longest alias when spans overlap, then restore query order.
    selected: list[CategoryMention] = []
    for mention in sorted(mentions, key=lambda value: (-(value.end - value.start), value.start)):
        if any(mention.start < prior.end and prior.start < mention.end for prior in selected):
            continue
        selected.append(mention)
    return tuple(sorted(selected, key=lambda value: (value.start, value.end, value.category)))


def parse_relation(
    query: str, target_category: str, available_categories: Sequence[str]
) -> RelationParse:
    text = query.casefold()
    relation = next(
        (name for name, pattern in RELATION_PATTERNS if re.search(pattern, text)), None
    )
    mentions = category_mentions(query, available_categories)
    non_target = []
    for mention in mentions:
        if mention.category != target_category and mention.category not in non_target:
            non_target.append(mention.category)
    if not non_target:
        target_mentions = [value for value in mentions if value.category == target_category]
        if len(target_mentions) >= 2:
            non_target.append(target_category)
    return RelationParse(relation, target_category, tuple(non_target), mentions)


def extract_reference_phrase(query: str) -> str | None:
    """Return the conservative query suffix governed by the first relation."""

    text = query.strip().casefold()
    for _, pattern in RELATION_PATTERNS:
        match = re.search(pattern, text)
        if not match:
            continue
        phrase = text[match.end() :].strip(" \t.,;:!?()[]{}\"")
        phrase = re.sub(r"^(?:the|a|an|his|her|their|this|that)\s+", "", phrase)
        return phrase or None
    return None


def normalized_box_features(
    box: Sequence[float], image_width: int, image_height: int
) -> list[float]:
    if image_width <= 0 or image_height <= 0 or len(box) != 4:
        raise ValueError("invalid image dimensions or box")
    x1, y1, x2, y2 = (float(value) for value in box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive extent")
    x1, x2 = x1 / image_width, x2 / image_width
    y1, y2 = y1 / image_height, y2 / image_height
    width, height = x2 - x1, y2 - y1
    return [
        x1,
        y1,
        x2,
        y2,
        width,
        height,
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        width * height,
        math.log((width + 1e-6) / (height + 1e-6)),
    ]


def candidate_reference_features(
    candidate_box: Sequence[float],
    reference_box: Sequence[float],
    reference_score: float,
    image_width: int,
    image_height: int,
) -> list[float]:
    candidate = normalized_box_features(candidate_box, image_width, image_height)
    reference = normalized_box_features(reference_box, image_width, image_height)
    dx, dy = candidate[6] - reference[6], candidate[7] - reference[7]
    intersection_x = max(0.0, min(candidate[2], reference[2]) - max(candidate[0], reference[0]))
    intersection_y = max(0.0, min(candidate[3], reference[3]) - max(candidate[1], reference[1]))
    intersection = intersection_x * intersection_y
    candidate_area, reference_area = candidate[8], reference[8]
    candidate_cover = intersection / candidate_area if candidate_area > 0 else 0.0
    reference_cover = intersection / reference_area if reference_area > 0 else 0.0
    return [
        *candidate,
        *reference,
        float(reference_score),
        dx,
        dy,
        abs(dx),
        abs(dy),
        math.hypot(dx, dy),
        math.log((candidate_area + 1e-6) / (reference_area + 1e-6)),
        box_iou(candidate[:4], reference[:4]),
        candidate_cover,
        reference_cover,
        intersection_x / min(candidate[4], reference[4]),
        intersection_y / min(candidate[5], reference[5]),
    ]


def relation_context_features(
    candidate_box: Sequence[float],
    proposals: Sequence[Mapping],
    relation: str,
    image_width: int,
    image_height: int,
) -> list[float]:
    relation_names = tuple(name for name, _ in RELATION_PATTERNS)
    if relation not in relation_names:
        raise ValueError(f"unknown relation: {relation}")
    one_hot = [float(name == relation) for name in relation_names]
    pair_rows = [
        candidate_reference_features(
            candidate_box,
            proposal["bbox_xyxy"],
            float(proposal["score"]),
            image_width,
            image_height,
        )
        for proposal in proposals
    ]
    if not pair_rows:
        raise ValueError("at least one reference proposal is required")
    scores = [max(0.0, float(proposal["score"])) for proposal in proposals]
    total_score = sum(scores)
    weights = (
        [score / total_score for score in scores]
        if total_score > 0
        else [1.0 / len(scores)] * len(scores)
    )
    weighted = [
        sum(weight * row[index] for weight, row in zip(weights, pair_rows, strict=True))
        for index in range(len(pair_rows[0]))
    ]
    maximum = [max(row[index] for row in pair_rows) for index in range(len(pair_rows[0]))]
    return [
        *one_hot,
        *weighted,
        *maximum,
        len(proposals) / 5.0,
        max(scores),
        sum(scores) / len(scores),
    ]
