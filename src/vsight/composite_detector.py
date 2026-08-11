"""One-forward composite GroundingDINO evidence adapter for CCV."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Mapping, Sequence

from .ccv import (
    AtomType,
    DetectorEvidence,
    DetectorProposal,
    ParsedClaim,
    PromptProvenance,
    TypedClaimParser,
)
from .cable import INVERSE_RELATIONS, SUPPORT_RELATIONS
from .attention_transport import RoleSpan, build_raft_evidence, infer_role_spans
from .trace_bind import TraceSpan, build_trace_bind_ledger
from .trajectory_ledger import build_trajectory_ledger


def _normalized(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().strip(". ").split())


@dataclass(frozen=True)
class PromptSegment:
    segment_id: str
    text: str
    atom_ids: tuple[str, ...]
    is_full: bool = False
    is_reference: bool = False
    provenance: PromptProvenance = PromptProvenance.CLAIM
    counterfactual_of: str | None = None
    relation: str | None = None
    role_swapped: bool = False


@dataclass(frozen=True)
class CompositePrompt:
    text: str
    segments: tuple[PromptSegment, ...]
    prompted_atom_ids: frozenset[str]
    counterfactual_atom_ids: frozenset[str] = frozenset()


RELATION_TEXT = {
    "to_the_left_of": "left of",
    "to_the_right_of": "right of",
    "in_front_of": "in front of",
    "next_to": "next to",
    "sitting_on": "sitting on",
    "standing_on": "standing on",
    "held_by": "held by",
    "holding": "holding",
    "wearing": "wearing",
    "riding": "riding",
    "between": "between",
    "behind": "behind",
    "above": "above",
    "below": "below",
    "beside": "beside",
    "on": "on",
    "with": "with",
    "by": "by",
}
BINARY_ATTRIBUTE_INVERSE = {"open": "closed", "closed": "open"}
TOKEN_SCORE_IGNORED_WORDS = frozenset(
    {"a", "an", "the", "this", "that", "his", "her", "their"}
)


def _relation_expression(target: str, relation: str, reference: str) -> str:
    phrase = RELATION_TEXT.get(relation, relation.replace("_", " "))
    return " ".join(value for value in (target, phrase, reference) if value)


class TypedCounterfactualPromptCompiler:
    """Compile claim and safe counterfactual segments into one detector text."""

    def compile(self, claim: ParsedClaim) -> CompositePrompt:
        return _build_composite_prompt(claim, include_counterfactuals=True)


def build_composite_prompt(claim: ParsedClaim) -> CompositePrompt:
    """Build a deduplicated detector prompt while retaining atom provenance."""

    return TypedCounterfactualPromptCompiler().compile(claim)


def build_raft_prompt(query: str, prompted_atom_ids: frozenset[str] = frozenset()) -> CompositePrompt:
    """Build a raw-query prompt for ontology-free attention role induction.

    RAFT deliberately does not expand the query with category, attribute, or
    relation vocabulary.  The detector sees the original expression once;
    role spans are inferred from its token-level attention after the forward.
    """

    text = _normalized(query)
    if not text:
        raise ValueError("query cannot be empty")
    segment = PromptSegment("full", text, (), is_full=True)
    return CompositePrompt(
        text=f"{text}.",
        segments=(segment,),
        prompted_atom_ids=prompted_atom_ids,
    )


def _build_composite_prompt(
    claim: ParsedClaim, *, include_counterfactuals: bool
) -> CompositePrompt:

    object_atom = next(
        (atom for atom in claim.known_atoms if atom.atom_type is AtomType.OBJECT), None
    )
    object_text = object_atom.text if object_atom else ""
    pending: list[PromptSegment] = []
    if object_atom:
        pending.append(PromptSegment("object", object_text, (object_atom.atom_id,)))
    for atom in claim.known_atoms:
        if atom.atom_type in {AtomType.ATTRIBUTE, AtomType.ACTION}:
            phrase = " ".join(value for value in (atom.text, object_text) if value)
            atom_ids = (
                (object_atom.atom_id, atom.atom_id)
                if object_atom else (atom.atom_id,)
            )
            pending.append(PromptSegment(atom.atom_id, phrase, atom_ids))
            inverse_attribute = BINARY_ATTRIBUTE_INVERSE.get(atom.text)
            if include_counterfactuals and inverse_attribute:
                pending.append(
                    PromptSegment(
                        f"inverse:{atom.atom_id}",
                        " ".join(value for value in (inverse_attribute, object_text) if value),
                        (),
                        provenance=PromptProvenance.INVERSE,
                        counterfactual_of=atom.atom_id,
                    )
                )
        elif atom.atom_type is AtomType.RELATION and atom.reference_text:
            pending.append(
                PromptSegment(
                    f"reference:{atom.atom_id}",
                    atom.reference_text,
                    (),
                    is_reference=True,
                )
            )
            if include_counterfactuals:
                inverse = INVERSE_RELATIONS.get(atom.relation or "")
                role_swapped = False
                inverse_target, inverse_reference = object_text, atom.reference_text
                if not inverse and (atom.relation or "") in SUPPORT_RELATIONS:
                    # Contact predicates do not have a lexical inverse.  A
                    # role-swapped edge is the only safe counterfactual slot.
                    inverse = atom.relation
                    inverse_target, inverse_reference = atom.reference_text, object_text
                    role_swapped = True
                if inverse:
                    pending.append(
                        PromptSegment(
                            f"inverse:{atom.atom_id}",
                            _relation_expression(inverse_target, inverse, inverse_reference),
                            (),
                            provenance=PromptProvenance.INVERSE,
                            counterfactual_of=atom.atom_id,
                            relation=inverse,
                            role_swapped=role_swapped,
                        )
                    )
    pending.append(
        PromptSegment("full", claim.full_expression, (), is_full=True)
    )

    # GroundingDINO can collapse repeated phrases. Merge their provenance so a
    # returned phrase remains sufficient for every atom that shared the text.
    merged: dict[tuple[str, bool, PromptProvenance], PromptSegment] = {}
    order: list[tuple[str, bool, PromptProvenance]] = []
    for row in pending:
        normalized = _normalized(row.text)
        if not normalized:
            continue
        key = (normalized, row.is_reference, row.provenance)
        if key not in merged:
            merged[key] = PromptSegment(
                row.segment_id,
                normalized,
                row.atom_ids,
                row.is_full,
                row.is_reference,
                row.provenance,
                row.counterfactual_of,
                row.relation,
                row.role_swapped,
            )
            order.append(key)
        else:
            prior = merged[key]
            merged[key] = PromptSegment(
                prior.segment_id,
                prior.text,
                tuple(dict.fromkeys((*prior.atom_ids, *row.atom_ids))),
                prior.is_full or row.is_full,
                prior.is_reference,
                prior.provenance,
                prior.counterfactual_of or row.counterfactual_of,
                prior.relation or row.relation,
                prior.role_swapped or row.role_swapped,
            )
    segments = tuple(merged[key] for key in order)
    prompted = frozenset(atom.atom_id for atom in claim.known_atoms)
    counterfactual = frozenset(
        str(row.counterfactual_of) for row in segments if row.counterfactual_of
    )
    return CompositePrompt(
        text=". ".join(row.text for row in segments) + ".",
        segments=segments,
        prompted_atom_ids=prompted,
        counterfactual_atom_ids=counterfactual,
    )


def _segment_for_label(
    label: str, segments: Sequence[PromptSegment]
) -> PromptSegment | None:
    normalized = _normalized(label)
    if not normalized:
        return None
    exact = [row for row in segments if normalized == _normalized(row.text)]
    if exact:
        return exact[0]
    label_tokens = set(re.findall(r"[a-z0-9]+", normalized)) - {"a", "an", "the"}
    if not label_tokens:
        return None
    ranked = []
    for row in segments:
        segment_tokens = set(re.findall(r"[a-z0-9]+", _normalized(row.text))) - {
            "a", "an", "the",
        }
        intersection = len(label_tokens & segment_tokens)
        union = len(label_tokens | segment_tokens)
        if intersection:
            ranked.append((intersection / union, intersection, len(segment_tokens), row))
    return max(ranked, key=lambda value: value[:3], default=(0, 0, 0, None))[3]


def _segment_token_indices(
    prompt: CompositePrompt,
    offsets: Sequence[Sequence[int]],
) -> dict[str, tuple[int, ...]]:
    """Map tokenizer offsets to the exact composite-prompt segment spans."""

    spans = []
    cursor = 0
    for segment in prompt.segments:
        start = cursor
        end = start + len(segment.text)
        spans.append((segment, start, end))
        cursor = end + 2  # ``CompositePrompt.text`` joins segments with ". ".
    output = {}
    for segment, segment_start, segment_end in spans:
        content = []
        fallback = []
        for index, raw_offset in enumerate(offsets):
            if len(raw_offset) != 2:
                raise ValueError("token offset must contain start and end")
            token_start, token_end = (int(value) for value in raw_offset)
            if token_end <= token_start:
                continue
            if token_start < segment_start or token_end > segment_end:
                continue
            token_text = prompt.text[token_start:token_end].casefold()
            if not re.search(r"[a-z0-9]", token_text):
                continue
            fallback.append(index)
            if token_text not in TOKEN_SCORE_IGNORED_WORDS:
                content.append(index)
        output[segment.segment_id] = tuple(content or fallback)
    return output


def normalize_detector_token_output(
    *,
    boxes: Sequence[Sequence[float]],
    global_scores: Sequence[float],
    token_scores: Sequence[Sequence[float]],
    token_offsets: Sequence[Sequence[int]],
    prompt: CompositePrompt,
    image_width: int,
    image_height: int,
    text_threshold: float,
    latency_ms: float | None = None,
    proposal_cap_per_segment: int = 5,
) -> DetectorEvidence:
    """Attribute boxes to typed segments from raw logits of one model pass.

    ``post_process_grounded_object_detection`` decodes every token above its
    threshold into one mixed phrase.  Overlapping composite segments can then
    be concatenated and misassigned to a single provenance slot.  This adapter
    instead averages the logits for the content tokens inside each exact
    segment span and retains the top boxes independently per segment.
    """

    if not (len(boxes) == len(global_scores) == len(token_scores)):
        raise ValueError("boxes, global scores, and token scores must have equal length")
    if not 0 <= text_threshold <= 1:
        raise ValueError("text threshold must be in [0, 1]")
    if proposal_cap_per_segment <= 0:
        raise ValueError("proposal cap must be positive")
    indices = _segment_token_indices(prompt, token_offsets)
    candidates: dict[str, list[tuple[float, int, Sequence[float], float]]] = {
        segment.segment_id: [] for segment in prompt.segments
    }
    for query_index, (raw_box, raw_global, raw_token_scores) in enumerate(
        zip(boxes, global_scores, token_scores, strict=True)
    ):
        global_score = min(1.0, max(0.0, float(raw_global)))
        for segment in prompt.segments:
            token_indices = indices[segment.segment_id]
            available = [
                min(1.0, max(0.0, float(raw_token_scores[index])))
                for index in token_indices
                if index < len(raw_token_scores)
            ]
            if not available:
                continue
            segment_score = sum(available) / len(available)
            if segment_score < text_threshold:
                continue
            # Global confidence is retained only as a deterministic tie-break;
            # the serialized score is segment-local and therefore auditable.
            candidates[segment.segment_id].append(
                (segment_score, query_index, raw_box, global_score)
            )

    proposals: list[DetectorProposal] = []
    for segment in prompt.segments:
        ranked = sorted(
            candidates[segment.segment_id],
            key=lambda value: (-value[0], -value[3], value[1]),
        )[:proposal_cap_per_segment]
        for segment_score, query_index, raw_box, _ in ranked:
            x1, y1, x2, y2 = (float(value) for value in raw_box)
            clipped = (
                max(0.0, min(float(image_width), x1)),
                max(0.0, min(float(image_height), y1)),
                max(0.0, min(float(image_width), x2)),
                max(0.0, min(float(image_height), y2)),
            )
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            proposals.append(
                DetectorProposal(
                    proposal_id=f"{segment.segment_id}:token:{query_index}",
                    box=clipped,
                    score=segment_score,
                    atom_scores={atom_id: segment_score for atom_id in segment.atom_ids},
                    full_score=segment_score if segment.is_full else None,
                    is_reference=segment.is_reference,
                    label=segment.text,
                    segment_id=segment.segment_id,
                    prompt_provenance=segment.provenance,
                    counterfactual_of=segment.counterfactual_of,
                    relation=segment.relation,
                    role_swapped=segment.role_swapped,
                )
            )
    return DetectorEvidence(
        proposals=tuple(proposals),
        image_width=image_width,
        image_height=image_height,
        evidence_complete=True,
        prompted_atom_ids=prompt.prompted_atom_ids,
        image_encoder_forwards=1,
        latency_ms=latency_ms,
        counterfactual_atom_ids=prompt.counterfactual_atom_ids,
    )


def normalize_detector_output(
    *,
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    labels: Sequence[str],
    prompt: CompositePrompt,
    image_width: int,
    image_height: int,
    latency_ms: float | None = None,
    proposal_cap_per_segment: int = 5,
) -> DetectorEvidence:
    """Convert post-processed model output into the stable CCV contract."""

    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("detector boxes, scores, and labels must have equal length")
    if proposal_cap_per_segment <= 0:
        raise ValueError("proposal cap must be positive")
    counts: dict[str, int] = {}
    proposals: list[DetectorProposal] = []
    for index, (raw_box, raw_score, label) in enumerate(
        zip(boxes, scores, labels, strict=True)
    ):
        segment = _segment_for_label(label, prompt.segments)
        if segment is None:
            continue
        count = counts.get(segment.segment_id, 0)
        if count >= proposal_cap_per_segment:
            continue
        counts[segment.segment_id] = count + 1
        x1, y1, x2, y2 = (float(value) for value in raw_box)
        clipped = (
            max(0.0, min(float(image_width), x1)),
            max(0.0, min(float(image_height), y1)),
            max(0.0, min(float(image_width), x2)),
            max(0.0, min(float(image_height), y2)),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        score = min(1.0, max(0.0, float(raw_score)))
        proposals.append(
            DetectorProposal(
                proposal_id=f"{segment.segment_id}:{index}",
                box=clipped,
                score=score,
                atom_scores={atom_id: score for atom_id in segment.atom_ids},
                full_score=score if segment.is_full else None,
                is_reference=segment.is_reference,
                label=str(label),
                segment_id=segment.segment_id,
                prompt_provenance=segment.provenance,
                counterfactual_of=segment.counterfactual_of,
                relation=segment.relation,
                role_swapped=segment.role_swapped,
            )
        )
    return DetectorEvidence(
        proposals=tuple(proposals),
        image_width=image_width,
        image_height=image_height,
        evidence_complete=True,
        prompted_atom_ids=prompt.prompted_atom_ids,
        image_encoder_forwards=1,
        latency_ms=latency_ms,
        counterfactual_atom_ids=prompt.counterfactual_atom_ids,
    )


class GroundingDINOCompositeDetector:
    """Lazy optional-dependency wrapper that performs exactly one model call."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda:0",
        box_threshold: float = 0.20,
        text_threshold: float = 0.20,
        proposal_cap_per_segment: int = 5,
        alignment: str = "legacy_label",
        raft_intervention: bool = False,
        trace_bind: bool = False,
        trace_candidate_cap: int = 5,
        parser: TypedClaimParser | None = None,
    ) -> None:
        if not 0 <= box_threshold <= 1 or not 0 <= text_threshold <= 1:
            raise ValueError("detector thresholds must be in [0, 1]")
        if alignment not in {"token_span", "legacy_label", "raft"}:
            raise ValueError("alignment must be token_span, legacy_label, or raft")
        if raft_intervention and alignment != "raft":
            raise ValueError("RAFT intervention requires alignment='raft'")
        if trace_bind and alignment != "raft":
            raise ValueError("TRACE-Bind capture requires alignment='raft'")
        if not 1 <= trace_candidate_cap <= 5:
            raise ValueError("TRACE candidate cap must be in [1, 5]")
        self.model_path = str(model_path)
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.proposal_cap_per_segment = proposal_cap_per_segment
        self.alignment = alignment
        self.raft_intervention = bool(raft_intervention)
        self.trace_bind = bool(trace_bind)
        self.trace_candidate_cap = int(trace_candidate_cap)
        self.parser = parser or TypedClaimParser()
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_path, local_files_only=True
        ).to(self.device).eval()

    def infer(
        self,
        image,
        query: str,
        *,
        upstream_box_xyxy: Sequence[float] | None = None,
        sample_id: str | None = None,
    ) -> tuple[ParsedClaim, DetectorEvidence]:
        import torch

        self._load()
        claim = self.parser.parse(query)
        prompt = (
            # RAFT evidence is deliberately independent of the legacy typed
            # parser.  The parsed claim is returned only for the compatibility
            # signature; no typed atom IDs enter the raw-query prompt.
            build_raft_prompt(query)
            if self.alignment == "raft" else build_composite_prompt(claim)
        )
        started = time.perf_counter()
        inputs = self._processor(
            images=image, text=prompt.text, return_tensors="pt"
        ).to(self.device)
        shape_capture: dict[str, object] = {}
        hook_handles = []
        if self.alignment == "raft":
            try:
                encoder_layers = getattr(self._model.model.encoder, "layers", ())
                if encoder_layers:
                    def capture_shapes(module, args, kwargs):
                        del module, args
                        shapes = kwargs.get("spatial_shapes")
                        if shapes is not None:
                            shape_capture["spatial_shapes"] = shapes.detach().cpu().tolist()

                    hook_handles.append(encoder_layers[0].register_forward_pre_hook(
                        capture_shapes, with_kwargs=True
                    ))
                decoder_module = self._model.model.decoder
                decoder_layers = getattr(decoder_module, "layers", ())
                if decoder_layers:
                    # GroundingDINO's decoder cross-attention is deformable.  The
                    # public output contains sampling weights but not locations;
                    # reproduce the location calculation in a pre-hook so the
                    # proposal-conditioned ledger can audit where each object
                    # query looked, still within the same detector forward.
                    def capture_decoder_sampling(module, args, kwargs):
                        hidden_states = kwargs.get("hidden_states")
                        if hidden_states is None and args:
                            hidden_states = args[0]
                        reference_points = kwargs.get("reference_points")
                        spatial_shapes = kwargs.get("spatial_shapes")
                        if (
                            hidden_states is None
                            or reference_points is None
                            or spatial_shapes is None
                        ):
                            return
                        position_embeddings = kwargs.get("position_embeddings")
                        if position_embeddings is not None:
                            # The hook is attached to deformable attention,
                            # while ``with_pos_embed`` belongs to its parent
                            # decoder layer in recent Transformers releases.
                            hidden_states = hidden_states + position_embeddings
                        batch_size, num_queries, _ = hidden_states.shape
                        offsets = module.sampling_offsets(hidden_states).view(
                            batch_size,
                            num_queries,
                            module.n_heads,
                            module.n_levels,
                            module.n_points,
                            2,
                        )
                        coordinates = reference_points.shape[-1]
                        if coordinates == 2:
                            normalizer = torch.stack(
                                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
                            )
                            locations = (
                                reference_points[:, :, None, :, None, :]
                                + offsets
                                / normalizer[None, None, None, :, None, :]
                            )
                        elif coordinates == 4:
                            locations = (
                                reference_points[:, :, None, :, None, :2]
                                + offsets
                                / module.n_points
                                * reference_points[:, :, None, :, None, 2:]
                                * 0.5
                            )
                        else:
                            return
                        shape_capture["decoder_sampling_locations"] = (
                            locations.detach().float().cpu().tolist()
                        )

                    hook_handles.append(
                        decoder_layers[-1].encoder_attn.register_forward_pre_hook(
                            capture_decoder_sampling, with_kwargs=True
                        )
                    )
                    if self.raft_intervention:
                        def capture_decoder_inputs(module, args, kwargs):
                            del module, args
                            # Retain the frozen encoder memory and decoder
                            # initialization tensors only until the optional
                            # batched role-mask replay completes.
                            shape_capture["decoder_inputs"] = dict(kwargs)

                        hook_handles.append(
                            decoder_module.register_forward_pre_hook(
                                capture_decoder_inputs, with_kwargs=True
                            )
                        )
            except (AttributeError, TypeError):
                hook_handles = []
        try:
            with torch.inference_mode():
                prediction = self._model(
                    **inputs, output_attentions=self.alignment == "raft"
                )  # The only detector/model forward.
        finally:
            for hook_handle in hook_handles:
                hook_handle.remove()
        result = self._processor.post_process_grounded_object_detection(
            prediction,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(self.device))
        elapsed_ms = (time.perf_counter() - started) * 1000
        labels = [str(value) for value in (result.get("text_labels") or [])]
        if len(labels) < len(result["scores"]):
            labels.extend([""] * (len(result["scores"]) - len(labels)))
        boxes = [value.tolist() for value in result["boxes"]]
        scores = [float(value) for value in result["scores"]]
        if self.alignment == "token_span":
            tokenized = self._processor.tokenizer(
                prompt.text,
                return_offsets_mapping=True,
                add_special_tokens=True,
            )
            token_ids = [int(value) for value in tokenized["input_ids"]]
            model_token_ids = [int(value) for value in inputs.input_ids[0].detach().cpu()]
            if token_ids != model_token_ids:
                raise ValueError("processor and tokenizer input ids disagree")
            probabilities = prediction.logits[0].sigmoid()
            keep = probabilities.max(dim=-1).values > self.box_threshold
            kept_probabilities = probabilities[keep].detach().cpu().tolist()
            if len(kept_probabilities) != len(boxes):
                raise ValueError("raw token logits do not align with post-processed boxes")
            evidence = normalize_detector_token_output(
                boxes=boxes,
                global_scores=scores,
                token_scores=kept_probabilities,
                token_offsets=tokenized["offset_mapping"],
                prompt=prompt,
                image_width=image.width,
                image_height=image.height,
                text_threshold=self.text_threshold,
                latency_ms=elapsed_ms,
                proposal_cap_per_segment=self.proposal_cap_per_segment,
            )
        elif self.alignment == "legacy_label":
            evidence = normalize_detector_output(
                boxes=boxes,
                scores=scores,
                labels=labels,
                prompt=prompt,
                image_width=image.width,
                image_height=image.height,
                latency_ms=elapsed_ms,
                proposal_cap_per_segment=self.proposal_cap_per_segment,
            )
        else:
            tokenized = self._processor.tokenizer(
                prompt.text,
                return_offsets_mapping=True,
                add_special_tokens=True,
            )
            token_ids = [int(value) for value in tokenized["input_ids"]]
            model_token_ids = [int(value) for value in inputs.input_ids[0].detach().cpu()]
            if token_ids != model_token_ids:
                raise ValueError("processor and tokenizer input ids disagree")
            probabilities = prediction.logits[0].sigmoid()
            keep = probabilities.max(dim=-1).values > self.box_threshold
            kept_probabilities = probabilities[keep].detach().cpu().tolist()
            kept_query_indices = (
                keep.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            )
            if len(kept_probabilities) != len(boxes):
                raise ValueError("raw token logits do not align with post-processed boxes")
            token_count = len(token_ids)
            token_offsets = [list(value) for value in tokenized["offset_mapping"]]
            token_maps, attention_metadata = _raft_token_attention(
                prediction,
                token_count=token_count,
                spatial_shapes=shape_capture.get("spatial_shapes"),
            )
            decoder_text_attention, decoder_visual_attention, decoder_metadata = (
                _raft_decoder_attention(
                    prediction,
                    token_count=token_count,
                    kept_query_indices=kept_query_indices,
                    sampling_locations=shape_capture.get("decoder_sampling_locations"),
                )
            )
            attention_metadata = {
                **attention_metadata,
                **decoder_metadata,
                "decoder_intervention_enabled": self.raft_intervention,
                "latency_cuda_synchronized": (
                    self.device.startswith("cuda") and torch.cuda.is_available()
                ),
            }
            if not token_maps:
                token_maps = [[0.0] for _ in range(token_count)]
                attention_metadata = {
                    **attention_metadata,
                    "status": "attention_unavailable",
                }
            roles = infer_role_spans(
                query=prompt.text.rstrip("."),
                token_offsets=token_offsets,
                token_maps=token_maps,
            )
            causal_role_effects = None
            if self.raft_intervention:
                causal_role_effects, intervention_metadata = (
                    _raft_decoder_role_intervention(
                        model=self._model,
                        decoder_inputs=shape_capture.get("decoder_inputs"),
                        roles=roles,
                        original_probabilities=probabilities,
                        original_hidden_state=(
                            prediction.intermediate_hidden_states[0, -1]
                        ),
                        kept_query_indices=kept_query_indices,
                    )
                )
                shape_capture.pop("decoder_inputs", None)
                attention_metadata = {
                    **attention_metadata,
                    **intervention_metadata,
                }
            raft_latency_ms = (time.perf_counter() - started) * 1000
            evidence = build_raft_evidence(
                query=prompt.text.rstrip("."),
                token_offsets=token_offsets,
                token_maps=token_maps,
                token_probabilities=[row[:token_count] for row in kept_probabilities],
                boxes=boxes,
                global_scores=scores,
                coordinates=_raft_coordinates(shape_capture.get("spatial_shapes")),
                image_width=image.width,
                image_height=image.height,
                prompted_atom_ids=prompt.prompted_atom_ids,
                text_threshold=self.text_threshold,
                proposal_cap_per_segment=self.proposal_cap_per_segment,
                proposal_query_indices=kept_query_indices,
                decoder_text_attention=decoder_text_attention,
                decoder_visual_attention=decoder_visual_attention,
                causal_role_effects=causal_role_effects,
                latency_ms=raft_latency_ms,
                attention_metadata=attention_metadata,
            )
            if self.trace_bind:
                trace_started = time.perf_counter()
                detector_only_ms = (trace_started - started) * 1000
                trace_ledger = _build_trace_ledger_from_prediction(
                    model=self._model,
                    prediction=prediction,
                    inputs=inputs,
                    query=prompt.text.rstrip("."),
                    token_offsets=token_offsets,
                    roles=roles,
                    upstream_box_xyxy=upstream_box_xyxy,
                    sample_id=sample_id or query,
                    image_width=image.width,
                    image_height=image.height,
                    detector_only_latency_ms=detector_only_ms,
                    candidate_cap=min(
                        self.trace_candidate_cap, self.proposal_cap_per_segment
                    ),
                )
                trace_added_ms = (time.perf_counter() - trace_started) * 1000
                trace_ledger["latency_ms"]["trace_added"] = trace_added_ms
                trace_ledger["latency_ms"]["total"] = (
                    (time.perf_counter() - started) * 1000
                )
                evidence = replace(
                    evidence,
                    latency_ms=float(trace_ledger["latency_ms"]["total"]),
                    trace_ledger=trace_ledger,
                )
        return claim, evidence


def _trace_span_from_role(
    role: RoleSpan,
    token_offsets: Sequence[Sequence[int]],
) -> TraceSpan:
    offsets = tuple(
        (int(token_offsets[index][0]), int(token_offsets[index][1]))
        for index in role.token_indices
        if 0 <= index < len(token_offsets)
        and len(token_offsets[index]) == 2
        and int(token_offsets[index][1]) > int(token_offsets[index][0])
    )
    indices = tuple(
        int(index)
        for index in role.token_indices
        if 0 <= index < len(token_offsets)
        and len(token_offsets[index]) == 2
        and int(token_offsets[index][1]) > int(token_offsets[index][0])
    )
    return TraceSpan(
        role=role.role,
        text=role.text,
        token_indices=indices,
        token_offsets=offsets,
        confidence=role.confidence,
        source=role.source,
    )


def _build_trace_ledger_from_prediction(
    *,
    model,
    prediction,
    inputs,
    query: str,
    token_offsets: Sequence[Sequence[int]],
    roles: Sequence[RoleSpan],
    upstream_box_xyxy: Sequence[float] | None,
    sample_id: str,
    image_width: int,
    image_height: int,
    detector_only_latency_ms: float,
    candidate_cap: int,
) -> dict[str, object]:
    """Reduce decoder states and per-layer text scores without a new forward."""

    import torch

    hidden_states = getattr(prediction, "intermediate_hidden_states", None)
    reference_points = getattr(prediction, "intermediate_reference_points", None)
    text_hidden = getattr(prediction, "encoder_last_hidden_state_text", None)
    attention_mask = getattr(inputs, "attention_mask", None)
    if hidden_states is None or reference_points is None or text_hidden is None or attention_mask is None:
        return build_trace_bind_ledger(
            sample_id=sample_id,
            query=query,
            upstream_box_xyxy=upstream_box_xyxy,
            spans=tuple(_trace_span_from_role(role, token_offsets) for role in roles),
            trajectory_ledger={"schema_version": "unavailable", "layer_count": 0},
            image_width=image_width,
            image_height=image_height,
            detector_only_latency_ms=detector_only_latency_ms,
            trace_added_latency_ms=None,
            candidate_cap=candidate_cap,
        )
    if hidden_states.ndim != 4 or reference_points.ndim != 4:
        raise ValueError("GroundingDINO decoder states have unexpected dimensions")
    hidden_states = hidden_states[0]
    reference_points = reference_points[0]
    layer_count, query_count, _ = hidden_states.shape
    token_count = len(token_offsets)
    role_indices = {
        role.role: tuple(
            int(index)
            for index in role.token_indices
            if 0 <= index < token_count
        )
        for role in roles
    }
    full_indices = tuple(
        index
        for index, raw_offset in enumerate(token_offsets)
        if len(raw_offset) == 2
        and int(raw_offset[1]) > int(raw_offset[0])
        and re.search(r"[a-z0-9]", query[int(raw_offset[0]):int(raw_offset[1])], re.I)
    )
    role_indices.setdefault("full_query", full_indices)
    role_indices = {
        role: indices for role, indices in role_indices.items() if indices
    }
    if "target" not in role_indices:
        role_indices["target"] = full_indices
    span_scores: dict[str, list[list[float]]] = {
        role: [] for role in role_indices
    }
    global_scores: list[list[float]] = []
    hidden_l2_norms = []
    hidden_cosines = []
    text_mask = attention_mask.bool()
    with torch.inference_mode():
        for layer_index in range(layer_count):
            logits = model.class_embed[layer_index](
                vision_hidden_state=hidden_states[layer_index : layer_index + 1],
                text_hidden_state=text_hidden,
                text_token_mask=text_mask,
            )
            probabilities = logits.sigmoid()[0]
            for role, indices in role_indices.items():
                valid = [index for index in indices if index < probabilities.shape[-1]]
                values = (
                    probabilities[:, valid].mean(dim=-1).detach().float().cpu().tolist()
                    if valid else [0.0] * query_count
                )
                span_scores[role].append([min(1.0, max(0.0, float(value))) for value in values])
            global_scores.append(
                probabilities[:, :token_count].max(dim=-1).values.detach().float().cpu().tolist()
            )
            hidden_l2_norms.append(hidden_states[layer_index].norm(dim=-1).detach().float().cpu().tolist())
            if layer_index == 0:
                hidden_cosines.append([None] * query_count)
            else:
                cosine = torch.nn.functional.cosine_similarity(
                    hidden_states[layer_index - 1], hidden_states[layer_index], dim=-1
                )
                hidden_cosines.append(cosine.detach().float().cpu().tolist())
    trajectory = build_trajectory_ledger(
        layer_boxes_cxcywh=reference_points.detach().float().cpu().tolist(),
        span_scores=span_scores,
        candidate_roles=("target", "reference"),
        candidate_cap=candidate_cap,
        global_scores=global_scores,
        hidden_l2_norms=hidden_l2_norms,
        hidden_cosine_to_previous=hidden_cosines,
    )
    spans = tuple(_trace_span_from_role(role, token_offsets) for role in roles)
    spans = (*spans, TraceSpan(
        role="full_query",
        text=query,
        token_indices=tuple(full_indices),
        token_offsets=tuple(
            (int(token_offsets[index][0]), int(token_offsets[index][1]))
            for index in full_indices
        ),
        confidence=1.0,
        source="raw_query",
    ))
    return build_trace_bind_ledger(
        sample_id=sample_id,
        query=query,
        upstream_box_xyxy=upstream_box_xyxy,
        spans=spans,
        trajectory_ledger=trajectory,
        image_width=image_width,
        image_height=image_height,
        detector_only_latency_ms=detector_only_latency_ms,
        trace_added_latency_ms=None,
        candidate_cap=candidate_cap,
    )


def _raft_token_attention(
    prediction,
    *,
    token_count: int,
    spatial_shapes: object,
) -> tuple[list[list[float]], dict[str, object]]:
    """Extract encoder text-to-vision attention without a second forward."""

    groups = getattr(prediction, "encoder_attentions", None)
    if not groups or len(groups) < 2 or not groups[1]:
        return [], {"status": "attention_unavailable"}
    layers = []
    head_groups = []
    for layer in groups[1]:
        tensor = layer.detach().float().cpu()
        if tensor.ndim == 4:
            # [batch, heads, text, vision] -> first batch, average heads.
            tensor = tensor[0]
        if tensor.ndim != 3:
            continue
        # Some Transformers versions omit batch for encoder attentions:
        # [heads, text, vision].
        head_groups.append(tensor[:, :token_count, :].tolist())
        layers.append(tensor[:, :token_count, :].mean(dim=0).tolist())
    if not layers:
        return [], {"status": "attention_unavailable"}
    width = len(layers[0][0]) if layers[0] else 0
    result = []
    for token_index in range(token_count):
        values = [
            sum(layer[token_index][position] for layer in layers) / len(layers)
            for position in range(width)
        ]
        total = sum(max(0.0, value) for value in values)
        result.append(
            [max(0.0, value) / total for value in values]
            if total > 0 else [0.0] * width
        )
    def agreement(vectors: Sequence[Sequence[float]]) -> float | None:
        if len(vectors) < 2:
            return None
        values = []
        for index, left in enumerate(vectors):
            for right in vectors[index + 1:]:
                values.append(
                    sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
                    / max(
                        1e-12,
                        (
                            sum(float(a) * float(a) for a in left)
                            * sum(float(b) * float(b) for b in right)
                        ) ** 0.5,
                    )
                )
        return sum(values) / len(values) if values else None

    layer_vectors = [
        [value for row in layer for value in row]
        for layer in layers
    ]
    head_vectors = [
        [value for row in head for value in row]
        for head in (head_groups[0] if head_groups else [])
    ]
    decoder_metadata: dict[str, object] = {}
    decoder_groups = getattr(prediction, "decoder_attentions", None)
    if decoder_groups and len(decoder_groups) >= 2 and decoder_groups[1]:
        sample = decoder_groups[1][0]
        decoder_metadata = {
            "decoder_text_attention_layers": len(decoder_groups[1]),
            "decoder_text_attention_shape": list(sample.shape),
        }
    return result, {
        "status": "ok",
        "encoder_layers": len(layers),
        "vision_tokens": width,
        "text_tokens": token_count,
        "aggregation": "encoder_text_to_vision_mean_heads_layers",
        "head_agreement": agreement(head_vectors),
        "layer_agreement": agreement(layer_vectors),
        **decoder_metadata,
    }


def _raft_decoder_attention(
    prediction,
    *,
    token_count: int,
    kept_query_indices: Sequence[int],
    sampling_locations: object,
) -> tuple[
    list[list[float]],
    list[list[list[float]]],
    dict[str, object],
]:
    """Extract proposal-conditioned decoder text/vision attention.

    GroundingDINO returns decoder text cross-attention as ``[B,H,Q,T]`` and
    deformable vision weights as ``[B,Q,H,L,P]``.  The latter is paired with
    sampling locations captured from the same final decoder layer, yielding a
    compact list of normalized ``(x, y, weight)`` samples per kept object
    query.  No second detector call or category vocabulary is used.
    """

    groups = getattr(prediction, "decoder_attentions", None)
    if not groups or len(groups) < 3:
        return [], [], {"decoder_binding_status": "attention_unavailable"}
    text_groups = groups[1]
    vision_groups = groups[2]
    if not text_groups or not vision_groups or not kept_query_indices:
        return [], [], {"decoder_binding_status": "attention_unavailable"}

    import torch

    text_layers = []
    text_shape = None
    for layer in text_groups:
        tensor = layer.detach().float().cpu()
        if tensor.ndim != 4:
            continue
        tensor = tensor[0]  # [heads, queries, text]
        text_shape = list(layer.shape)
        indices = torch.as_tensor(list(kept_query_indices), dtype=torch.long)
        text_layers.append(tensor[:, indices, :token_count].mean(dim=0))
    if not text_layers:
        return [], [], {"decoder_binding_status": "attention_unavailable"}
    text_stack = torch.stack(text_layers, dim=0)
    text_maps = text_stack.mean(dim=0)
    text_maps = text_maps / text_maps.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    layer_agreement = None
    if len(text_layers) >= 2:
        vectors = [row.reshape(-1) for row in text_layers]
        similarities = []
        for index, left in enumerate(vectors):
            for right in vectors[index + 1 :]:
                similarities.append(
                    float(
                        torch.nn.functional.cosine_similarity(
                            left.unsqueeze(0), right.unsqueeze(0), dim=-1
                        )[0]
                    )
                )
        layer_agreement = sum(similarities) / len(similarities) if similarities else None

    visual_layer = vision_groups[-1].detach().float().cpu()
    if visual_layer.ndim != 5:
        return (
            text_maps.tolist(),
            [[] for _ in kept_query_indices],
            {
                "decoder_binding_status": "text_attention_only",
                "decoder_text_attention_layers": len(text_layers),
                "decoder_text_attention_shape": text_shape,
                "decoder_text_layer_agreement": layer_agreement,
            },
        )
    visual_layer = visual_layer[0]  # [queries, heads, levels, points]
    visual_shape = list(vision_groups[-1].shape)
    locations = None
    if sampling_locations is not None:
        try:
            locations = torch.as_tensor(sampling_locations, dtype=torch.float32)
            if locations.ndim == 6:
                locations = locations[0]  # [queries, heads, levels, points, 2]
            if locations.ndim != 5:
                locations = None
        except (TypeError, ValueError):
            locations = None

    visual_samples: list[list[list[float]]] = []
    for query_index in kept_query_indices:
        if query_index < 0 or query_index >= visual_layer.shape[0]:
            visual_samples.append([])
            continue
        weights = visual_layer[query_index]
        if locations is None or query_index >= locations.shape[0]:
            visual_samples.append([])
            continue
        points = locations[query_index]
        heads = max(int(weights.shape[0]), 1)
        samples = []
        for head_index in range(weights.shape[0]):
            for level_index in range(weights.shape[1]):
                for point_index in range(weights.shape[2]):
                    x, y = points[head_index, level_index, point_index].tolist()
                    weight = float(weights[head_index, level_index, point_index]) / heads
                    samples.append([float(x), float(y), max(0.0, weight)])
        visual_samples.append(samples)
    return (
        text_maps.tolist(),
        visual_samples,
        {
            "decoder_binding_status": "ok" if locations is not None else "text_attention_only",
            "decoder_text_attention_layers": len(text_layers),
            "decoder_text_attention_shape": text_shape,
            "decoder_text_layer_agreement": layer_agreement,
            "decoder_vision_attention_shape": visual_shape,
            "decoder_sampling_locations": locations is not None,
            "decoder_binding_queries": len(kept_query_indices),
        },
    )


def _raft_decoder_role_intervention(
    *,
    model,
    decoder_inputs: object,
    roles: Sequence[RoleSpan],
    original_probabilities,
    original_hidden_state,
    kept_query_indices: Sequence[int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Replay the frozen decoder with batched role-token masks.

    Vision/text encoder memories and the initial object queries come from the
    original detector forward.  Target, predicate and reference masks are
    batched into one decoder replay; the detection head is evaluated with the
    original token mask so score changes reflect decoder access to the role,
    rather than trivially hiding that token from the classifier.
    """

    if not isinstance(decoder_inputs, Mapping):
        return [], {"decoder_intervention_status": "inputs_unavailable"}
    active_roles = [role for role in roles if role.token_indices]
    if not active_roles or not kept_query_indices:
        return [], {"decoder_intervention_status": "not_applicable"}
    required = {
        "inputs_embeds",
        "vision_encoder_hidden_states",
        "vision_encoder_attention_mask",
        "text_encoder_hidden_states",
        "text_encoder_attention_mask",
        "reference_points",
        "spatial_shapes",
        "spatial_shapes_list",
        "level_start_index",
        "valid_ratios",
    }
    if any(key not in decoder_inputs for key in required):
        return [], {"decoder_intervention_status": "inputs_incomplete"}

    import torch

    branches = len(active_roles)
    batch_keys = {
        "inputs_embeds",
        "vision_encoder_hidden_states",
        "vision_encoder_attention_mask",
        "text_encoder_hidden_states",
        "text_encoder_attention_mask",
        "reference_points",
        "valid_ratios",
    }
    replay_inputs = {}
    for key, value in decoder_inputs.items():
        if key in batch_keys and isinstance(value, torch.Tensor):
            repeats = (branches,) + (1,) * (value.ndim - 1)
            replay_inputs[key] = value.repeat(repeats)
        else:
            replay_inputs[key] = value
    original_decoder_mask = decoder_inputs["text_encoder_attention_mask"].bool()
    intervention_mask = original_decoder_mask.repeat(branches, 1)
    for branch_index, role in enumerate(active_roles):
        for token_index in role.token_indices:
            if 0 <= token_index < intervention_mask.shape[-1]:
                intervention_mask[branch_index, token_index] = True
    replay_inputs.update(
        text_encoder_attention_mask=intervention_mask,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    device = replay_inputs["inputs_embeds"].device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        replay = model.model.decoder(**replay_inputs)
        final_hidden = replay.intermediate_hidden_states[:, -1]
        original_token_mask = (~original_decoder_mask).bool().repeat(branches, 1)
        counterfactual_logits = model.class_embed[-1](
            vision_hidden_state=final_hidden,
            text_hidden_state=replay_inputs["text_encoder_hidden_states"],
            text_token_mask=original_token_mask,
        )
        counterfactual_probabilities = counterfactual_logits.sigmoid()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000

    effects = []
    drift_by_branch_query = {}
    selectivity_by_branch_query = {}
    for branch_index, _ in enumerate(active_roles):
        branch_drifts = {}
        for model_query_index in kept_query_indices:
            if (
                model_query_index < 0
                or model_query_index >= original_hidden_state.shape[0]
            ):
                continue
            similarity = torch.nn.functional.cosine_similarity(
                original_hidden_state[model_query_index].unsqueeze(0),
                final_hidden[branch_index, model_query_index].unsqueeze(0),
                dim=-1,
            )[0]
            branch_drifts[int(model_query_index)] = min(
                1.0, max(0.0, 1.0 - float(similarity))
            )
        total_drift = sum(branch_drifts.values())
        for model_query_index, drift in branch_drifts.items():
            drift_by_branch_query[(branch_index, model_query_index)] = drift
            selectivity_by_branch_query[(branch_index, model_query_index)] = (
                drift / total_drift if total_drift > 1e-12 else 0.0
            )
    for branch_index, role in enumerate(active_roles):
        valid_indices = [
            index
            for index in role.token_indices
            if 0 <= index < counterfactual_probabilities.shape[-1]
        ]
        if not valid_indices:
            continue
        for model_query_index in kept_query_indices:
            if model_query_index < 0 or model_query_index >= original_probabilities.shape[0]:
                continue
            original_score = float(
                original_probabilities[model_query_index, valid_indices].mean()
            )
            counterfactual_score = float(
                counterfactual_probabilities[
                    branch_index, model_query_index, valid_indices
                ].mean()
            )
            causal_drop = original_score - counterfactual_score
            relative_drop = causal_drop / max(original_score, 1e-12)
            hidden_drift = drift_by_branch_query.get(
                (branch_index, int(model_query_index)), 0.0
            )
            hidden_drift_selectivity = selectivity_by_branch_query.get(
                (branch_index, int(model_query_index)), 0.0
            )
            effects.append(
                {
                    "model_query_index": int(model_query_index),
                    "masked_role": role.role,
                    "original_score": min(1.0, max(0.0, original_score)),
                    "counterfactual_score": min(
                        1.0, max(0.0, counterfactual_score)
                    ),
                    "causal_drop": min(1.0, max(-1.0, causal_drop)),
                    "relative_drop": min(1.0, max(-1.0, relative_drop)),
                    "hidden_cosine_drift": hidden_drift,
                    "hidden_drift_selectivity": hidden_drift_selectivity,
                }
            )
    return effects, {
        "decoder_intervention_status": "ok",
        "decoder_counterfactual_branches": branches,
        "decoder_counterfactual_replays": 1,
        "decoder_counterfactual_latency_ms": latency_ms,
        "decoder_counterfactual_additional_image_encoder_forwards": 0,
    }


def _raft_coordinates(spatial_shapes: object) -> list[tuple[float, float]]:
    if not spatial_shapes:
        return []
    coordinates = []
    for shape in spatial_shapes:
        if len(shape) != 2:
            continue
        height, width = int(shape[0]), int(shape[1])
        if height <= 0 or width <= 0:
            continue
        coordinates.extend(
            ((column + 0.5) / width, (row + 0.5) / height)
            for row in range(height)
            for column in range(width)
        )
    return coordinates


def evidence_to_dict(evidence: DetectorEvidence) -> dict[str, object]:
    payload = {
        "schema_version": (
            "vsight_ccv_composite_evidence_v5"
            if evidence.trace_ledger is not None
            else (
            "vsight_ccv_composite_evidence_v4"
            if evidence.attention_ledger is not None
            else "vsight_ccv_composite_evidence_v3"
            )
        ),
        "image_width": evidence.image_width,
        "image_height": evidence.image_height,
        "evidence_complete": evidence.evidence_complete,
        "prompted_atom_ids": sorted(evidence.prompted_atom_ids),
        "null_support": evidence.null_support,
        "contradiction_support": evidence.contradiction_support,
        "image_encoder_forwards": evidence.image_encoder_forwards,
        "latency_ms": evidence.latency_ms,
        "counterfactual_atom_ids": sorted(evidence.counterfactual_atom_ids),
        "proposals": [
            {
                "proposal_id": row.proposal_id,
                "bbox_xyxy": list(row.box),
                "score": row.score,
                "atom_scores": dict(row.atom_scores),
                "full_score": row.full_score,
                "is_reference": row.is_reference,
                "label": row.label,
                "segment_id": row.segment_id,
                "prompt_provenance": row.prompt_provenance.value,
                "counterfactual_of": row.counterfactual_of,
                "relation": row.relation,
                "role_swapped": row.role_swapped,
            }
            for row in evidence.proposals
        ],
    }
    if evidence.attention_ledger is not None:
        payload["attention_ledger"] = dict(evidence.attention_ledger)
    if evidence.trace_ledger is not None:
        payload["trace_ledger"] = dict(evidence.trace_ledger)
    return payload


def evidence_from_dict(row: Mapping[str, object]) -> DetectorEvidence:
    return DetectorEvidence(
        proposals=tuple(
            DetectorProposal(
                proposal_id=str(item["proposal_id"]),
                box=tuple(item["bbox_xyxy"]),
                score=float(item["score"]),
                atom_scores=dict(item.get("atom_scores") or {}),
                full_score=(
                    float(item["full_score"])
                    if item.get("full_score") is not None else None
                ),
                is_reference=bool(item.get("is_reference")),
                label=str(item.get("label") or "") or None,
                segment_id=str(item.get("segment_id") or "") or None,
                prompt_provenance=str(item.get("prompt_provenance") or "claim"),
                counterfactual_of=str(item.get("counterfactual_of") or "") or None,
                relation=str(item.get("relation") or "") or None,
                role_swapped=bool(item.get("role_swapped")),
            )
            for item in row.get("proposals", [])
        ),
        image_width=int(row["image_width"]),
        image_height=int(row["image_height"]),
        evidence_complete=bool(row.get("evidence_complete")),
        prompted_atom_ids=frozenset(str(value) for value in row.get("prompted_atom_ids", [])),
        null_support=float(row.get("null_support") or 0.0),
        contradiction_support=float(row.get("contradiction_support") or 0.0),
        image_encoder_forwards=int(row.get("image_encoder_forwards") or 1),
        latency_ms=(float(row["latency_ms"]) if row.get("latency_ms") is not None else None),
        localized_attention=dict(row.get("localized_attention") or {}) or None,
        attention_ledger=dict(row.get("attention_ledger") or {}) or None,
        trace_ledger=dict(row.get("trace_ledger") or {}) or None,
        counterfactual_atom_ids=frozenset(
            str(value) for value in row.get("counterfactual_atom_ids", [])
        ),
    )
