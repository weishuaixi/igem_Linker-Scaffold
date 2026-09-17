from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from itertools import pairwise
from pathlib import Path

import torch

from rna_scaffold.geometry import validate_scaffold_length_limit
from rna_scaffold.utils import validate_rna_sequence

BASES = ("A", "U", "C", "G")


@dataclass(frozen=True)
class GenerationSettings:
    num_candidates: int = 256
    max_length: int = 512
    seed: int = 42
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 0.95
    denoise_steps: int = 12
    max_attempt_multiplier: int = 8
    max_attempts: int | None = None
    min_normalized_edit_distance: float = 0.0
    max_kmer_similarity: float | None = None
    kmer_size: int = 5
    max_homopolymer_run: int | None = None
    gc_min: float | None = None
    gc_max: float | None = None
    enforce_gc_bounds: bool = False
    min_scaffold_length: int = 8
    min_flank_length: int = 2
    short_flank_min: int | None = None
    short_flank_max: int | None = None
    long_flank_min: int | None = None
    long_flank_max: int | None = None
    length_sampling: str = "model"
    remask_strategy: str = "auto"
    self_conditioning: str = "auto"

    def __post_init__(self) -> None:
        if self.remask_strategy not in {"auto", "confidence", "random"}:
            raise ValueError("remask_strategy must be auto, confidence or random")
        if self.self_conditioning not in {"auto", "on", "off"}:
            raise ValueError("self_conditioning must be auto, on or off")
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if self.max_length < 5:
            raise ValueError("max_length must be at least 5")
        validate_scaffold_length_limit(self.max_length)
        if self.max_attempt_multiplier <= 0:
            raise ValueError("max_attempt_multiplier must be positive")
        if self.max_attempts is not None and self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if not 0 <= self.min_normalized_edit_distance <= 1:
            raise ValueError("min_normalized_edit_distance must be in [0, 1]")
        if self.max_kmer_similarity is not None and not 0 <= self.max_kmer_similarity <= 1:
            raise ValueError("max_kmer_similarity must be in [0, 1]")
        if self.kmer_size <= 0:
            raise ValueError("kmer_size must be positive")
        if self.max_homopolymer_run is not None and self.max_homopolymer_run <= 0:
            raise ValueError("max_homopolymer_run must be positive")
        if self.gc_min is not None and not 0 <= self.gc_min <= 1:
            raise ValueError("gc_min must be in [0, 1]")
        if self.gc_max is not None and not 0 <= self.gc_max <= 1:
            raise ValueError("gc_max must be in [0, 1]")
        if self.gc_min is not None and self.gc_max is not None and self.gc_min > self.gc_max:
            raise ValueError("gc_min must not exceed gc_max")
        if self.enforce_gc_bounds and self.gc_min is None and self.gc_max is None:
            raise ValueError("enforce_gc_bounds requires gc_min and/or gc_max")
        if self.length_sampling not in {"model", "uniform"}:
            raise ValueError("length_sampling must be 'model' or 'uniform'")
        if self.min_scaffold_length < 1:
            raise ValueError("min_scaffold_length must be positive")
        if self.min_flank_length < 0:
            raise ValueError("min_flank_length must be non-negative")
        flank_bounds = (
            self.short_flank_min,
            self.short_flank_max,
            self.long_flank_min,
            self.long_flank_max,
        )
        if any(value is not None for value in flank_bounds) and any(value is None for value in flank_bounds):
            raise ValueError("all four short/long flank bounds must be provided together")
        if all(value is not None for value in flank_bounds):
            if self.short_flank_min < 0 or self.long_flank_min < 0:
                raise ValueError("flank range minima must be non-negative")
            if self.short_flank_min > self.short_flank_max:
                raise ValueError("short_flank_min must not exceed short_flank_max")
            if self.long_flank_min > self.long_flank_max:
                raise ValueError("long_flank_min must not exceed long_flank_max")


@dataclass(frozen=True)
class ScaffoldCandidate:
    candidate_id: str
    full_sequence: str
    left_sequence: str
    motif: str
    right_sequence: str
    motif_start: int
    motif_end: int
    total_length: int
    normalized_log_probability: float
    checkpoint_sha256: str
    seed: int
    gc_fraction: float
    max_homopolymer: int
    base_entropy: float
    motif_preserved: bool
    valid: bool
    status: str
    generation_settings: dict = field(default_factory=dict)
    architecture_version: int = 2
    generation_audit: dict = field(default_factory=dict)
    gc_in_soft_range: bool = True
    gc_soft_penalty: float = 0.0
    flank_pair_log_probability: float = 0.0
    combined_normalized_log_probability: float | None = None
    mean_token_confidence: float | None = None
    linker_gc_fraction: float | None = None
    linker_max_homopolymer: int | None = None
    linker_composition_entropy: float | None = None
    constraint_forced_argmax_rate: float = 0.0
    constraint_log_probability_mass: float = 0.0


@dataclass(frozen=True)
class CandidateDiversityDecision:
    accepted: bool
    reason: str
    metadata: dict[str, float | bool]


@dataclass(frozen=True)
class CandidateGenerationAudit:
    requested: int
    max_attempts: int
    attempted: int
    accepted: int
    rejected: dict[str, int]


class GeneratedCandidates(list[ScaffoldCandidate]):
    """A list-compatible generated batch with reproducibility metadata."""

    def __init__(
        self,
        candidates: Sequence[ScaffoldCandidate] = (),
        audit: CandidateGenerationAudit | None = None,
        generation_settings: dict | None = None,
    ) -> None:
        super().__init__(candidates)
        self.audit = audit or CandidateGenerationAudit(
            requested=len(self), max_attempts=len(self), attempted=len(self), accepted=len(self), rejected={}
        )
        self.generation_settings = dict(
            generation_settings if generation_settings is not None else (self[0].generation_settings if self else {})
        )


def _candidate_model_score(candidate: ScaffoldCandidate) -> float:
    score = candidate.combined_normalized_log_probability
    return candidate.normalized_log_probability if score is None else score


def _combine_sequence_and_length_log_probability(
    sequence_normalized_log_probability: float,
    scaffold_length: int,
    flank_pair_log_probability: float,
) -> float:
    """Average log probability across scaffold tokens and one legal length-pair decision."""
    if scaffold_length <= 0:
        raise ValueError("scaffold_length must be positive")
    return (sequence_normalized_log_probability * scaffold_length + flank_pair_log_probability) / (scaffold_length + 1)


def _sequence_metrics(sequence: str) -> tuple[float, int, float]:
    if not sequence:
        return 0.0, 0, 0.0
    counts = Counter(sequence)
    gc_fraction = (counts["G"] + counts["C"]) / len(sequence)
    maximum_run = 1
    current_run = 1
    for previous, current in pairwise(sequence):
        current_run = current_run + 1 if current == previous else 1
        maximum_run = max(maximum_run, current_run)
    entropy = 0.0
    for base in BASES:
        probability = counts[base] / len(sequence)
        if probability:
            entropy -= probability * math.log2(probability)
    return gc_fraction, maximum_run, entropy


def _linker_sequence_metrics(left_sequence: str, right_sequence: str) -> tuple[float, int, float]:
    """Metrics over mutable linker bases without inventing adjacency across the motif."""
    linker_sequence = left_sequence + right_sequence
    gc_fraction, _, entropy = _sequence_metrics(linker_sequence)
    left_run = _sequence_metrics(left_sequence)[1]
    right_run = _sequence_metrics(right_sequence)[1]
    return gc_fraction, max(left_run, right_run), entropy


def _normalized_edit_distance(left: str, right: str) -> float:
    """Levenshtein distance scaled by the longer candidate length."""
    if not left and not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for left_index, left_base in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_base in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (left_base != right_base),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right))


def _kmer_similarity(left: str, right: str, k: int) -> float:
    left_kmers = {left[index : index + k] for index in range(max(0, len(left) - k + 1))}
    right_kmers = {right[index : index + k] for index in range(max(0, len(right) - k + 1))}
    if not left_kmers and not right_kmers:
        return 1.0
    return len(left_kmers & right_kmers) / len(left_kmers | right_kmers)


def _linker_normalized_edit_distance(
    left: tuple[str, str],
    right: tuple[str, str],
) -> float:
    left_scale = max(len(left[0]), len(right[0]))
    right_scale = max(len(left[1]), len(right[1]))
    scale = left_scale + right_scale
    if scale == 0:
        return 0.0
    return (
        _normalized_edit_distance(left[0], right[0]) * left_scale
        + _normalized_edit_distance(left[1], right[1]) * right_scale
    ) / scale


def _linker_kmer_similarity(
    left: tuple[str, str],
    right: tuple[str, str],
    k: int,
) -> float:
    def tagged_kmers(linker: tuple[str, str]) -> set[tuple[int, str]]:
        return {
            (side, sequence[index : index + k])
            for side, sequence in enumerate(linker)
            for index in range(max(0, len(sequence) - k + 1))
        }

    left_kmers = tagged_kmers(left)
    right_kmers = tagged_kmers(right)
    if not left_kmers and not right_kmers:
        return 0.0
    return len(left_kmers & right_kmers) / len(left_kmers | right_kmers)


def _gc_metadata(gc_fraction: float, settings: GenerationSettings) -> dict[str, float | bool]:
    if settings.gc_min is None and settings.gc_max is None:
        return {}
    lower_distance = max(0.0, (settings.gc_min or 0.0) - gc_fraction)
    upper_distance = max(0.0, gc_fraction - (settings.gc_max if settings.gc_max is not None else 1.0))
    return {
        "gc_in_soft_range": not (lower_distance or upper_distance),
        "gc_soft_penalty": lower_distance + upper_distance,
    }


def _hard_gc_count_in_bounds(sequence: str, settings: GenerationSettings) -> bool:
    count = sequence.count("G") + sequence.count("C")
    length = len(sequence)
    minimum = (
        0
        if settings.gc_min is None
        else int((Decimal(str(settings.gc_min)) * length).to_integral_value(rounding=ROUND_CEILING))
    )
    maximum = (
        length
        if settings.gc_max is None
        else int((Decimal(str(settings.gc_max)) * length).to_integral_value(rounding=ROUND_FLOOR))
    )
    return minimum <= count <= maximum


def assess_candidate_diversity(
    sequence: str,
    accepted_sequences: Sequence[str],
    settings: GenerationSettings,
) -> CandidateDiversityDecision:
    """Pure, ordered diversity policy for one proposed RNA sequence."""
    if sequence in accepted_sequences:
        return CandidateDiversityDecision(False, "duplicate", {})
    if any(
        _normalized_edit_distance(sequence, accepted) < settings.min_normalized_edit_distance
        for accepted in accepted_sequences
    ):
        return CandidateDiversityDecision(False, "near_duplicate", {})
    if settings.max_kmer_similarity is not None and any(
        _kmer_similarity(sequence, accepted, settings.kmer_size) > settings.max_kmer_similarity
        for accepted in accepted_sequences
    ):
        return CandidateDiversityDecision(False, "kmer_similarity", {})
    gc_fraction, maximum_run, _ = _sequence_metrics(sequence)
    if settings.max_homopolymer_run is not None and maximum_run > settings.max_homopolymer_run:
        return CandidateDiversityDecision(False, "homopolymer", {})
    metadata = _gc_metadata(gc_fraction, settings)
    if settings.enforce_gc_bounds and not _hard_gc_count_in_bounds(sequence, settings):
        return CandidateDiversityDecision(False, "gc_out_of_bounds", {})
    return CandidateDiversityDecision(True, "accepted", metadata)


def assess_linker_diversity(
    left_sequence: str,
    right_sequence: str,
    accepted_linkers: Sequence[tuple[str, str]],
    settings: GenerationSettings,
) -> CandidateDiversityDecision:
    """Assess only mutable linker bases while preserving the two real sides."""
    linker = (left_sequence, right_sequence)
    if linker in accepted_linkers:
        return CandidateDiversityDecision(False, "duplicate", {})
    if any(
        _linker_normalized_edit_distance(linker, accepted) < settings.min_normalized_edit_distance
        for accepted in accepted_linkers
    ):
        return CandidateDiversityDecision(False, "near_duplicate", {})
    if settings.max_kmer_similarity is not None and any(
        _linker_kmer_similarity(linker, accepted, settings.kmer_size) > settings.max_kmer_similarity
        for accepted in accepted_linkers
    ):
        return CandidateDiversityDecision(False, "kmer_similarity", {})
    gc_fraction, maximum_run, _ = _linker_sequence_metrics(left_sequence, right_sequence)
    if settings.max_homopolymer_run is not None and maximum_run > settings.max_homopolymer_run:
        return CandidateDiversityDecision(False, "homopolymer", {})
    metadata = _gc_metadata(gc_fraction, settings)
    linker_sequence = left_sequence + right_sequence
    if settings.enforce_gc_bounds and not _hard_gc_count_in_bounds(linker_sequence, settings):
        return CandidateDiversityDecision(False, "gc_out_of_bounds", {})
    return CandidateDiversityDecision(True, "accepted", metadata)


def resolve_generation_policies(settings: GenerationSettings, model: object) -> GenerationSettings:
    """Resolve checkpoint defaults and record the actual policy in output audit."""
    strategy = settings.remask_strategy
    conditioning = settings.self_conditioning
    if strategy == "auto":
        strategy = getattr(model, "generation_remask_strategy", "confidence")
    if conditioning == "auto":
        probability = getattr(model, "self_conditioning_probability", None)
        conditioning = "on" if probability is None or probability > 0 else "off"
    return replace(settings, remask_strategy=strategy, self_conditioning=conditioning)


def generate_candidates(
    checkpoint: str | Path,
    motif: str,
    settings: GenerationSettings | None = None,
    device: str | torch.device = "cpu",
    loaded_checkpoint: object | None = None,
) -> GeneratedCandidates:
    """Generate unique motif-preserving candidates with the learned checkpoint."""
    from rna_scaffold.checkpoints import load_scaffold_checkpoint
    from rna_scaffold.decoding import (
        DecodingSettings,
        build_flank_pair_sampler,
        iterative_denoise,
    )

    motif = motif.strip().upper().replace("T", "U")
    if len(motif) < 4 or not validate_rna_sequence(motif):
        raise ValueError("motif must contain at least four A/U/C/G nucleotides")
    settings = settings or GenerationSettings()
    loaded = loaded_checkpoint if loaded_checkpoint is not None else load_scaffold_checkpoint(checkpoint, device=device)
    settings = resolve_generation_policies(settings, loaded.model)
    maximum = min(settings.max_length, loaded.max_length)
    minimum_total_length = max(settings.min_scaffold_length, len(motif) + 2 * settings.min_flank_length)
    if minimum_total_length > maximum:
        raise ValueError("motif cannot fit with the required scaffold context")

    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device).manual_seed(settings.seed)
    decoding_settings = DecodingSettings(
        denoise_steps=settings.denoise_steps,
        temperature=settings.temperature,
        top_k=settings.top_k,
        top_p=settings.top_p,
        max_homopolymer_run=settings.max_homopolymer_run,
        gc_min=settings.gc_min,
        gc_max=settings.gc_max,
        enforce_gc_bounds=settings.enforce_gc_bounds,
        remask_strategy=settings.remask_strategy,
        use_self_conditioning=settings.self_conditioning == "on",
    )
    candidates: list[ScaffoldCandidate] = []
    max_attempts = settings.max_attempts or settings.num_candidates * settings.max_attempt_multiplier
    rejected: Counter[str] = Counter()
    attempts = 0
    loaded.model.eval()
    with torch.inference_mode():
        if settings.length_sampling == "model":
            motif_input = torch.tensor(
                [loaded.tokenizer.encode(motif)],
                dtype=torch.long,
                device=torch_device,
            )
            motif_attention = torch.ones_like(motif_input, dtype=torch.bool)
            placement_output = loaded.model(motif_input, motif_attention)
        else:
            placement_output = None
        pair_sampler = build_flank_pair_sampler(
            placement_output,
            motif_length=len(motif),
            max_length=maximum,
            min_scaffold_length=settings.min_scaffold_length,
            min_flank_length=settings.min_flank_length,
            short_flank_range=(settings.short_flank_min, settings.short_flank_max)
            if settings.short_flank_min is not None
            else None,
            long_flank_range=(settings.long_flank_min, settings.long_flank_max)
            if settings.long_flank_min is not None
            else None,
            motif=motif,
            decoding_settings=decoding_settings,
            length_sampling=settings.length_sampling,
            model_max_length=loaded.max_length,
            device=torch_device,
        )
        for _ in range(max_attempts):
            if len(candidates) >= settings.num_candidates:
                break
            attempts += 1
            left_length, right_length = pair_sampler.select(generator, sample=True)
            flank_pair_log_probability = pair_sampler.log_probability(left_length, right_length)
            total_length = left_length + len(motif) + right_length
            motif_start = left_length
            decoded = iterative_denoise(
                loaded.model.model,
                loaded.tokenizer,
                motif,
                total_length,
                motif_start,
                decoding_settings,
                generator,
                device=torch_device,
            )
            motif_end = motif_start + len(motif)
            left_sequence = decoded.sequence[:motif_start]
            right_sequence = decoded.sequence[motif_end:]
            decision = assess_linker_diversity(
                left_sequence,
                right_sequence,
                [(candidate.left_sequence, candidate.right_sequence) for candidate in candidates],
                settings,
            )
            if not decision.accepted:
                rejected[decision.reason] += 1
                continue
            preserved = decoded.sequence[motif_start:motif_end] == motif
            valid = preserved and validate_rna_sequence(decoded.sequence)
            gc_fraction, maximum_run, entropy = _sequence_metrics(decoded.sequence)
            linker_gc_fraction, linker_maximum_run, linker_entropy = _linker_sequence_metrics(
                left_sequence,
                right_sequence,
            )
            scaffold_length = left_length + right_length
            combined_normalized_log_probability = decoded.normalized_log_probability
            if settings.length_sampling == "model":
                combined_normalized_log_probability = _combine_sequence_and_length_log_probability(
                    decoded.normalized_log_probability,
                    scaffold_length,
                    flank_pair_log_probability,
                )
            candidates.append(
                ScaffoldCandidate(
                    candidate_id=f"candidate_{len(candidates) + 1:04d}",
                    full_sequence=decoded.sequence,
                    left_sequence=left_sequence,
                    motif=motif,
                    right_sequence=right_sequence,
                    motif_start=motif_start,
                    motif_end=motif_end,
                    total_length=total_length,
                    normalized_log_probability=decoded.normalized_log_probability,
                    checkpoint_sha256=loaded.checkpoint_sha256,
                    seed=settings.seed,
                    gc_fraction=gc_fraction,
                    max_homopolymer=maximum_run,
                    base_entropy=entropy,
                    motif_preserved=preserved,
                    valid=valid,
                    status="ok" if valid else "invalid",
                    generation_settings=asdict(settings),
                    gc_in_soft_range=bool(decision.metadata.get("gc_in_soft_range", True)),
                    gc_soft_penalty=float(decision.metadata.get("gc_soft_penalty", 0.0)),
                    flank_pair_log_probability=flank_pair_log_probability,
                    combined_normalized_log_probability=combined_normalized_log_probability,
                    mean_token_confidence=decoded.mean_token_confidence,
                    linker_gc_fraction=linker_gc_fraction,
                    linker_max_homopolymer=linker_maximum_run,
                    linker_composition_entropy=linker_entropy,
                    constraint_forced_argmax_rate=decoded.constraint_forced_argmax_rate,
                    constraint_log_probability_mass=decoded.constraint_log_probability_mass,
                )
            )
    audit = CandidateGenerationAudit(
        requested=settings.num_candidates,
        max_attempts=max_attempts,
        attempted=attempts,
        accepted=len(candidates),
        rejected=dict(sorted(rejected.items())),
    )
    status = "ok" if len(candidates) >= settings.num_candidates else "shortfall"
    candidates = [
        ScaffoldCandidate(
            **{
                **asdict(candidate),
                "status": candidate.status if candidate.status != "ok" else status,
                "generation_audit": asdict(audit),
            }
        )
        for candidate in candidates
    ]
    candidates.sort(
        key=lambda candidate: (
            candidate.gc_soft_penalty,
            -_candidate_model_score(candidate),
            candidate.candidate_id,
        )
    )
    return GeneratedCandidates(candidates, audit, asdict(settings))


def generate_rna_sequence(
    motif: str,
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    **settings,
) -> str:
    """Return the highest-likelihood sequence from a trained checkpoint."""
    candidates = generate_candidates(
        checkpoint=checkpoint,
        motif=motif,
        settings=GenerationSettings(**settings),
        device=device,
    )
    if not candidates:
        raise RuntimeError("generation produced no valid unique candidates")
    return max(
        candidates,
        key=_candidate_model_score,
    ).full_sequence


def _artifact_metadata(candidates: Sequence[ScaffoldCandidate]) -> dict:
    if isinstance(candidates, GeneratedCandidates):
        return {
            "architecture_version": 2,
            "generation_settings": candidates.generation_settings,
            "generation_audit": asdict(candidates.audit),
        }
    if candidates:
        first = candidates[0]
        audit = first.generation_audit or asdict(
            CandidateGenerationAudit(len(candidates), len(candidates), len(candidates), len(candidates), {})
        )
        return {
            "architecture_version": first.architecture_version,
            "generation_settings": first.generation_settings,
            "generation_audit": audit,
        }
    return {
        "architecture_version": 2,
        "generation_settings": {},
        "generation_audit": asdict(CandidateGenerationAudit(0, 0, 0, 0, {})),
    }


def write_candidates_jsonl(candidates: Sequence[ScaffoldCandidate], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    metadata = _artifact_metadata(candidates)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        if not candidates:
            handle.write(
                json.dumps(
                    {
                        "record_type": "generation_metadata",
                        **metadata,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        for candidate in candidates:
            row = asdict(candidate)
            row.update(metadata)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)


def write_candidates_fasta(candidates: Sequence[ScaffoldCandidate], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    metadata = _artifact_metadata(candidates)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "; " + json.dumps({"architecture_version": metadata["architecture_version"]}, sort_keys=True) + "\n"
        )
        handle.write("; " + json.dumps({"generation_settings": metadata["generation_settings"]}, sort_keys=True) + "\n")
        handle.write("; " + json.dumps({"generation_audit": metadata["generation_audit"]}, sort_keys=True) + "\n")
        for candidate in candidates:
            handle.write(f">{candidate.candidate_id}\n{candidate.full_sequence}\n")
    temporary.replace(output_path)
