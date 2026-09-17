from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from rna_scaffold.geometry import (
    minimum_legal_total_length,
    record_supports_legal_geometry,
    validate_scaffold_length_limit,
)
from rna_scaffold.records import RnaSequenceRecord
from rna_scaffold.splits import SplitManifest, build_cluster_disjoint_manifest
from rna_scaffold.tokenizer import RnaTokenizer

DEFAULT_MOTIF_LENGTH_BUCKETS = (
    (4, 15, 0.15),
    (16, 31, 0.30),
    (32, 63, 0.30),
    (64, 127, 0.20),
    (128, 512, 0.05),
)


@dataclass(frozen=True)
class MotifScaffoldExample:
    motif: str
    target_sequence: str
    motif_start: int
    source_start: int = 0

    @property
    def motif_end(self) -> int:
        return self.motif_start + len(self.motif)

    @property
    def total_length(self) -> int:
        return len(self.target_sequence)


def _normalize_preferred_motifs(
    preferred_motifs: tuple[str, ...] | list[str] | str | None,
) -> tuple[str, ...]:
    if preferred_motifs is None:
        return ()
    raw_motifs = (preferred_motifs,) if isinstance(preferred_motifs, str) else preferred_motifs
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_motif in raw_motifs:
        motif = str(raw_motif).strip().upper().replace("T", "U")
        if not motif or set(motif) - set("AUCG"):
            raise ValueError("preferred_motifs must contain only non-empty canonical RNA sequences")
        if motif not in seen:
            seen.add(motif)
            normalized.append(motif)
    return tuple(normalized)


def sample_motif_example(
    record: RnaSequenceRecord,
    generator: torch.Generator,
    min_motif_length: int = 4,
    max_motif_length: int | None = None,
    motif_length_buckets: tuple[tuple[int, int, float], ...] | list[list[float]] | None = None,
    min_flank_length: int = 2,
    min_total_scaffold_length: int = 8,
    preferred_total_scaffold_length: int = 24,
    short_flank_min: int | None = None,
    short_flank_max: int | None = None,
    long_flank_min: int | None = None,
    long_flank_max: int | None = None,
) -> MotifScaffoldExample:
    """Sample a reproducible, non-fixed motif while leaving scaffold context."""
    if min_motif_length < 1:
        raise ValueError("min_motif_length must be positive")
    if min_flank_length < 0:
        raise ValueError("min_flank_length must be non-negative")
    if min_total_scaffold_length < 0:
        raise ValueError("min_total_scaffold_length must be non-negative")
    minimum_total = minimum_legal_total_length(
        min_motif_length,
        min_flank_length,
        min_total_scaffold_length,
    )
    if len(record.sequence) < minimum_total:
        raise ValueError("sequence is too short for the requested motif and scaffold context")
    asymmetric_pairs = _asymmetric_flank_pairs(
        short_flank_min,
        short_flank_max,
        long_flank_min,
        long_flank_max,
        min_flank_length=min_flank_length,
    )
    if not asymmetric_pairs and preferred_total_scaffold_length < min_total_scaffold_length:
        raise ValueError("preferred_total_scaffold_length must be at least min_total_scaffold_length")
    minimum_flank_total = min(
        (left + right for left, right in asymmetric_pairs),
        default=2 * min_flank_length,
    )
    largest = len(record.sequence) - minimum_flank_total
    if max_motif_length is not None:
        largest = min(max_motif_length, largest)
    if largest < min_motif_length:
        raise ValueError("sequence is too short for the requested motif and scaffold context")
    motif_length = _sample_motif_length(
        generator,
        min_motif_length,
        largest,
        motif_length_buckets,
    )
    if asymmetric_pairs:
        legal_pairs = [
            pair
            for pair in asymmetric_pairs
            if min_total_scaffold_length <= motif_length + sum(pair) <= len(record.sequence)
        ]
        if not legal_pairs:
            raise ValueError("sequence is too short for the requested asymmetric linker geometry")
        pair_index = int(torch.randint(0, len(legal_pairs), (1,), generator=generator).item())
        left_length, right_length = legal_pairs[pair_index]
        target_length = left_length + motif_length + right_length
        source_start = int(torch.randint(0, len(record.sequence) - target_length + 1, (1,), generator=generator).item())
        target_sequence = record.sequence[source_start : source_start + target_length]
        motif = target_sequence[left_length : left_length + motif_length]
        return MotifScaffoldExample(motif, target_sequence, left_length, source_start)
    first_start = min_flank_length
    last_start = len(record.sequence) - motif_length - min_flank_length
    motif_start = int(torch.randint(first_start, last_start + 1, (1,), generator=generator).item())
    motif = record.sequence[motif_start : motif_start + motif_length]
    return MotifScaffoldExample(motif, record.sequence, motif_start)


def _asymmetric_flank_pairs(
    short_flank_min: int | None,
    short_flank_max: int | None,
    long_flank_min: int | None,
    long_flank_max: int | None,
    *,
    min_flank_length: int = 0,
) -> tuple[tuple[int, int], ...]:
    bounds = (short_flank_min, short_flank_max, long_flank_min, long_flank_max)
    if not any(value is not None for value in bounds):
        return ()
    if any(value is None for value in bounds):
        raise ValueError("all four short/long flank bounds must be provided together")
    short_min, short_max, long_min, long_max = (int(value) for value in bounds)
    if short_min < 0 or long_min < 0 or short_min > short_max or long_min > long_max:
        raise ValueError("invalid short/long flank ranges")
    oriented_pairs = {
        pair
        for short_length in range(short_min, short_max + 1)
        for long_length in range(long_min, long_max + 1)
        if long_length > short_length
        for pair in ((short_length, long_length), (long_length, short_length))
        if pair[0] >= min_flank_length and pair[1] >= min_flank_length
    }
    if not oriented_pairs:
        raise ValueError("short/long flank ranges contain no strictly asymmetric pair")
    return tuple(sorted(oriented_pairs))


def _sequence_occurrences(sequence: str, motif: str) -> tuple[int, ...]:
    starts: list[int] = []
    start = sequence.find(motif)
    while start >= 0:
        starts.append(start)
        start = sequence.find(motif, start + 1)
    return tuple(starts)


def _sample_preferred_motif_example(
    record: RnaSequenceRecord,
    generator: torch.Generator,
    preferred_motifs: tuple[str, ...],
    *,
    max_length: int,
    min_motif_length: int,
    max_motif_length: int | None,
    min_flank_length: int,
    min_total_scaffold_length: int,
    asymmetric_pairs: tuple[tuple[int, int], ...],
) -> MotifScaffoldExample | None:
    """Sample a real preferred-motif occurrence, returning ``None`` when none is feasible."""
    sequence = record.sequence
    maximum_target_length = min(len(sequence), int(max_length))
    feasible: list[tuple[str, int, tuple[tuple[int, int], ...]]] = []
    for motif in preferred_motifs:
        motif_length = len(motif)
        if motif_length < min_motif_length:
            continue
        if max_motif_length is not None and motif_length > max_motif_length:
            continue
        for motif_start in _sequence_occurrences(sequence, motif):
            if asymmetric_pairs:
                legal_pairs = tuple(
                    (left_length, right_length)
                    for left_length, right_length in asymmetric_pairs
                    if min_total_scaffold_length <= left_length + motif_length + right_length <= maximum_target_length
                    and left_length <= motif_start
                    and right_length <= len(sequence) - motif_start - motif_length
                )
                if legal_pairs:
                    feasible.append((motif, motif_start, legal_pairs))
                continue

            target_length = maximum_target_length
            if target_length < minimum_legal_total_length(
                motif_length,
                min_flank_length,
                min_total_scaffold_length,
            ):
                continue
            first_source_start = max(
                0,
                motif_start + motif_length + min_flank_length - target_length,
            )
            last_source_start = min(
                motif_start - min_flank_length,
                len(sequence) - target_length,
            )
            if first_source_start <= last_source_start:
                feasible.append((motif, motif_start, ((first_source_start, last_source_start),)))

    if not feasible:
        return None
    occurrence_index = int(torch.randint(0, len(feasible), (1,), generator=generator).item())
    motif, absolute_motif_start, choices = feasible[occurrence_index]
    if asymmetric_pairs:
        choice_index = int(torch.randint(0, len(choices), (1,), generator=generator).item())
        left_length, right_length = choices[choice_index]
        source_start = absolute_motif_start - left_length
        target_length = left_length + len(motif) + right_length
    else:
        first_source_start, last_source_start = choices[0]
        source_start = int(torch.randint(first_source_start, last_source_start + 1, (1,), generator=generator).item())
        left_length = absolute_motif_start - source_start
        target_length = maximum_target_length
    target_sequence = sequence[source_start : source_start + target_length]
    return MotifScaffoldExample(motif, target_sequence, left_length, source_start)


def _sample_motif_length(
    generator: torch.Generator,
    minimum: int,
    maximum: int,
    buckets: tuple[tuple[int, int, float], ...] | list[list[float]] | None,
) -> int:
    if not buckets:
        return int(torch.randint(minimum, maximum + 1, (1,), generator=generator).item())
    valid: list[tuple[int, int, float]] = []
    for raw_lower, raw_upper, raw_weight in buckets:
        lower = max(minimum, int(raw_lower))
        upper = min(maximum, int(raw_upper))
        weight = float(raw_weight)
        if weight < 0:
            raise ValueError("motif bucket weights must be non-negative")
        if lower <= upper and weight > 0:
            valid.append((lower, upper, weight))
    if not valid:
        raise ValueError("no motif length bucket overlaps the valid sequence range")
    weights = torch.tensor([bucket[2] for bucket in valid], dtype=torch.float64)
    bucket_index = int(torch.multinomial(weights, 1, generator=generator).item())
    lower, upper, _ = valid[bucket_index]
    return int(torch.randint(lower, upper + 1, (1,), generator=generator).item())


class RnaMotifDenoisingDataset(Dataset):
    """Joint left/right scaffold denoising with immutable motif positions."""

    def __init__(
        self,
        records: list[RnaSequenceRecord],
        tokenizer: RnaTokenizer,
        max_length: int = 512,
        min_motif_length: int = 4,
        max_motif_length: int | None = None,
        motif_length_buckets: tuple[tuple[int, int, float], ...]
        | list[list[float]]
        | None = DEFAULT_MOTIF_LENGTH_BUCKETS,
        min_flank_length: int = 2,
        min_total_scaffold_length: int = 8,
        preferred_total_scaffold_length: int = 24,
        full_mask_probability: float = 0.35,
        span_mask_probability: float = 0.35,
        min_random_mask_fraction: float = 0.30,
        max_random_mask_fraction: float = 0.80,
        mean_span_length: int = 8,
        iterative_mask_steps: int | None = None,
        context_noise_probability: float = 0.0,
        short_flank_min: int | None = None,
        short_flank_max: int | None = None,
        long_flank_min: int | None = None,
        long_flank_max: int | None = None,
        seed: int = 42,
        allow_empty: bool = False,
        views_per_record: int = 1,
        preferred_motifs: tuple[str, ...] | list[str] | str | None = None,
        preferred_motif_probability: float = 0.0,
        iterative_full_mask_probability: float | None = None,
        stage_aware_context_noise: bool = False,
    ) -> None:
        max_length = validate_scaffold_length_limit(max_length)
        self.input_record_count = len(records)
        self.records = eligible_denoising_records(
            records,
            max_length=max_length,
            min_motif_length=min_motif_length,
            min_flank_length=min_flank_length,
            min_total_scaffold_length=min_total_scaffold_length,
            short_flank_min=short_flank_min,
            short_flank_max=short_flank_max,
            long_flank_min=long_flank_min,
            long_flank_max=long_flank_max,
        )
        self.excluded_record_count = self.input_record_count - len(self.records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_motif_length = min_motif_length
        self.max_motif_length = max_motif_length
        self.motif_length_buckets = motif_length_buckets
        self.min_flank_length = min_flank_length
        self.min_total_scaffold_length = min_total_scaffold_length
        self.preferred_total_scaffold_length = preferred_total_scaffold_length
        if not 0 <= full_mask_probability <= 1:
            raise ValueError("full_mask_probability must be in [0, 1]")
        if not 0 <= span_mask_probability <= 1:
            raise ValueError("span_mask_probability must be in [0, 1]")
        if full_mask_probability + span_mask_probability > 1:
            raise ValueError("full and span mask probabilities must sum to at most one")
        if not 0 < min_random_mask_fraction <= max_random_mask_fraction <= 1:
            raise ValueError("random mask fractions must satisfy 0 < min <= max <= 1")
        if mean_span_length < 2:
            raise ValueError("mean_span_length must be at least two")
        if iterative_mask_steps is not None and iterative_mask_steps < 2:
            raise ValueError("iterative_mask_steps must be at least two")
        if iterative_mask_steps is not None and (full_mask_probability != 0 or span_mask_probability != 0):
            raise ValueError(
                "full_mask_probability and span_mask_probability must be zero when iterative_mask_steps is enabled"
            )
        if iterative_full_mask_probability is not None:
            if iterative_mask_steps is None:
                raise ValueError("iterative_full_mask_probability requires iterative_mask_steps")
            if not 0 <= iterative_full_mask_probability <= 1:
                raise ValueError("iterative_full_mask_probability must be in [0, 1]")
        if not 0 <= context_noise_probability < 1:
            raise ValueError("context_noise_probability must be in [0, 1)")
        if isinstance(views_per_record, bool) or int(views_per_record) != views_per_record or views_per_record < 1:
            raise ValueError("views_per_record must be a positive integer")
        if not 0 <= preferred_motif_probability <= 1:
            raise ValueError("preferred_motif_probability must be in [0, 1]")
        normalized_preferred_motifs = _normalize_preferred_motifs(preferred_motifs)
        if preferred_motif_probability and not normalized_preferred_motifs:
            raise ValueError("preferred_motifs must be provided when preferred_motif_probability is positive")
        self.full_mask_probability = full_mask_probability
        self.span_mask_probability = span_mask_probability
        self.min_random_mask_fraction = min_random_mask_fraction
        self.max_random_mask_fraction = max_random_mask_fraction
        self.mean_span_length = mean_span_length
        self.iterative_mask_steps = iterative_mask_steps
        self.iterative_full_mask_probability = iterative_full_mask_probability
        self.context_noise_probability = context_noise_probability
        self.stage_aware_context_noise = bool(stage_aware_context_noise)
        self.views_per_record = int(views_per_record)
        self.preferred_motifs = normalized_preferred_motifs
        self.preferred_motif_probability = float(preferred_motif_probability)
        self.short_flank_min = short_flank_min
        self.short_flank_max = short_flank_max
        self.long_flank_min = long_flank_min
        self.long_flank_max = long_flank_max
        self.asymmetric_flank_pairs = _asymmetric_flank_pairs(
            short_flank_min,
            short_flank_max,
            long_flank_min,
            long_flank_max,
            min_flank_length=min_flank_length,
        )
        self.seed = seed
        self.epoch = 0
        if not self.records and not allow_empty:
            raise ValueError("No records fit the denoising dataset length limit")

    def __len__(self) -> int:
        return self.record_count * self.views_per_record

    @property
    def record_count(self) -> int:
        """Number of independent source records, excluding virtual training views."""
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index // self.views_per_record]
        generator = torch.Generator().manual_seed(self.seed + self.epoch * max(1, len(self)) + index)
        example: MotifScaffoldExample | None = None
        if self.preferred_motifs and self.preferred_motif_probability:
            prefer_motif = float(torch.rand((), generator=generator).item()) < self.preferred_motif_probability
            if prefer_motif:
                example = _sample_preferred_motif_example(
                    record,
                    generator,
                    self.preferred_motifs,
                    max_length=self.max_length,
                    min_motif_length=self.min_motif_length,
                    max_motif_length=self.max_motif_length,
                    min_flank_length=self.min_flank_length,
                    min_total_scaffold_length=self.min_total_scaffold_length,
                    asymmetric_pairs=self.asymmetric_flank_pairs,
                )
        if example is None:
            crop_start = 0
            if len(record.sequence) > self.max_length:
                crop_start = int(
                    torch.randint(
                        0,
                        len(record.sequence) - self.max_length + 1,
                        (1,),
                        generator=generator,
                    ).item()
                )
                record = RnaSequenceRecord(
                    target_id=record.target_id,
                    sequence=record.sequence[crop_start : crop_start + self.max_length],
                    family=record.family,
                    source=record.source,
                )
            sampled = sample_motif_example(
                record,
                generator,
                self.min_motif_length,
                self.max_motif_length,
                self.motif_length_buckets,
                self.min_flank_length,
                self.min_total_scaffold_length,
                self.preferred_total_scaffold_length,
                self.short_flank_min,
                self.short_flank_max,
                self.long_flank_min,
                self.long_flank_max,
            )
            example = MotifScaffoldExample(
                motif=sampled.motif,
                target_sequence=sampled.target_sequence,
                motif_start=sampled.motif_start,
                source_start=crop_start + sampled.source_start,
            )
        length = example.total_length
        target = torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
        target[:length] = torch.tensor(self.tokenizer.encode(example.target_sequence))
        base_to_class = {"A": 0, "U": 1, "C": 2, "G": 3}
        target_base_ids = torch.full((self.max_length,), -100, dtype=torch.long)
        target_base_ids[:length] = torch.tensor(
            [base_to_class[base] for base in example.target_sequence], dtype=torch.long
        )
        input_ids = target.clone()
        fixed_mask = torch.zeros(self.max_length, dtype=torch.bool)
        fixed_mask[example.motif_start : example.motif_end] = True
        attention_mask = torch.arange(self.max_length) < length
        scaffold_mask = attention_mask & ~fixed_mask
        prediction_mask, mask_fraction = self._sample_prediction_mask(scaffold_mask, generator)
        input_ids[prediction_mask] = self.tokenizer.token_to_id[self.tokenizer.special.mask]
        context_noise_mask = torch.zeros_like(scaffold_mask)
        context_noise_probability = self._context_noise_rate(mask_fraction)
        if context_noise_probability:
            visible_scaffold = scaffold_mask & ~prediction_mask
            draws = torch.rand(self.max_length, generator=generator)
            context_noise_mask = visible_scaffold & draws.lt(context_noise_probability)
            if context_noise_mask.any():
                base_token_ids = torch.tensor(
                    [self.tokenizer.token_to_id[base] for base in "AUCG"],
                    dtype=torch.long,
                )
                original_classes = target_base_ids[context_noise_mask]
                offsets = torch.randint(1, 4, (int(context_noise_mask.sum().item()),), generator=generator)
                replacement_classes = (original_classes + offsets) % 4
                input_ids[context_noise_mask] = base_token_ids[replacement_classes]
        return {
            "input_ids": input_ids,
            "target_token_ids": target,
            "target_base_ids": target_base_ids,
            "fixed_mask": fixed_mask,
            "prediction_mask": prediction_mask,
            "attention_mask": attention_mask,
            "target_left_length": torch.tensor(example.motif_start, dtype=torch.long),
            "target_right_length": torch.tensor(
                length - example.motif_end,
                dtype=torch.long,
            ),
            "source_start": torch.tensor(example.source_start, dtype=torch.long),
            "noise_level": torch.tensor(mask_fraction, dtype=torch.float32),
        }

    def _context_noise_rate(self, mask_fraction: float) -> float:
        if not self.stage_aware_context_noise:
            return self.context_noise_probability
        return self.context_noise_probability * float(mask_fraction)

    def _sample_prediction_mask(
        self,
        scaffold_mask: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, float]:
        positions = scaffold_mask.nonzero(as_tuple=False).squeeze(-1)
        if positions.numel() == 0:
            raise ValueError("every example must contain at least one scaffold position")
        draw = float(torch.rand((), generator=generator).item())
        prediction = torch.zeros_like(scaffold_mask)
        if self.iterative_mask_steps is not None:
            if self.iterative_full_mask_probability is None:
                stage = int(torch.randint(1, self.iterative_mask_steps + 1, (1,), generator=generator).item())
            elif draw < self.iterative_full_mask_probability:
                stage = self.iterative_mask_steps
            else:
                stage = int(torch.randint(1, self.iterative_mask_steps, (1,), generator=generator).item())
            count = max(1, math.ceil(int(positions.numel()) * stage / self.iterative_mask_steps))
            order = torch.randperm(int(positions.numel()), generator=generator)[:count]
            prediction[positions[order]] = True
            return prediction, count / int(positions.numel())
        if draw < self.full_mask_probability:
            prediction = scaffold_mask.clone()
        elif draw < self.full_mask_probability + self.span_mask_probability:
            span_length = min(self.mean_span_length, int(positions.numel()))
            start = int(torch.randint(0, int(positions.numel()) - span_length + 1, (1,), generator=generator).item())
            prediction[positions[start : start + span_length]] = True
        else:
            fraction = self.min_random_mask_fraction + (
                self.max_random_mask_fraction - self.min_random_mask_fraction
            ) * float(torch.rand((), generator=generator).item())
            count = max(1, min(int(positions.numel()), round(int(positions.numel()) * fraction)))
            order = torch.randperm(int(positions.numel()), generator=generator)[:count]
            prediction[positions[order]] = True
        mask_fraction = int(prediction.sum().item()) / int(positions.numel())
        return prediction, mask_fraction


def eligible_denoising_records(
    records: list[RnaSequenceRecord],
    *,
    max_length: int,
    min_motif_length: int,
    min_flank_length: int,
    min_total_scaffold_length: int,
    short_flank_min: int | None = None,
    short_flank_max: int | None = None,
    long_flank_min: int | None = None,
    long_flank_max: int | None = None,
) -> list[RnaSequenceRecord]:
    """Apply the exact V2 geometry eligibility contract before splitting."""
    maximum = validate_scaffold_length_limit(max_length)
    asymmetric_pairs = _asymmetric_flank_pairs(
        short_flank_min,
        short_flank_max,
        long_flank_min,
        long_flank_max,
        min_flank_length=min_flank_length,
    )
    minimum_asymmetric_length = (
        min_motif_length + min(left + right for left, right in asymmetric_pairs) if asymmetric_pairs else 0
    )
    return [
        record
        for record in records
        if (
            record_supports_legal_geometry(
                len(record.sequence),
                max_length=maximum,
                min_motif_length=min_motif_length,
                min_flank_length=min_flank_length,
                min_total_scaffold_length=min_total_scaffold_length,
            )
            and min(len(record.sequence), maximum) >= minimum_asymmetric_length
        )
    ]


def build_partitioned_denoising_datasets(
    records: list[RnaSequenceRecord],
    cluster_by_id: dict[str, str],
    tokenizer: RnaTokenizer,
    max_length: int = 512,
    min_motif_length: int = 4,
    max_motif_length: int | None = None,
    motif_length_buckets: tuple[tuple[int, int, float], ...] | list[list[float]] | None = DEFAULT_MOTIF_LENGTH_BUCKETS,
    min_flank_length: int = 2,
    min_total_scaffold_length: int = 8,
    preferred_total_scaffold_length: int = 24,
    full_mask_probability: float = 0.35,
    span_mask_probability: float = 0.35,
    min_random_mask_fraction: float = 0.30,
    max_random_mask_fraction: float = 0.80,
    mean_span_length: int = 8,
    iterative_mask_steps: int | None = None,
    context_noise_probability: float = 0.0,
    short_flank_min: int | None = None,
    short_flank_max: int | None = None,
    long_flank_min: int | None = None,
    long_flank_max: int | None = None,
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    source_sha256: dict[str, str] | None = None,
    clustering_thresholds: dict[str, float] | None = None,
    expected_partition_counts: dict[str, int] | None = None,
    train_views_per_record: int = 1,
    validation_views_per_record: int = 1,
    test_views_per_record: int = 1,
    preferred_motifs: tuple[str, ...] | list[str] | str | None = None,
    preferred_motif_probability: float = 0.0,
    iterative_full_mask_probability: float | None = None,
    stage_aware_context_noise: bool = False,
) -> tuple[dict[str, RnaMotifDenoisingDataset], SplitManifest]:
    record_ids = {record.target_id for record in records}
    if len(record_ids) != len(records):
        raise ValueError("record target IDs must be unique")
    sequence_hashes = [record.sequence_sha256 for record in records]
    if len(sequence_hashes) != len(set(sequence_hashes)):
        raise ValueError("duplicate normalized sequence in formal training records")
    max_length = validate_scaffold_length_limit(max_length)
    eligible_records = eligible_denoising_records(
        records,
        max_length=max_length,
        min_motif_length=min_motif_length,
        min_flank_length=min_flank_length,
        min_total_scaffold_length=min_total_scaffold_length,
        short_flank_min=short_flank_min,
        short_flank_max=short_flank_max,
        long_flank_min=long_flank_min,
        long_flank_max=long_flank_max,
    )
    eligible_ids = {record.target_id for record in eligible_records}
    if set(cluster_by_id) != record_ids:
        unknown = sorted(set(cluster_by_id) - record_ids)
        missing = sorted(record_ids - set(cluster_by_id))
        if unknown:
            raise ValueError(f"cluster assignments contain unknown target IDs: {unknown}")
        raise ValueError(f"cluster assignments are incomplete; missing target IDs: {missing}")
    eligible_cluster_by_id = {
        target_id: cluster_id for target_id, cluster_id in cluster_by_id.items() if target_id in eligible_ids
    }
    audit_source_sha256 = dict(source_sha256 or {})
    cluster_manifest_sha256 = getattr(cluster_by_id, "source_sha256", None)
    if cluster_manifest_sha256:
        audit_source_sha256.setdefault("cluster_manifest", cluster_manifest_sha256)
    manifest = build_cluster_disjoint_manifest(
        eligible_records,
        eligible_cluster_by_id,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        source_sha256=audit_source_sha256,
        clustering_thresholds=clustering_thresholds,
        expected_partition_counts=expected_partition_counts,
    )
    by_id = {record.target_id: record for record in eligible_records}
    views_by_partition = {
        "train": train_views_per_record,
        "validation": validation_views_per_record,
        "test": test_views_per_record,
    }
    datasets = {
        partition: RnaMotifDenoisingDataset(
            records=[by_id[target_id] for target_id in manifest.partitions[partition]],
            tokenizer=tokenizer,
            max_length=max_length,
            min_motif_length=min_motif_length,
            max_motif_length=max_motif_length,
            motif_length_buckets=motif_length_buckets,
            min_flank_length=min_flank_length,
            min_total_scaffold_length=min_total_scaffold_length,
            preferred_total_scaffold_length=preferred_total_scaffold_length,
            full_mask_probability=full_mask_probability,
            span_mask_probability=span_mask_probability,
            min_random_mask_fraction=min_random_mask_fraction,
            max_random_mask_fraction=max_random_mask_fraction,
            mean_span_length=mean_span_length,
            iterative_mask_steps=iterative_mask_steps,
            context_noise_probability=context_noise_probability,
            short_flank_min=short_flank_min,
            short_flank_max=short_flank_max,
            long_flank_min=long_flank_min,
            long_flank_max=long_flank_max,
            seed=seed,
            allow_empty=True,
            views_per_record=views_by_partition[partition],
            preferred_motifs=preferred_motifs,
            preferred_motif_probability=preferred_motif_probability,
            iterative_full_mask_probability=iterative_full_mask_probability,
            stage_aware_context_noise=stage_aware_context_noise,
        )
        for partition in manifest.partitions
    }
    if sum(dataset.record_count for dataset in datasets.values()) != manifest.record_counts["total"]:
        raise ValueError("partition totals must equal the eligible dataset total")
    return datasets, manifest
