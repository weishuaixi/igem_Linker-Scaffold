from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise

import numpy as np


@dataclass(frozen=True)
class CandidateMetric:
    motif_id: str
    sequence: str
    valid: bool
    motif_preserved: bool
    total_length: int
    gc_fraction: float
    failure: str | None
    normalized_edit_diversity: float = 0.0
    kmer_diversity: float = 0.0
    nearest_training_kmer_similarity: float | None = None
    max_homopolymer_run: int = 0
    runtime_seconds: float = 0.0
    peak_cuda_memory_bytes: int | None = None
    rejection_reason: str | None = None
    rnafold_status: str | None = None
    rnafold_dot_bracket: str | None = None
    rnafold_mfe_kcal_mol: float | None = None
    rnafold_paired_fraction: float | None = None
    rnafold_motif_paired_fraction: float | None = None
    rnafold_runtime_seconds: float | None = None
    rnafold_version: str | None = None
    rnafold_error: str | None = None
    linker_length: int = 0
    linker_gc_fraction: float = 0.0
    linker_composition_entropy_bits: float = 0.0
    linker_max_homopolymer_run: int = 0


@dataclass(frozen=True)
class CandidateSummary:
    count: int
    valid_rate: float
    motif_preservation_rate: float
    unique_rate: float
    mean_length: float
    mean_gc_fraction: float
    failure_count: int
    mean_normalized_edit_diversity: float
    mean_kmer_diversity: float
    mean_nearest_training_kmer_similarity: float | None
    mean_max_homopolymer_run: float
    maximum_homopolymer_run: int
    runtime_seconds: float
    peak_cuda_memory_bytes: int | None
    rnafold_ok_rate: float | None
    mean_rnafold_mfe_kcal_mol: float | None
    mean_rnafold_paired_fraction: float | None
    mean_rnafold_motif_paired_fraction: float | None
    rnafold_failure_count: int
    mean_linker_length: float
    mean_linker_gc_fraction: float
    mean_linker_composition_entropy_bits: float
    mean_linker_max_homopolymer_run: float
    maximum_linker_homopolymer_run: int


@dataclass(frozen=True)
class BootstrapDifference:
    mean_difference: float
    lower: float
    upper: float
    samples: int
    seed: int


def summarize_candidates(rows: list[CandidateMetric]) -> CandidateSummary:
    if not rows:
        return CandidateSummary(
            count=0,
            valid_rate=0.0,
            motif_preservation_rate=0.0,
            unique_rate=0.0,
            mean_length=0.0,
            mean_gc_fraction=0.0,
            failure_count=0,
            mean_normalized_edit_diversity=0.0,
            mean_kmer_diversity=0.0,
            mean_nearest_training_kmer_similarity=None,
            mean_max_homopolymer_run=0.0,
            maximum_homopolymer_run=0,
            runtime_seconds=0.0,
            peak_cuda_memory_bytes=None,
            rnafold_ok_rate=None,
            mean_rnafold_mfe_kcal_mol=None,
            mean_rnafold_paired_fraction=None,
            mean_rnafold_motif_paired_fraction=None,
            rnafold_failure_count=0,
            mean_linker_length=0.0,
            mean_linker_gc_fraction=0.0,
            mean_linker_composition_entropy_bits=0.0,
            mean_linker_max_homopolymer_run=0.0,
            maximum_linker_homopolymer_run=0,
        )
    count = len(rows)
    nearest_training = [
        row.nearest_training_kmer_similarity for row in rows if row.nearest_training_kmer_similarity is not None
    ]
    peak_memory = [row.peak_cuda_memory_bytes for row in rows if row.peak_cuda_memory_bytes is not None]
    rnafold_rows = [row for row in rows if row.rnafold_status not in {None, "not_run"}]
    sequence_rows = [row for row in rows if row.sequence]
    linker_rows = sequence_rows
    return CandidateSummary(
        count=count,
        valid_rate=sum(row.valid for row in rows) / count,
        motif_preservation_rate=sum(row.motif_preserved for row in rows) / count,
        unique_rate=len({row.sequence for row in sequence_rows}) / count,
        mean_length=(sum(row.total_length for row in sequence_rows) / len(sequence_rows) if sequence_rows else 0.0),
        mean_gc_fraction=(sum(row.gc_fraction for row in sequence_rows) / len(sequence_rows) if sequence_rows else 0.0),
        failure_count=sum(row.failure is not None for row in rows),
        mean_normalized_edit_diversity=sum(row.normalized_edit_diversity for row in rows) / count,
        mean_kmer_diversity=sum(row.kmer_diversity for row in rows) / count,
        mean_nearest_training_kmer_similarity=(
            sum(nearest_training) / len(nearest_training) if nearest_training else None
        ),
        mean_max_homopolymer_run=sum(row.max_homopolymer_run for row in rows) / count,
        maximum_homopolymer_run=max(row.max_homopolymer_run for row in rows),
        runtime_seconds=sum(row.runtime_seconds for row in rows),
        peak_cuda_memory_bytes=max(peak_memory) if peak_memory else None,
        rnafold_ok_rate=(
            sum(row.rnafold_status == "ok" for row in rnafold_rows) / len(rnafold_rows) if rnafold_rows else None
        ),
        mean_rnafold_mfe_kcal_mol=_mean_available(row.rnafold_mfe_kcal_mol for row in rows),
        mean_rnafold_paired_fraction=_mean_available(row.rnafold_paired_fraction for row in rows),
        mean_rnafold_motif_paired_fraction=_mean_available(row.rnafold_motif_paired_fraction for row in rows),
        rnafold_failure_count=sum(row.rnafold_status not in {None, "ok", "not_run"} for row in rows),
        mean_linker_length=(sum(row.linker_length for row in linker_rows) / len(linker_rows) if linker_rows else 0.0),
        mean_linker_gc_fraction=(
            sum(row.linker_gc_fraction for row in linker_rows) / len(linker_rows) if linker_rows else 0.0
        ),
        mean_linker_composition_entropy_bits=(
            sum(row.linker_composition_entropy_bits for row in linker_rows) / len(linker_rows) if linker_rows else 0.0
        ),
        mean_linker_max_homopolymer_run=(
            sum(row.linker_max_homopolymer_run for row in linker_rows) / len(linker_rows) if linker_rows else 0.0
        ),
        maximum_linker_homopolymer_run=(
            max(row.linker_max_homopolymer_run for row in linker_rows) if linker_rows else 0
        ),
    )


def _mean_available(values) -> float | None:
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def normalized_edit_distance(left: str, right: str) -> float:
    """Exact global unit-cost Levenshtein distance using Myers bit vectors.

    Python integers support patterns longer than one machine word.
    """
    if left == right:
        return 0.0
    if not left or not right:
        return 1.0
    if len(left) > len(right):
        left, right = right, left
    masks = {}
    for index, character in enumerate(left):
        masks[character] = masks.get(character, 0) | (1 << index)
    width = (1 << len(left)) - 1
    high = 1 << (len(left) - 1)
    positive, negative, distance = width, 0, len(left)
    for character in right:
        matches = masks.get(character, 0)
        vertical = matches | negative
        horizontal = (((matches & positive) + positive) ^ positive) | matches
        plus = negative | ~(horizontal | positive)
        minus = positive & horizontal
        distance += bool(plus & high) - bool(minus & high)
        plus = (plus << 1) | 1
        minus <<= 1
        positive = (minus | ~(vertical | plus)) & width
        negative = plus & vertical & width
    return distance / len(right)


@lru_cache(maxsize=16384)
def _kmers(sequence: str, k: int) -> frozenset[str]:
    return frozenset(sequence[index : index + k] for index in range(max(0, len(sequence) - k + 1)))


def kmer_jaccard(left: str, right: str, k: int = 5) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    left_kmers = _kmers(left, k)
    right_kmers = _kmers(right, k)
    if not left_kmers and not right_kmers:
        return 1.0 if left == right else 0.0
    intersection = len(left_kmers & right_kmers)
    return intersection / (len(left_kmers) + len(right_kmers) - intersection)


class TrainingSimilarityIndex:
    """Exact Jaccard search using precomputed, collision-free k-mer bitsets."""

    def __init__(self, sequences: list[str], k: int = 5):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.vocabulary: dict[str, int] = {}
        references = set()
        self.empty_sequences = set()
        for sequence in sequences:
            words = _kmers(sequence, k)
            if not words:
                self.empty_sequences.add(sequence)
                continue
            bits = 0
            for word in sorted(words):
                index = self.vocabulary.setdefault(word, len(self.vocabulary))
                bits |= 1 << index
            references.add((bits, len(words)))
        self.references = tuple(references)

    def nearest(self, sequence: str) -> float | None:
        if not self.references and not self.empty_sequences:
            return None
        words = _kmers(sequence, self.k)
        if not words:
            return float(sequence in self.empty_sequences)
        bits = 0
        for word in words:
            index = self.vocabulary.get(word)
            if index is not None:
                bits |= 1 << index
        best = 0.0
        for reference, count in self.references:
            intersection = (bits & reference).bit_count()
            best = max(best, intersection / (len(words) + count - intersection))
            if best == 1.0:
                break
        return best


def nearest_training_similarity(sequence: str, training_sequences: list[str], k: int = 5) -> float | None:
    if not training_sequences:
        return None
    return max(kmer_jaccard(sequence, reference, k=k) for reference in training_sequences)


def maximum_homopolymer_run(sequence: str) -> int:
    if not sequence:
        return 0
    maximum = 1
    current = 1
    for previous, base in pairwise(sequence):
        current = current + 1 if base == previous else 1
        maximum = max(maximum, current)
    return maximum


def base_composition_entropy_bits(sequence: str) -> float:
    """Return Shannon entropy over A/U/C/G; 2 bits is perfectly balanced."""
    counts = [sequence.count(base) for base in "AUCG"]
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts if count)


def within_group_diversity(
    sequences: list[str],
    k: int = 5,
) -> tuple[list[float], list[float]]:
    if k <= 0:
        raise ValueError("k must be positive")
    count = len(sequences)
    edit_scores = [0.0] * count
    kmer_scores = [0.0] * count
    for index, sequence in enumerate(sequences):
        for peer_index in range(index + 1, count):
            peer = sequences[peer_index]
            edit = normalized_edit_distance(sequence, peer)
            kmer = 1.0 - kmer_jaccard(sequence, peer, k=k)
            edit_scores[index] += edit
            edit_scores[peer_index] += edit
            kmer_scores[index] += kmer
            kmer_scores[peer_index] += kmer
    if count > 1:
        edit_scores = [value / (count - 1) for value in edit_scores]
        kmer_scores = [value / (count - 1) for value in kmer_scores]
    return edit_scores, kmer_scores


def paired_bootstrap(
    left: list[float],
    right: list[float],
    seed: int = 42,
    samples: int = 10000,
) -> BootstrapDifference:
    if len(left) != len(right) or not left:
        raise ValueError("paired inputs must have equal non-zero length")
    if samples <= 0:
        raise ValueError("samples must be positive")
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(differences), size=(samples, len(differences)))
    bootstrapped = differences[indices].mean(axis=1)
    return BootstrapDifference(
        mean_difference=float(differences.mean()),
        lower=float(np.percentile(bootstrapped, 2.5)),
        upper=float(np.percentile(bootstrapped, 97.5)),
        samples=samples,
        seed=seed,
    )


def paired_bootstrap_by_motif(
    left_by_motif: dict[str, float],
    right_by_motif: dict[str, float],
    seed: int = 42,
    samples: int = 10000,
) -> tuple[tuple[str, ...], BootstrapDifference]:
    common_motif_ids = tuple(sorted(set(left_by_motif) & set(right_by_motif)))
    if not common_motif_ids:
        raise ValueError("paired methods must share at least one motif ID")
    difference = paired_bootstrap(
        [left_by_motif[motif_id] for motif_id in common_motif_ids],
        [right_by_motif[motif_id] for motif_id in common_motif_ids],
        seed=seed,
        samples=samples,
    )
    return common_motif_ids, difference
