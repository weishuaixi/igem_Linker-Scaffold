from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise

from rna_scaffold.utils import validate_rna_sequence

BASES = ("A", "U", "C", "G")


@dataclass
class RnaTrainingPrior:
    """Train-partition-only Markov statistics for benchmark baselines."""

    lengths: list[int]
    transition: dict[str, dict[str, float]]
    initial: dict[str, float]
    second_order_transition: dict[str, dict[str, float]] = field(default_factory=dict)
    reverse_transition: dict[str, dict[str, float]] = field(default_factory=dict)
    reverse_initial: dict[str, float] = field(default_factory=dict)
    reverse_second_order_transition: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> RnaTrainingPrior:
        return cls([], {}, {}, {}, {}, {}, {})

    @classmethod
    def from_sequences(cls, sequences: list[str]) -> RnaTrainingPrior:
        lengths: list[int] = []
        pair_counts = {base: Counter() for base in BASES}
        triplet_counts = {
            left + right: Counter() for left in BASES for right in BASES
        }
        start_counts: Counter[str] = Counter()
        reverse_pair_counts = {base: Counter() for base in BASES}
        reverse_triplet_counts = {
            left + right: Counter() for left in BASES for right in BASES
        }
        reverse_start_counts: Counter[str] = Counter()
        for raw_sequence in sequences:
            sequence = raw_sequence.strip().upper().replace("T", "U")
            if len(sequence) < 2 or not validate_rna_sequence(sequence):
                continue
            lengths.append(len(sequence))
            start_counts[sequence[0]] += 1
            for left, right in pairwise(sequence):
                pair_counts[left][right] += 1
            for first, second, third in zip(sequence, sequence[1:], sequence[2:]):
                triplet_counts[first + second][third] += 1
            reversed_sequence = sequence[::-1]
            reverse_start_counts[reversed_sequence[0]] += 1
            for left, right in pairwise(reversed_sequence):
                reverse_pair_counts[left][right] += 1
            for first, second, third in zip(
                reversed_sequence,
                reversed_sequence[1:],
                reversed_sequence[2:],
            ):
                reverse_triplet_counts[first + second][third] += 1

        return cls(
            lengths=lengths,
            transition=_smoothed_transitions(pair_counts),
            initial=_smoothed_initial(start_counts),
            second_order_transition=_smoothed_transitions(triplet_counts),
            reverse_transition=_smoothed_transitions(reverse_pair_counts),
            reverse_initial=_smoothed_initial(reverse_start_counts),
            reverse_second_order_transition=_smoothed_transitions(reverse_triplet_counts),
        )

    def has_statistics(self, direction: str = "forward") -> bool:
        transitions = self.transition if direction == "forward" else self.reverse_transition
        return bool(self.lengths) and bool(transitions)

    def sample_sequence(
        self,
        length: int,
        rng: random.Random,
        order: int = 1,
        prefix: str = "",
        direction: str = "forward",
    ) -> str:
        if order not in {1, 2}:
            raise ValueError("order must be 1 or 2")
        if direction not in {"forward", "reverse"}:
            raise ValueError("direction must be 'forward' or 'reverse'")
        prefix = prefix.strip().upper().replace("T", "U")
        if prefix and not validate_rna_sequence(prefix):
            raise ValueError("prefix must contain only A, U, C, and G")
        if length <= 0:
            return ""
        if not self.has_statistics(direction):
            return "".join(rng.choice(BASES) for _ in range(length))
        initial = self.initial if direction == "forward" else self.reverse_initial
        transition = self.transition if direction == "forward" else self.reverse_transition
        second_order = (
            self.second_order_transition
            if direction == "forward"
            else self.reverse_second_order_transition
        )
        history = list(prefix)
        sampled_bases: list[str] = []
        for _ in range(length):
            probabilities = initial
            if history:
                probabilities = transition.get(history[-1], {})
            if order == 2 and len(history) >= 2:
                probabilities = second_order.get("".join(history[-2:]), probabilities)
            sampled = _sample_probabilities(probabilities, rng)
            sampled_bases.append(sampled)
            history.append(sampled)
        return "".join(sampled_bases)


def _smoothed_initial(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values()) + len(BASES)
    return {base: (counts.get(base, 0) + 1) / total for base in BASES}


def _smoothed_transitions(
    counts_by_context: dict[str, Counter[str]],
) -> dict[str, dict[str, float]]:
    transitions: dict[str, dict[str, float]] = {}
    for context, counts in counts_by_context.items():
        total = sum(counts.values()) + len(BASES)
        transitions[context] = {
            base: (counts.get(base, 0) + 1) / total for base in BASES
        }
    return transitions


def _sample_probabilities(probabilities: dict[str, float], rng: random.Random) -> str:
    if not probabilities:
        return rng.choice(BASES)
    population, weights = zip(*probabilities.items())
    return rng.choices(population, weights=weights, k=1)[0]
