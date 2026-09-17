from __future__ import annotations

RNA_BASES = frozenset("AUCG")
_COMPLEMENT = str.maketrans({"A": "U", "U": "A", "C": "G", "G": "C"})


def validate_rna_sequence(sequence: str) -> bool:
    """Return True when a sequence contains only A/U/C/G bases."""
    return bool(sequence) and set(sequence.upper()).issubset(RNA_BASES)


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of an RNA sequence."""
    sequence = sequence.upper()
    if not validate_rna_sequence(sequence):
        raise ValueError("RNA sequence must contain only A, U, C, and G.")
    return sequence.translate(_COMPLEMENT)[::-1]


def complementarity_rate(left: str, right: str) -> float:
    """Score reverse-oriented Watson-Crick complementarity between left and right.

    The i-th base of left is compared against the reverse-complement orientation
    of right, matching the usual stem layout in L + motif + R.
    """
    left = left.upper()
    right = right.upper()
    if not left or not right:
        return 0.0
    if not validate_rna_sequence(left) or not validate_rna_sequence(right):
        raise ValueError("RNA sequences must contain only A, U, C, and G.")

    pairs = min(len(left), len(right))
    right_rc = reverse_complement(right)
    comparable_left = left[-pairs:]
    comparable_right = right_rc[-pairs:]
    matches = sum(a == b for a, b in zip(comparable_left, comparable_right))
    return matches / pairs


def gc_fraction(sequence: str) -> float:
    if not sequence:
        return 0.0
    sequence = sequence.upper()
    return (sequence.count("G") + sequence.count("C")) / len(sequence)
