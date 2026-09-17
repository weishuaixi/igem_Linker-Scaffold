from __future__ import annotations

import torch

MAX_SCAFFOLD_LENGTH = 512


def validate_scaffold_length_limit(value: int, *, field: str = "max_length") -> int:
    """Validate the single runtime canvas limit used by training and inference."""
    value = int(value)
    if value < 2:
        raise ValueError(f"{field} must be at least 2")
    if value > MAX_SCAFFOLD_LENGTH:
        raise ValueError(f"{field} must not exceed {MAX_SCAFFOLD_LENGTH}")
    return value


def minimum_legal_total_length(
    motif_length: int,
    min_flank_length: int,
    min_total_scaffold_length: int,
) -> int:
    """Return the smallest legal total canvas, including the fixed motif."""
    motif_length = int(motif_length)
    min_flank_length = int(min_flank_length)
    min_total_scaffold_length = int(min_total_scaffold_length)
    if motif_length < 1:
        raise ValueError("motif_length must be positive")
    if min_flank_length < 0:
        raise ValueError("min_flank_length must be non-negative")
    if min_total_scaffold_length < 0:
        raise ValueError("min_total_scaffold_length must be non-negative")
    return max(
        min_total_scaffold_length,
        motif_length + 2 * min_flank_length,
    )


def is_legal_scaffold_geometry(
    left_length: int,
    right_length: int,
    motif_length: int,
    *,
    min_flank_length: int,
    min_total_scaffold_length: int,
    requested_max_length: int,
    model_max_length: int | None = None,
) -> bool:
    """Evaluate the five V2 legal-pair predicates from the design."""
    requested_max_length = validate_scaffold_length_limit(
        requested_max_length,
        field="requested_max_length",
    )
    effective_maximum = requested_max_length
    if model_max_length is not None:
        effective_maximum = min(
            effective_maximum,
            validate_scaffold_length_limit(model_max_length, field="model_max_length"),
        )
    minimum_total = minimum_legal_total_length(
        motif_length,
        min_flank_length,
        min_total_scaffold_length,
    )
    total_length = int(left_length) + int(motif_length) + int(right_length)
    return (
        int(left_length) >= int(min_flank_length)
        and int(right_length) >= int(min_flank_length)
        and total_length >= minimum_total
        and total_length <= effective_maximum
    )


def legal_flank_pairs(
    motif_length: int,
    *,
    min_flank_length: int,
    min_total_scaffold_length: int,
    requested_max_length: int,
    model_max_length: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Build all legal ``(left, right)`` pairs as one vectorized tensor."""
    requested_max_length = validate_scaffold_length_limit(
        requested_max_length,
        field="requested_max_length",
    )
    model_max_length = validate_scaffold_length_limit(
        model_max_length,
        field="model_max_length",
    )
    minimum_total = minimum_legal_total_length(
        motif_length,
        min_flank_length,
        min_total_scaffold_length,
    )
    maximum_total = min(requested_max_length, model_max_length)
    largest_flank = maximum_total - int(motif_length) - int(min_flank_length)
    if largest_flank < int(min_flank_length) or minimum_total > maximum_total:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    lengths = torch.arange(
        int(min_flank_length),
        largest_flank + 1,
        dtype=torch.long,
        device=device,
    )
    pairs = torch.cartesian_prod(lengths, lengths)
    totals = pairs.sum(dim=1) + int(motif_length)
    return pairs[(totals >= minimum_total) & (totals <= maximum_total)]


def record_supports_legal_geometry(
    record_length: int,
    *,
    max_length: int,
    min_motif_length: int,
    min_flank_length: int,
    min_total_scaffold_length: int,
) -> bool:
    """Return whether a record (after dynamic cropping) can form a legal canvas."""
    maximum = validate_scaffold_length_limit(max_length)
    effective_total = min(int(record_length), maximum)
    return effective_total >= minimum_legal_total_length(
        min_motif_length,
        min_flank_length,
        min_total_scaffold_length,
    )
