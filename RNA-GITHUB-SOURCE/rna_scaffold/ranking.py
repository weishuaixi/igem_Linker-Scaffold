from __future__ import annotations

from dataclasses import dataclass

from rna_scaffold.generate import ScaffoldCandidate
from rna_scaffold.validators.rnafold import RnafoldResult

DEFAULT_WEIGHTS = {
    "likelihood": 0.25,
    "confidence_quality": 0.20,
    "gc_quality": 0.20,
    "homopolymer_quality": 0.15,
    "mfe_quality": 0.05,
    "paired_fraction": 0.05,
    "motif_accessibility": 0.10,
}


@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    candidate: ScaffoldCandidate
    composite_score: float
    raw_components: dict[str, float | None]
    normalized_components: dict[str, float]
    rnafold: RnafoldResult | None


def _raw(candidate: ScaffoldCandidate, fold: RnafoldResult | None) -> dict[str, float | None]:
    mfe_per_nt = (
        fold.mfe_kcal_mol / candidate.total_length
        if fold is not None and fold.status == "ok" and fold.mfe_kcal_mol is not None
        else None
    )
    linker_gc_fraction = candidate.gc_fraction if candidate.linker_gc_fraction is None else candidate.linker_gc_fraction
    linker_max_homopolymer = (
        candidate.max_homopolymer if candidate.linker_max_homopolymer is None else candidate.linker_max_homopolymer
    )
    linker_composition_entropy = (
        candidate.base_entropy if candidate.linker_composition_entropy is None else candidate.linker_composition_entropy
    )
    likelihood = (
        candidate.normalized_log_probability
        if candidate.combined_normalized_log_probability is None
        else candidate.combined_normalized_log_probability
    )
    return {
        "likelihood": likelihood,
        "base_entropy": candidate.base_entropy,
        "linker_gc_fraction": linker_gc_fraction,
        "linker_max_homopolymer": float(linker_max_homopolymer),
        "linker_composition_entropy": linker_composition_entropy,
        "constraint_forced_argmax_rate": candidate.constraint_forced_argmax_rate,
        "constraint_log_probability_mass": candidate.constraint_log_probability_mass,
        "confidence_quality": candidate.mean_token_confidence,
        "gc_quality": max(0.0, 1.0 - abs(linker_gc_fraction - 0.5) / 0.5),
        "homopolymer_quality": max(0.0, 1.0 - linker_max_homopolymer / 8.0),
        "mfe_per_nt": mfe_per_nt,
        "mfe_quality": None if mfe_per_nt is None else max(0.0, 1.0 - abs(mfe_per_nt + 0.3) / 0.7),
        "paired_fraction": fold.paired_fraction if fold is not None and fold.status == "ok" else None,
        "motif_accessibility": (
            1.0 - fold.motif_paired_fraction
            if fold is not None and fold.status == "ok" and fold.motif_paired_fraction is not None
            else None
        ),
    }


def rank_candidates(
    candidates: list[ScaffoldCandidate],
    rnafold_results: dict[str, RnafoldResult] | None = None,
    weights: dict[str, float] | None = None,
) -> list[RankedCandidate]:
    if not candidates:
        return []
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    folds = rnafold_results or {}
    raw_rows = [_raw(candidate, folds.get(candidate.candidate_id)) for candidate in candidates]
    active_weights = {
        component: weight
        for component, weight in weights.items()
        if any(row[component] is not None for row in raw_rows)
    }
    total_active_weight = sum(active_weights.values())
    if total_active_weight <= 0:
        raise ValueError("at least one ranking component must have a positive weight and data")
    normalized_rows: list[dict[str, float]] = [{} for _ in candidates]
    for component in active_weights:
        available = [row[component] for row in raw_rows if row[component] is not None]
        if not available:
            for normalized in normalized_rows:
                normalized[component] = 0.5
            continue
        lower, upper = min(available), max(available)
        for raw, normalized in zip(raw_rows, normalized_rows):
            value = raw[component]
            if value is None or upper == lower:
                normalized[component] = 0.5
            else:
                normalized[component] = float((value - lower) / (upper - lower))
    scored = []
    for candidate, raw, normalized in zip(candidates, raw_rows, normalized_rows):
        score = sum(active_weights[name] * normalized[name] for name in active_weights) / total_active_weight
        scored.append((score, candidate, raw, normalized, folds.get(candidate.candidate_id)))
    scored.sort(key=lambda item: (-item[0], item[1].candidate_id))
    return [
        RankedCandidate(rank, candidate, float(score), raw, normalized, fold)
        for rank, (score, candidate, raw, normalized, fold) in enumerate(scored, start=1)
    ]
