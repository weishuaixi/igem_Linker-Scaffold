from rna_scaffold.generate import ScaffoldCandidate
from rna_scaffold.ranking import rank_candidates
from rna_scaffold.validators.rnafold import RnafoldResult


def _candidate(
    candidate_id,
    sequence,
    likelihood,
    gc,
    run,
    entropy,
    *,
    confidence=0.5,
    combined_likelihood=None,
    linker_gc=None,
    linker_run=None,
    linker_entropy=None,
):
    return ScaffoldCandidate(
        candidate_id=candidate_id,
        full_sequence=sequence,
        left_sequence=sequence[:2],
        motif="GCGG",
        right_sequence=sequence[6:],
        motif_start=2,
        motif_end=6,
        total_length=len(sequence),
        normalized_log_probability=likelihood,
        checkpoint_sha256="abc",
        seed=42,
        gc_fraction=gc,
        max_homopolymer=run,
        base_entropy=entropy,
        motif_preserved=True,
        valid=True,
        status="ok",
        mean_token_confidence=confidence,
        combined_normalized_log_probability=combined_likelihood,
        linker_gc_fraction=linker_gc,
        linker_max_homopolymer=linker_run,
        linker_composition_entropy=linker_entropy,
    )


def test_ranking_retains_raw_components_and_penalizes_low_complexity():
    balanced = _candidate("balanced", "AUGCGGAU", -0.3, 0.5, 2, 1.9)
    repetitive = _candidate("repetitive", "GGGCGGGG", -0.2, 0.9, 4, 0.6)
    folds = {
        "balanced": RnafoldResult("ok", "((....))", -2.0, 0.5, 0.0, 0.1, "v", None),
        "repetitive": RnafoldResult("ok", "(((())))", -8.0, 1.0, 1.0, 0.1, "v", None),
    }

    ranked = rank_candidates([repetitive, balanced], folds)

    assert ranked[0].candidate.candidate_id == "balanced"
    assert "mfe_per_nt" in ranked[0].raw_components
    assert "base_entropy" in ranked[0].raw_components
    assert "confidence_quality" in ranked[0].normalized_components


def test_unavailable_rnafold_is_neutral_and_order_is_deterministic():
    first = _candidate("a", "AUGCGGAU", -0.2, 0.5, 2, 1.8)
    second = _candidate("b", "UUGCGGAA", -0.4, 0.5, 2, 1.8)
    unavailable = RnafoldResult("unavailable", None, None, None, None, 0.0, None, "missing")

    left = rank_candidates([second, first], {"a": unavailable, "b": unavailable})
    right = rank_candidates([second, first], {"a": unavailable, "b": unavailable})

    assert left == right
    assert left[0].candidate.candidate_id == "a"


def test_without_rnafold_ranking_uses_only_sequence_metrics_and_rewards_confidence():
    confident = _candidate("confident", "AUGCGGAU", -0.3, 0.5, 2, 0.2, confidence=0.9)
    uncertain = _candidate("uncertain", "UUGCGGAA", -0.3, 0.5, 2, 1.8, confidence=0.1)

    ranked = rank_candidates([uncertain, confident])

    assert ranked[0].candidate.candidate_id == "confident"
    assert "confidence_quality" in ranked[0].normalized_components
    assert "mfe_quality" not in ranked[0].normalized_components
    assert "paired_fraction" not in ranked[0].normalized_components
    assert "motif_accessibility" not in ranked[0].normalized_components


def test_ranking_prefers_combined_sequence_and_length_score_when_available():
    combined_best = _candidate(
        "combined_best",
        "AUGCGGAU",
        -5.0,
        0.5,
        2,
        1.8,
        combined_likelihood=-0.1,
    )
    token_only_best = _candidate(
        "token_only_best",
        "UUGCGGAA",
        -0.1,
        0.5,
        2,
        1.8,
        combined_likelihood=-1.0,
    )

    ranked = rank_candidates([token_only_best, combined_best], weights={"likelihood": 1.0})

    assert ranked[0].candidate.candidate_id == "combined_best"
    assert ranked[0].raw_components["likelihood"] == -0.1


def test_ranking_uses_linker_only_composition_and_direct_model_confidence():
    clean_linker = _candidate(
        "clean",
        "AAGGGGUU",
        -0.3,
        0.9,
        4,
        0.2,
        confidence=0.8,
        linker_gc=0.5,
        linker_run=2,
        linker_entropy=1.0,
    )
    poor_linker = _candidate(
        "poor",
        "AAGCGGUU",
        -0.3,
        0.5,
        2,
        1.8,
        confidence=0.2,
        linker_gc=1.0,
        linker_run=7,
        linker_entropy=0.0,
    )

    ranked = rank_candidates(
        [poor_linker, clean_linker],
        weights={"confidence_quality": 1.0, "gc_quality": 1.0, "homopolymer_quality": 1.0},
    )

    assert ranked[0].candidate.candidate_id == "clean"
    assert ranked[0].raw_components["confidence_quality"] == 0.8
    assert ranked[0].raw_components["linker_gc_fraction"] == 0.5
    assert ranked[0].raw_components["linker_max_homopolymer"] == 2.0
