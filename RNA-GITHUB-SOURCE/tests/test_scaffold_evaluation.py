import random

import pytest

from rna_scaffold.benchmarking import RnaTrainingPrior
from rna_scaffold.evaluation import (
    CandidateMetric,
    base_composition_entropy_bits,
    maximum_homopolymer_run,
    nearest_training_similarity,
    normalized_edit_distance,
    paired_bootstrap,
    paired_bootstrap_by_motif,
    summarize_candidates,
    within_group_diversity,
)


def test_training_prior_distinguishes_first_and_second_order_contexts():
    class HighestProbabilityRandom(random.Random):
        def choices(self, population, weights=None, *, cum_weights=None, k=1):
            assert weights is not None and cum_weights is None and k == 1
            return [population[max(range(len(weights)), key=weights.__getitem__)]]

    # The same one-base context A is followed by A, U, and G equally often,
    # while the two-base contexts AA, AU, and UA have distinct successors.
    prior = RnaTrainingPrior.from_sequences(["AAUAG"] * 20)

    assert prior.sample_sequence(5, HighestProbabilityRandom(7), order=1) == "AAAAA"
    assert prior.sample_sequence(5, HighestProbabilityRandom(7), order=2) == "AAUAG"


def test_training_prior_rejects_unsupported_markov_order():
    prior = RnaTrainingPrior.empty()

    with pytest.raises(ValueError, match="order must be 1 or 2"):
        prior.sample_sequence(4, random.Random(7), order=3)


def test_training_prior_prefix_continues_second_order_context():
    prior = RnaTrainingPrior.from_sequences(["AAGCGGUU"] * 20)

    try:
        generated = prior.sample_sequence(2, random.Random(4), order=2, prefix="AAGCGG")
    except TypeError:
        generated = None

    assert generated == "UU"


def test_training_prior_reverse_direction_conditions_left_boundary():
    prior = RnaTrainingPrior.from_sequences(["AAGCGGUU"] * 20)

    try:
        outward = prior.sample_sequence(2, random.Random(4), order=2, prefix="GGCG", direction="reverse")
    except TypeError:
        outward = None

    assert outward == "AA"


def test_candidate_summary_counts_invalid_and_duplicate_outputs():
    rows = [
        CandidateMetric("m1", "a", True, True, 20, 0.5, None),
        CandidateMetric("m1", "a", True, True, 20, 0.5, None),
        CandidateMetric("m1", "b", False, False, 18, 0.7, "invalid"),
    ]

    summary = summarize_candidates(rows)

    assert summary.valid_rate == pytest.approx(2 / 3)
    assert summary.motif_preservation_rate == pytest.approx(2 / 3)
    assert summary.unique_rate == pytest.approx(2 / 3)
    assert summary.failure_count == 1


def test_sequence_diversity_and_novelty_metrics_have_hand_checked_values():
    edit_diversity, kmer_diversity = within_group_diversity(
        ["AAAA", "AAAU", "UUUU"],
        k=2,
    )

    assert edit_diversity == pytest.approx([0.625, 0.5, 0.875])
    assert kmer_diversity == pytest.approx([0.75, 0.75, 1.0])
    assert nearest_training_similarity("AAAAC", ["AAAAG", "CCCCC"], k=2) == pytest.approx(1 / 3)
    assert maximum_homopolymer_run("AAAUCCCCGG") == 4
    assert maximum_homopolymer_run("") == 0
    assert base_composition_entropy_bits("AUCG") == pytest.approx(2.0)
    assert base_composition_entropy_bits("AAAA") == 0.0
    assert base_composition_entropy_bits("") == 0.0


def test_candidate_summary_aggregates_extended_metrics_without_fabricating_missing_values():
    rows = [
        CandidateMetric(
            "m1",
            "AAAAGCGG",
            True,
            True,
            8,
            0.5,
            None,
            normalized_edit_diversity=0.75,
            kmer_diversity=0.6,
            nearest_training_kmer_similarity=0.8,
            max_homopolymer_run=4,
            runtime_seconds=0.1,
            peak_cuda_memory_bytes=None,
            rnafold_status="ok",
            rnafold_mfe_kcal_mol=-2.0,
            rnafold_paired_fraction=0.5,
            rnafold_motif_paired_fraction=0.25,
            linker_length=4,
            linker_gc_fraction=0.25,
            linker_composition_entropy_bits=1.5,
            linker_max_homopolymer_run=3,
        ),
        CandidateMetric(
            "m1",
            "UUUUGCGG",
            False,
            True,
            8,
            0.5,
            "rejected",
            normalized_edit_diversity=0.25,
            kmer_diversity=0.2,
            nearest_training_kmer_similarity=None,
            max_homopolymer_run=3,
            runtime_seconds=0.2,
            peak_cuda_memory_bytes=1024,
            rnafold_status="unavailable",
            linker_length=6,
            linker_gc_fraction=0.75,
            linker_composition_entropy_bits=1.9,
            linker_max_homopolymer_run=2,
        ),
    ]

    summary = summarize_candidates(rows)

    assert summary.mean_normalized_edit_diversity == pytest.approx(0.5)
    assert summary.mean_kmer_diversity == pytest.approx(0.4)
    assert summary.mean_nearest_training_kmer_similarity == pytest.approx(0.8)
    assert summary.maximum_homopolymer_run == 4
    assert summary.runtime_seconds == pytest.approx(0.3)
    assert summary.peak_cuda_memory_bytes == 1024
    assert summary.rnafold_ok_rate == pytest.approx(0.5)
    assert summary.mean_rnafold_mfe_kcal_mol == pytest.approx(-2.0)
    assert summary.rnafold_failure_count == 1
    assert summary.mean_linker_length == pytest.approx(5.0)
    assert summary.mean_linker_gc_fraction == pytest.approx(0.5)
    assert summary.mean_linker_composition_entropy_bits == pytest.approx(1.7)
    assert summary.mean_linker_max_homopolymer_run == pytest.approx(2.5)
    assert summary.maximum_linker_homopolymer_run == 3


def test_candidate_summary_excludes_rnafold_not_run_from_attempt_rates_and_failures():
    rows = [
        CandidateMetric("m1", "AAAAGCGG", True, True, 8, 0.5, None, rnafold_status="ok"),
        CandidateMetric("m1", "UUUUGCGG", True, True, 8, 0.5, None, rnafold_status="not_run"),
    ]

    summary = summarize_candidates(rows)

    assert summary.rnafold_ok_rate == pytest.approx(1.0)
    assert summary.rnafold_failure_count == 0


def test_linker_summary_does_not_treat_generation_shortfall_as_zero_quality_linker():
    rows = [
        CandidateMetric(
            "m1",
            "AAAAGCGGCCCC",
            True,
            True,
            12,
            0.5,
            None,
            linker_length=8,
            linker_gc_fraction=0.5,
            linker_composition_entropy_bits=1.0,
            linker_max_homopolymer_run=4,
        ),
        CandidateMetric(
            "m1",
            "",
            False,
            False,
            0,
            0.0,
            "generation_shortfall",
        ),
    ]

    summary = summarize_candidates(rows)

    assert summary.count == 2
    assert summary.mean_linker_length == pytest.approx(8.0)
    assert summary.mean_linker_gc_fraction == pytest.approx(0.5)
    assert summary.mean_linker_composition_entropy_bits == pytest.approx(1.0)
    assert summary.mean_linker_max_homopolymer_run == pytest.approx(4.0)
    assert summary.maximum_linker_homopolymer_run == 4


def test_normalized_edit_distance_has_expected_bounds():
    assert normalized_edit_distance("AUGC", "AUGC") == 0.0
    assert normalized_edit_distance("AAAA", "UUUU") == 1.0
    assert normalized_edit_distance("", "") == 0.0


def test_paired_bootstrap_is_reproducible_and_paired():
    first = paired_bootstrap([1.0, 2.0, 3.0], [0.0, 1.0, 1.0], seed=42, samples=1000)
    second = paired_bootstrap([1.0, 2.0, 3.0], [0.0, 1.0, 1.0], seed=42, samples=1000)

    assert first == second
    assert first.mean_difference == pytest.approx(4 / 3)
    assert first.lower <= first.mean_difference <= first.upper


def test_paired_bootstrap_rejects_unpaired_inputs():
    with pytest.raises(ValueError, match="equal non-zero length"):
        paired_bootstrap([1.0], [1.0, 2.0])


def test_paired_bootstrap_by_motif_uses_only_sorted_common_motif_ids():
    common_ids, difference = paired_bootstrap_by_motif(
        {"m2": 3.0, "m1": 100.0},
        {"m3": -100.0, "m2": 1.0},
        seed=11,
        samples=100,
    )

    assert common_ids == ("m2",)
    assert difference.mean_difference == 2.0
    assert difference.seed == 11
    assert difference.samples == 100
