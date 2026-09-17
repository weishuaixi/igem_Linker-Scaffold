import math
from itertools import groupby

import pytest
import torch

from rna_scaffold.decoding import (
    DecodingSettings,
    build_flank_pair_sampler,
    iterative_denoise,
    select_flank_lengths,
)
from rna_scaffold.model import ScaffoldModelOutput
from rna_scaffold.tokenizer import RnaTokenizer


class DeterministicScaffoldModel(torch.nn.Module):
    max_length = 16

    def forward(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 4), -10.0, device=input_ids.device)
        preferred = torch.arange(length, device=input_ids.device) % 4
        logits.scatter_(2, preferred.view(1, length, 1).expand(batch, -1, -1), 10.0)
        left_length_logits = torch.full((batch, 17), -100.0, device=input_ids.device)
        left_length_logits[:, 3] = 10.0
        right_length_logits = torch.full((batch, 17), -100.0, device=input_ids.device)
        right_length_logits[:, 3] = 10.0
        return ScaffoldModelOutput(logits, left_length_logits, right_length_logits)


class ContextCorrectingScaffoldModel(torch.nn.Module):
    """Makes a low-confidence first guess that context corrects after remasking."""

    max_length = 16

    def __init__(self, mask_id: int) -> None:
        super().__init__()
        self.mask_id = mask_id
        self.seen_inputs: list[torch.Tensor] = []

    def forward(self, input_ids, attention_mask=None):
        self.seen_inputs.append(input_ids.detach().cpu().clone())
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 4), -20.0, device=input_ids.device)
        if not bool((input_ids == self.mask_id).any()):
            # A complete canvas would make a leaked self-scoring pass obvious.
            logits.zero_()
        elif len(self.seen_inputs) == 1:
            preferred = torch.tensor([1, 0, 0, 0, 2, 3], device=input_ids.device)
            logits.zero_()
            logits.scatter_(2, preferred.view(1, length, 1).expand(batch, -1, -1), 2.0)
            logits[:, 0, 1] = 20.0
        elif len(self.seen_inputs) == 2:
            preferred = torch.tensor([2, 0, 0, 0, 2, 3], device=input_ids.device)
            logits.scatter_(2, preferred.view(1, length, 1).expand(batch, -1, -1), 20.0)
            logits[:, 0] = 0.0
            logits[:, 0, 2] = 1.0
        else:
            preferred = torch.tensor([2, 0, 0, 0, 2, 3], device=input_ids.device)
            logits.scatter_(2, preferred.view(1, length, 1).expand(batch, -1, -1), 20.0)
        left_length_logits = torch.zeros((batch, 17), device=input_ids.device)
        right_length_logits = torch.zeros((batch, 17), device=input_ids.device)
        return ScaffoldModelOutput(logits, left_length_logits, right_length_logits)


class FilteringConfidenceModel(torch.nn.Module):
    """Exposes different raw confidences despite top-k deterministic sampling."""

    max_length = 16

    def __init__(self) -> None:
        super().__init__()
        self.seen_inputs: list[torch.Tensor] = []

    def forward(self, input_ids, attention_mask=None):
        self.seen_inputs.append(input_ids.detach().cpu().clone())
        batch, length = input_ids.shape
        logits = torch.zeros((batch, length, 4), device=input_ids.device)
        logits[:, 0, 0] = 10.0
        logits[:, 1, 1] = 0.2
        left_length_logits = torch.zeros((batch, 17), device=input_ids.device)
        right_length_logits = torch.zeros((batch, 17), device=input_ids.device)
        return ScaffoldModelOutput(logits, left_length_logits, right_length_logits)


class StrongGBiasModel(torch.nn.Module):
    """A collapsed model whose G preference must be bounded during decoding."""

    max_length = 512

    def forward(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        logits = torch.zeros((batch, length, 4), device=input_ids.device)
        logits[..., 3] = 20.0
        left_length_logits = torch.zeros((batch, self.max_length + 1), device=input_ids.device)
        right_length_logits = torch.zeros((batch, self.max_length + 1), device=input_ids.device)
        return ScaffoldModelOutput(logits, left_length_logits, right_length_logits)


class ConditioningAwareModel(torch.nn.Module):
    max_length = 16

    def __init__(self) -> None:
        super().__init__()
        self.contexts = []

    def forward(
        self,
        input_ids,
        attention_mask=None,
        fixed_mask=None,
        prediction_mask=None,
        denoise_step=None,
        self_condition_probs=None,
    ):
        self.contexts.append(
            (
                fixed_mask.detach().clone(),
                prediction_mask.detach().clone(),
                denoise_step.detach().clone(),
                self_condition_probs.detach().clone(),
            )
        )
        batch, length = input_ids.shape
        logits = torch.zeros((batch, length, 4), device=input_ids.device)
        logits[..., 0] = 2.0
        length_logits = torch.zeros((batch, self.max_length + 1), device=input_ids.device)
        return ScaffoldModelOutput(logits, length_logits, length_logits)


def _flank_output_with_illegal_marginal_maxima() -> ScaffoldModelOutput:
    token_logits = torch.zeros((1, 4, 4))
    left_length_logits = torch.full((1, 11), -100.0)
    right_length_logits = torch.full((1, 11), -100.0)
    left_length_logits[0, 0] = 100.0
    right_length_logits[0, 0] = 100.0
    left_length_logits[0, 2] = 90.0
    right_length_logits[0, 2] = 90.0
    return ScaffoldModelOutput(token_logits, left_length_logits, right_length_logits)


def test_select_flank_lengths_masks_illegal_joint_marginal_maxima():
    left_length, right_length = select_flank_lengths(
        _flank_output_with_illegal_marginal_maxima(),
        motif_length=4,
        max_length=10,
        generator=torch.Generator().manual_seed(7),
        sample=False,
        min_scaffold_length=4,
        min_flank_length=2,
    )

    assert (left_length, right_length) == (2, 2)


def test_select_flank_lengths_sampling_never_returns_illegal_joint_pair():
    selections = {
        select_flank_lengths(
            _flank_output_with_illegal_marginal_maxima(),
            motif_length=4,
            max_length=10,
            generator=torch.Generator().manual_seed(seed),
            sample=True,
            min_scaffold_length=4,
            min_flank_length=2,
        )
        for seed in range(20)
    }

    assert selections <= {(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (4, 2)}


def test_select_flank_lengths_applies_minimum_to_total_canvas_length():
    token_logits = torch.zeros((1, 20, 4))
    left_length_logits = torch.full((1, 31), -100.0)
    right_length_logits = torch.full((1, 31), -100.0)
    left_length_logits[0, 2] = 100.0
    right_length_logits[0, 3] = 100.0

    left_length, right_length = select_flank_lengths(
        ScaffoldModelOutput(token_logits, left_length_logits, right_length_logits),
        motif_length=20,
        max_length=30,
        generator=torch.Generator().manual_seed(7),
        sample=False,
        min_scaffold_length=25,
        min_flank_length=2,
    )

    assert (left_length, right_length) == (2, 3)


def test_select_flank_lengths_rejects_pairs_above_requested_maximum():
    token_logits = torch.zeros((1, 4, 4))
    left_length_logits = torch.full((1, 11), -100.0)
    right_length_logits = torch.full((1, 11), -100.0)
    left_length_logits[0, 2] = 100.0
    right_length_logits[0, 9] = 100.0
    right_length_logits[0, 4] = 90.0

    left_length, right_length = select_flank_lengths(
        ScaffoldModelOutput(token_logits, left_length_logits, right_length_logits),
        motif_length=4,
        max_length=10,
        generator=torch.Generator().manual_seed(7),
        sample=False,
        min_scaffold_length=1,
        min_flank_length=2,
    )

    assert (left_length, right_length) == (2, 4)


def test_select_flank_lengths_reports_when_no_pair_is_legal():
    with pytest.raises(ValueError, match="motif cannot fit"):
        select_flank_lengths(
            _flank_output_with_illegal_marginal_maxima(),
            motif_length=20,
            max_length=30,
            generator=torch.Generator().manual_seed(7),
            sample=False,
            min_scaffold_length=31,
            min_flank_length=2,
        )


def test_flank_pair_log_probability_is_conditioned_on_legal_pairs():
    sampler = build_flank_pair_sampler(
        _flank_output_with_illegal_marginal_maxima(),
        motif_length=4,
        max_length=10,
        min_scaffold_length=4,
        min_flank_length=2,
    )

    probabilities = [math.exp(sampler.log_probability(int(left), int(right))) for left, right in sampler.pairs.tolist()]
    assert sum(probabilities) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="not legal"):
        sampler.log_probability(0, 0)


def test_uniform_length_sampling_assigns_equal_probability_to_every_legal_pair():
    sampler = build_flank_pair_sampler(
        _flank_output_with_illegal_marginal_maxima(),
        motif_length=4,
        max_length=10,
        min_scaffold_length=4,
        min_flank_length=2,
        length_sampling="uniform",
    )

    log_probabilities = [sampler.log_probability(int(left), int(right)) for left, right in sampler.pairs.tolist()]
    assert log_probabilities == pytest.approx([-math.log(len(sampler.pairs))] * len(sampler.pairs))


def test_length_sampling_rejects_unknown_policy():
    with pytest.raises(ValueError, match="length_sampling"):
        build_flank_pair_sampler(
            _flank_output_with_illegal_marginal_maxima(),
            motif_length=4,
            max_length=10,
            length_sampling="biased",
        )


def test_model_length_sampling_requires_checkpoint_length_logits():
    output = ScaffoldModelOutput(torch.zeros((1, 4, 4)), None, None)
    with pytest.raises(ValueError, match="requires left and right length logits"):
        build_flank_pair_sampler(
            output,
            motif_length=4,
            max_length=10,
            length_sampling="model",
            model_max_length=10,
        )


def test_uniform_length_sampling_does_not_require_length_logits():
    sampler = build_flank_pair_sampler(
        None,
        motif_length=4,
        max_length=10,
        min_scaffold_length=4,
        min_flank_length=2,
        length_sampling="uniform",
        model_max_length=10,
        device="cpu",
    )

    assert len(sampler.pairs) == 6


def test_flank_pair_sampler_enforces_asymmetric_short_and_long_ranges():
    token_logits = torch.zeros((1, 4, 4))
    left_length_logits = torch.zeros((1, 101))
    right_length_logits = torch.zeros((1, 101))
    sampler = build_flank_pair_sampler(
        ScaffoldModelOutput(token_logits, left_length_logits, right_length_logits),
        motif_length=40,
        max_length=100,
        min_scaffold_length=1,
        min_flank_length=2,
        short_flank_range=(10, 20),
        long_flank_range=(15, 30),
    )

    pairs = {tuple(pair) for pair in sampler.pairs.tolist()}
    assert (10, 15) in pairs
    assert (30, 20) in pairs
    assert (10, 30) in pairs
    assert (30, 10) in pairs
    assert (10, 10) not in pairs
    assert (30, 30) not in pairs
    assert (20, 15) in pairs
    assert (15, 20) in pairs
    assert all(left + 40 + right <= 100 for left, right in pairs)


def test_iterative_denoise_is_reproducible_and_preserves_motif():
    tokenizer = RnaTokenizer()
    model = DeterministicScaffoldModel()
    settings = DecodingSettings(denoise_steps=4, temperature=1.0, top_k=1, top_p=1.0)

    def decode():
        return iterative_denoise(
            model,
            tokenizer,
            motif="GCGG",
            total_length=10,
            motif_start=3,
            settings=settings,
            generator=torch.Generator().manual_seed(11),
            device="cpu",
        )

    first = decode()
    second = decode()

    assert first == second
    assert first.sequence[3:7] == "GCGG"
    assert set(first.sequence) <= set("AUCG")
    assert first.masked_counts[-1] == 0
    assert all(later < earlier for earlier, later in zip(first.masked_counts, first.masked_counts[1:]))
    assert torch.isfinite(torch.tensor(first.normalized_log_probability))


def test_iterative_denoise_freezes_high_confidence_tokens_without_self_scoring_leakage():
    """Only low-confidence tokens returned to MASK may change on later passes."""
    tokenizer = RnaTokenizer()
    model = ContextCorrectingScaffoldModel(mask_id=tokenizer.token_to_id[tokenizer.special.mask])

    decoded = iterative_denoise(
        model,
        tokenizer,
        motif="GC",
        total_length=6,
        motif_start=2,
        settings=DecodingSettings(
            denoise_steps=3,
            temperature=1.0,
            top_p=1.0,
            mask_schedule="linear",
        ),
        generator=torch.Generator().manual_seed(19),
        device="cpu",
    )

    motif_ids = torch.tensor(tokenizer.encode("GC"))
    assert decoded.sequence[0] == "U"
    assert decoded.masked_counts == (4, 3, 2, 0)
    assert all(later < earlier for earlier, later in zip(decoded.masked_counts, decoded.masked_counts[1:]))
    assert set(decoded.sequence) <= set("AUCG")
    assert len(model.seen_inputs) == 3
    assert model.seen_inputs[1][0, 0].item() == tokenizer.token_to_id["U"]
    assert model.seen_inputs[2][0, 0].item() == tokenizer.token_to_id["U"]
    assert all(torch.equal(canvas[0, 2:4], motif_ids) for canvas in model.seen_inputs)
    assert decoded.normalized_log_probability > -1e-3
    assert decoded.mean_token_confidence > 0.999


def test_iterative_denoise_passes_masks_noise_fraction_and_self_conditioning():
    tokenizer = RnaTokenizer()
    model = ConditioningAwareModel()

    decoded = iterative_denoise(
        model,
        tokenizer,
        motif="GC",
        total_length=6,
        motif_start=2,
        settings=DecodingSettings(denoise_steps=2, top_k=1, top_p=1.0),
        generator=torch.Generator().manual_seed(5),
        device="cpu",
    )

    assert len(model.contexts) == 2
    first_fixed, first_prediction, first_noise, first_self_condition = model.contexts[0]
    _, second_prediction, second_noise, second_self_condition = model.contexts[1]
    assert torch.equal(first_fixed, torch.tensor([[False, False, True, True, False, False]]))
    assert torch.equal(first_prediction, ~first_fixed)
    assert first_noise.tolist() == pytest.approx([1.0])
    assert not bool(first_self_condition.any().item())
    assert second_prediction.sum().item() == 2
    assert second_noise.tolist() == pytest.approx([0.5])
    assert torch.allclose(
        second_self_condition.sum(dim=-1),
        torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0]]),
    )
    assert decoded.constraint_forced_argmax_rate == 0.0
    assert decoded.constraint_log_probability_mass == 0.0


def test_iterative_denoise_ranks_remasking_with_model_proposal_confidence():
    tokenizer = RnaTokenizer()
    model = FilteringConfidenceModel()

    iterative_denoise(
        model,
        tokenizer,
        motif="G",
        total_length=3,
        motif_start=2,
        settings=DecodingSettings(denoise_steps=2, top_p=1.0),
        generator=torch.Generator().manual_seed(3),
        device="cpu",
    )

    assert torch.equal(
        model.seen_inputs[1][0],
        torch.tensor(tokenizer.encode("A<MASK>G")),
    )


def test_top_k_sampling_does_not_inflate_reported_model_likelihood_or_confidence():
    decoded = iterative_denoise(
        FilteringConfidenceModel(),
        RnaTokenizer(),
        motif="G",
        total_length=3,
        motif_start=2,
        settings=DecodingSettings(denoise_steps=1, top_k=1, top_p=1.0),
        generator=torch.Generator().manual_seed(3),
        device="cpu",
    )

    # top-k=1 makes the sampling policy deterministic, but the second mutable
    # position still has only about 29% probability under the unfiltered model.
    assert decoded.mean_token_confidence < 0.7
    assert decoded.normalized_log_probability < -0.5


def test_iterative_denoise_rejects_a_canvas_without_scaffold_positions():
    with pytest.raises(ValueError, match="at least one scaffold position"):
        iterative_denoise(
            DeterministicScaffoldModel(),
            RnaTokenizer(),
            motif="GC",
            total_length=2,
            motif_start=0,
            settings=DecodingSettings(denoise_steps=1),
            generator=torch.Generator().manual_seed(1),
            device="cpu",
        )


def test_constrained_decoding_prevents_gc_and_homopolymer_collapse():
    tokenizer = RnaTokenizer()
    settings = DecodingSettings(
        denoise_steps=4,
        temperature=1.0,
        top_p=1.0,
        max_homopolymer_run=6,
        gc_min=0.30,
        gc_max=0.70,
        enforce_gc_bounds=True,
    )

    def decode():
        return iterative_denoise(
            StrongGBiasModel(),
            tokenizer,
            motif="GCGG",
            total_length=37,
            motif_start=10,
            settings=settings,
            generator=torch.Generator().manual_seed(23),
            device="cpu",
        )

    first = decode()
    second = decode()
    linker = first.sequence[:10] + first.sequence[14:]
    linker_gc_count = linker.count("G") + linker.count("C")
    left_maximum_run = max(sum(1 for _ in group) for _, group in groupby(first.sequence[:10]))
    right_maximum_run = max(sum(1 for _ in group) for _, group in groupby(first.sequence[14:]))

    assert first == second
    assert first.sequence[10:14] == "GCGG"
    assert 10 <= linker_gc_count <= 23
    assert left_maximum_run <= 6
    assert right_maximum_run <= 6
    assert first.constraint_forced_argmax_rate > 0.0
    assert first.constraint_log_probability_mass < 0.0


def test_constrained_decoding_excludes_the_fixed_motif_from_linker_run_limits():
    decoded = iterative_denoise(
        StrongGBiasModel(),
        RnaTokenizer(),
        motif="GGGG",
        total_length=12,
        motif_start=4,
        settings=DecodingSettings(denoise_steps=2, top_p=1.0, max_homopolymer_run=3),
        generator=torch.Generator().manual_seed(1),
        device="cpu",
    )

    assert decoded.sequence[4:8] == "GGGG"
    assert max(sum(1 for _ in group) for _, group in groupby(decoded.sequence[:4])) <= 3
    assert max(sum(1 for _ in group) for _, group in groupby(decoded.sequence[8:])) <= 3


def test_constrained_decoding_rejects_an_empty_integer_gc_interval():
    with pytest.raises(ValueError, match="no feasible integer"):
        iterative_denoise(
            StrongGBiasModel(),
            RnaTokenizer(),
            motif="AA",
            total_length=5,
            motif_start=1,
            settings=DecodingSettings(
                denoise_steps=2,
                gc_min=0.5,
                gc_max=0.5,
                enforce_gc_bounds=True,
            ),
            generator=torch.Generator().manual_seed(1),
            device="cpu",
        )


def test_constrained_decoding_supports_the_public_512_nt_limit_without_recursion():
    decoded = iterative_denoise(
        StrongGBiasModel(),
        RnaTokenizer(),
        motif="GCGG",
        total_length=512,
        motif_start=254,
        settings=DecodingSettings(
            denoise_steps=1,
            temperature=1.0,
            top_p=1.0,
            max_homopolymer_run=6,
        ),
        generator=torch.Generator().manual_seed(19),
        device="cpu",
    )

    left_maximum_run = max(sum(1 for _ in group) for _, group in groupby(decoded.sequence[:254]))
    right_maximum_run = max(sum(1 for _ in group) for _, group in groupby(decoded.sequence[258:]))
    assert len(decoded.sequence) == 512
    assert decoded.sequence[254:258] == "GCGG"
    assert left_maximum_run <= 6
    assert right_maximum_run <= 6


def test_flank_sampler_removes_gc_infeasible_lengths_before_sampling():
    output = ScaffoldModelOutput(
        torch.zeros((1, 4, 4)),
        torch.zeros((1, 11)),
        torch.zeros((1, 11)),
    )
    settings = DecodingSettings(
        denoise_steps=1,
        gc_min=0.5,
        gc_max=0.5,
        enforce_gc_bounds=True,
    )
    sampler = build_flank_pair_sampler(
        output,
        motif_length=4,
        max_length=10,
        min_scaffold_length=1,
        min_flank_length=2,
        motif="GCGC",
        decoding_settings=settings,
    )

    assert {left + 4 + right for left, right in sampler.pairs.tolist()} == {8, 10}


def test_constraint_conditioning_does_not_inflate_model_confidence():
    decoded = iterative_denoise(
        StrongGBiasModel(),
        RnaTokenizer(),
        motif="GGGG",
        total_length=10,
        motif_start=3,
        settings=DecodingSettings(
            denoise_steps=1,
            top_p=1.0,
            max_homopolymer_run=1,
        ),
        generator=torch.Generator().manual_seed(9),
        device="cpu",
    )

    assert decoded.sequence[3:7] == "GGGG"
    assert decoded.mean_token_confidence < 0.8
    assert decoded.normalized_log_probability < -5.0
    assert decoded.constraint_log_probability_mass < 0.0


def test_constraints_reject_an_incompatible_top_k_support():
    with pytest.raises(ValueError, match="sampling support"):
        iterative_denoise(
            StrongGBiasModel(),
            RnaTokenizer(),
            motif="GGGG",
            total_length=10,
            motif_start=3,
            settings=DecodingSettings(
                denoise_steps=1,
                top_k=1,
                top_p=1.0,
                max_homopolymer_run=1,
            ),
            generator=torch.Generator().manual_seed(9),
            device="cpu",
        )


def test_enforced_gc_requires_at_least_one_bound():
    with pytest.raises(ValueError, match="requires gc_min"):
        DecodingSettings(enforce_gc_bounds=True)
