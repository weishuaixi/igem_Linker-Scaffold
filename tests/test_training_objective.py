from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("lightning.pytorch")

from rna_scaffold.lightning_module import RnaScaffoldLitModule, _TokenDiagnostics
from rna_scaffold.model import MotifDenoisingTransformer, ScaffoldModelOutput, compute_denoising_losses


def _tiny_model(**overrides) -> MotifDenoisingTransformer:
    options = {
        "vocab_size": 12,
        "pad_token_id": 0,
        "d_model": 16,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 32,
        "dropout": 0.0,
        "max_length": 12,
    }
    options.update(overrides)
    return MotifDenoisingTransformer(**options)


def _tiny_lightning(**overrides) -> RnaScaffoldLitModule:
    options = {
        "vocab_size": 12,
        "pad_token_id": 0,
        "d_model": 16,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 32,
        "dropout": 0.0,
        "max_length": 8,
    }
    options.update(overrides)
    return RnaScaffoldLitModule(**options)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[3, 8, 11, 3, 0, 0, 0, 0]]),
        "target_base_ids": torch.tensor([[0, 0, 3, 1, -100, -100, -100, -100]]),
        "fixed_mask": torch.tensor([[False, True, True, False, False, False, False, False]]),
        "prediction_mask": torch.tensor([[True, False, False, True, False, False, False, False]]),
        "attention_mask": torch.tensor([[True, True, True, True, False, False, False, False]]),
        "target_left_length": torch.tensor([1]),
        "target_right_length": torch.tensor([1]),
        "denoise_step": torch.tensor([16]),
    }


def _reference_regularized_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    attention: torch.Tensor,
    fixed: torch.Tensor,
    prediction: torch.Tensor,
    *,
    label_smoothing: float,
    composition_weight: float,
    homopolymer_weight: float,
    homopolymer_max_run: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent reference matching the pre-optimization loss formulas."""
    scaffold_mask = prediction & attention & ~fixed
    safe_targets = targets.masked_fill(~scaffold_mask, 0)
    channel_first_logits = logits.float().transpose(1, 2)
    raw_token_losses = torch.nn.functional.cross_entropy(
        channel_first_logits,
        safe_targets,
        reduction="none",
    )
    base_token_losses = torch.nn.functional.cross_entropy(
        channel_first_logits,
        safe_targets,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    weights = scaffold_mask.float()
    counts = weights.sum(dim=1)

    def per_sequence_mean(token_losses: torch.Tensor) -> torch.Tensor:
        return ((token_losses * weights).sum(dim=1) / counts).mean()

    raw_base_nll = per_sequence_mean(raw_token_losses)
    base_loss = per_sequence_mean(base_token_losses)
    probabilities = logits.float().softmax(dim=-1)

    composition_weights = scaffold_mask.unsqueeze(-1).float()
    safe_composition_targets = targets.masked_fill(~scaffold_mask, 0)
    one_hot_targets = torch.nn.functional.one_hot(safe_composition_targets, num_classes=4).float()
    predicted_composition = (probabilities * composition_weights).sum(dim=1) / counts.unsqueeze(-1)
    target_composition = (one_hot_targets * composition_weights).sum(dim=1) / counts.unsqueeze(-1)
    midpoint = 0.5 * (predicted_composition + target_composition)
    epsilon = torch.finfo(midpoint.dtype).eps

    def js_terms(distribution: torch.Tensor) -> torch.Tensor:
        return torch.where(
            distribution > 0,
            distribution * (distribution.clamp_min(epsilon).log() - midpoint.clamp_min(epsilon).log()),
            torch.zeros_like(distribution),
        ).sum(dim=-1)

    composition_loss = 0.5 * (js_terms(predicted_composition) + js_terms(target_composition)).mean()

    window_length = homopolymer_max_run + 1
    safe_observed_targets = targets.masked_fill(~attention, 0)
    observed_probabilities = torch.nn.functional.one_hot(safe_observed_targets, num_classes=4).float()
    sequence_probabilities = torch.where(
        scaffold_mask.unsqueeze(-1),
        probabilities,
        observed_probabilities,
    )
    windows = sequence_probabilities.unfold(1, window_length, 1)
    same_base_probability = windows.prod(dim=-1).sum(dim=-1)
    valid_windows = attention.unfold(1, window_length, 1).all(dim=-1)
    linker_only_windows = ~fixed.unfold(1, window_length, 1).any(dim=-1)
    trainable_windows = scaffold_mask.unfold(1, window_length, 1).any(dim=-1)
    homopolymer_loss = same_base_probability[valid_windows & linker_only_windows & trainable_windows].mean()
    total_loss = base_loss + composition_weight * composition_loss + homopolymer_weight * homopolymer_loss
    return raw_base_nll, base_loss, composition_loss, homopolymer_loss, total_loss


def test_default_model_state_keeps_v2_v3_parameter_structure() -> None:
    state_keys = set(_tiny_model().state_dict())

    assert not any("role_embedding" in key for key in state_keys)
    assert not any("noise_level_projection" in key for key in state_keys)
    assert not any("self_condition_projection" in key for key in state_keys)
    assert not any("pretrained_gate" in key for key in state_keys)
    assert not any("pool_query" in key for key in state_keys)
    assert "left_length_head.weight" in state_keys
    assert "right_length_head.weight" in state_keys


def test_v4_conditioning_accepts_roles_sides_noise_and_self_conditioning() -> None:
    model = _tiny_model(use_conditioning_embeddings=True).eval()
    input_ids = torch.tensor([[3, 8, 11, 3]])
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    fixed = torch.tensor([[False, True, True, False]])
    prediction = torch.tensor([[True, False, False, True]])
    self_condition = torch.full((1, 4, 4), 0.25)

    output = model(
        input_ids,
        attention,
        fixed_mask=fixed,
        prediction_mask=prediction,
        denoise_step=torch.tensor([1.0]),
        self_condition_probs=self_condition,
    )

    assert output.token_logits.shape == (1, 4, 4)
    assert torch.isfinite(output.token_logits).all()
    with pytest.raises(ValueError, match="normalized values"):
        model(input_ids, attention, fixed_mask=fixed, prediction_mask=prediction, denoise_step=torch.tensor([2.0]))


def test_conditioning_infers_motif_only_inputs_when_masks_are_omitted() -> None:
    model = _tiny_model(use_conditioning_embeddings=True).eval()

    output = model(torch.tensor([[8, 11, 10]]), torch.ones(1, 3, dtype=torch.bool))

    assert output.token_logits.shape == (1, 3, 4)


def test_predict_flank_lengths_false_removes_heads_and_returns_none() -> None:
    model = _tiny_model(predict_flank_lengths=False).eval()

    output = model(torch.tensor([[3, 8, 3]]), torch.ones(1, 3, dtype=torch.bool))

    assert model.left_length_head is None
    assert model.right_length_head is None
    assert output.left_length_logits is None
    assert output.right_length_logits is None
    assert not any("length_head" in key for key in model.state_dict())


def test_attention_length_pooling_uses_independent_left_and_right_queries() -> None:
    model = _tiny_model(d_model=4, nhead=2, length_pooling="attention")
    hidden = torch.tensor([[[2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]]])
    attention = torch.ones(1, 2, dtype=torch.bool)
    with torch.no_grad():
        model.left_pool_query.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))
        model.right_pool_query.copy_(torch.tensor([-10.0, 0.0, 0.0, 0.0]))

    left, right = model._pool_for_lengths(hidden, attention)

    assert left[0, 0] > 1.9
    assert right[0, 0] < -1.9


def test_zero_length_weights_accept_missing_logits_and_skip_length_losses() -> None:
    output = ScaffoldModelOutput(torch.zeros(1, 3, 4), None, None)
    losses = compute_denoising_losses(
        output,
        target_base_ids=torch.tensor([[0, 1, 2]]),
        fixed_mask=torch.zeros(1, 3, dtype=torch.bool),
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        target_left_length=torch.tensor([1]),
        target_right_length=torch.tensor([1]),
        left_length_loss_weight=0.0,
        right_length_loss_weight=0.0,
    )

    assert losses.left_length_loss.item() == 0
    assert losses.right_length_loss.item() == 0
    assert torch.allclose(losses.total_loss, losses.base_loss)


def test_composition_and_expected_homopolymer_losses_detect_collapse() -> None:
    targets = torch.tensor([[0, 1, 0, 1]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    aligned_logits = torch.full((1, 4, 4), -8.0)
    aligned_logits.scatter_(2, targets.unsqueeze(-1), 8.0)
    collapsed_logits = torch.full((1, 4, 4), -8.0)
    collapsed_logits[..., 0] = 8.0

    def losses(logits: torch.Tensor):
        return compute_denoising_losses(
            ScaffoldModelOutput(logits, None, None),
            targets,
            fixed_mask=torch.zeros_like(mask),
            attention_mask=mask,
            target_left_length=torch.tensor([0]),
            target_right_length=torch.tensor([0]),
            left_length_loss_weight=0.0,
            right_length_loss_weight=0.0,
            composition_loss_weight=1.0,
            homopolymer_loss_weight=1.0,
            homopolymer_max_run=2,
        )

    aligned = losses(aligned_logits)
    collapsed = losses(collapsed_logits)

    assert collapsed.composition_loss > aligned.composition_loss
    assert collapsed.homopolymer_loss > aligned.homopolymer_loss


@pytest.mark.parametrize("label_smoothing", [0.0, 0.23])
def test_shared_fp32_probabilities_match_cross_entropy_and_regularizer_reference(
    label_smoothing: float,
) -> None:
    logits = torch.tensor(
        [
            [
                [2.0, -0.5, 0.3, 1.1],
                [-1.0, 1.7, 0.2, 0.5],
                [0.4, -0.3, 2.2, 0.1],
                [0.8, 0.2, -0.7, 1.4],
                [1.2, 0.9, -0.2, -0.6],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [-0.4, 0.3, 0.8, 1.9],
                [0.2, -0.1, 1.3, 1.1],
                [0.6, 1.6, -0.8, 0.4],
                [1.5, 0.1, -0.2, 0.7],
                [-0.9, 0.5, 1.8, 0.2],
                [0.3, 1.2, 0.6, -0.5],
            ],
        ],
        requires_grad=True,
    )
    reference_logits = logits.detach().clone().requires_grad_(True)
    targets = torch.tensor([[0, 1, 2, 3, 0, -100], [3, 3, 1, 0, 2, 1]])
    attention = torch.tensor([[True, True, True, True, True, False], [True, True, True, True, True, True]])
    fixed = torch.tensor([[False, True, False, False, False, False], [False, False, True, False, False, False]])
    prediction = torch.tensor([[True, False, True, False, True, False], [True, True, False, True, False, True]])
    composition_weight = 0.31
    homopolymer_weight = 0.27
    homopolymer_max_run = 2
    lengths = torch.zeros(2, dtype=torch.long)

    actual = compute_denoising_losses(
        ScaffoldModelOutput(logits, None, None),
        target_base_ids=targets,
        fixed_mask=fixed,
        attention_mask=attention,
        target_left_length=lengths,
        target_right_length=lengths,
        prediction_mask=prediction,
        left_length_loss_weight=0.0,
        right_length_loss_weight=0.0,
        label_smoothing=label_smoothing,
        normalize_base_loss_per_sequence=True,
        composition_loss_weight=composition_weight,
        homopolymer_loss_weight=homopolymer_weight,
        homopolymer_max_run=homopolymer_max_run,
    )
    expected = _reference_regularized_losses(
        reference_logits,
        targets,
        attention,
        fixed,
        prediction,
        label_smoothing=label_smoothing,
        composition_weight=composition_weight,
        homopolymer_weight=homopolymer_weight,
        homopolymer_max_run=homopolymer_max_run,
    )

    actual_values = (
        actual.raw_base_nll,
        actual.base_loss,
        actual.composition_loss,
        actual.homopolymer_loss,
        actual.total_loss,
    )
    for actual_value, expected_value in zip(actual_values, expected):
        assert torch.allclose(actual_value, expected_value, atol=1e-6, rtol=1e-6)
    actual.total_loss.backward()
    expected[-1].backward()
    assert torch.allclose(logits.grad, reference_logits.grad, atol=1e-6, rtol=1e-6)


def test_expected_homopolymer_loss_treats_fixed_motif_as_linker_separator() -> None:
    targets = torch.zeros((1, 5), dtype=torch.long)
    attention = torch.ones_like(targets, dtype=torch.bool)
    fixed_motif = torch.tensor([[False, False, True, False, False]])
    prediction_mask = ~fixed_motif
    collapsed_logits = torch.full((1, 5, 4), -20.0)
    collapsed_logits[..., 0] = 20.0

    def homopolymer_loss(fixed_mask: torch.Tensor) -> torch.Tensor:
        return compute_denoising_losses(
            ScaffoldModelOutput(collapsed_logits, None, None),
            targets,
            fixed_mask=fixed_mask,
            attention_mask=attention,
            target_left_length=torch.tensor([2]),
            target_right_length=torch.tensor([2]),
            prediction_mask=prediction_mask,
            left_length_loss_weight=0.0,
            right_length_loss_weight=0.0,
            homopolymer_loss_weight=1.0,
            homopolymer_max_run=2,
        ).homopolymer_loss

    # Both linkers are only two bases long, so the fixed middle motif must not
    # create a length-three run across either linker/motif boundary.
    assert homopolymer_loss(fixed_motif).item() == 0.0
    assert homopolymer_loss(torch.zeros_like(fixed_motif)).item() > 0.999


def test_lightning_skips_second_forward_when_length_losses_are_disabled(monkeypatch) -> None:
    class CountingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, input_ids, attention_mask=None):
            self.calls += 1
            logits = torch.zeros(*input_ids.shape, 4, device=input_ids.device)
            return ScaffoldModelOutput(logits, None, None)

    module = _tiny_lightning(
        predict_flank_lengths=False,
        left_length_loss_weight=0.0,
        right_length_loss_weight=0.0,
    )
    counting = CountingModel()
    module.model = counting
    logged_names: list[str] = []
    monkeypatch.setattr(module, "log", lambda name, *args, **kwargs: logged_names.append(name))

    output = module.training_step(_batch(), 0)

    assert counting.calls == 1
    assert output["left_length_loss"].item() == 0
    assert output["right_length_loss"].item() == 0
    assert not any("length" in name for name in logged_names)


def test_training_self_conditioning_uses_detached_draft_probabilities(monkeypatch) -> None:
    class CountingConditionedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_conditions = []

        def forward(self, input_ids, attention_mask=None, **kwargs):
            self.self_conditions.append(kwargs.get("self_condition_probs"))
            logits = torch.zeros(*input_ids.shape, 4, device=input_ids.device, requires_grad=True)
            return ScaffoldModelOutput(logits, None, None)

    module = _tiny_lightning(
        use_conditioning_embeddings=True,
        self_conditioning_probability=1.0,
        predict_flank_lengths=False,
        left_length_loss_weight=0.0,
        right_length_loss_weight=0.0,
    )
    counting = CountingConditionedModel()
    module.model = counting
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)

    module.training_step(_batch(), 0)

    assert len(counting.self_conditions) == 2
    assert counting.self_conditions[0] is None
    assert counting.self_conditions[1] is not None
    assert not counting.self_conditions[1].requires_grad


def test_validation_self_conditioning_flag_is_independent_of_training_probability() -> None:
    module = _tiny_lightning(
        use_conditioning_embeddings=True,
        self_conditioning_probability=0.0,
        self_conditioning_validation=True,
    )

    assert not module._uses_self_conditioning("train")
    assert module._uses_self_conditioning("val")


def test_validation_primary_metrics_use_first_pass_and_log_self_conditioned_pass_separately(
    monkeypatch,
) -> None:
    class TwoPassConditionedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, input_ids, attention_mask=None, **kwargs):
            self.calls += 1
            logits = torch.zeros(*input_ids.shape, 4, device=input_ids.device)
            if self.calls == 1:
                logits[0, 0, 0] = 8.0
                logits[0, 3, 1] = 8.0
            else:
                logits[0, 0, 2] = 8.0
                logits[0, 3, 2] = 8.0
            return ScaffoldModelOutput(logits, None, None)

    module = _tiny_lightning(
        use_conditioning_embeddings=True,
        self_conditioning_probability=0.0,
        self_conditioning_validation=True,
        predict_flank_lengths=False,
        left_length_loss_weight=0.0,
        right_length_loss_weight=0.0,
        label_smoothing=0.0,
    )
    counting = TwoPassConditionedModel()
    module.model = counting
    logged: dict[str, torch.Tensor] = {}
    monkeypatch.setattr(module, "log", lambda name, value, **kwargs: logged.setdefault(name, value))

    values = module.validation_step(_batch(), 0)

    assert counting.calls == 2
    assert values["base_loss"] < 0.01
    assert values["self_conditioned_base_loss"] > 7.0
    assert logged["val/base_loss"] is values["base_loss"]
    assert logged["val_self_conditioned/base_loss"] is values["self_conditioned_base_loss"]


def test_normalized_noise_level_takes_precedence_over_integer_denoise_step() -> None:
    module = _tiny_lightning(denoise_steps=16)
    batch = _batch()
    batch["noise_level"] = torch.tensor([0.375])
    mutable = batch["attention_mask"] & ~batch["fixed_mask"]

    normalized = module._normalized_denoise_step(batch, mutable, batch["prediction_mask"])

    assert normalized.item() == pytest.approx(0.375)


def test_token_diagnostics_aggregate_accuracy_by_token_count() -> None:
    metric = _TokenDiagnostics()
    metric.update(
        torch.tensor([[[10.0, 0.0, 0.0, 0.0]]]),
        torch.tensor([[0]]),
        torch.tensor([[True]]),
        torch.tensor([[True]]),
    )
    metric.update(
        torch.tensor([[[0.0, 10.0, 0.0, 0.0]]] * 3),
        torch.tensor([[0], [0], [0]]),
        torch.tensor([[True], [True], [True]]),
        torch.tensor([[True], [True], [True]]),
    )

    values = metric.compute()

    assert values["token_accuracy"].item() == pytest.approx(0.25)
    assert values["pred_fraction_A"].item() == pytest.approx(0.25)
    assert values["pred_fraction_U"].item() == pytest.approx(0.75)
    expected_nll = torch.nn.functional.cross_entropy(
        torch.tensor([[10.0, 0.0, 0.0, 0.0], *([[0.0, 10.0, 0.0, 0.0]] * 3)]),
        torch.tensor([0, 0, 0, 0]),
    )
    assert values["raw_base_nll"].item() == pytest.approx(expected_nll.item())
    assert values["perplexity"].item() == pytest.approx(expected_nll.exp().item())


def test_optimizer_groups_lora_lr_and_exclude_norm_bias_embedding_decay() -> None:
    module = _tiny_lightning(
        lora_lr_multiplier=0.25,
        exclude_norm_bias_embedding_from_weight_decay=True,
        lr=2e-4,
        weight_decay=0.05,
    )
    module.model.synthetic = nn.Module()
    module.model.synthetic.lora_A = nn.Linear(16, 2, bias=False)
    module._trainer = SimpleNamespace(estimated_stepping_batches=10)

    optimizer = module.configure_optimizers()["optimizer"]
    group_by_parameter = {id(parameter): group for group in optimizer.param_groups for parameter in group["params"]}

    embedding_group = group_by_parameter[id(module.model.token_embedding.weight)]
    bias_group = group_by_parameter[id(module.model.token_head.bias)]
    lora_group = group_by_parameter[id(module.model.synthetic.lora_A.weight)]
    assert embedding_group["weight_decay"] == 0
    assert bias_group["weight_decay"] == 0
    assert lora_group["initial_lr"] == pytest.approx(5e-5)
    assert lora_group["lr"] == pytest.approx(0.0)
    assert lora_group["weight_decay"] == pytest.approx(0.05)
