import torch

from rna_scaffold.model import (
    MotifDenoisingTransformer,
    ScaffoldModelOutput,
    compute_denoising_losses,
    restore_fixed_tokens,
)


def test_model_outputs_tokens_and_bilateral_flank_lengths():
    model = MotifDenoisingTransformer(
        vocab_size=12,
        pad_token_id=0,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        max_length=64,
    ).eval()
    input_ids = torch.tensor([[3, 3, 8, 11, 3, 3]])
    attention_mask = torch.ones(1, 6, dtype=torch.bool)

    output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert output.token_logits.shape == (1, 6, 4)
    assert output.left_length_logits.shape == (1, 65)
    assert output.right_length_logits.shape == (1, 65)
    assert not hasattr(output, "length_logits")
    assert not hasattr(output, "position_logits")


def test_restore_fixed_tokens_never_changes_motif():
    original = torch.tensor([[3, 8, 11, 3]])
    proposed = torch.tensor([[9, 9, 9, 9]])
    fixed = torch.tensor([[False, True, True, False]])

    restored = restore_fixed_tokens(proposed, original, fixed)

    assert restored.tolist() == [[9, 8, 11, 9]]


def test_padding_tokens_do_not_change_valid_position_outputs():
    torch.manual_seed(3)
    model = MotifDenoisingTransformer(
        vocab_size=12,
        pad_token_id=0,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        max_length=16,
    ).eval()
    short = model(
        input_ids=torch.tensor([[3, 8, 11, 3]]),
        attention_mask=torch.tensor([[True, True, True, True]]),
    )
    padded = model(
        input_ids=torch.tensor([[3, 8, 11, 3, 0, 0]]),
        attention_mask=torch.tensor([[True, True, True, True, False, False]]),
    )

    assert torch.allclose(short.token_logits, padded.token_logits[:, :4], atol=1e-6)
    assert torch.allclose(short.left_length_logits, padded.left_length_logits, atol=1e-6)
    assert torch.allclose(short.right_length_logits, padded.right_length_logits, atol=1e-6)


def test_denoising_loss_excludes_fixed_motif_positions():
    output = ScaffoldModelOutput(
        token_logits=torch.tensor([[[8.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]]]),
        left_length_logits=torch.tensor([[0.0, 0.0, 8.0, 0.0, 0.0]]),
        right_length_logits=torch.tensor([[0.0, 0.0, 0.0, 8.0, 0.0]]),
    )
    common = {
        "output": output,
        "fixed_mask": torch.tensor([[False, True]]),
        "attention_mask": torch.tensor([[True, True]]),
        "target_left_length": torch.tensor([2]),
        "target_right_length": torch.tensor([3]),
    }

    first = compute_denoising_losses(target_base_ids=torch.tensor([[0, 0]]), **common)
    changed_fixed_target = compute_denoising_losses(target_base_ids=torch.tensor([[0, 3]]), **common)

    assert torch.allclose(first.base_loss, changed_fixed_target.base_loss)
    assert torch.isfinite(first.total_loss)


def test_denoising_loss_excludes_fixed_motif_even_when_prediction_mask_includes_it():
    output = ScaffoldModelOutput(
        token_logits=torch.tensor([[[8.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]]]),
        left_length_logits=torch.tensor([[0.0, 0.0, 8.0, 0.0, 0.0]]),
        right_length_logits=torch.tensor([[0.0, 0.0, 0.0, 8.0, 0.0]]),
    )
    common = {
        "output": output,
        "fixed_mask": torch.tensor([[False, True]]),
        "attention_mask": torch.tensor([[True, True]]),
        "prediction_mask": torch.tensor([[True, True]]),
        "target_left_length": torch.tensor([2]),
        "target_right_length": torch.tensor([3]),
    }

    first = compute_denoising_losses(target_base_ids=torch.tensor([[0, 0]]), **common)
    changed_fixed_target = compute_denoising_losses(target_base_ids=torch.tensor([[0, 3]]), **common)

    assert torch.allclose(first.base_loss, changed_fixed_target.base_loss)


def test_denoising_losses_supervise_left_and_right_lengths_independently():
    output = ScaffoldModelOutput(
        token_logits=torch.tensor([[[8.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]]]),
        left_length_logits=torch.tensor([[0.0, 0.0, 8.0, 0.0, 0.0]]),
        right_length_logits=torch.tensor([[0.0, 0.0, 0.0, 8.0, 0.0]]),
    )
    common = {
        "output": output,
        "target_base_ids": torch.tensor([[0, 0]]),
        "fixed_mask": torch.tensor([[False, True]]),
        "attention_mask": torch.tensor([[True, True]]),
        "target_left_length": torch.tensor([2]),
        "target_right_length": torch.tensor([3]),
    }

    expected = compute_denoising_losses(**common)
    changed_left = compute_denoising_losses(**(common | {"target_left_length": torch.tensor([1])}))
    changed_right = compute_denoising_losses(**(common | {"target_right_length": torch.tensor([1])}))

    assert changed_left.left_length_loss > expected.left_length_loss
    assert torch.allclose(changed_left.right_length_loss, expected.right_length_loss)
    assert changed_right.right_length_loss > expected.right_length_loss
    assert torch.allclose(changed_right.left_length_loss, expected.left_length_loss)


def test_activation_checkpointing_backward_is_finite():
    model = MotifDenoisingTransformer(
        vocab_size=8,
        pad_token_id=0,
        d_model=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        max_length=16,
        activation_checkpointing=True,
    )
    model.train()
    output = model(
        input_ids=torch.tensor([[1, 2, 3, 4, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
    )
    output.token_logits.sum().backward()

    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
