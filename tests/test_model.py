import pytest
import torch
from torch.utils.data import DataLoader

pytest.importorskip("lightning.pytorch")

from rna_scaffold.data import RnaMotifDenoisingDataset
from rna_scaffold.lightning_module import RnaScaffoldLitModule, compact_fixed_motifs
from rna_scaffold.model import ScaffoldModelOutput, compute_denoising_losses
from rna_scaffold.records import RnaSequenceRecord
from rna_scaffold.tokenizer import RnaTokenizer


def test_lightning_module_returns_finite_joint_denoising_loss():
    tokenizer = RnaTokenizer()
    model = RnaScaffoldLitModule(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_length=8,
        lr=1e-3,
    )
    batch = {
        "input_ids": torch.tensor([[3, 8, 11, 3, 0, 0, 0, 0]]),
        "target_base_ids": torch.tensor([[0, 0, 3, 1, -100, -100, -100, -100]]),
        "fixed_mask": torch.tensor([[False, True, True, False, False, False, False, False]]),
        "attention_mask": torch.tensor([[True, True, True, True, False, False, False, False]]),
        "target_left_length": torch.tensor([1]),
        "target_right_length": torch.tensor([1]),
    }

    output = model.training_step(batch, batch_idx=0)

    assert set(output) >= {
        "loss",
        "base_loss",
        "left_length_loss",
        "right_length_loss",
        "token_accuracy",
        "left_length_accuracy",
        "right_length_accuracy",
    }
    assert all(torch.isfinite(output[name]) for name in output)


def test_lightning_token_accuracy_excludes_fixed_motif_in_prediction_mask():
    class FixedOutputs(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None):
            batch_size, sequence_length = input_ids.shape
            token_logits = torch.zeros(batch_size, sequence_length, 4, device=input_ids.device)
            token_logits[..., 0] = 10.0
            flank_logits = torch.zeros(batch_size, 5, device=input_ids.device)
            flank_logits[:, 1] = 10.0
            return ScaffoldModelOutput(token_logits, flank_logits, flank_logits)

    tokenizer = RnaTokenizer()
    model = RnaScaffoldLitModule(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_length=4,
    )
    model.model = FixedOutputs()
    batch = {
        "input_ids": torch.tensor([[3, 8, 3, 0]]),
        "target_base_ids": torch.tensor([[0, 3, 0, -100]]),
        "fixed_mask": torch.tensor([[False, True, False, False]]),
        "attention_mask": torch.tensor([[True, True, True, False]]),
        "prediction_mask": torch.tensor([[True, True, True, False]]),
        "target_left_length": torch.tensor([1]),
        "target_right_length": torch.tensor([1]),
    }

    output = model.training_step(batch, batch_idx=0)

    assert output["token_accuracy"] == pytest.approx(1.0)


def test_lightning_module_trains_on_a_real_v2_denoising_dataset_batch():
    tokenizer = RnaTokenizer()
    dataset = RnaMotifDenoisingDataset(
        records=[RnaSequenceRecord("x", "AACCGGUU", "RF1", "unit")],
        tokenizer=tokenizer,
        max_length=8,
        min_motif_length=4,
        max_motif_length=4,
        motif_length_buckets=None,
        min_flank_length=2,
        min_total_scaffold_length=0,
        preferred_total_scaffold_length=0,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    model = RnaScaffoldLitModule(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_length=8,
        lr=1e-3,
    )

    output = model.training_step(batch, batch_idx=0)

    assert torch.isfinite(output["loss"])


def test_compact_fixed_motifs_removes_flank_length_and_position_leakage():
    input_ids = torch.tensor([[3, 8, 11, 10, 3, 3], [3, 3, 9, 8, 11, 3]])
    fixed_mask = torch.tensor(
        [
            [False, True, True, True, False, False],
            [False, False, True, True, True, False],
        ]
    )

    motifs, attention = compact_fixed_motifs(input_ids, fixed_mask, pad_token_id=0)

    assert motifs.tolist() == [[8, 11, 10], [9, 8, 11]]
    assert attention.tolist() == [[True, True, True], [True, True, True]]


def test_per_sequence_base_loss_prevents_long_examples_from_dominating():
    logits = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]],
        ]
    )
    output = ScaffoldModelOutput(logits, torch.zeros(2, 4), torch.zeros(2, 4))
    targets = torch.tensor([[0, -100, -100], [0, 0, 0]])
    attention = targets.ne(-100)
    fixed = torch.zeros_like(attention)
    lengths = torch.zeros(2, dtype=torch.long)

    flattened = compute_denoising_losses(
        output,
        targets,
        fixed,
        attention,
        lengths,
        lengths,
        left_length_loss_weight=0.0,
        right_length_loss_weight=0.0,
    )
    normalized = compute_denoising_losses(
        output,
        targets,
        fixed,
        attention,
        lengths,
        lengths,
        left_length_loss_weight=0.0,
        right_length_loss_weight=0.0,
        normalize_base_loss_per_sequence=True,
    )

    assert normalized.base_loss > flattened.base_loss


def test_per_sequence_base_loss_rejects_an_unmasked_sample():
    output = ScaffoldModelOutput(
        torch.zeros((2, 2, 4)),
        torch.zeros((2, 3)),
        torch.zeros((2, 3)),
    )
    targets = torch.tensor([[0, -100], [-100, -100]])
    attention = targets.ne(-100)
    fixed = torch.zeros_like(attention)
    lengths = torch.zeros(2, dtype=torch.long)

    with pytest.raises(ValueError, match="every sample"):
        compute_denoising_losses(
            output,
            targets,
            fixed,
            attention,
            lengths,
            lengths,
            left_length_loss_weight=0.0,
            right_length_loss_weight=0.0,
            normalize_base_loss_per_sequence=True,
        )
