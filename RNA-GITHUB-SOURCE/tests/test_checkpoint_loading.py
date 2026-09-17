import hashlib

import pytest
import torch
from torch import nn

from rna_scaffold import pretrained
from rna_scaffold.checkpoints import CheckpointCompatibilityError, load_scaffold_checkpoint
from rna_scaffold.lightning_module import RnaScaffoldLitModule
from rna_scaffold.pretrained import LoRALinear, TrustedRnaFmCheckpoint
from rna_scaffold.tokenizer import RnaTokenizer


class TinyAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)


class TinyLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = TinyAttention(width)


class TinyRnaFmEncoder(nn.Module):
    output_dim = 6

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(12, self.output_dim)
        self.layers = nn.ModuleList([TinyLayer(self.output_dim) for _ in range(2)])

    def forward(self, input_ids, attention_mask):
        return self.embedding(input_ids) * attention_mask.unsqueeze(-1)


def _tiny_hparams() -> dict:
    tokenizer = RnaTokenizer()
    return {
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "d_model": 16,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 32,
        "dropout": 0.0,
        "max_length": 32,
        "activation_checkpointing": False,
        "pretrained": {"kind": "none"},
        "lr": 1e-3,
        "weight_decay": 0.0,
        "left_length_loss_weight": 0.25,
        "right_length_loss_weight": 0.25,
    }


def _write_tiny_checkpoint(path):
    hparams = _tiny_hparams()
    module = RnaScaffoldLitModule(**hparams)
    torch.save(
        {
            "architecture_version": 2,
            "state_dict": module.state_dict(),
            "hyper_parameters": hparams,
        },
        path,
    )


def _write_checkpoint_with_version(path, version):
    hparams = _tiny_hparams()
    module = RnaScaffoldLitModule(**hparams)
    payload = {"state_dict": module.state_dict(), "hyper_parameters": hparams}
    if version is not None:
        payload["architecture_version"] = version
    torch.save(payload, path)


def test_load_scaffold_checkpoint_reconstructs_exact_model(tmp_path):
    checkpoint = tmp_path / "tiny.ckpt"
    _write_tiny_checkpoint(checkpoint)

    loaded = load_scaffold_checkpoint(checkpoint)

    assert loaded.max_length == 32
    assert loaded.checkpoint_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert loaded.model.training is False
    assert loaded.tokenizer.vocab_size == RnaTokenizer().vocab_size


def test_load_scaffold_checkpoint_rejects_forged_pretrained_metadata(tmp_path):
    checkpoint = tmp_path / "forged-identity.ckpt"
    _write_tiny_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    payload["pretrained_encoder"] = {"kind": "rna_fm", "mode": "lora"}
    torch.save(payload, checkpoint)

    with pytest.raises(CheckpointCompatibilityError, match="pretrained identity mismatch"):
        load_scaffold_checkpoint(checkpoint)


def test_load_scaffold_checkpoint_strictly_round_trips_lora_state(monkeypatch, tmp_path):
    monkeypatch.setattr(pretrained, "load_rna_fm", lambda _: TinyRnaFmEncoder())
    rna_fm_checkpoint = tmp_path / "custom-rna-fm.pth"
    rna_fm_checkpoint.write_bytes(b"explicitly trusted test RNA-FM checkpoint")
    rna_fm_sha256 = hashlib.sha256(rna_fm_checkpoint.read_bytes()).hexdigest()
    hparams = _tiny_hparams()
    hparams["pretrained"] = {
        "kind": "rna_fm",
        "checkpoint": str(rna_fm_checkpoint),
        "expected_checkpoint_sha256": rna_fm_sha256,
        "mode": "lora",
        "lora_rank": 2,
        "lora_alpha": 4.0,
        "lora_dropout": 0.0,
        "lora_last_n_layers": 1,
    }
    module = RnaScaffoldLitModule(**hparams)
    with torch.no_grad():
        for adapter in module.modules():
            if isinstance(adapter, LoRALinear):
                adapter.lora_A.weight.fill_(0.125)
                adapter.lora_B.weight.fill_(0.25)
    expected_adapter_state = {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
        if ".lora_A." in name or ".lora_B." in name
    }
    checkpoint = tmp_path / "tiny-lora.ckpt"
    torch.save(
        {
            "architecture_version": 2,
            "state_dict": module.state_dict(),
            "hyper_parameters": hparams,
        },
        checkpoint,
    )

    loaded = load_scaffold_checkpoint(
        checkpoint,
        trusted_rna_fm_checkpoint=TrustedRnaFmCheckpoint(
            rna_fm_checkpoint,
            rna_fm_sha256,
        ),
    )

    loaded_encoder = loaded.model.model.pretrained_encoder
    assert loaded_encoder._rna_scaffold_lora_targets == (
        "layers.1.self_attn.q_proj",
        "layers.1.self_attn.k_proj",
        "layers.1.self_attn.v_proj",
        "layers.1.self_attn.out_proj",
    )
    loaded_state = loaded.model.state_dict()
    assert expected_adapter_state
    for name, expected in expected_adapter_state.items():
        torch.testing.assert_close(loaded_state[name], expected)


def test_scaffold_payload_cannot_authorize_an_arbitrary_rna_fm_pickle(
    monkeypatch,
    tmp_path,
):
    malicious_rna_fm = tmp_path / "payload-controlled.pth"
    malicious_rna_fm.write_bytes(b"not an authorized RNA-FM release")
    payload_digest = hashlib.sha256(malicious_rna_fm.read_bytes()).hexdigest()
    hparams = _tiny_hparams()
    hparams["pretrained"] = {
        "kind": "rna_fm",
        "checkpoint": str(malicious_rna_fm),
        "expected_checkpoint_sha256": payload_digest,
        "mode": "frozen",
    }
    checkpoint = tmp_path / "payload-controlled.ckpt"
    torch.save(
        {
            "architecture_version": 2,
            "state_dict": {},
            "hyper_parameters": hparams,
            "pretrained_encoder": {
                "kind": "rna_fm",
                "mode": "frozen",
                "checkpoint_sha256": payload_digest,
            },
        },
        checkpoint,
    )
    loader_called = False

    def fail_if_loaded(_):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("untrusted RNA-FM pickle must not be loaded")

    monkeypatch.setattr(pretrained, "load_rna_fm", fail_if_loaded)

    with pytest.raises(CheckpointCompatibilityError, match="controlled cache path"):
        load_scaffold_checkpoint(checkpoint)

    assert loader_called is False


def test_explicit_rna_fm_trust_must_match_payload_identity(monkeypatch, tmp_path):
    trusted_path = tmp_path / "trusted-rna-fm.pth"
    trusted_path.write_bytes(b"caller-reviewed RNA-FM checkpoint")
    trusted_digest = hashlib.sha256(trusted_path.read_bytes()).hexdigest()
    payload_digest = hashlib.sha256(b"different checkpoint").hexdigest()
    hparams = _tiny_hparams()
    hparams["pretrained"] = {
        "kind": "rna_fm",
        "checkpoint": str(trusted_path),
        "expected_checkpoint_sha256": payload_digest,
        "mode": "frozen",
    }
    checkpoint = tmp_path / "identity-mismatch.ckpt"
    torch.save(
        {
            "architecture_version": 2,
            "state_dict": {},
            "hyper_parameters": hparams,
        },
        checkpoint,
    )
    loader_called = False

    def fail_if_loaded(_):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("identity mismatch must be rejected before loading")

    monkeypatch.setattr(pretrained, "load_rna_fm", fail_if_loaded)

    with pytest.raises(CheckpointCompatibilityError, match="caller-authorized digest"):
        load_scaffold_checkpoint(
            checkpoint,
            trusted_rna_fm_checkpoint=TrustedRnaFmCheckpoint(
                trusted_path,
                trusted_digest,
            ),
        )

    assert loader_called is False


def test_lightning_checkpoint_metadata_declares_v2_architecture():
    module = RnaScaffoldLitModule(**_tiny_hparams())
    checkpoint = {}

    module.on_save_checkpoint(checkpoint)

    assert checkpoint["architecture_version"] == 2


def test_load_scaffold_checkpoint_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scaffold_checkpoint(tmp_path / "missing.ckpt")


def test_load_scaffold_checkpoint_rejects_missing_hyperparameters(tmp_path):
    checkpoint = tmp_path / "invalid.ckpt"
    torch.save({"architecture_version": 2, "state_dict": {}}, checkpoint)

    with pytest.raises(CheckpointCompatibilityError, match="hyper_parameters"):
        load_scaffold_checkpoint(checkpoint)


def test_load_scaffold_checkpoint_rejects_removed_state_even_when_marked_v2(tmp_path):
    checkpoint = tmp_path / "legacy-state.ckpt"
    _write_tiny_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    payload["state_dict"]["model.confidence_head.weight"] = torch.zeros(1, 16)
    payload["state_dict"]["model.confidence_head.bias"] = torch.zeros(1)
    torch.save(payload, checkpoint)

    with pytest.raises(CheckpointCompatibilityError, match="state_dict is incompatible"):
        load_scaffold_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "version",
    [None, 1, True, 2.0, "2"],
    ids=["missing", "v1", "boolean", "float", "string"],
)
def test_load_scaffold_checkpoint_rejects_non_v2_architecture(tmp_path, version):
    checkpoint = tmp_path / f"invalid-{version}.ckpt"
    _write_checkpoint_with_version(checkpoint, version)

    with pytest.raises(CheckpointCompatibilityError, match="architecture_version 2"):
        load_scaffold_checkpoint(checkpoint)
