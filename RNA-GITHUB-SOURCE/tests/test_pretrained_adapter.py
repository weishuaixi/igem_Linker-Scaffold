from __future__ import annotations

import copy
import hashlib
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from rna_scaffold import pretrained
from rna_scaffold.lightning_module import RnaScaffoldLitModule
from rna_scaffold.model import MotifDenoisingTransformer
from rna_scaffold.pretrained import (
    LoRALinear,
    PretrainedEncoderConfig,
    build_pretrained_encoder,
    inject_lora_adapters,
    pretrained_metadata,
)


class TinyEncoder(nn.Module):
    output_dim = 6

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(12, self.output_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids) * attention_mask.unsqueeze(-1)


class TinyAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        combined = self.q_proj(hidden) + self.k_proj(hidden) + self.v_proj(hidden)
        return self.out_proj(torch.tanh(combined))


class TinyRnaFmLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = TinyAttention(width)
        self.base_dropout = nn.Dropout(0.75)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.base_dropout(self.self_attn(hidden))


class TinyRnaFmEncoder(nn.Module):
    output_dim = 6

    def __init__(self, layer_count: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(12, self.output_dim)
        self.layers = nn.ModuleList([TinyRnaFmLayer(self.output_dim) for _ in range(layer_count)])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden * attention_mask.unsqueeze(-1)


def test_lora_injection_freezes_base_and_only_targets_last_layers():
    encoder = TinyRnaFmEncoder(layer_count=4)

    targets = inject_lora_adapters(
        encoder,
        last_n_layers=2,
        rank=2,
        alpha=4.0,
        dropout=0.1,
    )

    assert targets == (
        "layers.2.self_attn.q_proj",
        "layers.2.self_attn.k_proj",
        "layers.2.self_attn.v_proj",
        "layers.2.self_attn.out_proj",
        "layers.3.self_attn.q_proj",
        "layers.3.self_attn.k_proj",
        "layers.3.self_attn.v_proj",
        "layers.3.self_attn.out_proj",
    )
    assert isinstance(encoder.layers[0].self_attn.q_proj, nn.Linear)
    assert isinstance(encoder.layers[2].self_attn.q_proj, LoRALinear)
    adapter_parameters = [
        parameter for name, parameter in encoder.named_parameters() if ".lora_A." in name or ".lora_B." in name
    ]
    base_parameters = [
        parameter for name, parameter in encoder.named_parameters() if ".lora_A." not in name and ".lora_B." not in name
    ]
    assert adapter_parameters
    assert all(parameter.requires_grad for parameter in adapter_parameters)
    assert base_parameters
    assert not any(parameter.requires_grad for parameter in base_parameters)


def test_zero_initialized_lora_preserves_original_output():
    torch.manual_seed(7)
    original = TinyRnaFmEncoder(layer_count=2).eval()
    adapted = copy.deepcopy(original)
    inject_lora_adapters(adapted, last_n_layers=1, rank=3, alpha=6.0, dropout=0.5)
    inputs = torch.tensor([[8, 9, 10, 11]])
    attention = torch.ones_like(inputs, dtype=torch.bool)

    adapted.eval()

    torch.testing.assert_close(adapted(inputs, attention), original(inputs, attention))
    assert all(
        torch.count_nonzero(module.lora_B.weight).item() == 0
        for module in adapted.modules()
        if isinstance(module, LoRALinear)
    )


def test_lora_module_call_applies_dropout_to_inputs_and_backpropagates():
    torch.manual_seed(13)
    base = nn.Linear(5, 4)
    adapter = LoRALinear(base, rank=2, alpha=4.0, dropout=0.5)
    with torch.no_grad():
        adapter.lora_A.weight.fill_(0.25)
        adapter.lora_B.weight.fill_(0.5)
    inputs = torch.randn(3, 5)

    torch.manual_seed(29)
    dropped_inputs = F.dropout(inputs, p=0.5, training=True)
    expected = base(inputs) + adapter.scale * adapter.lora_B(adapter.lora_A(dropped_inputs))
    torch.manual_seed(29)

    adapted_output = adapter(inputs)
    adapted_output.square().mean().backward()

    torch.testing.assert_close(adapted_output, expected)
    assert base.weight.grad is None
    assert base.bias.grad is None
    assert adapter.lora_A.weight.grad is not None
    assert adapter.lora_B.weight.grad is not None
    assert torch.isfinite(adapter.lora_A.weight.grad).all()
    assert torch.isfinite(adapter.lora_B.weight.grad).all()
    assert torch.count_nonzero(adapter.lora_A.weight.grad) > 0
    assert torch.count_nonzero(adapter.lora_B.weight.grad) > 0


def test_lora_input_dropout_is_stochastic_only_in_train_mode():
    base = nn.Linear(5, 4)
    adapter = LoRALinear(base, rank=2, alpha=4.0, dropout=0.5)
    with torch.no_grad():
        adapter.lora_A.weight.fill_(1.0)
        adapter.lora_B.weight.fill_(0.5)
    inputs = torch.ones(16, 5)
    adapter.train()

    torch.manual_seed(17)
    first_train = adapter(inputs)
    torch.manual_seed(23)
    second_train = adapter(inputs)

    assert not torch.allclose(first_train, second_train)
    adapter.eval()
    first_eval = adapter(inputs)
    second_eval = adapter(inputs)
    torch.testing.assert_close(first_eval, second_eval)


class TinyFunctionalAttention(nn.Module):
    """RNA-FM-shaped branch: functional weights by default, modules as fallback."""

    def __init__(self, width: int = 4, heads: int = 2) -> None:
        super().__init__()
        self.embed_dim = width
        self.num_heads = heads
        self.dropout = 0.0
        self.enable_torch_version = True
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.enable_torch_version:
            output, _ = F.multi_head_attention_forward(
                hidden,
                hidden,
                hidden,
                self.embed_dim,
                self.num_heads,
                torch.empty(0, device=hidden.device, dtype=hidden.dtype),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                None,
                None,
                False,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training,
                None,
                False,
                None,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )
            return output
        combined = self.q_proj(hidden) + self.k_proj(hidden) + self.v_proj(hidden)
        return self.out_proj(torch.tanh(combined))


class TinyFunctionalLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = TinyFunctionalAttention()


class TinyFunctionalEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyFunctionalLayer()])


def test_rna_fm_functional_attention_switches_to_adapter_aware_module_path():
    encoder = TinyFunctionalEncoder()
    attention = encoder.layers[0].self_attn
    assert attention.enable_torch_version is True

    inject_lora_adapters(encoder, last_n_layers=1, rank=2, alpha=4.0, dropout=0.25)
    with torch.no_grad():
        for adapter in attention.modules():
            if isinstance(adapter, LoRALinear):
                adapter.lora_A.weight.fill_(0.25)
                adapter.lora_B.weight.fill_(0.5)
    inputs = torch.randn(3, 2, 4)

    output = attention(inputs)
    output.square().mean().backward()

    assert attention.enable_torch_version is False
    adapters = [module for module in attention.modules() if isinstance(module, LoRALinear)]
    assert len(adapters) == 4
    assert all(adapter.base.weight.grad is None for adapter in adapters)
    assert all(adapter.lora_A.weight.grad is not None for adapter in adapters)
    assert all(adapter.lora_B.weight.grad is not None for adapter in adapters)
    assert all(torch.isfinite(adapter.lora_A.weight.grad).all() for adapter in adapters)
    assert all(torch.isfinite(adapter.lora_B.weight.grad).all() for adapter in adapters)
    assert all(torch.count_nonzero(adapter.lora_A.weight.grad) > 0 for adapter in adapters)
    assert all(torch.count_nonzero(adapter.lora_B.weight.grad) > 0 for adapter in adapters)


def test_lora_injection_rejects_unsupported_layout_with_candidates():
    class UnsupportedEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(4, 4))])

    with pytest.raises(ValueError, match=r"supported.*attention projections.*blocks\.0\.0"):
        inject_lora_adapters(
            UnsupportedEncoder(),
            last_n_layers=1,
            rank=2,
            alpha=4.0,
            dropout=0.0,
        )


def test_none_pretrained_encoder_has_no_optional_dependency():
    assert build_pretrained_encoder(PretrainedEncoderConfig(kind="none")) is None


def test_frozen_pretrained_encoder_has_no_trainable_parameters(monkeypatch):
    monkeypatch.setattr(pretrained, "load_rna_fm", lambda _: TinyEncoder())
    encoder = build_pretrained_encoder(
        PretrainedEncoderConfig(
            kind="rna_fm",
            checkpoint="fake.pt",
            expected_checkpoint_sha256="0" * 64,
            mode="frozen",
        )
    )

    assert encoder is not None
    assert not any(parameter.requires_grad for parameter in encoder.parameters())


def test_hash_verified_local_checkpoint_enables_legacy_loader_only_during_load(monkeypatch, tmp_path):
    checkpoint = tmp_path / "rna-fm.pt"
    checkpoint.write_bytes(b"reviewed-rna-fm")
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    observed = {}
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)

    def fake_load(path):
        observed["path"] = path
        observed["compatibility"] = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        return TinyEncoder()

    monkeypatch.setattr(pretrained, "load_rna_fm", fake_load)

    build_pretrained_encoder(
        PretrainedEncoderConfig(
            kind="rna_fm",
            checkpoint=str(checkpoint),
            expected_checkpoint_sha256=expected,
            mode="frozen",
        )
    )

    assert observed == {"path": str(checkpoint), "compatibility": "1"}
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ


@pytest.mark.parametrize(
    "configured_digest",
    [None, pretrained.OFFICIAL_RNA_FM_SHA256],
    ids=["legacy-env-config", "direct-v4-config"],
)
def test_scaffold_rna_fm_policy_preserves_official_release_configs(
    monkeypatch,
    tmp_path,
    configured_digest,
):
    controlled_path = tmp_path / pretrained.RNA_FM_CHECKPOINT_FILENAME
    monkeypatch.setattr(
        pretrained,
        "default_rna_fm_checkpoint_paths",
        lambda: (controlled_path.resolve(),),
    )
    config = {
        "kind": "rna_fm",
        "checkpoint": ".cache/torch/hub/checkpoints/RNA-FM_pretrained.pth",
        "expected_checkpoint_sha256_env": "RNA_FM_EXPECTED_SHA256",
        "mode": "lora",
    }
    if configured_digest is not None:
        config["expected_checkpoint_sha256"] = configured_digest

    authorized = pretrained.authorize_scaffold_rna_fm_config(
        config,
        {
            "expected_checkpoint_sha256": pretrained.OFFICIAL_RNA_FM_SHA256,
            "checkpoint_sha256": pretrained.OFFICIAL_RNA_FM_SHA256,
        },
    )

    assert authorized["checkpoint"] == str(controlled_path.resolve())
    assert authorized["expected_checkpoint_sha256"] == pretrained.OFFICIAL_RNA_FM_SHA256
    assert authorized["expected_checkpoint_sha256_env"] is None


def test_pretrained_features_are_projected_into_generator_width():
    encoder = TinyEncoder()
    model = MotifDenoisingTransformer(
        vocab_size=12,
        pad_token_id=0,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        max_length=16,
        pretrained_encoder=encoder,
    )
    output = model(
        input_ids=torch.tensor([[3, 8, 11, 3]]),
        attention_mask=torch.ones(1, 4, dtype=torch.bool),
    )

    assert model.pretrained_projection.in_features == encoder.output_dim
    assert model.pretrained_projection.out_features == 16
    assert output.token_logits.shape == (1, 4, 4)


def test_frozen_pretrained_encoder_stays_in_eval_mode(monkeypatch):
    monkeypatch.setattr(pretrained, "load_rna_fm", lambda _: TinyEncoder())
    encoder = build_pretrained_encoder(PretrainedEncoderConfig(kind="rna_fm", mode="frozen"))
    model = MotifDenoisingTransformer(
        vocab_size=12,
        pad_token_id=0,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        max_length=16,
        pretrained_encoder=encoder,
    )

    model.train()

    assert encoder.training is False


def test_lora_mode_constructs_trainable_adapters(monkeypatch):
    monkeypatch.setattr(pretrained, "load_rna_fm", lambda _: TinyRnaFmEncoder(layer_count=3))

    encoder = build_pretrained_encoder(
        PretrainedEncoderConfig(
            kind="rna_fm",
            mode="lora",
            lora_rank=2,
            lora_alpha=4.0,
            lora_dropout=0.2,
            lora_last_n_layers=1,
        )
    )

    assert encoder is not None
    assert encoder._rna_scaffold_lora_targets == (
        "layers.2.self_attn.q_proj",
        "layers.2.self_attn.k_proj",
        "layers.2.self_attn.v_proj",
        "layers.2.self_attn.out_proj",
    )
    assert any(parameter.requires_grad for parameter in encoder.parameters())
    assert all(
        parameter.requires_grad == (".lora_A." in name or ".lora_B." in name)
        for name, parameter in encoder.named_parameters()
    )


def test_core_model_keeps_lora_base_eval_while_adapter_dropout_tracks_mode(monkeypatch):
    monkeypatch.setattr(pretrained, "load_rna_fm", lambda _: TinyRnaFmEncoder(layer_count=2))
    encoder = build_pretrained_encoder(
        PretrainedEncoderConfig(
            kind="rna_fm",
            mode="lora",
            lora_rank=2,
            lora_dropout=0.3,
            lora_last_n_layers=1,
        )
    )
    model = MotifDenoisingTransformer(
        vocab_size=12,
        pad_token_id=0,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_length=16,
        pretrained_encoder=encoder,
    )

    model.train()

    assert encoder.training is False
    assert not encoder.layers[-1].base_dropout.training
    assert encoder.layers[-1].self_attn.q_proj.dropout.training

    model.eval()

    assert not encoder.layers[-1].self_attn.q_proj.dropout.training


def test_v1_freeze_flag_is_rejected():
    with pytest.raises(ValueError, match=r"freeze.*removed.*mode"):
        build_pretrained_encoder({"kind": "rna_fm", "freeze": True})


def test_lora_checkpoint_metadata_records_adapter_contract(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "rna-fm.pt"
    checkpoint_path.write_bytes(b"tiny-rna-fm-checkpoint")
    monkeypatch.setattr(pretrained, "load_rna_fm", lambda _: TinyRnaFmEncoder(layer_count=2))
    config = PretrainedEncoderConfig(
        kind="rna_fm",
        checkpoint=str(checkpoint_path),
        expected_checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        mode="lora",
        lora_rank=3,
        lora_alpha=9.0,
        lora_dropout=0.25,
        lora_last_n_layers=1,
    )
    encoder = build_pretrained_encoder(config)

    metadata = pretrained_metadata(config, encoder)

    assert metadata["kind"] == "rna_fm"
    assert metadata["mode"] == "lora"
    assert metadata["lora_rank"] == 3
    assert metadata["lora_alpha"] == 9.0
    assert metadata["lora_dropout"] == 0.25
    assert metadata["lora_dropout_semantics"] == "input"
    assert metadata["lora_projection_execution"] == "module_forward"
    assert metadata["lora_last_n_layers"] == 1
    assert metadata["adapter_targets"] == list(encoder._rna_scaffold_lora_targets)
    assert metadata["checkpoint_sha256"] == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert metadata["parameter_counts"] == {
        "total": sum(parameter.numel() for parameter in encoder.parameters()),
        "trainable": sum(parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad),
    }


def test_optimizer_contains_all_and_only_trainable_parameters(monkeypatch):
    monkeypatch.setattr(pretrained, "load_rna_fm", lambda _: TinyRnaFmEncoder(layer_count=2))
    module = RnaScaffoldLitModule(
        vocab_size=12,
        pad_token_id=0,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_length=16,
        pretrained={
            "kind": "rna_fm",
            "mode": "lora",
            "lora_rank": 2,
            "lora_alpha": 4.0,
            "lora_dropout": 0.0,
            "lora_last_n_layers": 1,
        },
    )
    module._trainer = SimpleNamespace(estimated_stepping_batches=10)

    optimizer = module.configure_optimizers()["optimizer"]

    optimized_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    trainable_ids = {id(parameter) for parameter in module.parameters() if parameter.requires_grad}
    frozen_ids = {id(parameter) for parameter in module.parameters() if not parameter.requires_grad}
    assert optimized_ids == trainable_ids
    assert optimized_ids.isdisjoint(frozen_ids)
    assert any(
        id(parameter) in optimized_ids
        for name, parameter in module.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    )


def test_lightning_checkpoint_uses_constructed_pretrained_metadata(monkeypatch):
    monkeypatch.setattr(pretrained, "load_rna_fm", lambda _: TinyRnaFmEncoder(layer_count=2))
    module = RnaScaffoldLitModule(
        vocab_size=12,
        pad_token_id=0,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_length=16,
        pretrained={
            "kind": "rna_fm",
            "mode": "lora",
            "lora_rank": 2,
            "lora_last_n_layers": 1,
        },
    )
    checkpoint = {}

    module.on_save_checkpoint(checkpoint)

    metadata = checkpoint["pretrained_encoder"]
    assert metadata["adapter_targets"]
    assert metadata["parameter_counts"]["trainable"] > 0


def test_missing_rna_fm_explains_optional_install(monkeypatch):
    monkeypatch.setattr(pretrained, "_import_fm", lambda: (_ for _ in ()).throw(ImportError("missing")))

    try:
        pretrained.load_rna_fm(None)
    except RuntimeError as error:
        assert "pip install rna-fm" in str(error)
    else:
        raise AssertionError("missing RNA-FM must raise a useful error")


def test_local_rna_fm_checkpoint_uses_the_rna_alphabet(monkeypatch, tmp_path):
    captured = {}

    class FakeAlphabet:
        padding_idx = 0
        mask_idx = 1

        @staticmethod
        def get_idx(base):
            return {"A": 2, "U": 3, "C": 4, "G": 5}[base]

    class FakePretrained:
        @staticmethod
        def load_model_and_alphabet_local(path, *, theme):
            captured["path"] = path
            captured["theme"] = theme
            return nn.Identity(), FakeAlphabet()

    fake_fm = type("FakeFm", (), {"pretrained": FakePretrained})()
    monkeypatch.setattr(pretrained, "_import_fm", lambda: fake_fm)
    checkpoint = tmp_path / "RNA-FM_pretrained.pth"

    encoder = pretrained.load_rna_fm(str(checkpoint))

    assert captured == {"path": checkpoint, "theme": "rna"}
    assert isinstance(encoder, pretrained.RnaFmEncoder)
