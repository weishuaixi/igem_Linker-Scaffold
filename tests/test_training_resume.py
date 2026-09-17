import sys
from types import SimpleNamespace

import pytest
import torch

import train
from rna_scaffold.checkpoints import CheckpointCompatibilityError
from rna_scaffold.lightning_module import RnaScaffoldLitModule
from rna_scaffold.tokenizer import RnaTokenizer


def _write_versioned_checkpoint(path, version):
    payload = {}
    if version is not None:
        payload["architecture_version"] = version
    torch.save(payload, path)


def _run_train_main(monkeypatch, checkpoint, resume_source):
    fit_calls = []
    config = {
        "seed": 7,
        "data": {},
        "model": {},
        "trainer": {
            "args": {},
            "resume_from_checkpoint": str(checkpoint) if resume_source == "config" else None,
        },
        "wandb": {"project": "test", "log_model": False},
    }

    class RecordingTrainer:
        def __init__(self, **kwargs):
            pass

        def fit(self, model, datamodule, ckpt_path):
            fit_calls.append(ckpt_path)

    monkeypatch.setattr(train, "load_config", lambda _: config)
    monkeypatch.setattr(train.L, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.L, "Trainer", RecordingTrainer)
    monkeypatch.setattr(
        train,
        "RnaScaffoldDataModule",
        lambda **kwargs: SimpleNamespace(split_manifest=None, denoising_data_audit={}),
    )
    monkeypatch.setattr(
        train,
        "RnaScaffoldLitModule",
        lambda **kwargs: SimpleNamespace(pretrained_metadata={"kind": "none"}),
    )
    monkeypatch.setattr(
        train,
        "ModelCheckpoint",
        lambda **kwargs: SimpleNamespace(best_model_path="unused.ckpt"),
    )
    monkeypatch.setattr(train, "EarlyStopping", lambda **kwargs: object())
    monkeypatch.setattr(train, "LearningRateMonitor", lambda **kwargs: object())
    monkeypatch.setattr(train, "WandbLogger", lambda **kwargs: object())
    monkeypatch.setattr(train, "write_training_manifest", lambda *args, **kwargs: {})
    argv = ["src/train.py", "--config", "unused.yaml"]
    if resume_source == "cli":
        argv.extend(["--resume", str(checkpoint)])
    monkeypatch.setattr(sys, "argv", argv)

    train.main()
    return fit_calls


@pytest.mark.parametrize("resume_source", ["cli", "config"])
@pytest.mark.parametrize(
    "version",
    [None, 1, True, 2.0, "2"],
    ids=["missing", "v1", "boolean", "float", "string"],
)
def test_train_resume_rejects_non_v2_before_trainer_fit(
    monkeypatch,
    tmp_path,
    resume_source,
    version,
):
    checkpoint = tmp_path / f"{resume_source}-{version}.ckpt"
    _write_versioned_checkpoint(checkpoint, version)

    with pytest.raises(CheckpointCompatibilityError, match="architecture_version 2"):
        _run_train_main(monkeypatch, checkpoint, resume_source)


@pytest.mark.parametrize("resume_source", ["cli", "config"])
def test_train_resume_allows_v2_to_reach_trainer_fit(monkeypatch, tmp_path, resume_source):
    checkpoint = tmp_path / f"{resume_source}-v2.ckpt"
    _write_versioned_checkpoint(checkpoint, 2)

    fit_calls = _run_train_main(monkeypatch, checkpoint, resume_source)

    assert fit_calls == [str(checkpoint)]


@pytest.mark.parametrize(
    "version",
    [None, 1, True, 2.0, "2"],
    ids=["missing", "v1", "boolean", "float", "string"],
)
def test_lightning_resume_hook_rejects_non_v2_with_shared_error(version):
    tokenizer = RnaTokenizer()
    module = RnaScaffoldLitModule(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        max_length=16,
        pretrained={"kind": "none"},
    )
    payload = {}
    if version is not None:
        payload["architecture_version"] = version
    resume_hook = getattr(module, "on_load_checkpoint", lambda _: None)

    with pytest.raises(CheckpointCompatibilityError, match="architecture_version 2"):
        resume_hook(payload)


def test_lightning_resume_hook_accepts_v2():
    tokenizer = RnaTokenizer()
    module = RnaScaffoldLitModule(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        max_length=16,
        pretrained={"kind": "none"},
    )
    resume_hook = getattr(module, "on_load_checkpoint", lambda _: None)

    resume_hook({"architecture_version": 2})
