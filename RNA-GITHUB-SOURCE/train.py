from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch
import yaml

try:
    import lightning.pytorch as L
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
except ImportError:  # pragma: no cover
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, WandbLogger

from rna_scaffold.checkpoints import validate_scaffold_checkpoint_version
from rna_scaffold.datamodule import RnaScaffoldDataModule
from rna_scaffold.geometry import validate_scaffold_length_limit
from rna_scaffold.lightning_module import RnaScaffoldLitModule
from rna_scaffold.pretrained import PretrainedEncoderConfig
from rna_scaffold.tokenizer import RnaTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _json_value(value):
    return asdict(value) if is_dataclass(value) else value


def write_training_manifest(
    config_path: str | Path,
    config: dict,
    best_model_path: str | Path,
    *,
    pretrained_identity: dict | None = None,
    split_manifest=None,
    data_audit: dict | None = None,
) -> dict:
    """Atomically publish the hashed best-checkpoint contract consumed by benchmarks."""
    config_path = Path(config_path)
    checkpoint_path = Path(best_model_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"best Lightning checkpoint does not exist: {checkpoint_path}")
    checkpoint_dir = Path(str(config["trainer"]["checkpoint_dir"]))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / "training_manifest.json"
    resolved_checkpoint = checkpoint_path.resolve()
    resolved_manifest_dir = checkpoint_dir.resolve()
    try:
        checkpoint_reference = resolved_checkpoint.relative_to(resolved_manifest_dir).as_posix()
    except ValueError:
        checkpoint_reference = str(resolved_checkpoint)
    pretrained_config = dict(config["model"].get("pretrained") or {})
    pretrained_kind = str(pretrained_config.get("kind", "none"))
    identity = dict(pretrained_identity or {})
    identity.setdefault("kind", pretrained_kind)
    identity.setdefault(
        "mode",
        "none" if pretrained_kind == "none" else str(pretrained_config.get("mode", "frozen")),
    )
    identity["kind"] = str(identity["kind"])
    identity["mode"] = "none" if identity["kind"] == "none" else str(identity["mode"])
    manifest = {
        "status": "completed",
        "architecture_version": 2,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "best_checkpoint": {
            "path": checkpoint_reference,
            "sha256": _sha256(checkpoint_path),
        },
        "method_identity": {
            "pretrained_kind": identity["kind"],
            "pretrained_mode": identity["mode"],
        },
        "pretrained_encoder": identity,
        "split_manifest": _json_value(split_manifest),
        "data_audit": data_audit,
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "lightning": _package_version("lightning"),
            "rna_fm": _package_version("rna-fm"),
        },
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=checkpoint_dir,
            prefix=".training-manifest-",
            suffix=".tmp",
            delete=False,
            newline="",
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, manifest_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return manifest


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_training_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise TypeError("training config root must be a mapping")
    for section in ("data", "model", "trainer", "wandb"):
        if not isinstance(config.get(section), dict):
            raise TypeError(f"training config section {section!r} must be a mapping")

    tokenizer = RnaTokenizer()
    RnaScaffoldDataModule(tokenizer=tokenizer, **config["data"])
    model_config = dict(config["model"])
    pretrained_config = dict(model_config.get("pretrained") or {})
    if "freeze" in pretrained_config:
        raise ValueError("pretrained.freeze was removed in V2; use pretrained.mode")
    PretrainedEncoderConfig(**pretrained_config)
    model_config.setdefault("mask_token_id", tokenizer.token_to_id[tokenizer.special.mask])
    if model_config.get("generation_remask_strategy", "confidence") not in {"confidence", "random"}:
        raise ValueError("model.generation_remask_strategy must be confidence or random")
    inspect.signature(RnaScaffoldLitModule).bind(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        **model_config,
    )
    self_conditioning_probability = float(model_config.get("self_conditioning_probability", 0.0))
    if not 0 <= self_conditioning_probability <= 1:
        raise ValueError("model.self_conditioning_probability must be in [0, 1]")
    if self_conditioning_probability > 0 and not bool(model_config.get("use_conditioning_embeddings", False)):
        raise ValueError("model self-conditioning requires use_conditioning_embeddings=true")
    conditioning_clip = int(model_config.get("conditioning_relative_distance_clip", 32))
    if conditioning_clip < 1:
        raise ValueError("model.conditioning_relative_distance_clip must be positive")
    pretrained_fusion = str(model_config.get("pretrained_fusion", "add"))
    if pretrained_fusion not in {"add", "gated"}:
        raise ValueError("model.pretrained_fusion must be 'add' or 'gated'")
    length_pooling = str(model_config.get("length_pooling", "mean"))
    if length_pooling not in {"mean", "attention"}:
        raise ValueError("model.length_pooling must be 'mean' or 'attention'")
    predicts_lengths = bool(model_config.get("predict_flank_lengths", True))
    left_length_weight = float(model_config.get("left_length_loss_weight", 0.25))
    right_length_weight = float(model_config.get("right_length_loss_weight", 0.25))
    if not predicts_lengths and (left_length_weight > 0 or right_length_weight > 0):
        raise ValueError("model length loss weights must be zero when predict_flank_lengths=false")
    if pretrained_fusion == "gated" and pretrained_config.get("kind", "none") == "none":
        raise ValueError("model.pretrained_fusion='gated' requires a pretrained encoder")
    model_denoise_steps = int(model_config.get("denoise_steps", 16))
    if model_denoise_steps < 1:
        raise ValueError("model.denoise_steps must be positive")
    iterative_mask_steps = config["data"].get("iterative_mask_steps")
    if iterative_mask_steps is not None and int(iterative_mask_steps) != model_denoise_steps:
        raise ValueError(
            "data.iterative_mask_steps must equal model.denoise_steps: "
            f"{int(iterative_mask_steps)} != {model_denoise_steps}"
        )
    if float(model_config.get("composition_loss_weight", 0.0)) < 0:
        raise ValueError("model.composition_loss_weight must be non-negative")
    if float(model_config.get("homopolymer_loss_weight", 0.0)) < 0:
        raise ValueError("model.homopolymer_loss_weight must be non-negative")
    if int(model_config.get("homopolymer_max_run", 6)) < 1:
        raise ValueError("model.homopolymer_max_run must be positive")
    if float(model_config.get("lora_lr_multiplier", 1.0)) <= 0:
        raise ValueError("model.lora_lr_multiplier must be positive")
    data_max_length = validate_scaffold_length_limit(
        config["data"].get("max_target_length", 512),
        field="data.max_target_length",
    )
    model_max_length = validate_scaffold_length_limit(
        model_config.get("max_length", 512),
        field="model.max_length",
    )
    if data_max_length > model_max_length:
        raise ValueError(
            f"data.max_target_length must not exceed model.max_length: {data_max_length} > {model_max_length}"
        )
    if not isinstance(config["trainer"].get("args"), dict):
        raise TypeError("training config trainer.args must be a mapping")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RNA motif-conditioned scaffold model.")
    parser.add_argument("--config", default="configs/train_scaffold_5090_linker_v5.yaml")
    parser.add_argument("--resume", help="Resume from an existing Lightning checkpoint.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_training_config(cfg)
    resume = args.resume or cfg["trainer"].get("resume_from_checkpoint")
    if resume:
        validate_scaffold_checkpoint_version(resume)
    L.seed_everything(cfg.get("seed", 42), workers=True)

    tokenizer = RnaTokenizer()
    data = RnaScaffoldDataModule(tokenizer=tokenizer, **cfg["data"])
    model_config = dict(cfg["model"])
    model_config.setdefault("mask_token_id", tokenizer.token_to_id[tokenizer.special.mask])
    model = RnaScaffoldLitModule(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        **model_config,
    )

    monitor_metric = str(cfg["trainer"].get("monitor", "val/loss"))
    checkpoint = ModelCheckpoint(
        dirpath=cfg["trainer"].get("checkpoint_dir", "checkpoints"),
        filename=cfg["trainer"].get("checkpoint_filename", "rna-scaffold-{epoch:02d}-{val/loss:.4f}"),
        monitor=monitor_metric,
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        mode="min",
        patience=int(cfg["trainer"].get("early_stopping_patience", 12)),
        check_finite=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = WandbLogger(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"].get("name"),
        entity=cfg["wandb"].get("entity"),
        log_model=cfg["wandb"].get("log_model", False),
        config=cfg,
    )

    trainer = L.Trainer(
        logger=[logger, CSVLogger(save_dir=cfg["trainer"].get("checkpoint_dir", "checkpoints"), name="metrics")],
        callbacks=[checkpoint, early_stopping, lr_monitor],
        **cfg["trainer"]["args"],
    )
    trainer.fit(model, datamodule=data, ckpt_path=resume)
    write_training_manifest(
        args.config,
        cfg,
        checkpoint.best_model_path,
        pretrained_identity=model.pretrained_metadata,
        split_manifest=data.split_manifest,
        data_audit=data.denoising_data_audit,
    )


if __name__ == "__main__":
    main()
