from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from rna_scaffold.geometry import validate_scaffold_length_limit
from rna_scaffold.pretrained import TrustedRnaFmCheckpoint
from rna_scaffold.tokenizer import RnaTokenizer

if TYPE_CHECKING:
    from rna_scaffold.lightning_module import RnaScaffoldLitModule


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a checkpoint cannot safely reconstruct this generator."""


@dataclass(frozen=True)
class LoadedScaffoldModel:
    model: RnaScaffoldLitModule
    tokenizer: RnaTokenizer
    checkpoint_sha256: str
    max_length: int
    architecture_version: int
    pretrained_metadata: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_scaffold_checkpoint_payload(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError("checkpoint root must be a mapping")
    architecture_version = payload.get("architecture_version")
    if type(architecture_version) is not int or architecture_version != 2:
        raise CheckpointCompatibilityError(
            "checkpoint architecture_version 2 is required; the model must be retrained for V2"
        )
    hparams = payload.get("hyper_parameters")
    if isinstance(hparams, dict) and "max_length" in hparams:
        try:
            validate_scaffold_length_limit(
                hparams["max_length"],
                field="checkpoint hyper_parameters.max_length",
            )
        except (TypeError, ValueError) as error:
            raise CheckpointCompatibilityError(str(error)) from error
    return payload


def _read_checkpoint_payload(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[Path, dict]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"scaffold checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    return checkpoint_path, validate_scaffold_checkpoint_payload(payload)


def validate_scaffold_checkpoint_version(path: str | Path) -> None:
    _read_checkpoint_payload(path, device="cpu")


def load_scaffold_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    *,
    trusted_rna_fm_checkpoint: TrustedRnaFmCheckpoint | None = None,
) -> LoadedScaffoldModel:
    from rna_scaffold.lightning_module import RnaScaffoldLitModule
    from rna_scaffold.pretrained import authorize_scaffold_rna_fm_config

    checkpoint_path, payload = _read_checkpoint_payload(path, device=device)
    hparams = payload.get("hyper_parameters")
    state_dict = payload.get("state_dict")
    if not isinstance(hparams, dict):
        raise CheckpointCompatibilityError("checkpoint is missing hyper_parameters")
    if not isinstance(state_dict, dict):
        raise CheckpointCompatibilityError("checkpoint is missing state_dict")

    tokenizer = RnaTokenizer()
    constructor = dict(hparams)
    constructor.setdefault("vocab_size", tokenizer.vocab_size)
    constructor.setdefault("pad_token_id", tokenizer.pad_token_id)
    raw_metadata = payload.get("pretrained_encoder")
    configured = constructor.get("pretrained")
    if isinstance(configured, dict):
        try:
            constructor["pretrained"] = authorize_scaffold_rna_fm_config(
                configured,
                raw_metadata if isinstance(raw_metadata, dict) else None,
                trusted_checkpoint=trusted_rna_fm_checkpoint,
            )
        except (TypeError, ValueError) as error:
            raise CheckpointCompatibilityError(
                f"checkpoint RNA-FM trust policy rejected the pretrained identity: {error}"
            ) from error
    elif trusted_rna_fm_checkpoint is not None:
        raise CheckpointCompatibilityError(
            "a trusted RNA-FM checkpoint was supplied, but the scaffold checkpoint "
            "does not declare an RNA-FM pretrained encoder"
        )
    try:
        model = RnaScaffoldLitModule(**constructor)
    except (TypeError, ValueError, RuntimeError) as error:
        raise CheckpointCompatibilityError(f"checkpoint hyper_parameters are incompatible: {error}") from error

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise CheckpointCompatibilityError(f"checkpoint state_dict is incompatible: {error}") from error
    model.to(device)
    model.eval()
    from rna_scaffold.pretrained import LoRALinear

    configured = constructor.get("pretrained")
    configured = dict(configured) if isinstance(configured, dict) else {"kind": "none"}
    configured_kind = str(configured.get("kind", "none"))
    configured_mode = "none" if configured_kind == "none" else str(configured.get("mode", "frozen"))
    encoder = model.model.pretrained_encoder
    adapters = (
        tuple(name for name, module in encoder.named_modules() if isinstance(module, LoRALinear))
        if encoder is not None
        else ()
    )
    structure_mismatch = (
        (configured_kind == "none" and encoder is not None)
        or (configured_kind == "rna_fm" and encoder is None)
        or (configured_mode == "lora" and not adapters)
        or (configured_mode != "lora" and bool(adapters))
    )
    if structure_mismatch:
        raise CheckpointCompatibilityError(
            "checkpoint pretrained identity does not match the reconstructed model structure"
        )
    if isinstance(raw_metadata, dict):
        metadata_kind = str(raw_metadata.get("kind", "none"))
        metadata_mode = "none" if metadata_kind == "none" else str(raw_metadata.get("mode", "frozen"))
        if (metadata_kind, metadata_mode) != (configured_kind, configured_mode):
            raise CheckpointCompatibilityError(
                "checkpoint pretrained identity mismatch between metadata and hyper_parameters"
            )
    pretrained = dict(model.pretrained_metadata)
    pretrained["kind"] = configured_kind
    pretrained["mode"] = configured_mode
    if configured_mode == "lora":
        pretrained["adapter_targets"] = list(adapters)
    return LoadedScaffoldModel(
        model=model,
        tokenizer=tokenizer,
        checkpoint_sha256=_sha256(checkpoint_path),
        max_length=model.model.max_length,
        architecture_version=2,
        pretrained_metadata=pretrained,
    )
