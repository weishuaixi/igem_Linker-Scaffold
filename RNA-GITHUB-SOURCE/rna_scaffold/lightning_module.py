from __future__ import annotations

import math

import torch
from torch import nn
from torchmetrics import Metric

try:
    import lightning.pytorch as L
except ImportError:  # pragma: no cover
    import pytorch_lightning as L

from rna_scaffold.checkpoints import validate_scaffold_checkpoint_payload
from rna_scaffold.model import MotifDenoisingTransformer, ScaffoldModelOutput, compute_denoising_losses
from rna_scaffold.pretrained import (
    build_pretrained_encoder,
    pretrained_metadata,
)

_BASE_NAMES = ("A", "U", "C", "G")


class _TokenDiagnostics(Metric):
    """Distributed-safe epoch diagnostics over supervised scaffold tokens."""

    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "confusion",
            default=torch.zeros(4, 4, dtype=torch.long),
            dist_reduce_fx="sum",
            persistent=False,
        )
        self.add_state(
            "entropy_sum",
            default=torch.zeros((), dtype=torch.float64),
            dist_reduce_fx="sum",
            persistent=False,
        )
        self.add_state(
            "raw_nll_sum",
            default=torch.zeros((), dtype=torch.float64),
            dist_reduce_fx="sum",
            persistent=False,
        )
        self.add_state(
            "predicted_count",
            default=torch.zeros((), dtype=torch.long),
            dist_reduce_fx="sum",
            persistent=False,
        )
        self.add_state(
            "mutable_count",
            default=torch.zeros((), dtype=torch.long),
            dist_reduce_fx="sum",
            persistent=False,
        )

    def update(
        self,
        token_logits: torch.Tensor,
        target_base_ids: torch.Tensor,
        scaffold_mask: torch.Tensor,
        mutable_mask: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            selected_logits = token_logits.detach()[scaffold_mask].float()
            selected_targets = target_base_ids.detach()[scaffold_mask].long()
            if selected_targets.numel():
                predictions = selected_logits.argmax(dim=-1)
                cells = torch.bincount(selected_targets * 4 + predictions, minlength=16).reshape(4, 4)
                probabilities = selected_logits.softmax(dim=-1)
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
                raw_nll = nn.functional.cross_entropy(
                    selected_logits,
                    selected_targets,
                    reduction="sum",
                )
                self.confusion.add_(cells.to(self.confusion.device))
                self.entropy_sum.add_(entropy.double().sum().to(self.entropy_sum.device))
                self.raw_nll_sum.add_(raw_nll.double().to(self.raw_nll_sum.device))
                self.predicted_count.add_(selected_targets.numel())
            self.mutable_count.add_(mutable_mask.detach().sum().to(self.mutable_count.device))

    def compute(self) -> dict[str, torch.Tensor]:
        total = self.confusion.sum()
        safe_total = total.clamp_min(1).to(torch.float64)
        correct = self.confusion.diagonal().sum()
        target_counts = self.confusion.sum(dim=1)
        predicted_counts = self.confusion.sum(dim=0)
        recalls = self.confusion.diagonal().to(torch.float64) / target_counts.clamp_min(1).to(torch.float64)
        present = target_counts > 0
        macro_recall = recalls[present].mean() if bool(present.any().item()) else safe_total.new_zeros(())
        raw_base_nll = self.raw_nll_sum / self.predicted_count.clamp_min(1).to(torch.float64)
        metrics = {
            "token_accuracy": correct.to(torch.float64) / safe_total,
            "macro_recall": macro_recall,
            "raw_base_nll": raw_base_nll,
            "perplexity": raw_base_nll.clamp(max=20).exp(),
            "pred_entropy": self.entropy_sum / self.predicted_count.clamp_min(1).to(torch.float64),
            "mask_fraction": self.predicted_count.to(torch.float64) / self.mutable_count.clamp_min(1).to(torch.float64),
        }
        for index, base in enumerate(_BASE_NAMES):
            metrics[f"recall_{base}"] = recalls[index]
            metrics[f"pred_fraction_{base}"] = predicted_counts[index].to(torch.float64) / safe_total
            metrics[f"target_fraction_{base}"] = target_counts[index].to(torch.float64) / safe_total
        return metrics


def warmup_cosine_multiplier(
    step: int,
    total_steps: int,
    warmup_fraction: float = 0.05,
    min_fraction: float = 0.02,
) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if not 0 <= min_fraction <= 1:
        raise ValueError("min_fraction must be in [0, 1]")
    warmup_steps = max(1, round(total_steps * warmup_fraction))
    bounded_step = min(max(int(step), 0), total_steps)
    if bounded_step <= warmup_steps:
        return bounded_step / warmup_steps
    progress = (bounded_step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_fraction + (1.0 - min_fraction) * cosine


def compact_fixed_motifs(
    input_ids: torch.Tensor,
    fixed_mask: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack immutable motif tokens so placement heads cannot read target geometry."""
    if input_ids.shape != fixed_mask.shape or input_ids.ndim != 2:
        raise ValueError("input_ids and fixed_mask must have equal [batch, length] shapes")
    lengths = fixed_mask.long().sum(dim=1)
    if torch.any(lengths <= 0):
        raise ValueError("every item must contain a non-empty fixed motif")
    maximum = int(lengths.max().item())
    motifs = torch.full(
        (input_ids.shape[0], maximum),
        int(pad_token_id),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    attention = torch.zeros_like(motifs, dtype=torch.bool)
    for batch_index in range(input_ids.shape[0]):
        motif = input_ids[batch_index, fixed_mask[batch_index].bool()]
        motifs[batch_index, : motif.numel()] = motif
        attention[batch_index, : motif.numel()] = True
    return motifs, attention


class RnaScaffoldLitModule(L.LightningModule):
    """Lightning training wrapper for motif-protected scaffold denoising."""

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_length: int = 512,
        activation_checkpointing: bool = False,
        pretrained: dict | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        left_length_loss_weight: float = 0.25,
        right_length_loss_weight: float = 0.25,
        label_smoothing: float = 0.05,
        normalize_base_loss_per_sequence: bool = False,
        warmup_fraction: float = 0.05,
        min_lr_fraction: float = 0.02,
        use_conditioning_embeddings: bool = False,
        conditioning_relative_distance_clip: int = 32,
        pretrained_fusion: str = "add",
        length_pooling: str = "mean",
        predict_flank_lengths: bool = True,
        self_conditioning_probability: float = 0.0,
        self_conditioning_validation: bool = True,
        denoise_steps: int = 16,
        composition_loss_weight: float = 0.0,
        homopolymer_loss_weight: float = 0.0,
        homopolymer_max_run: int = 6,
        lora_lr_multiplier: float = 1.0,
        exclude_norm_bias_embedding_from_weight_decay: bool = False,
        mask_token_id: int = 3,
        generation_remask_strategy: str = "confidence",
    ) -> None:
        super().__init__()
        if generation_remask_strategy not in {"confidence", "random"}:
            raise ValueError("generation_remask_strategy must be confidence or random")
        if not 0 <= self_conditioning_probability <= 1:
            raise ValueError("self_conditioning_probability must be in [0, 1]")
        if self_conditioning_probability > 0 and not use_conditioning_embeddings:
            raise ValueError("self-conditioning requires use_conditioning_embeddings=True")
        if denoise_steps < 1:
            raise ValueError("denoise_steps must be positive")
        if not predict_flank_lengths and (left_length_loss_weight > 0 or right_length_loss_weight > 0):
            raise ValueError("length loss weights must be zero when predict_flank_lengths=False")
        if composition_loss_weight < 0 or homopolymer_loss_weight < 0:
            raise ValueError("regularization loss weights must be non-negative")
        if homopolymer_max_run < 1:
            raise ValueError("homopolymer_max_run must be positive")
        if lora_lr_multiplier <= 0:
            raise ValueError("lora_lr_multiplier must be positive")
        self.save_hyperparameters()
        self.pad_token_id = int(pad_token_id)
        self.generation_remask_strategy = generation_remask_strategy
        self.lr = lr
        self.weight_decay = weight_decay
        self.left_length_loss_weight = left_length_loss_weight
        self.right_length_loss_weight = right_length_loss_weight
        self.label_smoothing = label_smoothing
        self.normalize_base_loss_per_sequence = normalize_base_loss_per_sequence
        self.warmup_fraction = warmup_fraction
        self.min_lr_fraction = min_lr_fraction
        self.use_conditioning_embeddings = bool(use_conditioning_embeddings)
        self.predict_flank_lengths = bool(predict_flank_lengths)
        self.self_conditioning_probability = float(self_conditioning_probability)
        self.self_conditioning_validation = bool(self_conditioning_validation)
        self.denoise_steps = int(denoise_steps)
        self.composition_loss_weight = float(composition_loss_weight)
        self.homopolymer_loss_weight = float(homopolymer_loss_weight)
        self.homopolymer_max_run = int(homopolymer_max_run)
        self.lora_lr_multiplier = float(lora_lr_multiplier)
        self.exclude_norm_bias_embedding_from_weight_decay = bool(exclude_norm_bias_embedding_from_weight_decay)
        pretrained_encoder = build_pretrained_encoder(pretrained)
        self.pretrained_metadata = pretrained_metadata(pretrained, pretrained_encoder)
        self.model = MotifDenoisingTransformer(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_length=max_length,
            activation_checkpointing=activation_checkpointing,
            pretrained_encoder=pretrained_encoder,
            use_conditioning_embeddings=use_conditioning_embeddings,
            conditioning_relative_distance_clip=conditioning_relative_distance_clip,
            pretrained_fusion=pretrained_fusion,
            length_pooling=length_pooling,
            predict_flank_lengths=predict_flank_lengths,
            mask_token_id=mask_token_id,
        )
        self.train_token_diagnostics = _TokenDiagnostics()
        self.val_token_diagnostics = _TokenDiagnostics()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        prediction_mask: torch.Tensor | None = None,
        denoise_step: torch.Tensor | float | None = None,
        self_condition_probs: torch.Tensor | None = None,
    ) -> ScaffoldModelOutput:
        if fixed_mask is None and prediction_mask is None and denoise_step is None and self_condition_probs is None:
            return self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            fixed_mask=fixed_mask,
            prediction_mask=prediction_mask,
            denoise_step=denoise_step,
            self_condition_probs=self_condition_probs,
        )

    def _normalized_denoise_step(
        self,
        batch: dict[str, torch.Tensor],
        mutable_positions: torch.Tensor,
        prediction_mask: torch.Tensor,
    ) -> torch.Tensor:
        provided = batch.get("noise_level")
        already_normalized = provided is not None
        if provided is None:
            provided = batch.get("denoise_step")
        if provided is None:
            mutable_counts = mutable_positions.sum(dim=1).clamp_min(1)
            return prediction_mask.sum(dim=1).float() / mutable_counts.float()
        step = torch.as_tensor(provided, device=mutable_positions.device)
        if step.ndim == 0:
            step = step.expand(mutable_positions.shape[0])
        elif step.ndim == 2 and step.shape[1] == 1:
            step = step.squeeze(1)
        if step.shape != (mutable_positions.shape[0],):
            raise ValueError("batch denoise_step must be scalar or have shape [batch]")
        was_integer = not torch.is_floating_point(step)
        step = step.float()
        if not already_normalized and (was_integer or bool((step > 1).any().item())):
            step = step / self.denoise_steps
        if not bool(torch.isfinite(step).all().item()) or not bool(((step >= 0) & (step <= 1)).all().item()):
            raise ValueError("normalized denoise_step must be in [0, 1]")
        return step

    def _conditioned_scaffold_forward(
        self,
        batch: dict[str, torch.Tensor],
        prediction_mask: torch.Tensor,
        denoise_step: torch.Tensor,
        self_condition_probs: torch.Tensor | None = None,
    ) -> ScaffoldModelOutput:
        if not self.use_conditioning_embeddings:
            return self(batch["input_ids"], batch["attention_mask"])
        return self(
            batch["input_ids"],
            batch["attention_mask"],
            fixed_mask=batch["fixed_mask"],
            prediction_mask=prediction_mask,
            denoise_step=denoise_step,
            self_condition_probs=self_condition_probs,
        )

    def _uses_self_conditioning(self, stage: str) -> bool:
        if not self.use_conditioning_embeddings:
            return False
        if stage == "train":
            if self.self_conditioning_probability <= 0:
                return False
            return bool(torch.rand((), device=self.device).lt(self.self_conditioning_probability).item())
        return self.self_conditioning_validation

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> dict[str, torch.Tensor]:
        mutable_positions = batch["attention_mask"].bool() & ~batch["fixed_mask"].bool()
        supplied_prediction_mask = batch.get("prediction_mask")
        prediction_mask = (
            mutable_positions
            if supplied_prediction_mask is None
            else supplied_prediction_mask.bool() & mutable_positions
        )
        denoise_step = self._normalized_denoise_step(batch, mutable_positions, prediction_mask)
        self_conditioned_scaffold_output: ScaffoldModelOutput | None = None
        if self._uses_self_conditioning(stage):
            with torch.no_grad():
                draft_output = self._conditioned_scaffold_forward(batch, prediction_mask, denoise_step)
                self_condition_probs = draft_output.token_logits.float().softmax(dim=-1)
                self_condition_probs = self_condition_probs * mutable_positions.unsqueeze(-1)
            self_conditioned_scaffold_output = self._conditioned_scaffold_forward(
                batch,
                prediction_mask,
                denoise_step,
                self_condition_probs=self_condition_probs,
            )
            # Training optimizes the self-conditioned pass.  Validation keeps
            # the first pass as the primary metric because that is the output
            # used to begin an actual full-mask rollout; the second pass is
            # logged separately as an auxiliary diagnostic.
            scaffold_output = draft_output if stage == "val" else self_conditioned_scaffold_output
        else:
            scaffold_output = self._conditioned_scaffold_forward(batch, prediction_mask, denoise_step)

        length_losses_active = self.left_length_loss_weight > 0 or self.right_length_loss_weight > 0
        if length_losses_active:
            motif_ids, motif_attention = compact_fixed_motifs(
                batch["input_ids"],
                batch["fixed_mask"],
                self.pad_token_id,
            )
            if self.use_conditioning_embeddings:
                motif_fixed = motif_attention.clone()
                motif_prediction = torch.zeros_like(motif_attention)
                placement_output = self(
                    motif_ids,
                    motif_attention,
                    fixed_mask=motif_fixed,
                    prediction_mask=motif_prediction,
                    denoise_step=torch.zeros(motif_ids.shape[0], device=motif_ids.device),
                )
            else:
                placement_output = self(motif_ids, motif_attention)
            output = ScaffoldModelOutput(
                token_logits=scaffold_output.token_logits,
                left_length_logits=placement_output.left_length_logits,
                right_length_logits=placement_output.right_length_logits,
            )
        else:
            output = scaffold_output
        losses = compute_denoising_losses(
            output=output,
            target_base_ids=batch["target_base_ids"],
            fixed_mask=batch["fixed_mask"],
            attention_mask=batch["attention_mask"],
            target_left_length=batch["target_left_length"],
            target_right_length=batch["target_right_length"],
            prediction_mask=batch.get("prediction_mask"),
            left_length_loss_weight=self.left_length_loss_weight,
            right_length_loss_weight=self.right_length_loss_weight,
            label_smoothing=self.label_smoothing,
            normalize_base_loss_per_sequence=self.normalize_base_loss_per_sequence,
            composition_loss_weight=self.composition_loss_weight,
            homopolymer_loss_weight=self.homopolymer_loss_weight,
            homopolymer_max_run=self.homopolymer_max_run,
        )
        scaffold_mask = prediction_mask
        token_accuracy = (
            (output.token_logits.argmax(dim=-1)[scaffold_mask] == batch["target_base_ids"][scaffold_mask])
            .float()
            .mean()
        )
        zero = losses.base_loss.new_zeros(())
        left_length_accuracy = (
            (output.left_length_logits.argmax(dim=-1) == batch["target_left_length"]).float().mean()
            if self.left_length_loss_weight > 0
            else zero
        )
        right_length_accuracy = (
            (output.right_length_logits.argmax(dim=-1) == batch["target_right_length"]).float().mean()
            if self.right_length_loss_weight > 0
            else zero
        )
        diagnostics = self.train_token_diagnostics if stage == "train" else self.val_token_diagnostics
        diagnostics.update(output.token_logits, batch["target_base_ids"], scaffold_mask, mutable_positions)
        values = {
            "loss": losses.total_loss,
            "base_loss": losses.base_loss,
            "raw_base_nll": losses.raw_base_nll,
            "perplexity": losses.raw_base_nll.float().clamp(max=20).exp(),
            "left_length_loss": losses.left_length_loss,
            "right_length_loss": losses.right_length_loss,
            "composition_loss": losses.composition_loss,
            "homopolymer_loss": losses.homopolymer_loss,
            "weighted_composition_loss": self.composition_loss_weight * losses.composition_loss,
            "weighted_homopolymer_loss": self.homopolymer_loss_weight * losses.homopolymer_loss,
            "token_accuracy": token_accuracy,
            "left_length_accuracy": left_length_accuracy,
            "right_length_accuracy": right_length_accuracy,
            "mask_fraction": scaffold_mask.sum().float() / mutable_positions.sum().clamp_min(1).float(),
        }

        if stage == "val" and self_conditioned_scaffold_output is not None:
            auxiliary_output = ScaffoldModelOutput(
                token_logits=self_conditioned_scaffold_output.token_logits,
                left_length_logits=output.left_length_logits,
                right_length_logits=output.right_length_logits,
            )
            auxiliary_losses = compute_denoising_losses(
                output=auxiliary_output,
                target_base_ids=batch["target_base_ids"],
                fixed_mask=batch["fixed_mask"],
                attention_mask=batch["attention_mask"],
                target_left_length=batch["target_left_length"],
                target_right_length=batch["target_right_length"],
                prediction_mask=batch.get("prediction_mask"),
                left_length_loss_weight=self.left_length_loss_weight,
                right_length_loss_weight=self.right_length_loss_weight,
                label_smoothing=self.label_smoothing,
                normalize_base_loss_per_sequence=self.normalize_base_loss_per_sequence,
                composition_loss_weight=self.composition_loss_weight,
                homopolymer_loss_weight=self.homopolymer_loss_weight,
                homopolymer_max_run=self.homopolymer_max_run,
            )
            auxiliary_accuracy = (
                (auxiliary_output.token_logits.argmax(dim=-1)[scaffold_mask] == batch["target_base_ids"][scaffold_mask])
                .float()
                .mean()
            )
            auxiliary_values = {
                "loss": auxiliary_losses.total_loss,
                "base_loss": auxiliary_losses.base_loss,
                "raw_base_nll": auxiliary_losses.raw_base_nll,
                "token_accuracy": auxiliary_accuracy,
            }
            for name, value in auxiliary_values.items():
                self.log(
                    f"val_self_conditioned/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=int(batch["input_ids"].shape[0]),
                )
            values.update({f"self_conditioned_{name}": value for name, value in auxiliary_values.items()})

        inactive_length_metrics = (
            {
                "left_length_loss",
                "right_length_loss",
                "left_length_accuracy",
                "right_length_accuracy",
            }
            if not length_losses_active
            else set()
        )
        for name, value in values.items():
            if name.startswith("self_conditioned_"):
                continue
            if name in inactive_length_metrics:
                continue
            if name in {"token_accuracy", "mask_fraction", "raw_base_nll", "perplexity"}:
                continue
            self.log(
                f"{stage}/{name}",
                value,
                prog_bar=name == "loss",
                sync_dist=True,
                batch_size=int(batch["input_ids"].shape[0]),
            )
        if stage == "train":
            self.log(
                "train/token_accuracy_step",
                token_accuracy,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
                batch_size=int(scaffold_mask.sum().item()),
            )
        return values

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        return self._step(batch, "val")

    def on_train_epoch_start(self) -> None:
        self.train_token_diagnostics.reset()
        datamodule = getattr(self.trainer, "datamodule", None)
        dataset = getattr(datamodule, "train_dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        self.val_token_diagnostics.reset()

    def _log_epoch_diagnostics(self, stage: str, diagnostics: _TokenDiagnostics) -> None:
        computed = diagnostics.compute()
        for name, value in computed.items():
            self.log(
                f"{stage}/{name}",
                value.float(),
                on_step=False,
                on_epoch=True,
                prog_bar=name == "token_accuracy",
                sync_dist=False,
            )

    def on_train_epoch_end(self) -> None:
        self._log_epoch_diagnostics("train", self.train_token_diagnostics)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_diagnostics("val", self.val_token_diagnostics)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["architecture_version"] = 2
        checkpoint["pretrained_encoder"] = self.pretrained_metadata

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        validate_scaffold_checkpoint_payload(checkpoint)

    def configure_optimizers(self):
        trainable = [(name, parameter) for name, parameter in self.named_parameters() if parameter.requires_grad]
        if self.lora_lr_multiplier == 1.0 and not self.exclude_norm_bias_embedding_from_weight_decay:
            optimizer_parameters = (parameter for _, parameter in trainable)
        else:
            embedding_parameters = {
                id(parameter)
                for module in self.modules()
                if isinstance(module, nn.Embedding)
                for parameter in module.parameters(recurse=False)
            }
            grouped: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
            for name, parameter in trainable:
                is_lora = ".lora_A." in name or ".lora_B." in name
                learning_rate = self.lr * self.lora_lr_multiplier if is_lora else self.lr
                no_decay = self.exclude_norm_bias_embedding_from_weight_decay and (
                    parameter.ndim <= 1 or name.endswith(".bias") or id(parameter) in embedding_parameters
                )
                group_key = (learning_rate, 0.0 if no_decay else self.weight_decay)
                grouped.setdefault(group_key, []).append(parameter)
            optimizer_parameters = [
                {"params": parameters, "lr": learning_rate, "weight_decay": group_weight_decay}
                for (learning_rate, group_weight_decay), parameters in grouped.items()
            ]
        optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_cosine_multiplier(
                step,
                total_steps=total_steps,
                warmup_fraction=self.warmup_fraction,
                min_fraction=self.min_lr_fraction,
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
