from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rna_scaffold.geometry import validate_scaffold_length_limit
from rna_scaffold.pretrained import set_lora_adapter_train_mode


@dataclass(frozen=True)
class ScaffoldModelOutput:
    token_logits: torch.Tensor
    left_length_logits: torch.Tensor | None
    right_length_logits: torch.Tensor | None


@dataclass(frozen=True)
class DenoisingLosses:
    total_loss: torch.Tensor
    base_loss: torch.Tensor
    left_length_loss: torch.Tensor
    right_length_loss: torch.Tensor
    raw_base_nll: torch.Tensor
    composition_loss: torch.Tensor
    homopolymer_loss: torch.Tensor


def _masked_token_loss(
    token_losses: torch.Tensor,
    scaffold_mask: torch.Tensor,
    *,
    normalize_per_sequence: bool,
) -> torch.Tensor:
    weights = scaffold_mask.to(token_losses.dtype)
    if normalize_per_sequence:
        masked_counts = weights.sum(dim=1)
        if not bool(masked_counts.gt(0).all().item()):
            raise ValueError("every sample must contain at least one predicted scaffold position")
        return ((token_losses * weights).sum(dim=1) / masked_counts).mean()
    return token_losses[scaffold_mask].mean()


def _composition_js_loss(
    predicted_probabilities: torch.Tensor,
    target_base_ids: torch.Tensor,
    scaffold_mask: torch.Tensor,
) -> torch.Tensor:
    """Match predicted and target base composition without imposing a uniform prior."""
    weights = scaffold_mask.unsqueeze(-1).to(torch.float32)
    counts = weights.sum(dim=1)
    if not bool(counts.gt(0).all().item()):
        raise ValueError("every sample must contain at least one predicted scaffold position")
    safe_targets = target_base_ids.masked_fill(~scaffold_mask, 0)
    targets = F.one_hot(safe_targets, num_classes=4).to(torch.float32)
    predicted_composition = (predicted_probabilities * weights).sum(dim=1) / counts
    target_composition = (targets * weights).sum(dim=1) / counts
    midpoint = 0.5 * (predicted_composition + target_composition)
    epsilon = torch.finfo(midpoint.dtype).eps

    def kl_divergence(distribution: torch.Tensor) -> torch.Tensor:
        terms = torch.where(
            distribution > 0,
            distribution * (distribution.clamp_min(epsilon).log() - midpoint.clamp_min(epsilon).log()),
            torch.zeros_like(distribution),
        )
        return terms.sum(dim=-1)

    return (0.5 * (kl_divergence(predicted_composition) + kl_divergence(target_composition))).mean()


def _expected_homopolymer_loss(
    predicted_probabilities: torch.Tensor,
    target_base_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    scaffold_mask: torch.Tensor,
    fixed_mask: torch.Tensor,
    *,
    max_run: int,
) -> torch.Tensor:
    """Expected probability of a same-base run longer than ``max_run``.

    Non-predicted linker positions use their observed target base. Windows
    containing the fixed motif are excluded, so it separates the left and right
    linkers and never contributes to their run-length penalty. Windows containing
    no supervised position are excluded because they cannot provide a gradient.
    """
    if max_run < 1:
        raise ValueError("homopolymer_max_run must be positive")
    window_length = max_run + 1
    if predicted_probabilities.shape[1] < window_length:
        return predicted_probabilities.new_zeros(())

    safe_targets = target_base_ids.masked_fill(~attention_mask.bool(), 0)
    observed_probabilities = F.one_hot(safe_targets, num_classes=4).to(torch.float32)
    sequence_probabilities = torch.where(
        scaffold_mask.unsqueeze(-1),
        predicted_probabilities,
        observed_probabilities,
    )
    # [B, windows, bases, positions-within-window]
    windows = sequence_probabilities.unfold(1, window_length, 1)
    same_base_probability = windows.prod(dim=-1).sum(dim=-1)
    valid_windows = attention_mask.bool().unfold(1, window_length, 1).all(dim=-1)
    linker_only_windows = ~fixed_mask.bool().unfold(1, window_length, 1).any(dim=-1)
    trainable_windows = scaffold_mask.unfold(1, window_length, 1).any(dim=-1)
    included = valid_windows & linker_only_windows & trainable_windows
    if not bool(included.any().item()):
        return predicted_probabilities.new_zeros(())
    return same_base_probability[included].mean()


def compute_denoising_losses(
    output: ScaffoldModelOutput,
    target_base_ids: torch.Tensor,
    fixed_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    target_left_length: torch.Tensor,
    target_right_length: torch.Tensor,
    prediction_mask: torch.Tensor | None = None,
    left_length_loss_weight: float = 0.25,
    right_length_loss_weight: float = 0.25,
    label_smoothing: float = 0.0,
    normalize_base_loss_per_sequence: bool = False,
    composition_loss_weight: float = 0.0,
    homopolymer_loss_weight: float = 0.0,
    homopolymer_max_run: int = 6,
) -> DenoisingLosses:
    if left_length_loss_weight < 0 or right_length_loss_weight < 0:
        raise ValueError("length loss weights must be non-negative")
    if composition_loss_weight < 0 or homopolymer_loss_weight < 0:
        raise ValueError("regularization loss weights must be non-negative")
    if homopolymer_loss_weight and homopolymer_max_run < 1:
        raise ValueError("homopolymer_max_run must be positive")
    mutable_positions = attention_mask.bool() & ~fixed_mask.bool()
    scaffold_mask = prediction_mask.bool() & mutable_positions if prediction_mask is not None else mutable_positions
    if not scaffold_mask.any():
        raise ValueError("every batch must contain at least one scaffold position")
    safe_targets = target_base_ids.masked_fill(~scaffold_mask, 0)
    log_probabilities = F.log_softmax(output.token_logits.float(), dim=-1)
    raw_token_losses = -log_probabilities.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    raw_base_nll = _masked_token_loss(
        raw_token_losses,
        scaffold_mask,
        normalize_per_sequence=normalize_base_loss_per_sequence,
    )
    if label_smoothing:
        # PyTorch mixes the hard target with a uniform distribution over all
        # classes: (1 - epsilon) * NLL + epsilon * mean(-log p_class).
        uniform_token_losses = -log_probabilities.mean(dim=-1)
        smoothed_token_losses = (1.0 - label_smoothing) * raw_token_losses + label_smoothing * uniform_token_losses
        base_loss = _masked_token_loss(
            smoothed_token_losses,
            scaffold_mask,
            normalize_per_sequence=normalize_base_loss_per_sequence,
        )
    else:
        base_loss = raw_base_nll

    zero = base_loss.new_zeros(())
    if left_length_loss_weight > 0 and output.left_length_logits is None:
        raise ValueError("left_length_logits are required when left_length_loss_weight is positive")
    if right_length_loss_weight > 0 and output.right_length_logits is None:
        raise ValueError("right_length_logits are required when right_length_loss_weight is positive")
    left_length_loss = (
        F.cross_entropy(output.left_length_logits, target_left_length.long()) if left_length_loss_weight > 0 else zero
    )
    right_length_loss = (
        F.cross_entropy(output.right_length_logits, target_right_length.long())
        if right_length_loss_weight > 0
        else zero
    )
    predicted_probabilities = (
        log_probabilities.exp() if composition_loss_weight > 0 or homopolymer_loss_weight > 0 else None
    )
    composition_loss = (
        _composition_js_loss(predicted_probabilities, target_base_ids, scaffold_mask)
        if composition_loss_weight > 0
        else zero
    )
    homopolymer_loss = (
        _expected_homopolymer_loss(
            predicted_probabilities,
            target_base_ids,
            attention_mask,
            scaffold_mask,
            fixed_mask,
            max_run=homopolymer_max_run,
        )
        if homopolymer_loss_weight > 0
        else zero
    )
    total_loss = (
        base_loss
        + left_length_loss_weight * left_length_loss
        + right_length_loss_weight * right_length_loss
        + composition_loss_weight * composition_loss
        + homopolymer_loss_weight * homopolymer_loss
    )
    return DenoisingLosses(
        total_loss,
        base_loss,
        left_length_loss,
        right_length_loss,
        raw_base_nll,
        composition_loss,
        homopolymer_loss,
    )


def restore_fixed_tokens(
    proposed_ids: torch.Tensor,
    original_ids: torch.Tensor,
    fixed_mask: torch.Tensor,
) -> torch.Tensor:
    if proposed_ids.shape != original_ids.shape or proposed_ids.shape != fixed_mask.shape:
        raise ValueError("proposed_ids, original_ids, and fixed_mask must have identical shapes")
    return torch.where(fixed_mask.bool(), original_ids, proposed_ids)


class MotifDenoisingTransformer(nn.Module):
    """Bidirectional RNA scaffold model with token and bilateral flank-length heads."""

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
        pretrained_encoder: nn.Module | None = None,
        use_conditioning_embeddings: bool = False,
        conditioning_relative_distance_clip: int = 32,
        pretrained_fusion: str = "add",
        length_pooling: str = "mean",
        predict_flank_lengths: bool = True,
        mask_token_id: int = 3,
    ) -> None:
        super().__init__()
        if conditioning_relative_distance_clip < 1:
            raise ValueError("conditioning_relative_distance_clip must be positive")
        if pretrained_fusion not in {"add", "gated"}:
            raise ValueError("pretrained_fusion must be 'add' or 'gated'")
        if length_pooling not in {"mean", "attention"}:
            raise ValueError("length_pooling must be 'mean' or 'attention'")
        if pretrained_fusion == "gated" and pretrained_encoder is None:
            raise ValueError("gated pretrained fusion requires a pretrained encoder")
        self.max_length = validate_scaffold_length_limit(max_length)
        self.pad_token_id = pad_token_id
        self.mask_token_id = int(mask_token_id)
        self.activation_checkpointing = activation_checkpointing
        self.pretrained_encoder = pretrained_encoder
        self.use_conditioning_embeddings = bool(use_conditioning_embeddings)
        self.conditioning_relative_distance_clip = int(conditioning_relative_distance_clip)
        self.pretrained_fusion = pretrained_fusion
        self.length_pooling = length_pooling
        self.predict_flank_lengths = bool(predict_flank_lengths)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(self.max_length, d_model)
        self.pretrained_projection = (
            nn.Linear(int(pretrained_encoder.output_dim), d_model, bias=False)
            if pretrained_encoder is not None
            else None
        )
        if pretrained_encoder is not None and pretrained_fusion == "gated":
            self.pretrained_norm = nn.LayerNorm(d_model)
            self.pretrained_gate = nn.Parameter(torch.zeros(d_model))
        else:
            self.pretrained_norm = None
            self.register_parameter("pretrained_gate", None)
        if self.use_conditioning_embeddings:
            self.role_embedding = nn.Embedding(3, d_model)
            self.side_embedding = nn.Embedding(3, d_model)
            self.relative_motif_embedding = nn.Embedding(2 * self.conditioning_relative_distance_clip + 1, d_model)
            self.noise_level_projection = nn.Sequential(
                nn.Linear(1, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            self.self_condition_projection = nn.Linear(4, d_model, bias=False)
        else:
            self.role_embedding = None
            self.side_embedding = None
            self.relative_motif_embedding = None
            self.noise_level_projection = None
            self.self_condition_projection = None
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)
        self.token_head = nn.Linear(d_model, 4)
        if self.predict_flank_lengths:
            self.left_length_head = nn.Linear(d_model, self.max_length + 1)
            self.right_length_head = nn.Linear(d_model, self.max_length + 1)
        else:
            self.left_length_head = None
            self.right_length_head = None
        if self.predict_flank_lengths and length_pooling == "attention":
            self.left_pool_query = nn.Parameter(torch.empty(d_model))
            self.right_pool_query = nn.Parameter(torch.empty(d_model))
            nn.init.normal_(self.left_pool_query, std=d_model**-0.5)
            nn.init.normal_(self.right_pool_query, std=d_model**-0.5)
        else:
            self.register_parameter("left_pool_query", None)
            self.register_parameter("right_pool_query", None)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.pretrained_encoder is not None and getattr(self.pretrained_encoder, "_rna_scaffold_frozen", False):
            self.pretrained_encoder.eval()
        elif self.pretrained_encoder is not None and getattr(self.pretrained_encoder, "_rna_scaffold_lora", False):
            set_lora_adapter_train_mode(self.pretrained_encoder, mode)
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        prediction_mask: torch.Tensor | None = None,
        denoise_step: torch.Tensor | float | None = None,
        self_condition_probs: torch.Tensor | None = None,
    ) -> ScaffoldModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, length]")
        batch_size, length = input_ids.shape
        if length > self.max_length:
            raise ValueError(f"input length {length} exceeds max_length {self.max_length}")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        attention_mask = attention_mask.bool()
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        if self.use_conditioning_embeddings:
            hidden = hidden + self._conditioning_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                fixed_mask=fixed_mask,
                prediction_mask=prediction_mask,
                denoise_step=denoise_step,
                self_condition_probs=self_condition_probs,
                dtype=hidden.dtype,
            )
        elif self_condition_probs is not None:
            raise ValueError("self_condition_probs requires use_conditioning_embeddings=True")
        if self.pretrained_encoder is not None:
            pretrained_features = self.pretrained_encoder(input_ids, attention_mask)
            projected_features = self.pretrained_projection(pretrained_features)
            if self.pretrained_fusion == "gated":
                gate = self.pretrained_gate.sigmoid().to(projected_features.dtype)
                hidden = hidden + gate * self.pretrained_norm(projected_features)
            else:
                hidden = hidden + projected_features
        padding_mask = ~attention_mask
        if self.activation_checkpointing and self.training:
            for layer in self.encoder.layers:
                hidden = checkpoint(
                    layer,
                    hidden,
                    src_key_padding_mask=padding_mask,
                    use_reentrant=False,
                )
            if self.encoder.norm is not None:
                hidden = self.encoder.norm(hidden)
        else:
            hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.final_norm(hidden)
        left_length_logits = None
        right_length_logits = None
        if self.predict_flank_lengths:
            left_pooled, right_pooled = self._pool_for_lengths(hidden, attention_mask)
            left_length_logits = self.left_length_head(left_pooled)
            right_length_logits = self.right_length_head(right_pooled)
        return ScaffoldModelOutput(
            token_logits=self.token_head(hidden),
            left_length_logits=left_length_logits,
            right_length_logits=right_length_logits,
        )

    def _pool_for_lengths(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.length_pooling == "mean":
            weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            return pooled, pooled

        scale = hidden.shape[-1] ** -0.5

        def attention_pool(query: torch.Tensor) -> torch.Tensor:
            scores = torch.einsum("bld,d->bl", hidden, query.to(hidden.dtype)) * scale
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
            weights = scores.float().softmax(dim=-1).to(hidden.dtype)
            return torch.einsum("bl,bld->bd", weights, hidden)

        return attention_pool(self.left_pool_query), attention_pool(self.right_pool_query)

    def _conditioning_features(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fixed_mask: torch.Tensor | None,
        prediction_mask: torch.Tensor | None,
        denoise_step: torch.Tensor | float | None,
        self_condition_probs: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, length = attention_mask.shape
        device = attention_mask.device
        inferred_prediction = attention_mask & input_ids.eq(self.mask_token_id)
        fixed = attention_mask & ~inferred_prediction if fixed_mask is None else fixed_mask.bool()
        predicted = inferred_prediction if prediction_mask is None else prediction_mask.bool()
        if fixed.shape != attention_mask.shape:
            raise ValueError("fixed_mask must match input_ids")
        if predicted.shape != attention_mask.shape:
            raise ValueError("prediction_mask must match input_ids")
        fixed = fixed & attention_mask
        predicted = predicted & attention_mask & ~fixed

        # 0: mutable-visible, 1: fixed motif, 2: mutable-masked/predicted.
        role_ids = torch.zeros_like(attention_mask, dtype=torch.long)
        role_ids[fixed] = 1
        role_ids[predicted] = 2

        positions = torch.arange(length, device=device).unsqueeze(0).expand(batch_size, -1)
        has_motif = fixed.any(dim=1)
        first_motif = torch.where(fixed, positions, length).amin(dim=1)
        last_motif = torch.where(fixed, positions, -1).amax(dim=1)
        first_motif = torch.where(has_motif, first_motif, torch.zeros_like(first_motif))
        last_motif = torch.where(has_motif, last_motif, torch.full_like(last_motif, length - 1))

        left = positions < first_motif.unsqueeze(1)
        right = positions > last_motif.unsqueeze(1)
        side_ids = torch.ones_like(positions)
        side_ids[left] = 0
        side_ids[right] = 2
        side_ids = torch.where(has_motif.unsqueeze(1), side_ids, torch.ones_like(side_ids))

        relative_distance = torch.where(
            left,
            positions - first_motif.unsqueeze(1),
            torch.where(right, positions - last_motif.unsqueeze(1), torch.zeros_like(positions)),
        )
        clip = self.conditioning_relative_distance_clip
        relative_ids = relative_distance.clamp(-clip, clip) + clip

        if denoise_step is None:
            noise_level = torch.zeros(batch_size, device=device, dtype=torch.float32)
        else:
            noise_level = torch.as_tensor(denoise_step, device=device, dtype=torch.float32)
            if noise_level.ndim == 0:
                noise_level = noise_level.expand(batch_size)
            elif noise_level.ndim == 2 and noise_level.shape[1] == 1:
                noise_level = noise_level.squeeze(1)
            if noise_level.shape != (batch_size,):
                raise ValueError("denoise_step must be a scalar or have shape [batch]")
            if not bool(torch.isfinite(noise_level).all().item()) or not bool(
                ((noise_level >= 0) & (noise_level <= 1)).all().item()
            ):
                raise ValueError("denoise_step must contain normalized values in [0, 1]")

        features = (
            self.role_embedding(role_ids)
            + self.side_embedding(side_ids)
            + self.relative_motif_embedding(relative_ids)
            + self.noise_level_projection(noise_level.unsqueeze(-1)).unsqueeze(1)
        )
        if self_condition_probs is not None:
            if self_condition_probs.shape != (batch_size, length, 4):
                raise ValueError("self_condition_probs must have shape [batch, length, 4]")
            probabilities = self_condition_probs.to(device=device, dtype=torch.float32)
            if not bool(torch.isfinite(probabilities).all().item()) or bool((probabilities < 0).any().item()):
                raise ValueError("self_condition_probs must be finite and non-negative")
            normalizer = probabilities.sum(dim=-1, keepdim=True)
            probabilities = torch.where(
                normalizer > 0,
                probabilities / normalizer.clamp_min(torch.finfo(probabilities.dtype).eps),
                torch.zeros_like(probabilities),
            )
            features = features + self.self_condition_projection(probabilities)
        return features.to(dtype) * attention_mask.unsqueeze(-1).to(dtype)
