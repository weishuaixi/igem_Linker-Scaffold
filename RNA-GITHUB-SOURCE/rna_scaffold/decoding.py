from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

import torch

from rna_scaffold.geometry import legal_flank_pairs
from rna_scaffold.model import ScaffoldModelOutput
from rna_scaffold.tokenizer import RnaTokenizer


@dataclass(frozen=True)
class DecodingSettings:
    denoise_steps: int = 12
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 0.95
    mask_schedule: str = "linear"
    max_homopolymer_run: int | None = None
    gc_min: float | None = None
    gc_max: float | None = None
    enforce_gc_bounds: bool = False
    remask_strategy: str = "confidence"
    use_self_conditioning: bool = True

    def __post_init__(self) -> None:
        if self.remask_strategy not in {"confidence", "random"}:
            raise ValueError("remask_strategy must be 'confidence' or 'random'")
        if self.denoise_steps <= 0:
            raise ValueError("denoise_steps must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k is not None and not 1 <= self.top_k <= 4:
            raise ValueError("top_k must be between 1 and 4")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.mask_schedule not in {"linear", "cosine"}:
            raise ValueError("mask_schedule must be 'linear' or 'cosine'")
        if self.max_homopolymer_run is not None and self.max_homopolymer_run <= 0:
            raise ValueError("max_homopolymer_run must be positive")
        if self.gc_min is not None and not 0 <= self.gc_min <= 1:
            raise ValueError("gc_min must be in [0, 1]")
        if self.gc_max is not None and not 0 <= self.gc_max <= 1:
            raise ValueError("gc_max must be in [0, 1]")
        if self.gc_min is not None and self.gc_max is not None and self.gc_min > self.gc_max:
            raise ValueError("gc_min must not exceed gc_max")
        if self.enforce_gc_bounds and self.gc_min is None and self.gc_max is None:
            raise ValueError("enforce_gc_bounds requires gc_min and/or gc_max")


@dataclass(frozen=True)
class DecodedScaffold:
    sequence: str
    normalized_log_probability: float
    masked_counts: tuple[int, ...]
    mean_token_confidence: float = 0.0
    constraint_forced_argmax_rate: float = 0.0
    constraint_log_probability_mass: float = 0.0


@dataclass
class FlankPairSampler:
    """Cached joint flank distribution with an encoded without-replacement mask."""

    pairs: torch.Tensor
    pair_scores: torch.Tensor
    pair_ids: torch.Tensor
    used_pair_id_mask: torch.Tensor
    _sample_order: torch.Tensor | None = field(default=None, init=False, repr=False)
    _sample_offset: int = field(default=0, init=False, repr=False)

    def _refill_sample_order(self, generator: torch.Generator) -> None:
        available = ~self.used_pair_id_mask[self.pair_ids]
        if not bool(available.any().item()):
            # More candidates than legal geometries may still explore different
            # sequences. Start a new without-replacement cycle.
            self.used_pair_id_mask.zero_()
            available = torch.ones_like(available)
        # A Gumbel-top-k ordering is a weighted sample without replacement.
        # Building it once per cycle avoids an O(number_of_pairs) softmax for
        # every generated candidate.
        uniforms = torch.rand(
            self.pair_scores.shape,
            dtype=torch.float32,
            device=self.pair_scores.device,
            generator=generator,
        ).clamp_min_(torch.finfo(torch.float32).tiny)
        keys = self.pair_scores.float() - torch.log(-torch.log(uniforms))
        keys = keys.masked_fill(~available, float("-inf"))
        order = torch.argsort(keys, descending=True)
        self._sample_order = order[available[order]]
        self._sample_offset = 0

    def select(
        self,
        generator: torch.Generator,
        *,
        sample: bool = True,
    ) -> tuple[int, int]:
        if sample:
            if self._sample_order is None or self._sample_offset >= int(self._sample_order.numel()):
                self._refill_sample_order(generator)
            assert self._sample_order is not None
            index = int(self._sample_order[self._sample_offset].item())
            self._sample_offset += 1
            pair_id = int(self.pair_ids[index].item())
            self.used_pair_id_mask[pair_id] = True
            return int(self.pairs[index, 0].item()), int(self.pairs[index, 1].item())

        available = ~self.used_pair_id_mask[self.pair_ids]
        if not bool(available.any().item()):
            # More candidates than legal geometries may still explore different
            # sequences. Start a new without-replacement cycle without rebuilding
            # or copying the legal-pair tensor.
            self.used_pair_id_mask.zero_()
            available = torch.ones_like(available)
        available_scores = self.pair_scores.masked_fill(~available, float("-inf"))
        index = int(torch.argmax(available_scores).item())
        self._sample_order = None
        self._sample_offset = 0
        pair_id = int(self.pair_ids[index].item())
        self.used_pair_id_mask[pair_id] = True
        return int(self.pairs[index, 0].item()), int(self.pairs[index, 1].item())

    def log_probability(self, left_length: int, right_length: int) -> float:
        """Return the selected pair's log probability after legal-pair conditioning."""
        stride = int(self.used_pair_id_mask.numel() ** 0.5)
        pair_id = int(left_length) * stride + int(right_length)
        matches = self.pair_ids.eq(pair_id)
        if not bool(matches.any().item()):
            raise ValueError("flank-length pair is not legal for this sampler")
        log_probabilities = torch.log_softmax(self.pair_scores.float(), dim=0)
        return float(log_probabilities[matches.nonzero(as_tuple=False)[0, 0]].item())


def build_flank_pair_sampler(
    output: ScaffoldModelOutput | None,
    motif_length: int,
    max_length: int,
    *,
    min_scaffold_length: int = 1,
    min_flank_length: int = 0,
    short_flank_range: tuple[int, int] | None = None,
    long_flank_range: tuple[int, int] | None = None,
    excluded_pairs: set[tuple[int, int]] | None = None,
    motif: str | None = None,
    decoding_settings: DecodingSettings | None = None,
    length_sampling: str = "model",
    model_max_length: int | None = None,
    device: str | torch.device | None = None,
) -> FlankPairSampler:
    if length_sampling not in {"model", "uniform"}:
        raise ValueError("length_sampling must be 'model' or 'uniform'")
    left_log_probs: torch.Tensor | None = None
    right_log_probs: torch.Tensor | None = None
    if length_sampling == "model":
        if output is None or output.left_length_logits is None or output.right_length_logits is None:
            raise ValueError("length_sampling='model' requires left and right length logits from the checkpoint")
        if output.left_length_logits.shape[0] != 1 or output.right_length_logits.shape[0] != 1:
            raise ValueError("flank-length selection currently requires batch size one")
        left_log_probs = output.left_length_logits[0].float().log_softmax(dim=-1)
        right_log_probs = output.right_length_logits[0].float().log_softmax(dim=-1)
        inferred_max_length = min(
            left_log_probs.shape[-1] - 1,
            right_log_probs.shape[-1] - 1,
        )
        model_max_length = (
            inferred_max_length if model_max_length is None else min(int(model_max_length), inferred_max_length)
        )
        pair_device = left_log_probs.device
    else:
        if model_max_length is None:
            if output is None or output.left_length_logits is None or output.right_length_logits is None:
                raise ValueError(
                    "length_sampling='uniform' requires model_max_length when no length logits are provided"
                )
            model_max_length = min(
                output.left_length_logits.shape[-1] - 1,
                output.right_length_logits.shape[-1] - 1,
            )
        pair_device = (
            torch.device(device)
            if device is not None
            else (output.token_logits.device if output is not None else torch.device("cpu"))
        )
    model_max_length = int(model_max_length)
    pairs = legal_flank_pairs(
        motif_length,
        min_flank_length=min_flank_length,
        min_total_scaffold_length=min_scaffold_length,
        requested_max_length=int(max_length),
        model_max_length=model_max_length,
        device=pair_device,
    )
    # Iterative denoising needs at least one mutable scaffold position.
    pairs = pairs[pairs.sum(dim=1) > 0]
    if (short_flank_range is None) != (long_flank_range is None):
        raise ValueError("short_flank_range and long_flank_range must be provided together")
    if short_flank_range is not None and long_flank_range is not None:
        short_min, short_max = map(int, short_flank_range)
        long_min, long_max = map(int, long_flank_range)
        if short_min < 0 or long_min < 0 or short_min > short_max or long_min > long_max:
            raise ValueError("invalid short/long flank ranges")
        left, right = pairs[:, 0], pairs[:, 1]
        left_short_right_long = (
            (left >= short_min) & (left <= short_max) & (right >= long_min) & (right <= long_max) & (right > left)
        )
        left_long_right_short = (
            (left >= long_min) & (left <= long_max) & (right >= short_min) & (right <= short_max) & (left > right)
        )
        pairs = pairs[left_short_right_long | left_long_right_short]
    if pairs.numel() == 0:
        raise ValueError("motif cannot fit within the requested length, flank, and sequence constraints")
    constraints_active = decoding_settings is not None and (
        decoding_settings.max_homopolymer_run is not None or decoding_settings.enforce_gc_bounds
    )
    if constraints_active:
        if motif is None or len(motif) != int(motif_length):
            raise ValueError("motif must match motif_length when filtering constrained flank pairs")
        totals = pairs.sum(dim=1) + int(motif_length)
        feasible_by_total = torch.zeros(
            int(totals.max().item()) + 1,
            dtype=torch.bool,
            device=pairs.device,
        )
        for total_length in totals.unique().tolist():
            feasible_by_total[int(total_length)] = _fixed_motif_constraints_feasible(
                motif,
                total_length=int(total_length),
                motif_start=0,
                settings=decoding_settings,
            )
        pairs = pairs[feasible_by_total[totals]]
    if pairs.numel() == 0:
        raise ValueError("motif cannot fit within the requested length, flank, and sequence constraints")
    if length_sampling == "uniform":
        pair_scores = torch.zeros(pairs.shape[0], dtype=torch.float32, device=pairs.device)
    else:
        assert left_log_probs is not None and right_log_probs is not None
        pair_scores = left_log_probs[pairs[:, 0]] + right_log_probs[pairs[:, 1]]
    stride = model_max_length + 1
    pair_ids = pairs[:, 0] * stride + pairs[:, 1]
    used_pair_id_mask = torch.zeros(
        stride * stride,
        dtype=torch.bool,
        device=pairs.device,
    )
    if excluded_pairs:
        encoded = [
            int(left_length) * stride + int(right_length)
            for left_length, right_length in excluded_pairs
            if 0 <= int(left_length) < stride and 0 <= int(right_length) < stride
        ]
        if encoded:
            used_pair_id_mask[torch.tensor(encoded, dtype=torch.long, device=pairs.device)] = True
    return FlankPairSampler(pairs, pair_scores, pair_ids, used_pair_id_mask)


def select_flank_lengths(
    output: ScaffoldModelOutput,
    motif_length: int,
    max_length: int,
    generator: torch.Generator,
    sample: bool = True,
    min_scaffold_length: int = 1,
    min_flank_length: int = 0,
    short_flank_range: tuple[int, int] | None = None,
    long_flank_range: tuple[int, int] | None = None,
    excluded_pairs: set[tuple[int, int]] | None = None,
    length_sampling: str = "model",
) -> tuple[int, int]:
    sampler = build_flank_pair_sampler(
        output,
        motif_length,
        max_length,
        min_scaffold_length=min_scaffold_length,
        min_flank_length=min_flank_length,
        short_flank_range=short_flank_range,
        long_flank_range=long_flank_range,
        excluded_pairs=excluded_pairs,
        length_sampling=length_sampling,
    )
    return sampler.select(generator, sample=sample)


def _probabilities_from_scaled_logits(
    scaled_logits: torch.Tensor,
    settings: DecodingSettings,
    allowed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    filtered = scaled_logits.float()
    if allowed_mask is not None:
        if allowed_mask.shape != filtered.shape:
            raise ValueError("allowed_mask must match logits")
        if not bool(allowed_mask.any(dim=-1).all().item()):
            raise ValueError("every sampled position must allow at least one base")
        filtered = filtered.masked_fill(~allowed_mask.bool(), float("-inf"))
    if settings.top_k is not None and settings.top_k < filtered.shape[-1]:
        threshold = torch.topk(filtered, settings.top_k, dim=-1).values[..., -1:]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if settings.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative - sorted_probs >= settings.top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf")).scatter(
            -1,
            sorted_indices,
            sorted_logits,
        )
    return torch.softmax(filtered, dim=-1)


def _filtered_probabilities(
    logits: torch.Tensor,
    settings: DecodingSettings,
    allowed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return _probabilities_from_scaled_logits(
        logits.float() / settings.temperature,
        settings,
        allowed_mask,
    )


def _gc_count_bounds(total_length: int, settings: DecodingSettings) -> tuple[int, int] | None:
    if not settings.enforce_gc_bounds:
        return None
    lower = (
        0
        if settings.gc_min is None
        else int((Decimal(str(settings.gc_min)) * total_length).to_integral_value(rounding=ROUND_CEILING))
    )
    upper = (
        total_length
        if settings.gc_max is None
        else int((Decimal(str(settings.gc_max)) * total_length).to_integral_value(rounding=ROUND_FLOOR))
    )
    if lower > upper:
        raise ValueError("GC bounds contain no feasible integer count for this sequence length")
    return lower, upper


def _fixed_motif_constraints_feasible(
    motif: str,
    total_length: int,
    motif_start: int,
    settings: DecodingSettings,
) -> bool:
    """Return whether the two mutable linkers can satisfy hard constraints.

    The fixed motif is deliberately outside the GC denominator and breaks
    homopolymer adjacency between the left and right linkers.
    """
    motif = motif.strip().upper().replace("T", "U")
    if not motif or set(motif) - set("AUCG"):
        return False
    motif_end = int(motif_start) + len(motif)
    if total_length < len(motif) or motif_start < 0 or motif_end > total_length:
        return False
    linker_length = total_length - len(motif)
    try:
        _gc_count_bounds(linker_length, settings)
    except ValueError:
        return False
    return True


def _logsumexp(values: list[float]) -> float:
    if not values:
        return float("-inf")
    maximum = max(values)
    if maximum == float("-inf"):
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _cpu_sampling_generator(
    generator: torch.Generator,
) -> torch.Generator:
    generator_device = torch.device(getattr(generator, "device", "cpu"))
    if generator_device.type == "cpu":
        return generator
    seed = int(
        torch.randint(
            0,
            2**63 - 1,
            (1,),
            generator=generator,
            device=generator_device,
            dtype=torch.int64,
        )
        .cpu()
        .item()
    )
    return torch.Generator(device="cpu").manual_seed(seed)


def _sample_masked_classes_with_constraints(
    logits: torch.Tensor,
    canvas: torch.Tensor,
    fixed_mask: torch.Tensor,
    mask_id: int,
    base_token_ids: torch.Tensor,
    settings: DecodingSettings,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
    """Sample MASK positions from the model distribution conditioned on hard constraints."""
    if logits.ndim != 2 or logits.shape[0] != canvas.numel() or logits.shape[1] != 4:
        raise ValueError("logits must have shape [sequence_length, 4]")
    constrained = settings.max_homopolymer_run is not None or settings.enforce_gc_bounds
    output_device = logits.device
    logits_cpu = logits.detach().float().cpu()
    if not bool(torch.isfinite(logits_cpu).all().item()):
        raise RuntimeError("model produced non-finite token logits during decoding")
    canvas_cpu = canvas.detach().cpu()
    fixed_mask_cpu = fixed_mask.detach().bool().cpu()
    if fixed_mask_cpu.shape != canvas_cpu.shape:
        raise ValueError("fixed_mask must match canvas")
    base_token_ids_cpu = base_token_ids.detach().cpu()
    cpu_generator = _cpu_sampling_generator(generator)
    masked_positions = canvas_cpu.eq(mask_id).nonzero(as_tuple=False).squeeze(-1)
    # Keep the model distribution separate from the temperature/top-k/top-p
    # proposal.  Sampling and remasking are policy decisions; reported model
    # confidence/likelihood must remain comparable across decoding settings.
    model_probabilities = torch.softmax(logits_cpu, dim=-1)
    proposal_probabilities = _filtered_probabilities(logits_cpu, settings)
    if not constrained:
        probabilities = proposal_probabilities[masked_positions]
        sampled_classes = torch.multinomial(
            probabilities,
            1,
            generator=cpu_generator,
        ).squeeze(-1)
        sampled_policy_confidences = probabilities.gather(
            1,
            sampled_classes.unsqueeze(-1),
        ).squeeze(-1)
        sampled_model_confidences = (
            model_probabilities[masked_positions]
            .gather(
                1,
                sampled_classes.unsqueeze(-1),
            )
            .squeeze(-1)
        )
        return (
            masked_positions.to(output_device),
            sampled_classes.to(output_device),
            sampled_policy_confidences.to(output_device),
            sampled_model_confidences.to(output_device),
            0,
            0.0,
        )

    total_length = int(canvas_cpu.numel())
    linker_length = int((~fixed_mask_cpu).sum().item())
    gc_bounds = _gc_count_bounds(linker_length, settings)
    track_gc = gc_bounds is not None
    min_gc, max_gc = gc_bounds or (0, linker_length)
    max_run = settings.max_homopolymer_run
    token_to_class = {int(token_id): index for index, token_id in enumerate(base_token_ids_cpu.tolist())}
    fixed_classes: list[int] = []
    for token_id, is_motif in zip(canvas_cpu.tolist(), fixed_mask_cpu.tolist()):
        token_id = int(token_id)
        if is_motif:
            if token_id not in token_to_class:
                raise ValueError("fixed motif positions must contain RNA base tokens")
            # -2 marks motif positions: they contribute no GC and reset the
            # homopolymer state, so the two linkers are never made adjacent.
            fixed_classes.append(-2)
        elif token_id == mask_id:
            fixed_classes.append(-1)
        elif token_id in token_to_class:
            fixed_classes.append(token_to_class[token_id])
        else:
            raise ValueError("constrained decoding canvas may contain only MASK or RNA base tokens")

    fixed_gc_suffix = [0] * (total_length + 1)
    mutable_suffix = [0] * (total_length + 1)
    for position in range(total_length - 1, -1, -1):
        fixed_class = fixed_classes[position]
        fixed_gc_suffix[position] = fixed_gc_suffix[position + 1] + int(fixed_class in {2, 3})
        mutable_suffix[position] = mutable_suffix[position + 1] + int(fixed_class == -1)

    proposal_log_probabilities = proposal_probabilities.log().tolist()

    State = tuple[int, int, int, int]

    def is_pruned(state: State) -> bool:
        position, _, _, gc_count = state
        if track_gc:
            minimum_possible = gc_count + fixed_gc_suffix[position]
            maximum_possible = minimum_possible + mutable_suffix[position]
            if minimum_possible > max_gc or maximum_possible < min_gc:
                return True
        return False

    def successors(state: State) -> list[tuple[State, float, int]]:
        position, previous_class, run_length, gc_count = state
        if position >= total_length:
            return []
        fixed_class = fixed_classes[position]
        if fixed_class == -2:
            # The fixed motif is excluded from linker-only constraints and
            # separates the left and right homopolymer runs.
            return [((position + 1, -1, 0, gc_count), 0.0, -2)]
        choices = range(4) if fixed_class < 0 else (fixed_class,)
        result: list[tuple[State, float, int]] = []
        for base_class in choices:
            next_run = run_length + 1 if base_class == previous_class else 1
            if max_run is not None and next_run > max_run:
                continue
            next_gc = gc_count + int(base_class in {2, 3}) if track_gc else 0
            if track_gc and next_gc > max_gc:
                continue
            next_state = (
                position + 1,
                base_class if max_run is not None else -1,
                next_run if max_run is not None else 0,
                next_gc,
            )
            log_weight = float(proposal_log_probabilities[position][base_class]) if fixed_class < 0 else 0.0
            if log_weight == float("-inf"):
                continue
            result.append((next_state, log_weight, base_class))
        return result

    root: State = (0, -1, 0, 0)
    log_partition: dict[State, float] = {}
    stack: list[tuple[State, bool]] = [(root, False)]
    while stack:
        state, expanded = stack.pop()
        if state in log_partition:
            continue
        position = state[0]
        if is_pruned(state):
            log_partition[state] = float("-inf")
            continue
        if position == total_length:
            gc_count = state[3]
            log_partition[state] = 0.0 if not track_gc or min_gc <= gc_count <= max_gc else float("-inf")
            continue
        child_states = successors(state)
        if not expanded:
            stack.append((state, True))
            stack.extend((child_state, False) for child_state, _, _ in child_states if child_state not in log_partition)
            continue
        log_partition[state] = _logsumexp(
            [log_weight + log_partition[child_state] for child_state, log_weight, _ in child_states]
        )

    if log_partition[root] == float("-inf"):
        raise ValueError(
            "linker constraints cannot be satisfied within the configured temperature/top-k/top-p sampling support"
        )

    constraint_log_probability_mass = min(
        0.0,
        float(log_partition[root]),
    )

    sampled_positions: list[int] = []
    sampled_classes: list[int] = []
    sampled_policy_confidences: list[float] = []
    sampled_model_confidences: list[float] = []
    forced_argmax_count = 0
    state = root
    for position, fixed_class in enumerate(fixed_classes):
        available = successors(state)
        if fixed_class == -2:
            if not available or log_partition[available[0][0]] == float("-inf"):
                raise RuntimeError("fixed motif separator became infeasible during constrained sampling")
            state = available[0][0]
        elif fixed_class >= 0:
            base_class = fixed_class
            matching = [child_state for child_state, _, choice in available if choice == base_class]
            if not matching or log_partition[matching[0]] == float("-inf"):
                raise RuntimeError("committed linker context became infeasible during constrained sampling")
            state = matching[0]
        else:
            conditional_scores = torch.full((4,), float("-inf"), dtype=torch.float64)
            next_state_by_class: dict[int, State] = {}
            for child_state, log_weight, candidate_class in available:
                suffix_score = log_partition[child_state]
                if suffix_score != float("-inf"):
                    conditional_scores[candidate_class] = log_weight + suffix_score
                    next_state_by_class[candidate_class] = child_state
            proposal_argmax = int(proposal_probabilities[position].argmax().item())
            constrained_argmax = int(conditional_scores.argmax().item())
            forced_argmax_count += int(constrained_argmax != proposal_argmax)
            probabilities = torch.softmax(conditional_scores, dim=-1)
            sampled = torch.multinomial(probabilities, 1, generator=cpu_generator)
            base_class = int(sampled.item())
            sampled_positions.append(position)
            sampled_classes.append(base_class)
            # Remasking follows the configured proposal, not the DP-renormalized
            # posterior.  Reporting follows the raw model distribution, so a
            # rule-forced or top-k-forced base never appears spuriously certain.
            sampled_policy_confidences.append(float(proposal_probabilities[position, base_class].item()))
            sampled_model_confidences.append(float(model_probabilities[position, base_class].item()))
            state = next_state_by_class[base_class]

    return (
        torch.tensor(sampled_positions, dtype=torch.long, device=output_device),
        torch.tensor(sampled_classes, dtype=torch.long, device=output_device),
        torch.tensor(sampled_policy_confidences, dtype=torch.float32, device=output_device),
        torch.tensor(sampled_model_confidences, dtype=torch.float32, device=output_device),
        forced_argmax_count,
        constraint_log_probability_mass,
    )


def _forward_with_optional_denoising_context(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    fixed_mask: torch.Tensor,
    prediction_mask: torch.Tensor,
    denoise_step: torch.Tensor,
    self_condition_probs: torch.Tensor,
) -> ScaffoldModelOutput:
    """Call old and new model classes without hiding errors raised inside ``forward``."""
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError, AttributeError):
        parameters = {}
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    optional = {
        "fixed_mask": fixed_mask,
        "prediction_mask": prediction_mask,
        "denoise_step": denoise_step,
        "self_condition_probs": self_condition_probs,
    }
    if getattr(model, "use_conditioning_embeddings", None) is False:
        # New class definitions can load old checkpoints with conditioning
        # disabled; those models intentionally reject self-conditioning input.
        optional.pop("self_condition_probs")
    supported = optional if accepts_kwargs else {name: value for name, value in optional.items() if name in parameters}
    return model(input_ids=input_ids, attention_mask=attention_mask, **supported)


def _scheduled_mask_count(
    initial_masked_count: int,
    step: int,
    settings: DecodingSettings,
) -> int:
    """Return the deterministic number of scaffold tokens to remask after a pass."""
    if step == settings.denoise_steps - 1:
        return 0
    completed_fraction = (step + 1) / settings.denoise_steps
    if settings.mask_schedule == "linear":
        remaining_fraction = 1.0 - completed_fraction
    else:
        remaining_fraction = math.cos(completed_fraction * math.pi / 2)
    return max(0, min(initial_masked_count, math.ceil(initial_masked_count * remaining_fraction)))


@torch.inference_mode()
def iterative_denoise(
    model,
    tokenizer: RnaTokenizer,
    motif: str,
    total_length: int,
    motif_start: int,
    settings: DecodingSettings,
    generator: torch.Generator,
    device: str | torch.device = "cpu",
) -> DecodedScaffold:
    motif = motif.strip().upper().replace("T", "U")
    if len(motif) < 1 or set(motif) - set("AUCG"):
        raise ValueError("motif must contain only A, U, C, and G")
    motif_end = motif_start + len(motif)
    if total_length <= len(motif):
        raise ValueError("canvas must contain at least one scaffold position")
    if total_length > int(model.max_length) or motif_start < 0 or motif_end > total_length:
        raise ValueError("invalid motif coordinates for the requested canvas")

    device = torch.device(device)
    mask_id = tokenizer.token_to_id[tokenizer.special.mask]
    canvas = torch.full((1, total_length), mask_id, dtype=torch.long, device=device)
    motif_ids = torch.tensor(tokenizer.encode(motif), dtype=torch.long, device=device)
    canvas[0, motif_start:motif_end] = motif_ids
    fixed = torch.zeros(total_length, dtype=torch.bool, device=device)
    fixed[motif_start:motif_end] = True
    attention = torch.ones_like(canvas, dtype=torch.bool)
    base_token_ids = torch.tensor([tokenizer.token_to_id[base] for base in "AUCG"], dtype=torch.long, device=device)
    scaffold_positions = (~fixed).nonzero(as_tuple=False).squeeze(-1)
    initial_masked_count = int(scaffold_positions.numel())
    masked_counts = [initial_masked_count]
    finalization_log_probs = torch.full((total_length,), float("nan"), dtype=torch.float32, device=device)
    finalization_confidences = torch.full((total_length,), float("nan"), dtype=torch.float32, device=device)
    self_condition_probs = torch.zeros((1, total_length, 4), dtype=torch.float32, device=device)
    total_sampled_count = 0
    forced_argmax_count = 0
    initial_constraint_log_probability_mass = 0.0

    for step in range(settings.denoise_steps):
        masked_positions = ((canvas[0] == mask_id) & ~fixed).nonzero(as_tuple=False).squeeze(-1)
        if masked_positions.numel() == 0:
            masked_counts.extend([0] * (settings.denoise_steps - step))
            break
        prediction_mask = canvas.eq(mask_id) & ~fixed.unsqueeze(0)
        noise_fraction = float(masked_positions.numel()) / initial_masked_count
        denoise_step = torch.full((1,), noise_fraction, dtype=torch.float32, device=device)
        output = _forward_with_optional_denoising_context(
            model,
            input_ids=canvas,
            attention_mask=attention,
            fixed_mask=fixed.unsqueeze(0),
            prediction_mask=prediction_mask,
            denoise_step=denoise_step,
            self_condition_probs=self_condition_probs,
        )
        if settings.use_self_conditioning:
            self_condition_probs = torch.softmax(output.token_logits.detach().float(), dim=-1)
            self_condition_probs = self_condition_probs * (attention & ~fixed.unsqueeze(0)).unsqueeze(-1)
        (
            sampled_positions,
            sampled_classes,
            sampled_policy_confidences,
            sampled_model_confidences,
            step_forced_argmax_count,
            step_constraint_log_probability_mass,
        ) = _sample_masked_classes_with_constraints(
            output.token_logits[0],
            canvas[0],
            fixed,
            mask_id,
            base_token_ids,
            settings,
            generator,
        )
        if not torch.equal(sampled_positions, masked_positions):
            raise RuntimeError("constrained sampler returned unexpected MASK positions")
        canvas[0, masked_positions] = base_token_ids[sampled_classes]
        total_sampled_count += int(sampled_positions.numel())
        forced_argmax_count += step_forced_argmax_count
        if step == 0:
            initial_constraint_log_probability_mass = step_constraint_log_probability_mass
        next_masked_count = _scheduled_mask_count(initial_masked_count, step, settings)
        committed = torch.ones(sampled_positions.numel(), dtype=torch.bool, device=device)
        if next_masked_count:
            if settings.remask_strategy == "random":
                # Commit positions independently of their sampled base. Ranking
                # sampled-token probabilities repeatedly accepts common bases
                # and rejects rare ones, sharpening even a constant predictor.
                remask_indices = torch.randperm(sampled_positions.numel(), generator=generator, device=device)[
                    :next_masked_count
                ]
            else:
                remask_indices = torch.topk(
                    sampled_policy_confidences,
                    next_masked_count,
                    largest=False,
                ).indices
            canvas[0, masked_positions[remask_indices]] = mask_id
            committed[remask_indices] = False
        committed_positions = sampled_positions[committed]
        committed_confidences = sampled_model_confidences[committed].clamp_min(1e-12)
        finalization_log_probs[committed_positions] = committed_confidences.log()
        finalization_confidences[committed_positions] = committed_confidences
        canvas[0, motif_start:motif_end] = motif_ids
        masked_counts.append(next_masked_count)

    if masked_counts[-1] != 0:
        raise RuntimeError("denoising steps ended with unresolved scaffold positions")
    sequence = tokenizer.decode(canvas[0].tolist())
    scaffold_log_probs = finalization_log_probs[scaffold_positions]
    scaffold_confidences = finalization_confidences[scaffold_positions]
    if not bool(torch.isfinite(scaffold_log_probs).all().item()):
        raise RuntimeError("denoising ended without a recorded conditional probability for every scaffold position")
    normalized = float(scaffold_log_probs.mean().item())
    mean_token_confidence = float(scaffold_confidences.mean().item())
    forced_argmax_rate = forced_argmax_count / total_sampled_count if total_sampled_count else 0.0
    return DecodedScaffold(
        sequence,
        normalized,
        tuple(masked_counts),
        mean_token_confidence,
        float(forced_argmax_rate),
        initial_constraint_log_probability_mass,
    )
