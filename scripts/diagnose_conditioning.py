"""Validation-only full-mask motif ablation and matched marginal NLL diagnostic."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from benchmark_scaffolds import _checkpoint_from_training_manifest
from rna_scaffold.benchmarking import RnaTrainingPrior
from rna_scaffold.checkpoints import load_scaffold_checkpoint
from rna_scaffold.datamodule import RnaScaffoldDataModule
from rna_scaffold.decoding import _forward_with_optional_denoising_context
from rna_scaffold.evaluation import paired_bootstrap
from rna_scaffold.tokenizer import RnaTokenizer

BASES = "AUCG"


def markov_marginals(prior, length, left, motif):
    """Exact first-order marginals given ONLY visible motif, via scaled forward/backward."""
    transition = np.array([[prior.transition[a][b] for b in BASES] for a in BASES])
    initial = np.array([prior.initial[b] for b in BASES])
    evidence = np.ones((length, 4))
    for i, base in enumerate(motif, left):
        evidence[i] = 0
        evidence[i, BASES.index(base)] = 1
    forward = np.zeros_like(evidence)
    forward[0] = initial * evidence[0]
    forward[0] /= forward[0].sum()
    for i in range(1, length):
        forward[i] = (forward[i - 1] @ transition) * evidence[i]
        forward[i] /= forward[i].sum()
    backward = np.ones_like(evidence)
    for i in range(length - 2, -1, -1):
        backward[i] = transition @ (evidence[i + 1] * backward[i + 1])
        backward[i] /= backward[i].sum()
    posterior = forward * backward
    return posterior / posterior.sum(axis=1, keepdims=True)


def score(probabilities, targets, mutable):
    p = probabilities[mutable]
    y = targets[mutable]
    return {
        "nll": float(-np.log(p[np.arange(len(y)), y].clip(1e-300)).mean()),
        "accuracy": float((p.argmax(axis=1) == y).mean()),
    }


def shuffled_motif(motif, rng):
    if len(set(motif)) == 1:
        return None
    chars = list(motif)
    for _ in range(100):
        rng.shuffle(chars)
        if "".join(chars) != motif:
            return "".join(chars)
    return motif[1:] + motif[:1]


@torch.inference_mode()
def model_probabilities(loaded, length, left, motif, device):
    tokenizer = loaded.tokenizer
    canvas = torch.full((1, length), tokenizer.token_to_id[tokenizer.special.mask], device=device, dtype=torch.long)
    canvas[0, left : left + len(motif)] = torch.tensor(tokenizer.encode(motif), device=device)
    fixed = torch.zeros_like(canvas, dtype=torch.bool)
    fixed[:, left : left + len(motif)] = True
    output = _forward_with_optional_denoising_context(
        loaded.model.model,
        input_ids=canvas,
        attention_mask=torch.ones_like(fixed),
        fixed_mask=fixed,
        prediction_mask=~fixed,
        denoise_step=torch.ones(1, device=device),
        self_condition_probs=torch.zeros((1, length, 4), device=device),
    )
    return output.token_logits[0].float().softmax(-1).cpu().numpy().astype(float)


def build_report(rows, seed):
    names = sorted({name for row in rows for name in row["scores"]})
    means = {}
    for name in names:
        values = [r["scores"][name] for r in rows if name in r["scores"]]
        means[name] = {
            "clusters": len(values),
            **{key: statistics.mean(v[key] for v in values) for key in ("nll", "accuracy")},
        }
    comparisons = {}
    for other in names:
        if other == "model_correct":
            continue
        matched = [r for r in rows if other in r["scores"]]
        comparisons[other + "_minus_model_correct"] = {
            "clusters": len(matched),
            **asdict(
                paired_bootstrap(
                    [r["scores"][other]["nll"] for r in matched],
                    [r["scores"]["model_correct"]["nll"] for r in matched],
                    seed=seed,
                )
            ),
        }
    return {
        "means": means,
        "paired_nll_differences": comparisons,
        "difference_sign": "positive favors model_correct; interval spanning zero is inconclusive",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--output", default="outputs/linker_v5_conditioning")
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.clusters < 2:
        parser.error("clusters must be at least 2")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dm = RnaScaffoldDataModule(tokenizer=RnaTokenizer(), **cfg["data"])
    dm.setup("validate")
    checkpoint, manifest = _checkpoint_from_training_manifest(
        {
            "training_manifest": str(Path(cfg["trainer"]["checkpoint_dir"]) / "training_manifest.json"),
            "training_config": args.config,
        },
        expected_split_manifest=asdict(dm.split_manifest),
    )
    training = [r.sequence for r in dm.train_dataset.records]
    prior = RnaTrainingPrior.from_sequences(training)
    counts = Counter("".join(training))
    frequency = np.array([counts[b] + 1 for b in BASES], dtype=float)
    frequency /= frequency.sum()
    dataset = dm.val_dataset
    indices = list(range(dataset.record_count))
    random.Random(args.seed).shuffle(indices)
    contexts, seen = [], set()
    for index in indices:
        record = dataset.records[index]
        cluster = dm.cluster_by_id[record.target_id]
        if cluster in seen:
            continue
        seen.add(cluster)
        item = dataset[index * dataset.views_per_record]
        n = int(item["attention_mask"].sum())
        left, right = int(item["target_left_length"]), int(item["target_right_length"])
        sequence = "".join(BASES[b] for b in item["target_base_ids"][:n].tolist())
        contexts.append(
            {
                "target_id": record.target_id,
                "cluster": cluster,
                "sequence": sequence,
                "left": left,
                "right": right,
                "motif": sequence[left : n - right],
            }
        )
        if len(contexts) == args.clusters:
            break
    if len(contexts) != args.clusters:
        raise ValueError(f"Only {len(contexts)} validation clusters available")
    loaded = load_scaffold_checkpoint(checkpoint, device=args.device)
    loaded.model.eval()
    rows = []
    with (output / "contexts.jsonl").open("w", encoding="utf-8") as handle:
        for i, context in enumerate(contexts):
            motif, left = context["motif"], context["left"]
            n = len(context["sequence"])
            targets = np.array([BASES.index(b) for b in context["sequence"]])
            mutable = np.ones(n, dtype=bool)
            mutable[left : left + len(motif)] = False
            rng = random.Random(args.seed + i)
            shuffled = shuffled_motif(motif, rng)
            # Same-length contiguous segment from another validation cluster, not an artificial padded motif.
            donors = [c for c in contexts if c["cluster"] != context["cluster"] and len(c["motif"]) >= len(motif)]
            swaps = [
                (c["target_id"], c["motif"][s : s + len(motif)])
                for c in donors
                for s in range(len(c["motif"]) - len(motif) + 1)
                if c["motif"][s : s + len(motif)] != motif
            ]
            donor_id, swapped = rng.choice(swaps) if swaps else (None, None)
            scores = {}
            for label, supplied in (("correct", motif), ("shuffled", shuffled), ("swapped", swapped)):
                if supplied is None:
                    continue
                scores["model_" + label] = score(
                    model_probabilities(loaded, n, left, supplied, args.device), targets, mutable
                )
                scores["markov1_" + label] = score(markov_marginals(prior, n, left, supplied), targets, mutable)
            scores["uniform"] = score(np.full((n, 4), 0.25), targets, mutable)
            scores["train_frequency"] = score(np.tile(frequency, (n, 1)), targets, mutable)
            row = {
                **context,
                "shuffled_motif": shuffled,
                "swapped_motif": swapped,
                "donor_target_id": donor_id,
                "scored_tokens": int(mutable.sum()),
                "scores": scores,
            }
            rows.append(row)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(f"SAVED {i + 1}/{len(contexts)} {context['target_id']}", flush=True)
    report = {
        "status": "completed",
        "partition": "validation",
        "clusters": len(rows),
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": manifest["best_checkpoint"]["sha256"],
        "training_records": len(training),
        "train_frequency_AUCG": frequency.tolist(),
        "protocol": "One context per cluster; all linker tokens masked; first forward pass; zero self-conditioning; unsmoothed marginal NLL in nats/token, equal cluster weights. No teacher-forcing of unknown linker bases. Not joint sequence likelihood.",
        "limitations": "Exploratory validation reused for model selection. One ablation draw per context. Shuffle preserves composition; swap does not. No structural/function proof; ablations do not by themselves isolate long-range learning.",
        **build_report(rows, args.seed),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
