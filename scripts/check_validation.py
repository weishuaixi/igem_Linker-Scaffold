"""Matched validation-context decoder diagnostic; no test-set tuning."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_scaffolds import _checkpoint_from_training_manifest, _markov, _uniform
from rna_scaffold.benchmarking import RnaTrainingPrior
from rna_scaffold.checkpoints import load_scaffold_checkpoint
from rna_scaffold.datamodule import RnaScaffoldDataModule
from rna_scaffold.decoding import DecodingSettings, iterative_denoise
from rna_scaffold.evaluation import (
    TrainingSimilarityIndex,
    base_composition_entropy_bits,
    maximum_homopolymer_run,
    within_group_diversity,
)
from rna_scaffold.tokenizer import RnaTokenizer


def linker_metrics(left, right):
    sequence = left + right
    run = max(maximum_homopolymer_run(left), maximum_homopolymer_run(right))
    return {
        "gc": (sequence.count("G") + sequence.count("C")) / len(sequence),
        "composition_entropy": base_composition_entropy_bits(sequence),
        "max_run": run,
        "run_gt6": int(run > 6),
    }


def summarize(rows):
    return {
        "count": len(rows),
        **{
            f"mean_{key}": statistics.mean(row[key] for row in rows)
            for key in ("gc", "composition_entropy", "max_run", "run_gt6")
        },
        "maximum_run": max(row["max_run"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--output", default="outputs/linker_v5_validation")
    parser.add_argument("--motifs", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--expanded", action="store_true", help="Add train-only baselines and sequence diversity diagnostics"
    )
    args = parser.parse_args()
    if args.motifs < 1 or args.candidates < 1:
        parser.error("motifs and candidates must be positive")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dm = RnaScaffoldDataModule(tokenizer=RnaTokenizer(), **cfg["data"])
    dm.setup("validate")
    dataset = dm.val_dataset
    indices = list(range(dataset.record_count))
    random.Random(args.seed).shuffle(indices)
    selected = []
    clusters = set()
    for index in indices:
        record = dataset.records[index]
        cluster = dm.cluster_by_id[record.target_id]
        if cluster in clusters:
            continue
        clusters.add(cluster)
        selected.append(index)
        if len(selected) == args.motifs:
            break
    if len(selected) != args.motifs:
        raise ValueError("not enough distinct validation clusters")
    checkpoint, manifest = _checkpoint_from_training_manifest(
        {
            "training_manifest": str(Path(cfg["trainer"]["checkpoint_dir"]) / "training_manifest.json"),
            "training_config": args.config,
        },
        expected_split_manifest=asdict(dm.split_manifest),
    )
    loaded = load_scaffold_checkpoint(checkpoint, device=args.device)
    loaded.model.eval()
    conditioning = loaded.model.self_conditioning_probability > 0
    variants = [
        ("confidence_16", "confidence", 16, conditioning),
        ("random_16", "random", 16, conditioning),
        ("random_4", "random", 4, conditioning),
        ("single_pass", "random", 1, False),
    ]
    if conditioning:
        variants.append(("random_16_no_sc", "random", 16, False))
    if args.expanded:
        variants.extend((name, None, 0, False) for name in ("uniform", "markov1", "markov2"))
    training_sequences = [record.sequence for record in dm.train_dataset.records]
    prior = RnaTrainingPrior.from_sequences(training_sequences) if args.expanded else None
    similarity = TrainingSimilarityIndex(training_sequences) if args.expanded else None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source_rows = []
    results = {name: [] for name, *_ in variants}
    group_reports = []
    # Flush every draw: interrupted diagnostics retain inspectable raw results.
    with (output / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for motif_index, index in enumerate(selected):
            item = dataset[index * dataset.views_per_record]
            n = int(item["attention_mask"].sum())
            left = int(item["target_left_length"])
            right = int(item["target_right_length"])
            sequence = "".join("AUCG"[base] for base in item["target_base_ids"][:n].tolist())
            motif = sequence[left : n - right]
            source = {
                "target_id": dataset.records[index].target_id,
                "motif": motif,
                "target_sequence": sequence,
                "left_length": left,
                "right_length": right,
                **linker_metrics(sequence[:left], sequence[n - right :]),
            }
            source_rows.append(source)
            for name, strategy, steps, use_sc in variants:
                started = time.perf_counter()
                sequences = []
                print(f"validation {motif_index + 1}/{len(selected)} {name}: {args.candidates} draws", flush=True)
                for draw in range(args.candidates):
                    seed = args.seed + motif_index * 100000 + draw
                    if strategy is None:
                        if name == "uniform":
                            generated, _ = _uniform(motif, n, random.Random(seed), flank_pair=(left, right))
                        else:
                            generated, _ = _markov(
                                motif, n, prior, random.Random(seed), order=int(name[-1]), flank_pair=(left, right)
                            )
                    else:
                        decoded = iterative_denoise(
                            loaded.model.model,
                            loaded.tokenizer,
                            motif,
                            n,
                            left,
                            DecodingSettings(
                                denoise_steps=steps,
                                top_p=1,
                                temperature=1,
                                remask_strategy=strategy,
                                use_self_conditioning=use_sc,
                            ),
                            torch.Generator(device=args.device).manual_seed(seed),
                            args.device,
                        )
                        generated = decoded.sequence
                    assert generated[left : n - right] == motif
                    sequences.append(generated)
                    metric = linker_metrics(generated[:left], generated[n - right :])
                    results[name].append(metric)
                    row = {
                        "variant": name,
                        "target_id": source["target_id"],
                        "seed": seed,
                        "sequence": generated,
                        "motif": motif,
                        "left_length": left,
                        "right_length": right,
                        "self_conditioning": use_sc,
                        **metric,
                    }
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                if args.expanded:
                    edit, kmer = within_group_diversity(sequences)
                    group = {
                        "variant": name,
                        "target_id": source["target_id"],
                        "cluster_id": dm.cluster_by_id[source["target_id"]],
                        **summarize(results[name][-args.candidates :]),
                        "unique_rate": len(set(sequences)) / len(sequences),
                        "mean_edit_diversity": statistics.mean(edit),
                        "mean_kmer_diversity": statistics.mean(kmer),
                        "mean_nearest_training_kmer_similarity": statistics.mean(
                            similarity.nearest(s) for s in sequences
                        ),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                    group_reports.append(group)
                    with (output / "groups.jsonl").open("a", encoding="utf-8") as groups_file:
                        groups_file.write(json.dumps(group) + "\n")
                    print(f"SAVED {name} context {motif_index + 1}/{len(selected)}", flush=True)
    report = {
        "status": "completed",
        "partition": "validation",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": manifest["best_checkpoint"]["sha256"],
        "sampling": "raw independent draws; no hard filtering or deduplication",
        "seed": args.seed,
        "source_contexts": source_rows,
        "real_validation_linkers": summarize(source_rows),
        "variants": {name: summarize(rows) for name, rows in results.items()},
        "interpretation": "Descriptive diagnostic, not a test-set or biological-function claim.",
    }
    if args.expanded:
        report["baseline_training_records"] = len(training_sequences)
        report["metric_scope"] = (
            "GC/entropy/run: mutable flanks; diversity/nearest-training 5-mer Jaccard: full sequence including fixed motif. Equal-weight context means; draws are not independent biological replicates."
        )
        report["rnafold_status"] = "not_run; no structural or functional claims"
        report["groups"] = group_reports
        for name in results:
            groups = [g for g in group_reports if g["variant"] == name]
            report["variants"][name].update(
                {
                    key: statistics.mean(g[key] for g in groups)
                    for key in (
                        "unique_rate",
                        "mean_edit_diversity",
                        "mean_kmer_diversity",
                        "mean_nearest_training_kmer_similarity",
                    )
                }
            )
            report["variants"][name]["elapsed_seconds"] = sum(g["elapsed_seconds"] for g in groups)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
