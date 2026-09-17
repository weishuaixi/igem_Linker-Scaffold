from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sequence_quality(sequence: str) -> dict[str, float | int]:
    sequence = sequence.strip().upper().replace("T", "U")
    if not sequence or set(sequence) - set("AUCG"):
        raise ValueError("sequence must contain only A, U, C, and G")
    counts = Counter(sequence)
    length = len(sequence)
    entropy = -sum((counts[base] / length) * math.log2(counts[base] / length) for base in "AUCG" if counts[base])
    return {
        "length": length,
        "gc_fraction": (counts["G"] + counts["C"]) / length,
        "max_homopolymer_run": max(sum(1 for _ in run) for _, run in groupby(sequence)),
        "base_entropy": entropy,
    }


def filter_sequence_table(
    input_csv: Path,
    output_csv: Path,
    *,
    gc_min: float = 0.10,
    gc_max: float = 0.90,
    max_homopolymer_run: int = 12,
    min_base_entropy: float = 1.20,
    min_length: int = 1,
) -> tuple[dict[str, int | float], set[str]]:
    if not 0 <= gc_min <= gc_max <= 1:
        raise ValueError("GC bounds must satisfy 0 <= min <= max <= 1")
    if max_homopolymer_run <= 0:
        raise ValueError("max_homopolymer_run must be positive")
    if not 0 <= min_base_entropy <= 2:
        raise ValueError("min_base_entropy must be in [0, 2]")
    if min_length < 1:
        raise ValueError("min_length must be positive")

    stats: defaultdict[str, int | float] = defaultdict(int)
    kept_rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_sequences: set[str] = set()
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"target_id", "sequence"}.issubset(reader.fieldnames):
            raise ValueError("input CSV must contain target_id and sequence columns")
        for row in reader:
            stats["input_records"] += 1
            target_id = (row.get("target_id") or "").strip()
            sequence = (row.get("sequence") or "").strip().upper().replace("T", "U")
            if not target_id or not sequence or set(sequence) - set("AUCG"):
                stats["invalid"] += 1
                continue
            if target_id in seen_ids:
                stats["duplicate_target_id"] += 1
                continue
            if sequence in seen_sequences:
                stats["duplicate_sequence"] += 1
                continue
            metrics = sequence_quality(sequence)
            reasons: list[str] = []
            if int(metrics["length"]) < min_length:
                reasons.append("short")
            if not gc_min <= float(metrics["gc_fraction"]) <= gc_max:
                reasons.append("gc")
            if int(metrics["max_homopolymer_run"]) > max_homopolymer_run:
                reasons.append("homopolymer")
            if float(metrics["base_entropy"]) < min_base_entropy:
                reasons.append("entropy")
            if reasons:
                stats["rejected_quality"] += 1
                for reason in reasons:
                    stats[f"rejected_{reason}"] += 1
                continue
            seen_ids.add(target_id)
            seen_sequences.add(sequence)
            kept_rows.append(
                {
                    "target_id": target_id,
                    "sequence": sequence,
                    "family": (row.get("family") or "").strip(),
                    "source": (row.get("source") or "").strip(),
                }
            )

    kept_rows.sort(key=lambda row: row["target_id"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target_id", "sequence", "family", "source"])
        writer.writeheader()
        writer.writerows(kept_rows)
    stats["output_records"] = len(kept_rows)
    stats.update(
        {
            "gc_min": gc_min,
            "gc_max": gc_max,
            "max_homopolymer_run": max_homopolymer_run,
            "min_base_entropy": min_base_entropy,
            "min_length": min_length,
        }
    )
    return dict(stats), {row["target_id"] for row in kept_rows}


def filter_cluster_manifest(
    input_tsv: Path,
    output_tsv: Path,
    kept_ids: set[str],
    *,
    expected_input_ids: set[str] | None = None,
) -> dict[str, int]:
    members_by_cluster: defaultdict[str, list[str]] = defaultdict(list)
    all_members_by_cluster: defaultdict[str, list[str]] = defaultdict(list)
    declared_sizes: dict[str, int] = {}
    all_seen_ids: set[str] = set()
    with input_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"cluster_id", "sequence_id", "cluster_size"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("cluster TSV must contain cluster_id, sequence_id, and cluster_size columns")
        for line_number, row in enumerate(reader, start=2):
            sequence_id = (row.get("sequence_id") or "").strip()
            cluster_id = (row.get("cluster_id") or "").strip()
            raw_size = (row.get("cluster_size") or "").strip()
            if not sequence_id or not cluster_id or not raw_size:
                raise ValueError(f"blank cluster field at line {line_number}")
            if sequence_id in all_seen_ids:
                raise ValueError(f"duplicate cluster assignment for {sequence_id}")
            try:
                declared_size = int(raw_size)
            except ValueError as error:
                raise ValueError(f"invalid cluster_size at line {line_number}") from error
            if declared_size < 1:
                raise ValueError(f"invalid cluster_size at line {line_number}")
            previous_size = declared_sizes.setdefault(cluster_id, declared_size)
            if previous_size != declared_size:
                raise ValueError(f"inconsistent cluster_size for {cluster_id}")
            all_seen_ids.add(sequence_id)
            all_members_by_cluster[cluster_id].append(sequence_id)
            if sequence_id in kept_ids:
                members_by_cluster[cluster_id].append(sequence_id)
    for cluster_id, members in all_members_by_cluster.items():
        if declared_sizes[cluster_id] != len(members):
            raise ValueError(f"cluster_size does not match rows for {cluster_id}")
    if expected_input_ids is not None and all_seen_ids != expected_input_ids:
        missing_input = sorted(expected_input_ids - all_seen_ids)
        unknown_input = sorted(all_seen_ids - expected_input_ids)
        raise ValueError(
            f"input CSV and cluster manifest IDs differ: missing={missing_input[:10]}, unknown={unknown_input[:10]}"
        )
    missing = sorted(kept_ids - all_seen_ids)
    if missing:
        raise ValueError(f"cluster manifest is missing {len(missing)} retained sequence IDs")

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("cluster_id\tsequence_id\tcluster_size\n")
        for cluster_id in sorted(members_by_cluster):
            members = sorted(members_by_cluster[cluster_id])
            for sequence_id in members:
                handle.write(f"{cluster_id}\t{sequence_id}\t{len(members)}\n")
    return {"sequence_count": len(kept_ids), "cluster_count": len(members_by_cluster)}


def _input_target_ids(input_csv: Path) -> set[str]:
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "target_id" not in reader.fieldnames:
            raise ValueError("input CSV must contain target_id")
        identifiers = [(row.get("target_id") or "").strip() for row in reader]
    if any(not identifier for identifier in identifiers):
        raise ValueError("input CSV contains a blank target_id")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("input CSV contains duplicate target_id values")
    return set(identifiers)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a quality-controlled RNA scaffold training release.")
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--input-clusters", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-clusters", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--gc-min", type=float, default=0.10)
    parser.add_argument("--gc-max", type=float, default=0.90)
    parser.add_argument("--max-homopolymer-run", type=int, default=12)
    parser.add_argument("--min-base-entropy", type=float, default=1.20)
    parser.add_argument("--min-length", type=int, default=1)
    args = parser.parse_args()

    sequence_stats, kept_ids = filter_sequence_table(
        args.input_csv,
        args.output_csv,
        gc_min=args.gc_min,
        gc_max=args.gc_max,
        max_homopolymer_run=args.max_homopolymer_run,
        min_base_entropy=args.min_base_entropy,
        min_length=args.min_length,
    )
    cluster_stats = filter_cluster_manifest(
        args.input_clusters,
        args.output_clusters,
        kept_ids,
        expected_input_ids=_input_target_ids(args.input_csv),
    )
    audit = {
        "source": {
            "sequence_table": _portable_path(args.input_csv),
            "cluster_manifest": _portable_path(args.input_clusters),
        },
        "sequence_stats": sequence_stats,
        "cluster_stats": cluster_stats,
        "sha256": {
            "input_csv": _sha256(args.input_csv),
            "input_clusters": _sha256(args.input_clusters),
            "output_csv": _sha256(args.output_csv),
            "output_clusters": _sha256(args.output_clusters),
        },
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
