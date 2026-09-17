from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_sequence_table(source: Path, output_csv: Path, output_fasta: Path) -> dict[str, int]:
    """Normalize and sequence-deduplicate Stanford RNA records deterministically."""
    candidates: list[dict[str, str]] = []
    stats = defaultdict(int)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"target_id", "sequence"}.issubset(reader.fieldnames):
            raise ValueError("Stanford sequence CSV must contain target_id and sequence columns")
        for row in reader:
            stats["input_records"] += 1
            target_id = (row.get("target_id") or "").strip()
            sequence = (row.get("sequence") or "").strip().upper().replace("T", "U")
            if not target_id or len(sequence) < 8 or set(sequence) - set("AUCG"):
                stats["invalid_or_too_short"] += 1
                continue
            candidates.append(
                {
                    "target_id": target_id,
                    "sequence": sequence,
                    "family": "",
                    "source": "stanford_rna_3d_folding_train_v2",
                }
            )
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_sequences: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item["target_id"]):
        if candidate["target_id"] in seen_ids:
            stats["duplicate_target_id"] += 1
            continue
        if candidate["sequence"] in seen_sequences:
            stats["duplicate_sequence"] += 1
            continue
        seen_ids.add(candidate["target_id"])
        seen_sequences.add(candidate["sequence"])
        records.append(candidate)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target_id", "sequence", "family", "source"])
        writer.writeheader()
        writer.writerows(records)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    with output_fasta.open("w", encoding="ascii", newline="\n") as handle:
        for record in records:
            handle.write(f">{record['target_id']}\n{record['sequence']}\n")
    stats["output_records"] = len(records)
    return dict(stats)


def write_connected_components(mmseqs_tsv: Path, sequence_csv: Path, output_tsv: Path) -> dict[str, int]:
    """Convert representative/member pairs into strict connected components."""
    with sequence_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        sequence_ids = {row["target_id"] for row in csv.DictReader(handle)}
    adjacency: dict[str, set[str]] = {sequence_id: set() for sequence_id in sequence_ids}
    with mmseqs_tsv.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.rstrip("\r\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"invalid MMseqs2 cluster row at line {line_number}")
            representative, member = fields
            if representative not in adjacency or member not in adjacency:
                raise ValueError(f"unknown sequence ID in MMseqs2 output at line {line_number}")
            adjacency[representative].add(member)
            adjacency[member].add(representative)
    components: list[list[str]] = []
    unseen = set(sequence_ids)
    while unseen:
        start = min(unseen)
        stack = [start]
        component: list[str] = []
        unseen.remove(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda members: members[0])
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("cluster_id\tsequence_id\tcluster_size\n")
        for index, members in enumerate(components, start=1):
            cluster_id = f"cluster_{index:06d}"
            for member in members:
                handle.write(f"{cluster_id}\t{member}\t{len(members)}\n")
    return {"sequence_count": len(sequence_ids), "cluster_count": len(components)}


def run_mmseqs(
    mmseqs: str,
    fasta: Path,
    result_prefix: Path,
    temp_dir: Path,
    threads: int,
) -> Path:
    result_prefix.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    database = result_prefix.parent / f"{result_prefix.name}_db"
    clustering = result_prefix.parent / f"{result_prefix.name}_clu"
    cluster_tsv = result_prefix.parent / f"{result_prefix.name}_cluster.tsv"
    subprocess.run([mmseqs, "createdb", str(fasta), str(database), "--dbtype", "2"], check=True)
    subprocess.run(
        [
            mmseqs,
            "linclust",
            str(database),
            str(clustering),
            str(temp_dir),
            "--min-seq-id",
            "0.8",
            "-c",
            "0.8",
            "--cov-mode",
            "0",
            "--cluster-mode",
            "1",
            "--threads",
            str(threads),
        ],
        check=True,
    )
    subprocess.run(
        [
            mmseqs,
            "createtsv",
            str(database),
            str(database),
            str(clustering),
            str(cluster_tsv),
            "--threads",
            str(threads),
        ],
        check=True,
    )
    if not cluster_tsv.is_file():
        raise FileNotFoundError(f"MMseqs2 did not create {cluster_tsv}")
    return cluster_tsv


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stanford RNA sequences for Scaffold Generator V2.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-csv", default=Path("data/processed/rna_merged_mmseqs80.csv"), type=Path)
    parser.add_argument("--output-fasta", default=Path("project_artifacts/mmseqs80_search/rna80.fasta"), type=Path)
    parser.add_argument(
        "--output-clusters", default=Path("project_artifacts/mmseqs80_search/rna80_connected_components.tsv"), type=Path
    )
    parser.add_argument("--mmseqs", default="mmseqs")
    parser.add_argument("--mmseqs-cluster-tsv", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--audit", default=Path("project_artifacts/mmseqs80_search/data_preparation_audit.json"), type=Path)
    args = parser.parse_args()

    sequence_stats = prepare_sequence_table(args.source, args.output_csv, args.output_fasta)
    cluster_tsv = args.mmseqs_cluster_tsv
    if cluster_tsv is None:
        cluster_tsv = run_mmseqs(
            args.mmseqs,
            args.output_fasta,
            args.output_fasta.parent / "rna80",
            args.output_fasta.parent / "tmp",
            args.threads,
        )
    cluster_stats = write_connected_components(cluster_tsv, args.output_csv, args.output_clusters)
    audit = {
        "source": str(args.source.resolve()),
        "parameters": {
            "workflow": "linclust",
            "cluster_mode": 1,
            "min_sequence_identity": 0.8,
            "min_bidirectional_coverage": 0.8,
            "cov_mode": 0,
        },
        "sequence_stats": sequence_stats,
        "cluster_stats": cluster_stats,
        "sha256": {
            "source": _sha256(args.source),
            "sequence_table": _sha256(args.output_csv),
            "cluster_manifest": _sha256(args.output_clusters),
        },
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
