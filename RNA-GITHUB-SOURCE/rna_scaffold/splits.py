from __future__ import annotations

import csv
import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rna_scaffold.records import RnaSequenceRecord

PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitManifest:
    partitions: dict[str, tuple[str, ...]]
    sequence_hashes: dict[str, str]
    seed: int
    record_counts: dict[str, int] = field(default_factory=dict)
    cluster_counts: dict[str, int] = field(default_factory=dict)
    source_sha256: dict[str, str] = field(default_factory=dict)
    clustering_thresholds: dict[str, float] = field(default_factory=dict)

    def partition_for(self, target_id: str) -> str:
        matches = [name for name, members in self.partitions.items() if target_id in members]
        if len(matches) != 1:
            raise KeyError(f"target_id must occur in exactly one partition: {target_id}")
        return matches[0]


class ClusterAssignments(dict[str, str]):
    """Dictionary-compatible cluster assignments with immutable source audit data."""

    def __init__(self, assignments: Mapping[str, str], source_sha256: str) -> None:
        super().__init__(assignments)
        self.source_sha256 = source_sha256


def _group_key(record: RnaSequenceRecord) -> str:
    return f"family:{record.family}" if record.family else f"sequence:{record.sequence_sha256}"


def _partition_for_group(group_key: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{group_key}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def build_family_disjoint_manifest(
    records: Sequence[RnaSequenceRecord], seed: int = 42
) -> SplitManifest:
    grouped: dict[str, list[RnaSequenceRecord]] = {}
    for record in records:
        grouped.setdefault(_group_key(record), []).append(record)
    partitions: dict[str, list[str]] = {name: [] for name in PARTITIONS}
    for group_key in sorted(grouped):
        partition = _partition_for_group(group_key, seed)
        partitions[partition].extend(record.target_id for record in grouped[group_key])
    return validate_manifest(records, partitions, seed=seed)


def load_cluster_assignments(path: str | Path) -> dict[str, str]:
    """Load one strict MMseqs2 connected-component assignment per sequence."""
    source = Path(path)
    required_fields = {"cluster_id", "sequence_id", "cluster_size"}
    assignments: dict[str, str] = {}
    declared_sizes: dict[str, int] = {}
    members_by_cluster: dict[str, list[str]] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not required_fields.issubset(reader.fieldnames):
            raise ValueError(
                "cluster manifest must contain cluster_id, sequence_id, and cluster_size columns"
            )
        for line_number, row in enumerate(reader, start=2):
            cluster_id = (row.get("cluster_id") or "").strip()
            sequence_id = (row.get("sequence_id") or "").strip()
            cluster_size = (row.get("cluster_size") or "").strip()
            if not cluster_id or not sequence_id or not cluster_size:
                raise ValueError(f"blank cluster assignment field at line {line_number}")
            try:
                declared_size = int(cluster_size)
                if declared_size < 1:
                    raise ValueError
            except ValueError as error:
                raise ValueError(f"invalid cluster_size at line {line_number}") from error
            if sequence_id in assignments:
                raise ValueError(f"duplicate cluster assignment for sequence_id: {sequence_id}")
            previous_size = declared_sizes.setdefault(cluster_id, declared_size)
            if previous_size != declared_size:
                raise ValueError(f"inconsistent cluster_size for cluster_id: {cluster_id}")
            assignments[sequence_id] = cluster_id
            members_by_cluster.setdefault(cluster_id, []).append(sequence_id)
    for cluster_id, members in members_by_cluster.items():
        if declared_sizes[cluster_id] != len(members):
            raise ValueError(f"cluster_size does not match rows for cluster_id: {cluster_id}")
    return ClusterAssignments(assignments, hashlib.sha256(source.read_bytes()).hexdigest())


def build_cluster_disjoint_manifest(
    records: Sequence[RnaSequenceRecord],
    cluster_by_id: Mapping[str, str],
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    source_sha256: Mapping[str, str] | None = None,
    clustering_thresholds: Mapping[str, float] | None = None,
    expected_partition_counts: Mapping[str, int] | None = None,
) -> SplitManifest:
    """Assign complete clusters to deterministic train, validation, and test partitions."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must be non-negative and leave a train fraction")
    _validate_record_integrity(records)
    _validate_cluster_assignments(records, cluster_by_id)
    grouped: dict[str, list[RnaSequenceRecord]] = {}
    for record in records:
        grouped.setdefault(cluster_by_id[record.target_id], []).append(record)

    total_records = len(records)
    target_counts = {
        "train": total_records * (1 - val_fraction - test_fraction),
        "validation": total_records * val_fraction,
        "test": total_records * test_fraction,
    }
    counts = {name: 0 for name in PARTITIONS}
    partitions: dict[str, list[str]] = {name: [] for name in PARTITIONS}
    cluster_ids = sorted(grouped)
    random.Random(seed).shuffle(cluster_ids)
    for cluster_id in cluster_ids:
        partition = max(PARTITIONS, key=lambda name: target_counts[name] - counts[name])
        members = grouped[cluster_id]
        partitions[partition].extend(record.target_id for record in members)
        counts[partition] += len(members)
    counts = {name: len(partitions[name]) for name in PARTITIONS}
    if expected_partition_counts is not None:
        expected = {name: int(expected_partition_counts.get(name, -1)) for name in PARTITIONS}
        if set(expected_partition_counts) != set(PARTITIONS) or counts != expected:
            raise ValueError(
                f"partition counts do not match expected: actual={counts}, expected={expected_partition_counts}"
            )
    audit_hashes = dict(source_sha256 or {})
    audit_hashes.setdefault("records", _records_sha256(records))
    manifest_hash = getattr(cluster_by_id, "source_sha256", None)
    if manifest_hash:
        audit_hashes.setdefault("cluster_manifest", manifest_hash)
    cluster_counts = {
        name: len({cluster_by_id[target_id] for target_id in partitions[name]})
        for name in PARTITIONS
    }
    return validate_manifest(
        records,
        partitions,
        seed=seed,
        cluster_by_id=cluster_by_id,
        record_counts={**counts, "total": len(records)},
        cluster_counts={**cluster_counts, "total": len(grouped)},
        source_sha256=audit_hashes,
        clustering_thresholds=dict(clustering_thresholds or {}),
    )


def validate_manifest(
    records: Sequence[RnaSequenceRecord],
    partitions: Mapping[str, Sequence[str]],
    seed: int = 42,
    cluster_by_id: Mapping[str, str] | None = None,
    record_counts: Mapping[str, int] | None = None,
    cluster_counts: Mapping[str, int] | None = None,
    source_sha256: Mapping[str, str] | None = None,
    clustering_thresholds: Mapping[str, float] | None = None,
) -> SplitManifest:
    if set(partitions) != set(PARTITIONS):
        raise ValueError(f"manifest partitions must be {PARTITIONS}")
    _validate_record_integrity(records)
    by_id = {record.target_id: record for record in records}
    owner: dict[str, str] = {}
    for partition, members in partitions.items():
        for target_id in members:
            if target_id not in by_id:
                raise ValueError(f"unknown target ID in manifest: {target_id}")
            if target_id in owner:
                raise ValueError(f"target ID occurs in multiple partitions: {target_id}")
            owner[target_id] = partition
    if set(owner) != set(by_id):
        missing = sorted(set(by_id) - set(owner))
        raise ValueError(f"manifest omits target IDs: {missing}")

    if cluster_by_id is not None:
        _validate_cluster_assignments(records, cluster_by_id)
        cluster_owner: dict[str, str] = {}
        for target_id, partition in owner.items():
            cluster_id = cluster_by_id[target_id]
            previous = cluster_owner.setdefault(cluster_id, partition)
            if previous != partition:
                raise ValueError(f"cluster crosses partitions: {cluster_id}")

    family_owner: dict[str, str] = {}
    sequence_owner: dict[str, str] = {}
    for target_id, partition in owner.items():
        record = by_id[target_id]
        if record.family:
            previous = family_owner.setdefault(record.family, partition)
            if previous != partition:
                raise ValueError(f"family crosses partitions: {record.family}")
        if record.sequence_sha256 in sequence_owner:
            raise ValueError(f"duplicate normalized sequence / exact sequence overlap: {target_id}")
        sequence_owner[record.sequence_sha256] = partition

    normalized = {
        name: tuple(sorted(partitions[name]))
        for name in PARTITIONS
    }
    hashes = {record.target_id: record.sequence_sha256 for record in records}
    default_record_counts = {name: len(normalized[name]) for name in PARTITIONS}
    default_record_counts["total"] = len(records)
    return SplitManifest(
        normalized,
        hashes,
        seed,
        dict(record_counts or default_record_counts),
        dict(cluster_counts or {}),
        dict(source_sha256 or {"records": _records_sha256(records)}),
        dict(clustering_thresholds or {}),
    )


def _validate_cluster_assignments(
    records: Sequence[RnaSequenceRecord], cluster_by_id: Mapping[str, str]
) -> None:
    record_ids = {record.target_id for record in records}
    if len(record_ids) != len(records):
        raise ValueError("record target IDs must be unique")
    assignment_ids = set(cluster_by_id)
    unknown = sorted(assignment_ids - record_ids)
    if unknown:
        raise ValueError(f"cluster assignments contain unknown target IDs: {unknown}")
    missing = sorted(record_ids - assignment_ids)
    if missing:
        raise ValueError(f"cluster assignments are incomplete; missing target IDs: {missing}")
    blank = sorted(target_id for target_id, cluster_id in cluster_by_id.items() if not cluster_id.strip())
    if blank:
        raise ValueError(f"cluster assignments contain blank cluster IDs: {blank}")


def _validate_record_integrity(records: Sequence[RnaSequenceRecord]) -> None:
    target_ids = {record.target_id for record in records}
    if len(target_ids) != len(records):
        raise ValueError("record target IDs must be unique")
    sequence_ids: dict[str, str] = {}
    for record in records:
        previous = sequence_ids.setdefault(record.sequence_sha256, record.target_id)
        if previous != record.target_id:
            raise ValueError(
                f"duplicate normalized sequence / exact sequence overlap: {previous}, {record.target_id}"
            )


def _records_sha256(records: Sequence[RnaSequenceRecord]) -> str:
    canonical = "".join(
        f"{record.target_id}\t{record.sequence}\n" for record in sorted(records, key=lambda item: item.target_id)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
