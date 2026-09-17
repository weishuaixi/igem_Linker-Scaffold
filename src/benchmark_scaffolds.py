from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from rna_scaffold.benchmarking import BASES, RnaTrainingPrior
from rna_scaffold.checkpoints import load_scaffold_checkpoint
from rna_scaffold.data import eligible_denoising_records
from rna_scaffold.evaluation import (
    CandidateMetric,
    CandidateSummary,
    TrainingSimilarityIndex,
    base_composition_entropy_bits,
    maximum_homopolymer_run,
    nearest_training_similarity,
    paired_bootstrap_by_motif,
    summarize_candidates,
    within_group_diversity,
)
from rna_scaffold.geometry import (
    MAX_SCAFFOLD_LENGTH,
    is_legal_scaffold_geometry,
    minimum_legal_total_length,
    validate_scaffold_length_limit,
)
from rna_scaffold.records import RnaSequenceRecord, load_sequence_records
from rna_scaffold.splits import (
    SplitManifest,
    build_cluster_disjoint_manifest,
    load_cluster_assignments,
)
from rna_scaffold.utils import validate_rna_sequence
from rna_scaffold.validators.rnafold import RnafoldResult, run_rnafold

ARTIFACT_NAMES = (
    "candidate_results.csv",
    "motif_summary.csv",
    "model_summary.csv",
    "paired_bootstrap.json",
    "run_manifest.json",
    "benchmark_report.md",
)
BOOTSTRAP_METRICS = (
    "valid_rate",
    "motif_preservation_rate",
    "unique_rate",
    "mean_normalized_edit_diversity",
    "mean_kmer_diversity",
    "mean_nearest_training_kmer_similarity",
)
CANDIDATE_RESULT_FIELDS = (
    "method",
    "primary",
    "motif_id",
    "source_partition",
    "source_target_id",
    "source_sequence_sha256",
    "source_start",
    "source_length",
    "length_stratum",
    "gc_stratum",
    "candidate_index",
    "motif_start",
    "motif_end",
    "seed",
    *(field.name for field in fields(CandidateMetric) if field.name != "motif_id"),
)
MOTIF_SUMMARY_FIELDS = (
    "method",
    "primary",
    "motif_id",
    "source_partition",
    "source_target_id",
    "source_sequence_sha256",
    "source_start",
    "source_length",
    "length_stratum",
    "gc_stratum",
    "status",
    "error_reason",
    "seed",
    "rejection_reasons",
    *(field.name for field in fields(CandidateSummary)),
)
MODEL_SUMMARY_FIELDS = (
    "method",
    "kind",
    "primary",
    "status",
    "error_reason",
    "seed",
    "actual_pretrained_kind",
    "actual_pretrained_mode",
    "decoder_mode",
    "denoise_steps",
    "rejection_reasons",
    *(field.name for field in fields(CandidateSummary)),
    "setup_runtime_seconds",
    "generation_runtime_seconds",
    "validation_runtime_seconds",
)


@dataclass(frozen=True)
class BenchmarkGeometry:
    min_flank_length: int = 2
    min_total_scaffold_length: int = 8
    preferred_total_scaffold_length: int = 24
    max_length: int = MAX_SCAFFOLD_LENGTH
    short_flank_min: int | None = None
    short_flank_max: int | None = None
    long_flank_min: int | None = None
    long_flank_max: int | None = None

    def __post_init__(self) -> None:
        validate_scaffold_length_limit(self.max_length, field="geometry.max_length")
        minimum_legal_total_length(
            1,
            self.min_flank_length,
            self.min_total_scaffold_length,
        )
        if self.preferred_total_scaffold_length < self.min_total_scaffold_length:
            raise ValueError(
                "geometry.preferred_total_scaffold_length must be at least geometry.min_total_scaffold_length"
            )
        if self.preferred_total_scaffold_length > self.max_length:
            raise ValueError("geometry.preferred_total_scaffold_length must not exceed geometry.max_length")
        bounds = (
            self.short_flank_min,
            self.short_flank_max,
            self.long_flank_min,
            self.long_flank_max,
        )
        if any(value is not None for value in bounds) and any(value is None for value in bounds):
            raise ValueError("all four geometry short/long flank bounds must be provided together")
        if all(value is not None for value in bounds):
            assert self.short_flank_min is not None
            assert self.short_flank_max is not None
            assert self.long_flank_min is not None
            assert self.long_flank_max is not None
            if self.short_flank_min < 0 or self.long_flank_min < 0:
                raise ValueError("geometry flank range minima must be non-negative")
            if self.short_flank_min > self.short_flank_max:
                raise ValueError("geometry.short_flank_min must not exceed short_flank_max")
            if self.long_flank_min > self.long_flank_max:
                raise ValueError("geometry.long_flank_min must not exceed long_flank_max")
            if not any(
                _matches_asymmetric_flank_ranges(short, long, self)
                or _matches_asymmetric_flank_ranges(long, short, self)
                for short in range(self.short_flank_min, self.short_flank_max + 1)
                for long in range(self.long_flank_min, self.long_flank_max + 1)
            ):
                raise ValueError("geometry short/long flank ranges contain no strictly asymmetric pair")


def _matches_asymmetric_flank_ranges(
    left_length: int,
    right_length: int,
    geometry: BenchmarkGeometry,
) -> bool:
    if geometry.short_flank_min is None:
        return True
    assert geometry.short_flank_max is not None
    assert geometry.long_flank_min is not None
    assert geometry.long_flank_max is not None
    left_short_right_long = (
        geometry.short_flank_min <= left_length <= geometry.short_flank_max
        and geometry.long_flank_min <= right_length <= geometry.long_flank_max
        and right_length > left_length
    )
    left_long_right_short = (
        geometry.long_flank_min <= left_length <= geometry.long_flank_max
        and geometry.short_flank_min <= right_length <= geometry.short_flank_max
        and left_length > right_length
    )
    return left_short_right_long or left_long_right_short


def _minimum_benchmark_total_length(motif_length: int, geometry: BenchmarkGeometry) -> int:
    minimum = minimum_legal_total_length(
        motif_length,
        geometry.min_flank_length,
        geometry.min_total_scaffold_length,
    )
    if geometry.short_flank_min is None:
        return minimum
    assert geometry.short_flank_max is not None
    assert geometry.long_flank_min is not None
    assert geometry.long_flank_max is not None
    asymmetric_sums = [
        short + long
        for short in range(geometry.short_flank_min, geometry.short_flank_max + 1)
        for long in range(geometry.long_flank_min, geometry.long_flank_max + 1)
        if _matches_asymmetric_flank_ranges(short, long, geometry)
        or _matches_asymmetric_flank_ranges(long, short, geometry)
    ]
    if not asymmetric_sums:
        raise ValueError("geometry short/long flank ranges contain no strictly asymmetric pair")
    return max(minimum, motif_length + min(asymmetric_sums))


@dataclass(frozen=True)
class BenchmarkPartitions:
    train_records: tuple[RnaSequenceRecord, ...]
    validation_records: tuple[RnaSequenceRecord, ...]
    test_records: tuple[RnaSequenceRecord, ...]
    manifest: SplitManifest
    training_data: Path
    cluster_manifest: Path


def _benchmark_geometry(config: dict[str, Any]) -> BenchmarkGeometry:
    return BenchmarkGeometry(**dict(config.get("geometry") or {}))


def _split_source_config(config: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    training_config_path = config.get("training_config")
    if training_config_path:
        training_config = yaml.safe_load(Path(str(training_config_path)).read_text(encoding="utf-8"))
        if not isinstance(training_config, dict) or not isinstance(training_config.get("data"), dict):
            raise TypeError("training_config must contain a data mapping")
        data = dict(training_config["data"])
        return (
            Path(str(data["train_data"])),
            Path(str(data["cluster_manifest"])),
            data,
        )
    split_contract = dict(config.get("split_contract") or {})
    if not config.get("training_data") or not config.get("cluster_manifest"):
        raise ValueError("formal benchmark requires training_config or training_data plus cluster_manifest")
    return (
        Path(str(config["training_data"])),
        Path(str(config["cluster_manifest"])),
        split_contract,
    )


def _load_benchmark_partitions(config: dict[str, Any]) -> BenchmarkPartitions:
    """Reconstruct the exact cluster-disjoint split used by V2 training."""
    training_data, cluster_manifest, split = _split_source_config(config)
    records = load_sequence_records(training_data, skip_invalid=False)
    assignments = load_cluster_assignments(cluster_manifest)
    max_length = int(split.get("max_target_length", split.get("max_length", MAX_SCAFFOLD_LENGTH)))
    eligible = eligible_denoising_records(
        records,
        max_length=max_length,
        min_motif_length=int(split.get("min_motif_length", 4)),
        min_flank_length=int(split.get("min_flank_length", 2)),
        min_total_scaffold_length=int(split.get("min_total_scaffold_length", 8)),
    )
    eligible_ids = {record.target_id for record in eligible}
    eligible_assignments = {
        target_id: cluster_id for target_id, cluster_id in assignments.items() if target_id in eligible_ids
    }
    manifest = build_cluster_disjoint_manifest(
        eligible,
        eligible_assignments,
        seed=int(split.get("seed", config.get("seed", 42))),
        val_fraction=float(split.get("val_fraction", 0.1)),
        test_fraction=float(split.get("test_fraction", 0.1)),
        source_sha256={
            "sequence_table": _sha256(training_data),
            "cluster_manifest": _sha256(cluster_manifest),
        },
        clustering_thresholds=dict(split.get("cluster_thresholds", split.get("clustering_thresholds", {})) or {}),
        expected_partition_counts=split.get("expected_partition_counts"),
    )
    expected_cluster_count = split.get("expected_cluster_count")
    if expected_cluster_count is not None and manifest.cluster_counts.get("total") != int(expected_cluster_count):
        raise ValueError(
            "eligible cluster count does not match expected: "
            f"actual={manifest.cluster_counts.get('total')}, expected={expected_cluster_count}"
        )
    by_id = {record.target_id: record for record in eligible}
    return BenchmarkPartitions(
        train_records=tuple(by_id[target_id] for target_id in manifest.partitions["train"]),
        validation_records=tuple(by_id[target_id] for target_id in manifest.partitions["validation"]),
        test_records=tuple(by_id[target_id] for target_id in manifest.partitions["test"]),
        manifest=manifest,
        training_data=training_data,
        cluster_manifest=cluster_manifest,
    )


def _length_stratum(length: int) -> str:
    if length <= 64:
        return "short_2_64"
    if length <= 256:
        return "medium_65_256"
    return "long_257_512_plus"


def _gc_stratum(sequence: str) -> str:
    fraction = (sequence.count("G") + sequence.count("C")) / len(sequence)
    if fraction < 0.4:
        return "low_gc"
    if fraction > 0.6:
        return "high_gc"
    return "mid_gc"


def build_held_out_motif_panel(
    records: list[RnaSequenceRecord] | tuple[RnaSequenceRecord, ...],
    *,
    panel_size: int,
    seed: int,
    motif_lengths: tuple[int, ...] = (4, 8, 16, 32),
    min_flank_length: int = 2,
    geometry: BenchmarkGeometry | None = None,
) -> list[dict[str, Any]]:
    """Deterministically derive a provenance-rich panel from held-out test rows."""
    if panel_size <= 0:
        raise ValueError("motif panel size must be positive")
    lengths = tuple(sorted({int(length) for length in motif_lengths}))
    if not lengths or lengths[0] < 4:
        raise ValueError("motif panel lengths must contain values of at least four")
    candidates: list[dict[str, Any]] = []
    effective_min_flank = max(
        int(min_flank_length),
        geometry.min_flank_length if geometry is not None else 0,
    )
    asymmetric_pairs: tuple[tuple[int, int], ...] = ()
    if geometry is not None and geometry.short_flank_min is not None:
        assert geometry.short_flank_max is not None
        assert geometry.long_flank_min is not None
        assert geometry.long_flank_max is not None
        asymmetric_pairs = tuple(
            pair
            for short in range(geometry.short_flank_min, geometry.short_flank_max + 1)
            for long in range(geometry.long_flank_min, geometry.long_flank_max + 1)
            for pair in ((short, long), (long, short))
            if _matches_asymmetric_flank_ranges(pair[0], pair[1], geometry)
        )
    for record in sorted(records, key=lambda item: item.target_id):
        for motif_length in lengths:
            possible_starts = list(
                range(
                    effective_min_flank,
                    len(record.sequence) - motif_length - effective_min_flank + 1,
                )
            )
            if asymmetric_pairs:
                possible_starts = [
                    start
                    for start in possible_starts
                    if any(
                        left <= start and right <= len(record.sequence) - start - motif_length
                        for left, right in asymmetric_pairs
                    )
                ]
            if not possible_starts:
                continue
            placement_digest = hashlib.sha256(f"{seed}:{record.target_id}:{motif_length}".encode()).digest()
            start = possible_starts[int.from_bytes(placement_digest[:8], "big") % len(possible_starts)]
            motif = record.sequence[start : start + motif_length]
            identity = hashlib.sha256(f"{record.target_id}:{start}:{motif_length}:{motif}".encode()).hexdigest()
            candidates.append(
                {
                    "id": f"motif_{identity[:16]}",
                    "sequence": motif,
                    "source_partition": "test",
                    "source_target_id": record.target_id,
                    "source_sequence_sha256": record.sequence_sha256,
                    "source_start": start,
                    "source_length": len(record.sequence),
                    "motif_length": motif_length,
                    "length_stratum": _length_stratum(len(record.sequence)),
                    "gc_stratum": _gc_stratum(record.sequence),
                }
            )
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in candidates:
        key = (row["length_stratum"], row["gc_stratum"], row["motif_length"])
        groups.setdefault(key, []).append(row)
    for key, rows in groups.items():
        rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{key}:{row['id']}".encode()).hexdigest())
    selected: list[dict[str, Any]] = []
    seen_sequences: set[str] = set()
    while groups and len(selected) < panel_size:
        for key in sorted(groups):
            rows = groups[key]
            while rows and rows[0]["sequence"] in seen_sequences:
                rows.pop(0)
            if rows:
                row = rows.pop(0)
                selected.append(row)
                seen_sequences.add(row["sequence"])
                if len(selected) == panel_size:
                    break
            if not rows:
                groups.pop(key, None)
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_sha256(path: Path) -> str:
    if path.is_file():
        return _sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(f"benchmark data does not exist: {path}")
    digest = hashlib.sha256()
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(child)))
    return digest.hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            newline="",
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_benchmark_artifact_directory(path: Path) -> dict[str, Any]:
    actual_names = {child.name for child in path.iterdir()}
    expected_names = set(ARTIFACT_NAMES)
    if actual_names != expected_names:
        raise ValueError(
            "benchmark artifact directory is incomplete: "
            f"expected={sorted(expected_names)}, actual={sorted(actual_names)}"
        )
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") not in {"completed", "failed"}:
        raise ValueError("benchmark manifest must have a terminal status")
    expected_hashes = manifest.get("artifact_sha256")
    hash_names = expected_names - {"run_manifest.json"}
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != hash_names:
        raise ValueError("benchmark manifest artifact hashes are incomplete")
    for name, expected in expected_hashes.items():
        actual = _sha256(path / name)
        if actual != expected:
            raise ValueError(f"benchmark artifact SHA-256 mismatch for {name}: expected={expected}, actual={actual}")
    return manifest


def _cleanup_staging_directory(staging: Path, parent: Path) -> None:
    resolved_parent = parent.resolve()
    resolved_staging = staging.resolve()
    if resolved_staging.parent != resolved_parent or not staging.name.startswith(".benchmark-staging-"):
        raise RuntimeError(f"refusing to clean unexpected staging directory: {staging}")
    for child in staging.iterdir():
        if not child.is_file():
            raise RuntimeError(f"unexpected nested path in benchmark staging directory: {child}")
        child.unlink()
    staging.rmdir()


def _publish_benchmark_run(
    output_dir: Path,
    artifact_content: dict[str, str],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Publish one complete six-file run with a single sibling-directory rename."""
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".benchmark-staging-", dir=parent))
    published = False
    try:
        for name, content in artifact_content.items():
            _atomic_write_text(staging / name, content)
        committed_manifest = dict(manifest)
        committed_manifest["artifact_sha256"] = {name: _sha256(staging / name) for name in artifact_content}
        _atomic_write_text(
            staging / "run_manifest.json",
            json.dumps(committed_manifest, indent=2, sort_keys=True) + "\n",
        )
        _validate_benchmark_artifact_directory(staging)
        _fsync_directory(staging)
        _require_new_or_empty_output_directory(output_dir)
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(staging, output_dir)
        published = True
        _fsync_directory(parent)
        return committed_manifest
    finally:
        if not published and staging.exists():
            _cleanup_staging_directory(staging, parent)


def _csv_text(rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _legal_motif_start(
    motif: str,
    length: int,
    rng: random.Random,
    geometry: BenchmarkGeometry,
) -> int:
    scaffold_length = length - len(motif)
    minimum_total = _minimum_benchmark_total_length(len(motif), geometry)
    if length < minimum_total or length > geometry.max_length:
        raise ValueError("benchmark target length violates the shared geometry contract")
    maximum_start = scaffold_length - geometry.min_flank_length
    if maximum_start < geometry.min_flank_length:
        raise ValueError("motif cannot fit the shared benchmark flank minimum")
    if geometry.short_flank_min is None:
        return rng.randint(geometry.min_flank_length, maximum_start)
    legal_starts = [
        start
        for start in range(geometry.min_flank_length, maximum_start + 1)
        if _matches_asymmetric_flank_ranges(start, scaffold_length - start, geometry)
    ]
    if not legal_starts:
        raise ValueError("benchmark target length has no legal asymmetric flank placement")
    return legal_starts[rng.randrange(len(legal_starts))]


def _legal_benchmark_flank_pairs(
    motif_length: int,
    geometry: BenchmarkGeometry,
) -> tuple[tuple[int, int], ...]:
    pairs = []
    maximum_flank = geometry.max_length - int(motif_length)
    for left_length in range(geometry.min_flank_length, maximum_flank + 1):
        for right_length in range(geometry.min_flank_length, maximum_flank - left_length + 1):
            if not is_legal_scaffold_geometry(
                left_length,
                right_length,
                motif_length,
                min_flank_length=geometry.min_flank_length,
                min_total_scaffold_length=geometry.min_total_scaffold_length,
                requested_max_length=geometry.max_length,
            ):
                continue
            if _matches_asymmetric_flank_ranges(left_length, right_length, geometry):
                pairs.append((left_length, right_length))
    if not pairs:
        raise ValueError("motif cannot fit the shared benchmark geometry contract")
    return tuple(pairs)


def _uniform(
    motif: str,
    length: int,
    rng: random.Random,
    geometry: BenchmarkGeometry | None = None,
    flank_pair: tuple[int, int] | None = None,
) -> tuple[str, int]:
    geometry = geometry or BenchmarkGeometry()
    if flank_pair is None:
        scaffold_length = length - len(motif)
        start = _legal_motif_start(motif, length, rng, geometry)
    else:
        start, right_length = map(int, flank_pair)
        scaffold_length = start + right_length
        if (
            length != scaffold_length + len(motif)
            or not is_legal_scaffold_geometry(
                start,
                right_length,
                len(motif),
                min_flank_length=geometry.min_flank_length,
                min_total_scaffold_length=geometry.min_total_scaffold_length,
                requested_max_length=geometry.max_length,
            )
            or not _matches_asymmetric_flank_ranges(start, right_length, geometry)
        ):
            raise ValueError("explicit flank pair violates the shared benchmark geometry contract")
    left = "".join(rng.choice(BASES) for _ in range(start))
    right = "".join(rng.choice(BASES) for _ in range(scaffold_length - start))
    return left + motif + right, start


def _markov(
    motif: str,
    length: int,
    prior: RnaTrainingPrior,
    rng: random.Random,
    order: int,
    geometry: BenchmarkGeometry | None = None,
    flank_pair: tuple[int, int] | None = None,
) -> tuple[str, int]:
    geometry = geometry or BenchmarkGeometry()
    if flank_pair is None:
        scaffold_length = length - len(motif)
        start = _legal_motif_start(motif, length, rng, geometry)
    else:
        start, right_length = map(int, flank_pair)
        scaffold_length = start + right_length
        if (
            length != scaffold_length + len(motif)
            or not is_legal_scaffold_geometry(
                start,
                right_length,
                len(motif),
                min_flank_length=geometry.min_flank_length,
                min_total_scaffold_length=geometry.min_total_scaffold_length,
                requested_max_length=geometry.max_length,
            )
            or not _matches_asymmetric_flank_ranges(start, right_length, geometry)
        ):
            raise ValueError("explicit flank pair violates the shared benchmark geometry contract")
    # Generate the left flank from the motif boundary outward using the
    # reverse-corpus prior, then restore conventional 5'->3' orientation.
    left_outward = prior.sample_sequence(
        start,
        rng,
        order=order,
        prefix=motif[::-1],
        direction="reverse",
    )
    left = left_outward[::-1]
    # The right flank continues the complete left+motif history, so the fixed
    # motif junction does not reset first- or second-order context.
    right = prior.sample_sequence(
        scaffold_length - start,
        rng,
        order=order,
        prefix=left + motif,
    )
    return left + motif + right, start


def _normalise_method_specs(config: dict[str, Any], smoke_test: bool) -> list[dict[str, Any]]:
    raw_methods: list[Any]
    if smoke_test:
        raw_methods = [
            {"name": "uniform", "kind": "uniform"},
            {"name": "markov1", "kind": "markov1"},
            {"name": "markov2", "kind": "markov2"},
        ]
    else:
        raw_methods = list(config.get("methods") or [])
    if not raw_methods:
        raise ValueError("benchmark requires at least one method")

    specs: list[dict[str, Any]] = []
    for raw_method in raw_methods:
        method = {"name": raw_method, "kind": raw_method} if isinstance(raw_method, str) else dict(raw_method)
        method.setdefault("kind", method.get("name"))
        method.setdefault("primary", False)
        if not method.get("name"):
            raise ValueError("every benchmark method requires a name")
        if method["kind"] not in {"uniform", "markov1", "markov2", "checkpoint"}:
            raise ValueError(f"unknown benchmark method kind: {method['kind']}")
        if method["kind"] == "checkpoint" and not (method.get("checkpoint") or method.get("training_manifest")):
            raise ValueError(f"checkpoint method {method['name']!r} requires checkpoint or training_manifest")
        if method["kind"] == "checkpoint" and not smoke_test and not config.get("fixture_mode", False):
            if not method.get("training_manifest"):
                raise ValueError(f"formal checkpoint method {method['name']!r} requires training_manifest")
            required_identity = (
                "expected_pretrained_kind",
                "expected_pretrained_mode",
                "expected_decoder_mode",
                "expected_denoise_steps",
            )
            missing = [field for field in required_identity if field not in method]
            if missing:
                raise ValueError(f"formal checkpoint method {method['name']!r} is missing identity fields: {missing}")
        specs.append(method)

    names = [str(method["name"]) for method in specs]
    if len(names) != len(set(names)):
        raise ValueError("benchmark method names must be unique")
    primary_methods = [method for method in specs if method["primary"]]
    if smoke_test or config.get("fixture_mode", False):
        if len(primary_methods) > 1:
            raise ValueError("at most one benchmark method may be primary")
    else:
        if len(primary_methods) != 1:
            raise ValueError("formal benchmark requires exactly one primary method")
        primary = primary_methods[0]
        if not (
            primary["kind"] == "checkpoint"
            and primary["expected_pretrained_kind"] == "rna_fm"
            and primary["expected_pretrained_mode"] == "lora"
            and primary["expected_decoder_mode"] == "iterative_remasking"
            and int(primary["expected_denoise_steps"]) > 1
        ):
            raise ValueError("formal primary must be exactly one LoRA iterative checkpoint method")
    return specs


def _checkpoint_from_training_manifest(
    method: dict[str, Any],
    *,
    expected_split_manifest: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(str(method["training_manifest"]))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"training manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("architecture_version") != 2:
        raise ValueError(f"training manifest is not a completed V2 run: {manifest_path}")
    producer_config = method.get("training_config")
    if producer_config:
        producer_path = Path(str(producer_config))
        if not producer_path.is_file():
            raise FileNotFoundError(f"training producer config does not exist: {producer_path}")
        expected_config_digest = _sha256(producer_path)
        actual_config_digest = manifest.get("config_sha256")
        if actual_config_digest != expected_config_digest:
            raise ValueError(
                "training manifest config SHA-256 mismatch: "
                f"expected={expected_config_digest}, actual={actual_config_digest}"
            )
    if expected_split_manifest is not None:
        actual_split_manifest = manifest.get("split_manifest")
        actual_split_json = json.dumps(actual_split_manifest, sort_keys=True, separators=(",", ":"))
        expected_split_json = json.dumps(expected_split_manifest, sort_keys=True, separators=(",", ":"))
        if actual_split_json != expected_split_json:
            raise ValueError("training manifest split manifest mismatch")
    best = manifest.get("best_checkpoint")
    if not isinstance(best, dict) or not best.get("path") or not best.get("sha256"):
        raise ValueError(f"training manifest best_checkpoint is incomplete: {manifest_path}")
    checkpoint = Path(str(best["path"]))
    if not checkpoint.is_absolute():
        checkpoint = manifest_path.parent / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"manifest best checkpoint does not exist: {checkpoint}")
    actual_digest = _sha256(checkpoint)
    if actual_digest != str(best["sha256"]):
        raise ValueError(
            f"training manifest checkpoint SHA-256 mismatch: expected={best['sha256']}, actual={actual_digest}"
        )
    manifest_identity = dict(manifest.get("method_identity") or {})
    expected_identity = {
        "pretrained_kind": method.get("expected_pretrained_kind"),
        "pretrained_mode": method.get("expected_pretrained_mode"),
    }
    mismatches = [
        f"{field}: expected={expected!r}, actual={manifest_identity.get(field)!r}"
        for field, expected in expected_identity.items()
        if expected is not None and manifest_identity.get(field) != expected
    ]
    if mismatches:
        raise ValueError("training manifest identity mismatch: " + "; ".join(mismatches))
    return checkpoint, manifest


def _actual_method_identity(method: dict[str, Any], loaded: object) -> dict[str, Any]:
    metadata = getattr(loaded, "pretrained_metadata", None)
    if not isinstance(metadata, dict):
        raise TypeError("actual checkpoint pretrained identity metadata is missing")
    actual_kind = str(metadata.get("kind", "none"))
    actual_mode = "none" if actual_kind == "none" else str(metadata.get("mode", ""))
    generation = dict(method.get("generation") or {})
    actual_steps = int(generation.get("denoise_steps", 12))
    actual_decoder = "single_pass" if actual_steps == 1 else "iterative_remasking"
    from rna_scaffold.generate import GenerationSettings, resolve_generation_policies

    policies = resolve_generation_policies(GenerationSettings(**generation), getattr(loaded, "model", None))
    return {
        "architecture_version": getattr(loaded, "architecture_version", None),
        "pretrained_kind": actual_kind,
        "pretrained_mode": actual_mode,
        "decoder_mode": actual_decoder,
        "denoise_steps": actual_steps,
        "remask_strategy": policies.remask_strategy,
        "self_conditioning": policies.self_conditioning,
    }


def _validate_method_identity(method: dict[str, Any], loaded: object) -> dict[str, Any]:
    """Bind a benchmark label to the checkpoint and decoder identity actually used."""
    actual = _actual_method_identity(method, loaded)
    expected = {
        "pretrained_kind": str(method.get("expected_pretrained_kind")),
        "pretrained_mode": str(method.get("expected_pretrained_mode")),
        "decoder_mode": str(method.get("expected_decoder_mode")),
        "denoise_steps": int(method.get("expected_denoise_steps", -1)),
    }
    mismatches = [
        f"{field}: expected={expected[field]!r}, actual={actual[field]!r}"
        for field in expected
        if expected[field] != actual[field]
    ]
    if actual["architecture_version"] != 2:
        mismatches.append(f"architecture_version: expected=2, actual={actual['architecture_version']!r}")
    if mismatches:
        raise ValueError("benchmark method identity mismatch: " + "; ".join(mismatches))
    return actual


def _normalise_motifs(config: dict[str, Any]) -> list[dict[str, Any]]:
    motifs = []
    for raw_row in config.get("motifs") or []:
        row = dict(raw_row)
        row["id"] = str(row["id"])
        row["sequence"] = str(row["sequence"]).strip().upper().replace("T", "U")
        motifs.append(row)
    if not motifs:
        raise ValueError("benchmark requires at least one motif")
    if len({row["id"] for row in motifs}) != len(motifs):
        raise ValueError("benchmark motif IDs must be unique")
    for row in motifs:
        if not validate_rna_sequence(row["sequence"]):
            raise ValueError(f"invalid benchmark motif {row['id']!r}")
    return motifs


def _resolve_motifs(
    config: dict[str, Any],
    *,
    smoke_test: bool,
    partitions: BenchmarkPartitions | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if smoke_test:
        smoke_config = {"motifs": config.get("smoke_motifs") or config.get("motifs")}
        motifs = _normalise_motifs(smoke_config)
        canonical = json.dumps(motifs, sort_keys=True, separators=(",", ":"))
        return motifs, {
            "source_partition": "smoke_fixture",
            "size": len(motifs),
            "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "grade": "exploratory_not_release_grade",
        }
    if config.get("fixture_mode", False):
        motifs = _normalise_motifs(config)
        canonical = json.dumps(motifs, sort_keys=True, separators=(",", ":"))
        return motifs, {
            "source_partition": "test_fixture",
            "size": len(motifs),
            "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "grade": "exploratory_not_release_grade",
        }
    if partitions is None:
        raise ValueError("formal motif panel requires reconstructed benchmark partitions")
    if config.get("motifs"):
        raise ValueError("formal benchmark motifs must be derived from motif_panel, not handwritten")
    panel_config = dict(config.get("motif_panel") or {})
    if panel_config.get("source_partition", "test") != "test":
        raise ValueError("formal motif_panel.source_partition must be 'test'")
    panel_size = int(panel_config.get("size", 0))
    motif_lengths = tuple(int(value) for value in panel_config.get("motif_lengths", (4, 8, 16, 32)))
    geometry = _benchmark_geometry(config)
    panel_min_flank = int(panel_config.get("min_flank_length", geometry.min_flank_length))
    if panel_min_flank != geometry.min_flank_length:
        raise ValueError("motif_panel.min_flank_length must match geometry.min_flank_length")
    motifs = build_held_out_motif_panel(
        partitions.test_records,
        panel_size=panel_size,
        seed=int(panel_config.get("seed", config.get("seed", 42))),
        motif_lengths=motif_lengths,
        min_flank_length=panel_min_flank,
        geometry=geometry,
    )
    if len(motifs) != panel_size:
        raise ValueError(
            "held-out test split cannot produce the requested unique motif panel: "
            f"requested={panel_size}, produced={len(motifs)}"
        )
    canonical = json.dumps(motifs, sort_keys=True, separators=(",", ":"))
    return motifs, {
        "source_partition": "test",
        "size": len(motifs),
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "config": panel_config,
        "motifs": motifs,
    }


def _motif_provenance(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "source_partition",
            "source_target_id",
            "source_sequence_sha256",
            "source_start",
            "source_length",
            "length_stratum",
            "gc_stratum",
        )
    }


def _configured_primary(config: dict[str, Any]) -> str | None:
    for raw_method in config.get("methods") or []:
        if isinstance(raw_method, dict) and raw_method.get("primary"):
            return str(raw_method.get("name"))
    return None


def _data_hashes(
    config: dict[str, Any],
    partitions: BenchmarkPartitions | None = None,
) -> dict[str, str]:
    paths: list[Path] = []
    if config.get("training_config"):
        paths.append(Path(str(config["training_config"])))
    if partitions is not None:
        paths.extend((partitions.training_data, partitions.cluster_manifest))
    for key in ("training_data", "cluster_manifest"):
        if config.get(key):
            paths.append(Path(str(config[key])))
    configured_data = config.get("data_files") or {}
    values = configured_data.values() if isinstance(configured_data, dict) else configured_data
    paths.extend(Path(str(value)) for value in values)
    return {str(path): _path_sha256(path) for path in dict.fromkeys(paths)}


def _load_fixture_training_sequences(config: dict[str, Any]) -> list[str]:
    """Load an explicit test fixture without reintroducing a runtime dataset API."""
    raw_path = config.get("training_data")
    if not raw_path:
        return []
    path = Path(str(raw_path))
    if path.suffix.lower() == ".csv":
        return [record.sequence for record in load_sequence_records(path, skip_invalid=False)]
    sequences: list[str] = []
    current: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                sequences.append("".join(current).upper().replace("T", "U"))
                current = []
            continue
        current.append(line)
    if current:
        sequences.append("".join(current).upper().replace("T", "U"))
    if not sequences or any(not validate_rna_sequence(sequence) for sequence in sequences):
        raise ValueError(f"invalid benchmark fixture training data: {path}")
    return sequences


def _candidate_metric(
    motif_id: str,
    motif: str,
    sequence: str,
    start: int | None,
    training_sequences: list[str],
    normalized_edit_diversity: float,
    kmer_diversity: float,
    runtime_seconds: float,
    peak_cuda_memory_bytes: int | None,
    failure: str | None,
    rejection_reason: str | None,
    rnafold: RnafoldResult | None,
    geometry: BenchmarkGeometry,
    similarity_index: TrainingSimilarityIndex | None = None,
) -> CandidateMetric:
    preserved = start is not None and sequence[start : start + len(motif)] == motif
    right_length = len(sequence) - start - len(motif) if start is not None else -1
    geometry_legal = (
        bool(sequence)
        and start is not None
        and is_legal_scaffold_geometry(
            start,
            right_length,
            len(motif),
            min_flank_length=geometry.min_flank_length,
            min_total_scaffold_length=geometry.min_total_scaffold_length,
            requested_max_length=geometry.max_length,
        )
        and _matches_asymmetric_flank_ranges(start, right_length, geometry)
    )
    valid = bool(sequence) and validate_rna_sequence(sequence) and preserved and geometry_legal and failure is None
    gc_fraction = (sequence.count("G") + sequence.count("C")) / len(sequence) if sequence else 0.0
    left_linker = sequence[:start] if sequence and start is not None and preserved else ""
    right_linker = sequence[start + len(motif) :] if sequence and start is not None and preserved else ""
    linker = left_linker + right_linker
    linker_gc_fraction = (linker.count("G") + linker.count("C")) / len(linker) if linker else 0.0
    if failure is None and not geometry_legal:
        failure = "illegal_geometry"
    elif failure is None and not valid:
        failure = "invalid_sequence"
    return CandidateMetric(
        motif_id=motif_id,
        sequence=sequence,
        valid=valid,
        motif_preserved=preserved,
        total_length=len(sequence),
        gc_fraction=gc_fraction,
        failure=failure,
        normalized_edit_diversity=normalized_edit_diversity,
        kmer_diversity=kmer_diversity,
        nearest_training_kmer_similarity=(
            (
                similarity_index.nearest(sequence)
                if similarity_index is not None
                else nearest_training_similarity(sequence, training_sequences, k=5)
            )
            if sequence
            else None
        ),
        max_homopolymer_run=maximum_homopolymer_run(sequence),
        runtime_seconds=runtime_seconds,
        peak_cuda_memory_bytes=peak_cuda_memory_bytes,
        rejection_reason=rejection_reason,
        rnafold_status=rnafold.status if rnafold else None,
        rnafold_dot_bracket=rnafold.dot_bracket if rnafold else None,
        rnafold_mfe_kcal_mol=rnafold.mfe_kcal_mol if rnafold else None,
        rnafold_paired_fraction=rnafold.paired_fraction if rnafold else None,
        rnafold_motif_paired_fraction=rnafold.motif_paired_fraction if rnafold else None,
        rnafold_runtime_seconds=rnafold.runtime_seconds if rnafold else None,
        rnafold_version=rnafold.version if rnafold else None,
        rnafold_error=rnafold.error if rnafold else None,
        linker_length=len(linker),
        linker_gc_fraction=linker_gc_fraction,
        linker_composition_entropy_bits=base_composition_entropy_bits(linker),
        linker_max_homopolymer_run=max(
            maximum_homopolymer_run(left_linker),
            maximum_homopolymer_run(right_linker),
        ),
    )


def _generate_method_motif(
    method: dict[str, Any],
    motif: str,
    length: int,
    budget: int,
    seed: int,
    prior: RnaTrainingPrior,
    geometry: BenchmarkGeometry,
    loaded_checkpoint: object | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], int | None]:
    kind = str(method["kind"])
    rng = random.Random(seed)
    if kind in {"uniform", "markov1", "markov2"}:
        generated = []
        legal_pairs = (
            list(_legal_benchmark_flank_pairs(len(motif), geometry)) if geometry.short_flank_min is not None else []
        )
        remaining_pairs: list[tuple[int, int]] = []
        for _ in range(budget):
            flank_pair = None
            candidate_length = length
            if legal_pairs:
                if not remaining_pairs:
                    remaining_pairs = legal_pairs.copy()
                    rng.shuffle(remaining_pairs)
                flank_pair = remaining_pairs.pop()
                candidate_length = flank_pair[0] + len(motif) + flank_pair[1]
            sequence, start = (
                _uniform(
                    motif,
                    candidate_length,
                    rng,
                    geometry,
                    flank_pair=flank_pair,
                )
                if kind == "uniform"
                else _markov(
                    motif,
                    candidate_length,
                    prior,
                    rng,
                    order=1 if kind == "markov1" else 2,
                    geometry=geometry,
                    flank_pair=flank_pair,
                )
            )
            generated.append({"sequence": sequence, "motif_start": start, "failure": None, "rejection_reason": None})
        return generated, {}, None

    from rna_scaffold.generate import GenerationSettings, generate_candidates

    generation = dict(method.get("generation") or {})
    generation.update(
        num_candidates=budget,
        seed=seed,
        max_length=geometry.max_length,
        min_scaffold_length=geometry.min_total_scaffold_length,
        min_flank_length=geometry.min_flank_length,
    )
    if geometry.short_flank_min is not None:
        shared_bounds = {
            "short_flank_min": geometry.short_flank_min,
            "short_flank_max": geometry.short_flank_max,
            "long_flank_min": geometry.long_flank_min,
            "long_flank_max": geometry.long_flank_max,
        }
        mismatches = [
            f"{name}: generation={generation[name]!r}, geometry={expected!r}"
            for name, expected in shared_bounds.items()
            if name in generation and int(generation[name]) != expected
        ]
        if mismatches:
            raise ValueError(
                "checkpoint generation flank bounds differ from benchmark geometry: " + "; ".join(mismatches)
            )
        generation.update(shared_bounds)
    device = str(method.get("device", "cpu"))
    using_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if using_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    learned_candidates = generate_candidates(
        method["checkpoint"],
        motif,
        GenerationSettings(**generation),
        device=device,
        loaded_checkpoint=loaded_checkpoint,
    )
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if using_cuda else None
    generated = [
        {
            "sequence": candidate.full_sequence,
            "motif_start": candidate.motif_start,
            "failure": None if candidate.valid else candidate.status,
            "rejection_reason": None,
        }
        for candidate in learned_candidates
    ]
    rejection_counts = dict(learned_candidates.audit.rejected)
    shortfall = budget - len(generated)
    if shortfall > 0:
        rejection_reason = (
            ";".join(f"{reason}:{count}" for reason, count in sorted(rejection_counts.items()))
            or "candidate_budget_exhausted"
        )
        generated.extend(
            {
                "sequence": "",
                "motif_start": None,
                "failure": "generation_shortfall",
                "rejection_reason": rejection_reason,
            }
            for _ in range(shortfall)
        )
    return generated, rejection_counts, peak_memory


def _rnafold_results(
    generated: list[dict[str, Any]],
    motif_length: int,
    config: dict[str, Any],
    budget: int,
) -> list[RnafoldResult | None]:
    rnafold_config = dict(config.get("rnafold") or {})
    if not rnafold_config.get("enabled", False):
        return [None] * len(generated)
    structure_budget = int(rnafold_config.get("candidate_budget_per_motif", budget))
    if structure_budget < 0:
        raise ValueError("rnafold candidate budget must be non-negative")
    results: list[RnafoldResult | None] = []
    used = 0
    for candidate in generated:
        start = candidate["motif_start"]
        if not candidate["sequence"] or start is None or used >= structure_budget:
            results.append(RnafoldResult("not_run", None, None, None, None, 0.0, None, None))
            continue
        results.append(
            run_rnafold(
                candidate["sequence"],
                motif_start=start,
                motif_end=start + motif_length,
                executable=rnafold_config.get("executable", "RNAfold"),
                timeout_seconds=float(rnafold_config.get("timeout_seconds", 30.0)),
            )
        )
        used += 1
    return results


def _rnafold_run_status(config: dict[str, Any], candidate_rows: list[dict[str, Any]]) -> str:
    if not (config.get("rnafold") or {}).get("enabled", False):
        return "disabled"
    attempted = [
        str(row["rnafold_status"]) for row in candidate_rows if row.get("rnafold_status") not in {None, "not_run"}
    ]
    if not attempted:
        return "not_run"
    if all(status == "ok" for status in attempted):
        return "ok"
    if all(status == "unavailable" for status in attempted):
        return "unavailable"
    if any(status == "ok" for status in attempted):
        return "partial"
    return "failed"


def _paired_bootstrap_artifact(
    motif_summaries: list[dict[str, Any]],
    methods: list[str],
    seed: int,
    samples: int,
    min_common_motifs_for_release: int = 20,
) -> dict[str, Any]:
    if min_common_motifs_for_release <= 0:
        raise ValueError("min_common_motifs_for_release must be positive")
    comparison_rows: list[dict[str, Any]] = []
    by_method = {
        method: {
            row["motif_id"]: row
            for row in motif_summaries
            if row["method"] == method and row["status"] in {"ok", "partial"}
        }
        for method in methods
    }
    for left_method, right_method in combinations(methods, 2):
        for metric in BOOTSTRAP_METRICS:
            left_values = {
                motif_id: float(row[metric])
                for motif_id, row in by_method[left_method].items()
                if row.get(metric) is not None
            }
            right_values = {
                motif_id: float(row[metric])
                for motif_id, row in by_method[right_method].items()
                if row.get(metric) is not None
            }
            if not set(left_values) & set(right_values):
                continue
            common_ids, difference = paired_bootstrap_by_motif(
                left_values,
                right_values,
                seed=seed,
                samples=samples,
            )
            comparison_rows.append(
                {
                    "left_method": left_method,
                    "right_method": right_method,
                    "metric": metric,
                    "common_motif_ids": list(common_ids),
                    "common_motif_n": len(common_ids),
                    "grade": (
                        "release_grade_sample_size"
                        if len(common_ids) >= min_common_motifs_for_release
                        else "exploratory_not_release_grade"
                    ),
                    **asdict(difference),
                }
            )
    return {
        "seed": seed,
        "samples": samples,
        "min_common_motifs_for_release": min_common_motifs_for_release,
        "comparisons": comparison_rows,
    }


def _report(
    model_summaries: list[dict[str, Any]],
    primary_name: str | None,
    budget: int,
    seed: int,
) -> str:
    primary_line = f"Primary model: `{primary_name}`" if primary_name else "Primary model: none configured"
    lines = [
        "# RNA Scaffold Benchmark",
        "",
        primary_line,
        "",
        f"Candidate budget per motif: {budget}; deterministic seed: {seed}.",
        "",
        ("RNAfold-derived MFE and pairing metrics are a plausibility proxy only; they are not proof of function."),
        "",
        "| Method | Primary | Status | Valid rate | Unique rate | Edit diversity | Failures | Runtime (s) | Error |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in model_summaries:
        method = f"**{row['method']}**" if row["primary"] else row["method"]
        lines.append(
            "| {method} | {primary} | {status} | {valid:.4f} | {unique:.4f} | "
            "{diversity:.4f} | {failures} | {runtime:.4f} | {error} |".format(
                method=method,
                primary="yes" if row["primary"] else "no",
                status=row["status"],
                valid=row["valid_rate"],
                unique=row["unique_rate"],
                diversity=row["mean_normalized_edit_diversity"],
                failures=row["failure_count"],
                runtime=row["runtime_seconds"],
                error=row["error_reason"] or "",
            )
        )
    return "\n".join(lines) + "\n"


def _software_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pyyaml": yaml.__version__,
    }
    for key, distribution in (("lightning", "lightning"), ("rna_fm", "rna-fm")):
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "unavailable"
    return versions


def _require_new_or_empty_output_directory(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ValueError(f"benchmark output path is not a directory: {output_dir}")
    if next(output_dir.iterdir(), None) is not None:
        raise ValueError(f"benchmark output directory must be empty; refusing to remove existing files: {output_dir}")


def validate_benchmark_config(config_path: str | Path) -> dict[str, Any]:
    """Validate the formal contract without pretending external runs exist."""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("benchmark configuration must be a mapping")
    if int(config.get("candidate_count", 0)) <= 0:
        raise ValueError("benchmark candidate_count must be positive")
    if int(config.get("bootstrap_samples", 0)) <= 0:
        raise ValueError("benchmark bootstrap_samples must be positive")
    geometry = _benchmark_geometry(config)
    methods = _normalise_method_specs(config, smoke_test=False)
    panel = dict(config.get("motif_panel") or {})
    if panel.get("source_partition") != "test" or int(panel.get("size", 0)) <= 0:
        raise ValueError("formal motif_panel requires a positive held-out test panel")
    panel_min_flank = int(panel.get("min_flank_length", geometry.min_flank_length))
    if panel_min_flank != geometry.min_flank_length:
        raise ValueError("motif_panel.min_flank_length must match geometry.min_flank_length")
    if config.get("motifs"):
        raise ValueError("formal benchmark may not contain handwritten motifs")
    _normalise_motifs({"motifs": config.get("smoke_motifs")})
    training_data, cluster_manifest, _ = _split_source_config(config)
    prerequisites = [training_data, cluster_manifest]
    for method in methods:
        if method["kind"] != "checkpoint":
            continue
        producer_path = Path(str(method.get("training_config", "")))
        if not producer_path.is_file():
            raise ValueError(f"formal method {method['name']!r} requires an existing training_config producer")
        from train import validate_training_config

        producer = yaml.safe_load(producer_path.read_text(encoding="utf-8"))
        validate_training_config(producer)
        generation = dict(method.get("generation") or {})
        if (
            not bool(producer["model"].get("predict_flank_lengths", True))
            and generation.get("length_sampling", "model") != "uniform"
        ):
            raise ValueError(
                f"benchmark method {method['name']!r} disables length heads and requires "
                "generation.length_sampling='uniform'"
            )
        if geometry.short_flank_min is not None:
            for field in (
                "short_flank_min",
                "short_flank_max",
                "long_flank_min",
                "long_flank_max",
            ):
                configured = generation.get(field, getattr(geometry, field))
                if configured is None or int(configured) != getattr(geometry, field):
                    raise ValueError(
                        f"benchmark method {method['name']!r} generation.{field} must match geometry.{field}"
                    )
        expected_manifest = Path(str(producer["trainer"]["checkpoint_dir"])) / "training_manifest.json"
        configured_manifest = Path(str(method["training_manifest"]))
        if configured_manifest != expected_manifest:
            raise ValueError(
                f"benchmark method {method['name']!r} does not consume its producer manifest: "
                f"expected={expected_manifest}, configured={configured_manifest}"
            )
        pretrained = dict(producer["model"].get("pretrained") or {})
        producer_kind = str(pretrained.get("kind", "none"))
        producer_mode = "none" if producer_kind == "none" else str(pretrained.get("mode", "frozen"))
        if method["expected_pretrained_kind"] != producer_kind or method["expected_pretrained_mode"] != producer_mode:
            raise ValueError(f"benchmark method {method['name']!r} identity differs from producer config")
        prerequisites.append(configured_manifest)
        if pretrained.get("checkpoint"):
            prerequisites.append(Path(str(pretrained["checkpoint"])))
    return {
        "status": "validated_external_inputs_pending",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "geometry": asdict(geometry),
        "motif_panel": panel,
        "methods": [str(method["name"]) for method in methods],
        "external_prerequisites": {
            str(path): "available" if path.is_file() else "pending" for path in dict.fromkeys(prerequisites)
        },
        "scientific_release_status": "not_run",
    }


def _progress(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _atomic_progress(path: Path, payload: dict) -> None:
    content = json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False)
    envelope = {"sha256": hashlib.sha256(content.encode()).hexdigest(), "payload": payload}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(envelope, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_progress(path: Path) -> dict:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    content = json.dumps(envelope["payload"], sort_keys=True, ensure_ascii=True, allow_nan=False)
    if hashlib.sha256(content.encode()).hexdigest() != envelope["sha256"]:
        raise ValueError(f"progress checksum mismatch: {path}")
    return envelope["payload"]


def run_benchmark(
    config_path: Path,
    output_dir: Path,
    smoke_test: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    _require_new_or_empty_output_directory(output_dir)
    _progress("Preparing benchmark: validating data, split and checkpoint hashes")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("benchmark configuration must be a mapping")
    seed = int(config.get("seed", 42))
    budget = 4 if smoke_test else int(config.get("candidate_count", 16))
    if budget <= 0:
        raise ValueError("candidate_count must be positive")
    bootstrap_samples = 100 if smoke_test else int(config.get("bootstrap_samples", 10000))
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    geometry = _benchmark_geometry(config)
    method_specs = _normalise_method_specs(config, smoke_test)
    fixture_mode = bool(config.get("fixture_mode", False))
    if not smoke_test and not fixture_mode:
        validate_benchmark_config(config_path)
    partitions = None if smoke_test or fixture_mode else _load_benchmark_partitions(config)
    motifs, motif_panel = _resolve_motifs(
        config,
        smoke_test=smoke_test,
        partitions=partitions,
    )
    primary_name = _configured_primary(config)
    if not smoke_test:
        primary_name = next(
            (str(method["name"]) for method in method_specs if method["primary"]),
            primary_name,
        )
    data_sha256 = {} if smoke_test else _data_hashes(config, partitions)
    checkpoint_sha256: dict[str, str | None] = {}
    for method in method_specs:
        if method["kind"] != "checkpoint":
            continue
        if method.get("training_manifest"):
            try:
                checkpoint, training_manifest = _checkpoint_from_training_manifest(
                    method,
                    expected_split_manifest=(asdict(partitions.manifest) if partitions is not None else None),
                )
            except Exception as error:  # noqa: BLE001 - converted to a terminal method result below
                method["_checkpoint_resolution_error"] = f"{type(error).__name__}: {error}"
                method["_checkpoint_resolution_status"] = (
                    "unavailable" if isinstance(error, FileNotFoundError) else "failed"
                )
                checkpoint_sha256[str(method["name"])] = None
                continue
            method["checkpoint"] = str(checkpoint)
            method["_training_manifest"] = training_manifest
        checkpoint = Path(str(method["checkpoint"]))
        checkpoint_sha256[str(method["name"])] = _sha256(checkpoint) if checkpoint.is_file() else None

    if partitions is not None:
        training_sequences = [record.sequence for record in partitions.train_records]
    elif fixture_mode:
        training_sequences = _load_fixture_training_sequences(config)
    else:
        training_sequences = []
    prior = RnaTrainingPrior.from_sequences(training_sequences) if training_sequences else RnaTrainingPrior.empty()
    similarity_index = TrainingSimilarityIndex(training_sequences)
    # Sidecar stays separate from the six immutable final release artifacts.
    progress_dir = output_dir.with_name(output_dir.name + ".progress")
    progress_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parent
    source_paths = [Path(__file__).resolve(), *sorted((source_root / "rna_scaffold").rglob("*.py"))]
    binding = {
        "schema": 1,
        "config_sha256": _sha256(config_path),
        "data_sha256": data_sha256,
        "training_sequences_sha256": hashlib.sha256(json.dumps(training_sequences).encode()).hexdigest(),
        "checkpoint_sha256": checkpoint_sha256,
        "smoke_test": smoke_test,
        "motifs": motifs,
        "code_sha256": {str(path.relative_to(source_root)): _sha256(path) for path in source_paths},
        "software_versions": _software_versions(),
        "training_manifests": {
            str(method["name"]): _sha256(Path(str(method["training_manifest"])))
            for method in method_specs
            if method.get("training_manifest") and Path(str(method["training_manifest"])).is_file()
        },
    }
    binding_path = progress_dir / "binding.json"
    if binding_path.exists():
        if not resume:
            raise ValueError(f"progress already exists: {progress_dir}; use --resume or a new output directory")
        if _read_progress(binding_path) != binding:
            raise ValueError("resume input/code/software mismatch; use a new output directory")
    else:
        if resume:
            raise ValueError(f"no resumable progress exists: {progress_dir}")
        _atomic_progress(binding_path, binding)
    resumed_motifs = 0
    _progress(
        f"Ready: {len(method_specs)} methods x {len(motifs)} motifs x {budget} candidates; progress={progress_dir}"
    )

    started = time.perf_counter()
    candidate_rows: list[dict[str, Any]] = []
    motif_summaries: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    manifest_methods: list[dict[str, Any]] = []

    for method in method_specs:
        method_name = str(method["name"])
        method_kind = str(method["kind"])
        _progress(f"METHOD {method_name} ({method_kind}) starting")
        method_started = time.perf_counter()
        setup_runtime = 0.0
        generation_runtime_total = 0.0
        validation_runtime_total = 0.0
        method_status = "ok"
        method_error: str | None = None
        loaded_checkpoint: object | None = None
        actual_identity: dict[str, Any] | None = None
        method_metrics: list[CandidateMetric] = []
        method_candidate_rows: list[dict[str, Any]] = []
        method_motif_summaries: list[dict[str, Any]] = []
        method_rejections: Counter[str] = Counter()
        motif_seeds = {motif_row["id"]: seed + motif_index * 1000 for motif_index, motif_row in enumerate(motifs)}
        method_partial = False

        if method_kind == "checkpoint":
            setup_started = time.perf_counter()
            if method.get("_checkpoint_resolution_error"):
                method_status = str(method["_checkpoint_resolution_status"])
                method_error = str(method["_checkpoint_resolution_error"])
            else:
                checkpoint = Path(str(method["checkpoint"]))
            if method_status == "ok" and not checkpoint.is_file():
                method_status = "unavailable"
                method_error = f"checkpoint for benchmark method {method_name!r} does not exist: {checkpoint}"
            elif method_status == "ok":
                try:
                    loaded_checkpoint = load_scaffold_checkpoint(
                        str(checkpoint),
                        device=str(method.get("device", "cpu")),
                    )
                    identity_fields = {
                        "expected_pretrained_kind",
                        "expected_pretrained_mode",
                        "expected_decoder_mode",
                        "expected_denoise_steps",
                    }
                    if identity_fields <= method.keys():
                        actual_identity = _actual_method_identity(method, loaded_checkpoint)
                        _validate_method_identity(method, loaded_checkpoint)
                # A third-party/deserialization loader can raise several exception
                # types; every ordinary failure must become a terminal method row.
                except Exception as error:  # noqa: BLE001
                    method_status = "failed"
                    method_error = f"{type(error).__name__}: {error}"
            setup_runtime = time.perf_counter() - setup_started

        if method_status == "ok":
            try:
                for motif_index, motif_row in enumerate(motifs):
                    motif_id = motif_row["id"]
                    motif = motif_row["sequence"]
                    motif_seed = motif_seeds[motif_id]
                    label = f"{method_name} motif {motif_index + 1}/{len(motifs)} ({motif_id})"
                    cache_key = hashlib.sha256(json.dumps([method_name, motif_id]).encode()).hexdigest()
                    cache_path = progress_dir / f"{cache_key}.json"
                    if resume and cache_path.exists():
                        cached = _read_progress(cache_path)
                        if cached["method"] != method_name or cached["motif_id"] != motif_id:
                            raise ValueError("progress motif identity mismatch")
                        method_metrics.extend(CandidateMetric(**row) for row in cached["metrics"])
                        method_candidate_rows.extend(cached["candidate_rows"])
                        method_motif_summaries.append(cached["summary"])
                        method_rejections.update(cached["rejections"])
                        method_partial = method_partial or cached["partial"]
                        generation_runtime_total += cached["generation_runtime"]
                        validation_runtime_total += cached["validation_runtime"]
                        resumed_motifs += 1
                        _progress(f"{label}: RESUMED (saved timings reused)")
                        continue
                    _progress(f"{label}: generating {budget} candidates")
                    motif_started = time.perf_counter()
                    candidate_start = len(method_candidate_rows)
                    length = max(
                        geometry.preferred_total_scaffold_length,
                        _minimum_benchmark_total_length(len(motif), geometry),
                    )
                    if length > geometry.max_length:
                        raise ValueError(f"motif {motif_id!r} cannot fit the benchmark geometry contract")
                    generation_started = time.perf_counter()
                    generated, rejection_counts, peak_memory = _generate_method_motif(
                        method,
                        motif,
                        length,
                        budget,
                        motif_seed,
                        prior,
                        geometry,
                        loaded_checkpoint=loaded_checkpoint,
                    )
                    generation_runtime = time.perf_counter() - generation_started
                    generation_runtime_total += generation_runtime
                    _progress(f"{label}: generation done in {generation_runtime:.2f}s; computing diversity")
                    method_rejections.update(rejection_counts)
                    motif_partial = any(row["failure"] == "generation_shortfall" for row in generated)
                    method_partial = method_partial or motif_partial
                    actual_sequences = [row["sequence"] for row in generated if row["sequence"]]
                    edit_scores, kmer_scores = within_group_diversity(actual_sequences, k=5)
                    _progress(f"{label}: diversity done; computing training similarity and validation")
                    actual_index = 0
                    validation_started = time.perf_counter()
                    rnafold_results = _rnafold_results(generated, len(motif), config, budget)
                    validation_runtime = time.perf_counter() - validation_started
                    validation_runtime_total += validation_runtime
                    motif_metrics: list[CandidateMetric] = []
                    per_candidate_runtime = generation_runtime / budget
                    for candidate_index, (generated_row, rnafold_result) in enumerate(zip(generated, rnafold_results)):
                        has_sequence = bool(generated_row["sequence"])
                        edit_diversity = edit_scores[actual_index] if has_sequence else 0.0
                        kmer_diversity = kmer_scores[actual_index] if has_sequence else 0.0
                        if has_sequence:
                            actual_index += 1
                        metric = _candidate_metric(
                            motif_id,
                            motif,
                            generated_row["sequence"],
                            generated_row["motif_start"],
                            training_sequences,
                            edit_diversity,
                            kmer_diversity,
                            per_candidate_runtime,
                            peak_memory,
                            generated_row["failure"],
                            generated_row["rejection_reason"],
                            rnafold_result,
                            geometry,
                            similarity_index=similarity_index,
                        )
                        method_metrics.append(metric)
                        motif_metrics.append(metric)
                        method_candidate_rows.append(
                            {
                                "method": method_name,
                                "primary": bool(method["primary"]),
                                "motif_id": motif_id,
                                **_motif_provenance(motif_row),
                                "candidate_index": candidate_index,
                                "motif_start": generated_row["motif_start"],
                                "motif_end": (
                                    generated_row["motif_start"] + len(motif)
                                    if generated_row["motif_start"] is not None
                                    else None
                                ),
                                "seed": motif_seed,
                                **asdict(metric),
                            }
                        )
                    method_motif_summaries.append(
                        {
                            "method": method_name,
                            "primary": bool(method["primary"]),
                            "motif_id": motif_id,
                            **_motif_provenance(motif_row),
                            "status": "partial" if motif_partial else "ok",
                            "error_reason": None,
                            "seed": motif_seed,
                            "rejection_reasons": json.dumps(rejection_counts, sort_keys=True),
                            **asdict(summarize_candidates(motif_metrics)),
                        }
                    )
                    _atomic_progress(
                        cache_path,
                        {
                            "method": method_name,
                            "motif_id": motif_id,
                            "metrics": [asdict(metric) for metric in motif_metrics],
                            "candidate_rows": method_candidate_rows[candidate_start:],
                            "summary": method_motif_summaries[-1],
                            "rejections": dict(rejection_counts),
                            "partial": motif_partial,
                            "generation_runtime": generation_runtime,
                            "validation_runtime": validation_runtime,
                        },
                    )
                    _progress(f"{label}: SAVED; elapsed={time.perf_counter() - motif_started:.2f}s")
            except Exception as error:
                if method_kind != "checkpoint":
                    raise
                method_status = "failed"
                method_error = f"{type(error).__name__}: {error}"
                method_metrics.clear()
                method_candidate_rows.clear()
                method_motif_summaries.clear()
                method_rejections.clear()

        if method_status in {"unavailable", "failed"}:
            empty_summary = asdict(summarize_candidates([]))
            method_motif_summaries = [
                {
                    "method": method_name,
                    "primary": bool(method["primary"]),
                    "motif_id": motif_row["id"],
                    **_motif_provenance(motif_row),
                    "status": method_status,
                    "error_reason": method_error,
                    "seed": motif_seeds[motif_row["id"]],
                    "rejection_reasons": "{}",
                    **empty_summary,
                }
                for motif_row in motifs
            ]
        elif method_partial:
            method_status = "partial"

        candidate_rows.extend(method_candidate_rows)
        motif_summaries.extend(method_motif_summaries)
        method_runtime = time.perf_counter() - method_started
        method_summary = {
            "method": method_name,
            "kind": method_kind,
            "primary": bool(method["primary"]),
            "status": method_status,
            "error_reason": method_error,
            "seed": seed,
            "actual_pretrained_kind": (actual_identity.get("pretrained_kind") if actual_identity else None),
            "actual_pretrained_mode": (actual_identity.get("pretrained_mode") if actual_identity else None),
            "decoder_mode": actual_identity.get("decoder_mode") if actual_identity else None,
            "denoise_steps": actual_identity.get("denoise_steps") if actual_identity else None,
            "rejection_reasons": json.dumps(dict(sorted(method_rejections.items())), sort_keys=True),
            **asdict(summarize_candidates(method_metrics)),
            "setup_runtime_seconds": setup_runtime,
            "generation_runtime_seconds": generation_runtime_total,
            "validation_runtime_seconds": validation_runtime_total,
        }
        method_summary["runtime_seconds"] = method_runtime
        model_summaries.append(method_summary)
        manifest_methods.append(
            {
                "name": method_name,
                "kind": method_kind,
                "primary": bool(method["primary"]),
                "status": method_status,
                "error_reason": method_error,
                "seed": seed,
                "motif_seeds": motif_seeds,
                "runtime_seconds": method_runtime,
                "setup_runtime_seconds": setup_runtime,
                "generation_runtime_seconds": generation_runtime_total,
                "validation_runtime_seconds": validation_runtime_total,
                "checkpoint_sha256": checkpoint_sha256.get(method_name),
                "expected_identity": (
                    {
                        "pretrained_kind": method.get("expected_pretrained_kind"),
                        "pretrained_mode": method.get("expected_pretrained_mode"),
                        "decoder_mode": method.get("expected_decoder_mode"),
                        "denoise_steps": method.get("expected_denoise_steps"),
                    }
                    if method_kind == "checkpoint"
                    else None
                ),
                "actual_identity": actual_identity,
                "training_manifest_sha256": (
                    _sha256(Path(str(method["training_manifest"])))
                    if method.get("training_manifest") and Path(str(method["training_manifest"])).is_file()
                    else None
                ),
                "rejection_reasons": dict(sorted(method_rejections.items())),
            }
        )

    _progress("All methods processed; computing paired bootstrap and publishing reports")
    paired_artifact = _paired_bootstrap_artifact(
        motif_summaries,
        [str(method["name"]) for method in method_specs],
        seed,
        bootstrap_samples,
        min_common_motifs_for_release=int((config.get("motif_panel") or {}).get("min_common_motifs_for_release", 20)),
    )
    report = _report(model_summaries, primary_name, budget, seed)
    comparison_counts = sorted({int(row["common_motif_n"]) for row in paired_artifact["comparisons"]})
    comparison_grades = sorted({str(row["grade"]) for row in paired_artifact["comparisons"]})
    report += (
        "\n## Statistical evidence grade\n\n"
        f"Common motif N values: {comparison_counts or [0]}. "
        f"Grades: {comparison_grades or ['exploratory_not_release_grade']}.\n"
    )

    artifact_content = {
        "candidate_results.csv": _csv_text(candidate_rows, CANDIDATE_RESULT_FIELDS),
        "motif_summary.csv": _csv_text(motif_summaries, MOTIF_SUMMARY_FIELDS),
        "model_summary.csv": _csv_text(model_summaries, MODEL_SUMMARY_FIELDS),
        "paired_bootstrap.json": json.dumps(paired_artifact, indent=2, sort_keys=True) + "\n",
        "benchmark_report.md": report,
    }
    runtime_seconds = time.perf_counter() - started
    software_versions = _software_versions()
    rnafold_status = _rnafold_run_status(config, candidate_rows)
    rnafold_versions = sorted({str(row["rnafold_version"]) for row in candidate_rows if row.get("rnafold_version")})
    if rnafold_versions:
        software_versions["rnafold"] = ", ".join(rnafold_versions)
    elif (config.get("rnafold") or {}).get("enabled", False):
        software_versions["rnafold"] = rnafold_status
    overall_status = (
        "failed" if any(method["status"] in {"unavailable", "failed"} for method in manifest_methods) else "completed"
    )
    manifest = {
        "status": overall_status,
        "configuration": config,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "data_sha256": data_sha256,
        "split_manifest": asdict(partitions.manifest) if partitions is not None else None,
        "training_reference": {
            "partition": "train" if partitions is not None else "none",
            "record_count": len(training_sequences),
            "target_ids_sha256": (
                hashlib.sha256(
                    "\n".join(sorted(record.target_id for record in partitions.train_records)).encode()
                ).hexdigest()
                if partitions is not None
                else None
            ),
        },
        "motif_panel": motif_panel,
        "geometry": asdict(geometry),
        "checkpoint_sha256": checkpoint_sha256,
        "seed": seed,
        "candidate_budget_per_motif": budget,
        "rnafold_status": rnafold_status,
        "rnafold_candidate_budget_per_motif": int(
            (config.get("rnafold") or {}).get("candidate_budget_per_motif", budget)
        ),
        "bootstrap_samples": bootstrap_samples,
        "smoke_test": smoke_test,
        "scientific_release_status": "not_release_grade" if smoke_test else "pending_external_review",
        "command": list(sys.argv),
        "software_versions": software_versions,
        "methods": manifest_methods,
        "runtime_seconds": runtime_seconds,
        "resumed_motifs": resumed_motifs,
        "timing_note": (
            "Current invocation wall time; resumed motif generation/validation timings originate from prior invocations."
            if resumed_motifs
            else "All timings from this invocation."
        ),
    }
    published = _publish_benchmark_run(output_dir, artifact_content, manifest)
    _progress(f"Benchmark {published['status']}: {output_dir}")
    return published


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RNA scaffold generators fairly.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="Reuse verified completed motifs for this output directory"
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if args.dry_run:
        print(json.dumps(validate_benchmark_config(config_path), indent=2, sort_keys=True))
        return
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/scaffold_benchmark"))
    manifest = run_benchmark(config_path, output_dir, smoke_test=args.smoke_test, resume=args.resume)
    if manifest["status"] == "failed":
        for method in manifest["methods"]:
            if method["status"] in {"unavailable", "failed"}:
                print(
                    f"{method['name']}: {method['status']}: {method['error_reason']}",
                    file=sys.stderr,
                )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
