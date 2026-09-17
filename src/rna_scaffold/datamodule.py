from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    import lightning.pytorch as L
except ImportError:  # pragma: no cover
    import pytorch_lightning as L

from rna_scaffold.data import (
    _asymmetric_flank_pairs,
    _normalize_preferred_motifs,
    build_partitioned_denoising_datasets,
)
from rna_scaffold.geometry import validate_scaffold_length_limit
from rna_scaffold.records import load_sequence_records
from rna_scaffold.splits import load_cluster_assignments
from rna_scaffold.tokenizer import RnaTokenizer


class RnaScaffoldDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer: RnaTokenizer,
        train_data: str | None = None,
        cluster_manifest: str | None = None,
        task: str = "motif_denoising",
        min_motif_length: int = 4,
        max_motif_length: int | None = None,
        motif_length_buckets: list[list[float]] | None = None,
        min_flank_length: int = 2,
        min_total_scaffold_length: int = 8,
        preferred_total_scaffold_length: int = 24,
        full_mask_probability: float = 0.35,
        span_mask_probability: float = 0.35,
        min_random_mask_fraction: float = 0.30,
        max_random_mask_fraction: float = 0.80,
        mean_span_length: int = 8,
        iterative_mask_steps: int | None = None,
        context_noise_probability: float = 0.0,
        validation_full_mask: bool = False,
        short_flank_min: int | None = None,
        short_flank_max: int | None = None,
        long_flank_min: int | None = None,
        long_flank_max: int | None = None,
        max_target_length: int = 256,
        batch_size: int = 64,
        num_workers: int = 4,
        val_fraction: float = 0.05,
        test_fraction: float = 0.1,
        expected_partition_counts: dict[str, int] | None = None,
        expected_cluster_count: int | None = None,
        cluster_thresholds: dict[str, float] | None = None,
        seed: int = 42,
        train_views_per_record: int = 1,
        validation_views_per_record: int = 1,
        test_views_per_record: int = 1,
        cluster_balanced_sampling: bool = False,
        preferred_motifs: tuple[str, ...] | list[str] | str | None = None,
        preferred_motif_probability: float = 0.0,
        iterative_full_mask_probability: float | None = None,
        stage_aware_context_noise: bool = False,
    ) -> None:
        super().__init__()
        if task != "motif_denoising":
            raise ValueError("task must be 'motif_denoising' in Scaffold Generator V2")
        asymmetric_geometry_requested = any(
            value is not None for value in (short_flank_min, short_flank_max, long_flank_min, long_flank_max)
        )
        if not asymmetric_geometry_requested and preferred_total_scaffold_length < min_total_scaffold_length:
            raise ValueError("preferred_total_scaffold_length must be at least min_total_scaffold_length")
        self.train_data = Path(train_data or "")
        if not str(self.train_data):
            raise ValueError("train_data must be provided")
        self.cluster_manifest = Path(cluster_manifest) if cluster_manifest else None
        if self.cluster_manifest is None:
            raise ValueError("cluster_manifest must be provided for motif_denoising training")
        self.tokenizer = tokenizer
        self.task = task
        self.min_motif_length = min_motif_length
        self.max_motif_length = max_motif_length
        self.motif_length_buckets = motif_length_buckets
        self.min_flank_length = min_flank_length
        self.min_total_scaffold_length = min_total_scaffold_length
        self.preferred_total_scaffold_length = preferred_total_scaffold_length
        self.full_mask_probability = full_mask_probability
        self.span_mask_probability = span_mask_probability
        self.min_random_mask_fraction = min_random_mask_fraction
        self.max_random_mask_fraction = max_random_mask_fraction
        self.mean_span_length = mean_span_length
        if iterative_mask_steps is not None and iterative_mask_steps < 2:
            raise ValueError("iterative_mask_steps must be at least two")
        if iterative_mask_steps is not None and (full_mask_probability != 0 or span_mask_probability != 0):
            raise ValueError(
                "full_mask_probability and span_mask_probability must be zero when iterative_mask_steps is enabled"
            )
        if not 0 <= context_noise_probability < 1:
            raise ValueError("context_noise_probability must be in [0, 1)")
        if iterative_full_mask_probability is not None:
            if iterative_mask_steps is None:
                raise ValueError("iterative_full_mask_probability requires iterative_mask_steps")
            if not 0 <= iterative_full_mask_probability <= 1:
                raise ValueError("iterative_full_mask_probability must be in [0, 1]")
        views_by_partition = {
            "train_views_per_record": train_views_per_record,
            "validation_views_per_record": validation_views_per_record,
            "test_views_per_record": test_views_per_record,
        }
        for name, value in views_by_partition.items():
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= preferred_motif_probability <= 1:
            raise ValueError("preferred_motif_probability must be in [0, 1]")
        normalized_preferred_motifs = _normalize_preferred_motifs(preferred_motifs)
        if preferred_motif_probability and not normalized_preferred_motifs:
            raise ValueError("preferred_motifs must be provided when preferred_motif_probability is positive")
        if not isinstance(cluster_balanced_sampling, bool):
            raise TypeError("cluster_balanced_sampling must be a boolean")
        if not isinstance(stage_aware_context_noise, bool):
            raise TypeError("stage_aware_context_noise must be a boolean")
        self.iterative_mask_steps = iterative_mask_steps
        self.context_noise_probability = context_noise_probability
        self.validation_full_mask = validation_full_mask
        self.short_flank_min = short_flank_min
        self.short_flank_max = short_flank_max
        self.long_flank_min = long_flank_min
        self.long_flank_max = long_flank_max
        self.max_target_length = validate_scaffold_length_limit(
            max_target_length,
            field="max_target_length",
        )
        flank_pairs = _asymmetric_flank_pairs(
            short_flank_min,
            short_flank_max,
            long_flank_min,
            long_flank_max,
            min_flank_length=min_flank_length,
        )
        self.uses_asymmetric_geometry = bool(flank_pairs)
        if flank_pairs:
            motif_upper = min(
                int(max_motif_length or self.max_target_length),
                self.max_target_length,
            )
            jointly_feasible = any(
                min_total_scaffold_length <= motif_length + left_length + right_length <= self.max_target_length
                for motif_length in range(int(min_motif_length), motif_upper + 1)
                for left_length, right_length in flank_pairs
            )
            if not jointly_feasible:
                raise ValueError("motif, flank, total-length, and max_target_length constraints have no joint solution")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.expected_partition_counts = expected_partition_counts
        self.expected_cluster_count = expected_cluster_count
        self.cluster_thresholds = cluster_thresholds
        self.seed = seed
        self.train_views_per_record = int(train_views_per_record)
        self.validation_views_per_record = int(validation_views_per_record)
        self.test_views_per_record = int(test_views_per_record)
        self.cluster_balanced_sampling = cluster_balanced_sampling
        self.preferred_motifs = normalized_preferred_motifs
        self.preferred_motif_probability = float(preferred_motif_probability)
        self.iterative_full_mask_probability = iterative_full_mask_probability
        self.stage_aware_context_noise = stage_aware_context_noise

    def setup(self, stage: str | None = None) -> None:
        if self.cluster_manifest is None:
            raise ValueError("cluster_manifest must be provided for motif_denoising training")
        cluster_by_id = load_cluster_assignments(self.cluster_manifest)
        self.cluster_by_id = cluster_by_id
        records = load_sequence_records(self.train_data, skip_invalid=False)
        datasets, self.split_manifest = build_partitioned_denoising_datasets(
            records=records,
            cluster_by_id=cluster_by_id,
            tokenizer=self.tokenizer,
            max_length=self.max_target_length,
            min_motif_length=self.min_motif_length,
            max_motif_length=self.max_motif_length,
            motif_length_buckets=self.motif_length_buckets,
            min_flank_length=self.min_flank_length,
            min_total_scaffold_length=self.min_total_scaffold_length,
            preferred_total_scaffold_length=self.preferred_total_scaffold_length,
            full_mask_probability=self.full_mask_probability,
            span_mask_probability=self.span_mask_probability,
            min_random_mask_fraction=self.min_random_mask_fraction,
            max_random_mask_fraction=self.max_random_mask_fraction,
            mean_span_length=self.mean_span_length,
            iterative_mask_steps=self.iterative_mask_steps,
            context_noise_probability=self.context_noise_probability,
            short_flank_min=self.short_flank_min,
            short_flank_max=self.short_flank_max,
            long_flank_min=self.long_flank_min,
            long_flank_max=self.long_flank_max,
            seed=self.seed,
            val_fraction=self.val_fraction,
            test_fraction=self.test_fraction,
            source_sha256={"sequence_table": hashlib.sha256(self.train_data.read_bytes()).hexdigest()},
            clustering_thresholds=self.cluster_thresholds,
            expected_partition_counts=self.expected_partition_counts,
            train_views_per_record=self.train_views_per_record,
            validation_views_per_record=self.validation_views_per_record,
            test_views_per_record=self.test_views_per_record,
            preferred_motifs=self.preferred_motifs,
            preferred_motif_probability=self.preferred_motif_probability,
            iterative_full_mask_probability=self.iterative_full_mask_probability,
            stage_aware_context_noise=self.stage_aware_context_noise,
        )
        if self.validation_full_mask:
            for partition in ("validation", "test"):
                dataset = datasets[partition]
                dataset.iterative_mask_steps = None
                dataset.iterative_full_mask_probability = None
                dataset.full_mask_probability = 1.0
                dataset.span_mask_probability = 0.0
                dataset.context_noise_probability = 0.0
        actual_cluster_count = self.split_manifest.cluster_counts.get("total")
        if self.expected_cluster_count is not None and actual_cluster_count != self.expected_cluster_count:
            raise ValueError(
                "eligible cluster count does not match expected: "
                f"actual={actual_cluster_count}, expected={self.expected_cluster_count}"
            )
        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["validation"]
        self.test_dataset = datasets["test"]
        eligible_records = sum(dataset.record_count for dataset in datasets.values())
        self.denoising_data_audit = {
            "input_records": len(records),
            "eligible_records": eligible_records,
            "excluded_by_length_or_context": len(records) - eligible_records,
            "min_motif_length": self.min_motif_length,
            "max_motif_length": self.max_motif_length,
            "min_flank_length": self.min_flank_length,
            "min_total_scaffold_length": self.min_total_scaffold_length,
            "short_flank_range": [self.short_flank_min, self.short_flank_max],
            "long_flank_range": [self.long_flank_min, self.long_flank_max],
            "iterative_mask_steps": self.iterative_mask_steps,
            "context_noise_probability": self.context_noise_probability,
            "validation_full_mask": self.validation_full_mask,
            "views_per_record": {
                "train": self.train_views_per_record,
                "validation": self.validation_views_per_record,
                "test": self.test_views_per_record,
            },
            "virtual_examples": {partition: len(dataset) for partition, dataset in datasets.items()},
            "cluster_balanced_sampling": self.cluster_balanced_sampling,
            "preferred_motifs": self.preferred_motifs,
            "preferred_motif_probability": self.preferred_motif_probability,
            "iterative_full_mask_probability": self.iterative_full_mask_probability,
            "stage_aware_context_noise": self.stage_aware_context_noise,
        }
        if not self.uses_asymmetric_geometry:
            self.denoising_data_audit["preferred_total_scaffold_length"] = self.preferred_total_scaffold_length
        print("denoising_data_audit=" + ", ".join(f"{key}={value}" for key, value in self.denoising_data_audit.items()))
        if not len(self.train_dataset):
            raise ValueError("Cluster-disjoint split produced an empty training partition")
        expected_counts = self.expected_partition_counts or {}
        empty_required = [
            partition
            for partition, expected_count in expected_counts.items()
            if expected_count > 0 and not len(datasets[partition])
        ]
        if empty_required:
            raise ValueError(f"expected non-empty partitions are empty: {empty_required}")

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if self.cluster_balanced_sampling:
            cluster_sizes = Counter(self.cluster_by_id[record.target_id] for record in self.train_dataset.records)
            weights = [
                1.0
                / cluster_sizes[
                    self.cluster_by_id[
                        self.train_dataset.records[virtual_index // self.train_dataset.views_per_record].target_id
                    ]
                ]
                for virtual_index in range(len(self.train_dataset))
            ]
            sampler = WeightedRandomSampler(
                weights,
                num_samples=len(self.train_dataset),
                replacement=True,
                generator=torch.Generator().manual_seed(self.seed),
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        dataset = getattr(self, "test_dataset", self.val_dataset)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
