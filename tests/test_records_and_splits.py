import pytest
import torch

from rna_scaffold.data import build_partitioned_denoising_datasets, sample_motif_example
from rna_scaffold.datamodule import RnaScaffoldDataModule
from rna_scaffold.records import RnaSequenceRecord, load_sequence_records
from rna_scaffold.splits import (
    build_cluster_disjoint_manifest,
    build_family_disjoint_manifest,
    load_cluster_assignments,
    validate_manifest,
)
from rna_scaffold.tokenizer import RnaTokenizer


def test_record_normalizes_thymine_and_whitespace():
    record = RnaSequenceRecord("a", "  atgcau  ", "RF1", "unit")

    assert record.sequence == "AUGCAU"


def test_record_rejects_ambiguous_but_accepts_long_canonical_sequences():
    with pytest.raises(ValueError, match="invalid RNA sequence"):
        RnaSequenceRecord("ambiguous", "AUGN", None, "unit")

    record = RnaSequenceRecord("long", "A" * 4298, None, "unit")

    assert len(record.sequence) == 4298


def test_csv_record_loader_keeps_family_and_source_metadata(tmp_path):
    source = tmp_path / "rfam.csv"
    source.write_text(
        "target_id,sequence,family,source\n"
        "x,ATGCAU,RF00001,Rfam\n",
        encoding="utf-8",
    )

    assert load_sequence_records(source) == [
        RnaSequenceRecord("x", "AUGCAU", "RF00001", "Rfam")
    ]


def test_csv_loader_can_skip_noncanonical_rows_without_dropping_long_rna(tmp_path):
    source = tmp_path / "mixed.csv"
    source.write_text(
        "target_id,sequence\n"
        f"long,{'A' * 600}\n"
        "ambiguous,AUGN\n",
        encoding="utf-8",
    )

    records = load_sequence_records(source, skip_invalid=True)

    assert [record.target_id for record in records] == ["long"]
    assert len(records[0].sequence) == 600


def test_strict_csv_record_loader_rejects_empty_sequence_rows(tmp_path):
    source = tmp_path / "empty.csv"
    source.write_text("target_id,sequence\nempty,\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty RNA sequence"):
        load_sequence_records(source)


def test_family_members_never_cross_partitions():
    records = [
        RnaSequenceRecord("a", "AUGCAUGC", "RF1", "unit"),
        RnaSequenceRecord("b", "AUGCAUGG", "RF1", "unit"),
        RnaSequenceRecord("c", "CCCCAAAA", "RF2", "unit"),
    ]

    manifest = build_family_disjoint_manifest(records, seed=7)

    assert manifest.partition_for("a") == manifest.partition_for("b")


def test_manifest_rejects_exact_sequence_overlap():
    records = [
        RnaSequenceRecord("a", "AUGCAUGC", "RF1", "unit"),
        RnaSequenceRecord("b", "AUGCAUGC", "RF2", "unit"),
    ]

    with pytest.raises(ValueError, match="exact sequence overlap"):
        validate_manifest(records, {"train": ["a"], "validation": [], "test": ["b"]})


def test_cluster_assignment_tsv_keeps_cluster_members_in_one_partition(tmp_path):
    assignment_path = tmp_path / "clusters.tsv"
    assignment_path.write_text(
        "cluster_id\tsequence_id\tcluster_size\n"
        "cluster_a\ta\t2\n"
        "cluster_a\tb\t2\n"
        "cluster_b\tc\t1\n",
        encoding="utf-8",
    )
    records = [
        RnaSequenceRecord("a", "AUGCAUGC", "RF1", "unit"),
        RnaSequenceRecord("b", "AUGCAUGG", "RF2", "unit"),
        RnaSequenceRecord("c", "CCCCAAAA", "RF3", "unit"),
    ]

    manifest = build_cluster_disjoint_manifest(
        records,
        load_cluster_assignments(assignment_path),
        seed=7,
        val_fraction=0.1,
        test_fraction=0.1,
    )

    assert manifest.partition_for("a") == manifest.partition_for("b")
    assert {target_id for members in manifest.partitions.values() for target_id in members} == {"a", "b", "c"}


def test_cluster_manifest_rejects_missing_assignments():
    records = [
        RnaSequenceRecord("a", "AUGCAUGC", "RF1", "unit"),
        RnaSequenceRecord("b", "AUGCAUGG", "RF2", "unit"),
    ]

    with pytest.raises(ValueError, match="incomplete"):
        build_cluster_disjoint_manifest(records, {"a": "cluster_a"})


def test_cluster_assignment_loader_rejects_duplicate_sequence_ids(tmp_path):
    assignment_path = tmp_path / "clusters.tsv"
    assignment_path.write_text(
        "cluster_id\tsequence_id\tcluster_size\n"
        "cluster_a\ta\t1\n"
        "cluster_b\ta\t1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_cluster_assignments(assignment_path)


def test_cluster_assignment_loader_rejects_inconsistent_declared_cluster_sizes(tmp_path):
    assignment_path = tmp_path / "clusters.tsv"
    assignment_path.write_text(
        "cluster_id\tsequence_id\tcluster_size\n"
        "cluster_a\ta\t2\n"
        "cluster_a\tb\t3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inconsistent cluster_size"):
        load_cluster_assignments(assignment_path)


def test_cluster_assignment_loader_rejects_declared_size_that_does_not_match_rows(tmp_path):
    assignment_path = tmp_path / "clusters.tsv"
    assignment_path.write_text(
        "cluster_id\tsequence_id\tcluster_size\n"
        "cluster_a\ta\t3\n"
        "cluster_a\tb\t3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cluster_size does not match rows"):
        load_cluster_assignments(assignment_path)


def test_manifest_rejects_cluster_in_two_requested_partitions():
    records = [
        RnaSequenceRecord("a", "AUGCAUGC", "RF1", "unit"),
        RnaSequenceRecord("b", "AUGCAUGG", "RF2", "unit"),
    ]

    with pytest.raises(ValueError, match="cluster crosses partitions"):
        validate_manifest(
            records,
            {"train": ["a"], "validation": ["b"], "test": []},
            cluster_by_id={"a": "cluster_a", "b": "cluster_a"},
        )


def test_cluster_manifest_rejects_duplicate_normalized_sequences():
    records = [
        RnaSequenceRecord("a", "ATGCAUGC", "RF1", "unit"),
        RnaSequenceRecord("b", "AUGCAUGC", "RF2", "unit"),
    ]

    with pytest.raises(ValueError, match="duplicate normalized sequence"):
        build_cluster_disjoint_manifest(records, {"a": "cluster_a", "b": "cluster_a"})


def test_cluster_manifest_records_audit_metadata_and_expected_partition_totals(tmp_path):
    assignment_path = tmp_path / "clusters.tsv"
    assignment_path.write_text(
        "cluster_id\tsequence_id\tcluster_size\n"
        "cluster_a\ta\t2\n"
        "cluster_a\tb\t2\n"
        "cluster_b\tc\t1\n",
        encoding="utf-8",
    )
    records = [
        RnaSequenceRecord("a", "AUGCAUGC", "RF1", "unit"),
        RnaSequenceRecord("b", "AUGCAUGG", "RF2", "unit"),
        RnaSequenceRecord("c", "CCCCAAAA", "RF3", "unit"),
    ]

    manifest = build_cluster_disjoint_manifest(
        records,
        load_cluster_assignments(assignment_path),
        seed=7,
        val_fraction=0,
        test_fraction=0,
        source_sha256={"sequence_table": "records-sha"},
        clustering_thresholds={"min_sequence_identity": 0.8},
        expected_partition_counts={"train": 3, "validation": 0, "test": 0},
    )

    assert manifest.record_counts == {"train": 3, "validation": 0, "test": 0, "total": 3}
    assert manifest.cluster_counts == {"train": 2, "validation": 0, "test": 0, "total": 2}
    assert manifest.source_sha256["sequence_table"] == "records-sha"
    assert "cluster_manifest" in manifest.source_sha256
    assert manifest.clustering_thresholds == {"min_sequence_identity": 0.8}


def test_cluster_manifest_rejects_unexpected_partition_totals():
    records = [RnaSequenceRecord("a", "AUGCAUGC", "RF1", "unit")]

    with pytest.raises(ValueError, match="partition counts do not match expected"):
        build_cluster_disjoint_manifest(
            records,
            {"a": "cluster_a"},
            expected_partition_counts={"train": 0, "validation": 1, "test": 0},
        )


def test_variable_motif_sampling_is_reproducible_and_not_center_only():
    record = RnaSequenceRecord("a", "AUGCAUGCAUGCAUGCAUGC", "RF1", "unit")
    first_generator = torch.Generator().manual_seed(11)
    second_generator = torch.Generator().manual_seed(11)

    first = [
        sample_motif_example(
            record, first_generator, 4, 8, min_flank_length=0, min_total_scaffold_length=1
        )
        for _ in range(12)
    ]
    second = [
        sample_motif_example(
            record, second_generator, 4, 8, min_flank_length=0, min_total_scaffold_length=1
        )
        for _ in range(12)
    ]

    assert first == second
    assert all(4 <= len(example.motif) <= 8 for example in first)
    assert all(example.target_sequence[example.motif_start : example.motif_end] == example.motif for example in first)
    assert len({example.motif_start for example in first}) > 1
    assert any(example.motif_start != (example.total_length - len(example.motif)) // 2 for example in first)


def test_weighted_motif_sampling_covers_short_and_extended_inputs():
    record = RnaSequenceRecord("long", "AUGC" * 100, "RF1", "unit")
    generator = torch.Generator().manual_seed(23)
    buckets = (
        (8, 15, 0.15),
        (16, 31, 0.30),
        (32, 63, 0.30),
        (64, 127, 0.20),
        (128, 256, 0.05),
    )

    examples = [
        sample_motif_example(
            record,
            generator,
            min_motif_length=8,
            max_motif_length=256,
            motif_length_buckets=buckets,
            min_flank_length=4,
            min_total_scaffold_length=24,
        )
        for _ in range(1000)
    ]
    lengths = [len(example.motif) for example in examples]

    assert all(8 <= length <= 256 for length in lengths)
    assert 20 <= sum(length >= 128 for length in lengths) <= 90
    assert all(example.motif_start >= 4 for example in examples)
    assert all(example.total_length - example.motif_end >= 4 for example in examples)
    assert all(example.total_length - len(example.motif) >= 24 for example in examples)


def test_motif_sampling_rejects_sequence_without_required_scaffold_context():
    record = RnaSequenceRecord("short", "AUGC" * 5, "RF1", "unit")

    with pytest.raises(ValueError, match="scaffold context"):
        sample_motif_example(
            record,
            torch.Generator().manual_seed(1),
            min_motif_length=8,
            max_motif_length=256,
            min_flank_length=4,
            min_total_scaffold_length=24,
        )


def test_denoising_datasets_use_manifest_instead_of_random_record_split(tmp_path):
    source = tmp_path / "records.csv"
    source.write_text(
        "target_id,sequence,family,source\n"
        "a,AUGCAUGC,RF1,unit\n"
        "b,AUGCAUGG,RF1,unit\n"
        "c,CCCCAAAA,RF2,unit\n",
        encoding="utf-8",
    )
    records = load_sequence_records(source)
    datasets, manifest = build_partitioned_denoising_datasets(
        records=records,
        cluster_by_id={"a": "cluster_a", "b": "cluster_a", "c": "cluster_b"},
        tokenizer=RnaTokenizer(),
        min_motif_length=4,
        max_motif_length=6,
        motif_length_buckets=None,
        min_flank_length=0,
        min_total_scaffold_length=1,
        max_length=32,
        seed=7,
        val_fraction=0.1,
        test_fraction=0.1,
    )

    assert manifest.partition_for("a") == manifest.partition_for("b")
    assert sum(len(dataset) for dataset in datasets.values()) == 3


def test_denoising_manifest_counts_only_records_eligible_for_motif_context():
    records = [
        RnaSequenceRecord("eligible", "AUGCAUGC", "RF1", "unit"),
        RnaSequenceRecord("short", "AUGC", "RF2", "unit"),
    ]

    datasets, manifest = build_partitioned_denoising_datasets(
        records=records,
        cluster_by_id={"eligible": "cluster_a", "short": "cluster_b"},
        tokenizer=RnaTokenizer(),
        min_motif_length=4,
        max_motif_length=4,
        motif_length_buckets=None,
        min_flank_length=2,
        min_total_scaffold_length=0,
        preferred_total_scaffold_length=0,
        max_length=32,
    )

    assert manifest.record_counts["total"] == 1
    assert sum(len(dataset) for dataset in datasets.values()) == 1


def test_denoising_builder_rejects_duplicate_target_ids_before_eligibility_filtering():
    records = [
        RnaSequenceRecord("duplicate", "AUGC", "RF1", "unit"),
        RnaSequenceRecord("duplicate", "GGGG", "RF2", "unit"),
    ]

    with pytest.raises(ValueError, match="record target IDs must be unique"):
        build_partitioned_denoising_datasets(
            records=records,
            cluster_by_id={"duplicate": "cluster_a"},
            tokenizer=RnaTokenizer(),
            min_motif_length=4,
            max_motif_length=4,
            motif_length_buckets=None,
            min_flank_length=2,
            min_total_scaffold_length=0,
            preferred_total_scaffold_length=0,
        )


def test_formal_datamodule_rejects_invalid_source_rows(tmp_path):
    source = tmp_path / "records.csv"
    source.write_text("target_id,sequence\na,AUGN\n", encoding="utf-8")
    cluster_manifest = tmp_path / "clusters.tsv"
    cluster_manifest.write_text(
        "cluster_id\tsequence_id\tcluster_size\ncluster_a\ta\t1\n",
        encoding="utf-8",
    )
    module = RnaScaffoldDataModule(
        tokenizer=RnaTokenizer(),
        train_data=str(source),
        cluster_manifest=str(cluster_manifest),
        task="motif_denoising",
    )

    with pytest.raises(ValueError, match="invalid RNA sequence"):
        module.setup()


def test_formal_datamodule_requires_cluster_manifest_at_construction(tmp_path):
    source = tmp_path / "records.csv"
    source.write_text("target_id,sequence\na,AUGCAUGC\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cluster_manifest must be provided"):
        RnaScaffoldDataModule(
            tokenizer=RnaTokenizer(),
            train_data=str(source),
            task="motif_denoising",
        )


def test_formal_datamodule_rejects_unexpected_eligible_cluster_count(tmp_path):
    source = tmp_path / "records.csv"
    source.write_text(
        "target_id,sequence,family,source\n"
        "a,AUGCAUGC,RF1,unit\n"
        "b,CCGGCCGG,RF2,unit\n",
        encoding="utf-8",
    )
    assignments = tmp_path / "clusters.tsv"
    assignments.write_text(
        "cluster_id\tsequence_id\tcluster_size\n"
        "cluster_a\ta\t1\n"
        "cluster_b\tb\t1\n",
        encoding="utf-8",
    )
    module = RnaScaffoldDataModule(
        tokenizer=RnaTokenizer(),
        train_data=str(source),
        cluster_manifest=str(assignments),
        task="motif_denoising",
        min_motif_length=4,
        max_motif_length=4,
        motif_length_buckets=None,
        min_flank_length=2,
        min_total_scaffold_length=0,
        preferred_total_scaffold_length=0,
        max_target_length=8,
        val_fraction=0,
        test_fraction=0,
        expected_cluster_count=3,
        seed=7,
    )

    with pytest.raises(ValueError, match="cluster count does not match expected"):
        module.setup("fit")
