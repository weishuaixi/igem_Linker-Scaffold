import csv
import hashlib
import json
from pathlib import Path

from scripts.filter_scaffold_training_data import filter_cluster_manifest, filter_sequence_table


def test_quality_filter_is_audited_and_preserves_cluster_assignments(tmp_path: Path):
    source = tmp_path / "source.csv"
    source.write_text(
        "target_id,sequence,family,source\n"
        "good,AUGCAUGCAUGC,,unit\n"
        "poly,AUGCAAAAAAAAAAAAA,,unit\n"
        "extreme,GGGGGCGGGGG,,unit\n",
        encoding="utf-8",
    )
    clusters = tmp_path / "clusters.tsv"
    clusters.write_text(
        "cluster_id\tsequence_id\tcluster_size\nc1\tgood\t2\nc1\tpoly\t2\nc2\textreme\t1\n",
        encoding="utf-8",
    )
    cleaned = tmp_path / "cleaned.csv"
    cleaned_clusters = tmp_path / "cleaned.tsv"

    stats, kept_ids = filter_sequence_table(
        source,
        cleaned,
        gc_min=0.10,
        gc_max=0.90,
        max_homopolymer_run=6,
        min_base_entropy=1.20,
    )
    cluster_stats = filter_cluster_manifest(clusters, cleaned_clusters, kept_ids)

    rows = list(csv.DictReader(cleaned.open(encoding="utf-8")))
    assert [row["target_id"] for row in rows] == ["good"]
    assert stats["rejected_quality"] == 2
    assert stats["rejected_homopolymer"] == 1
    assert stats["rejected_gc"] == 1
    assert cluster_stats == {"sequence_count": 1, "cluster_count": 1}
    assert cleaned_clusters.read_text(encoding="utf-8").splitlines()[-1] == "c1\tgood\t1"


def test_checked_in_qc_audit_matches_the_versioned_release():
    root = Path(__file__).parents[1]
    audit_path = root / "data" / "processed" / "rna_linker_v3_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    csv_path = root / "data" / "processed" / "rna_linker_v3_training.csv"
    cluster_path = root / "data" / "processed" / "rna_linker_v3_clusters.tsv"

    assert audit["sequence_stats"]["input_records"] == 4125
    assert audit["sequence_stats"]["output_records"] == 3442
    assert audit["sequence_stats"]["min_length"] == 29
    assert audit["cluster_stats"] == {"cluster_count": 1572, "sequence_count": 3442}
    assert audit["sha256"]["output_csv"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert audit["sha256"]["output_clusters"] == hashlib.sha256(cluster_path.read_bytes()).hexdigest()

    with csv_path.open(encoding="utf-8", newline="") as handle:
        sequence_ids = {row["target_id"] for row in csv.DictReader(handle)}
    with cluster_path.open(encoding="utf-8", newline="") as handle:
        cluster_ids = {row["sequence_id"] for row in csv.DictReader(handle, delimiter="\t")}
    assert sequence_ids == cluster_ids
