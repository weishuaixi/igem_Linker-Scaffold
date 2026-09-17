import csv
from pathlib import Path

from scripts.prepare_stanford_scaffold_data import prepare_sequence_table, write_connected_components


def test_prepare_sequence_table_normalizes_filters_and_deduplicates(tmp_path: Path):
    source = tmp_path / "source.csv"
    source.write_text(
        "target_id,sequence\nrna_b,AUCTAUGC\nrna_a,aucuaugc\nrna_duplicate,AUCNAUGC\nrna_c,AUGCAUGC\nrna_d,AUGCAUGC\n",
        encoding="utf-8",
    )
    output = tmp_path / "processed.csv"
    fasta = tmp_path / "processed.fasta"

    stats = prepare_sequence_table(source, output, fasta)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["target_id"] for row in rows] == ["rna_a", "rna_c"]
    assert [row["sequence"] for row in rows] == ["AUCUAUGC", "AUGCAUGC"]
    assert stats["output_records"] == 2
    assert fasta.read_text(encoding="ascii") == ">rna_a\nAUCUAUGC\n>rna_c\nAUGCAUGC\n"


def test_write_connected_components_includes_singletons(tmp_path: Path):
    sequence_csv = tmp_path / "sequences.csv"
    sequence_csv.write_text(
        "target_id,sequence,family,source\na,AUGCAUGC,,test\nb,AUGCAUGU,,test\nc,CCCCCCCC,,test\n",
        encoding="utf-8",
    )
    mmseqs = tmp_path / "clusters.tsv"
    mmseqs.write_text("a\ta\na\tb\nc\tc\n", encoding="utf-8")
    output = tmp_path / "components.tsv"

    stats = write_connected_components(mmseqs, sequence_csv, output)

    assert stats == {"sequence_count": 3, "cluster_count": 2}
    assert output.read_text(encoding="utf-8").splitlines() == [
        "cluster_id\tsequence_id\tcluster_size",
        "cluster_000001\ta\t2",
        "cluster_000001\tb\t2",
        "cluster_000002\tc\t1",
    ]
