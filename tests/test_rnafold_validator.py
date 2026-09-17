import pytest

from rna_scaffold.validators.rnafold import parse_rnafold_output, run_rnafold


def test_parse_rnafold_output_extracts_structure_and_mfe():
    result = parse_rnafold_output(
        "AUGCAUGC\n((....)) (-2.40)\n",
        motif_start=2,
        motif_end=6,
        runtime_seconds=0.1,
        version="RNAfold 2.7",
    )

    assert result.status == "ok"
    assert result.dot_bracket == "((....))"
    assert result.mfe_kcal_mol == pytest.approx(-2.4)
    assert result.paired_fraction == pytest.approx(0.5)
    assert result.motif_paired_fraction == 0.0


def test_missing_rnafold_is_reported_as_unavailable():
    result = run_rnafold(
        "AUGC",
        motif_start=0,
        motif_end=4,
        executable="definitely-not-an-rnafold-executable",
    )

    assert result.status == "unavailable"
    assert result.mfe_kcal_mol is None
    assert result.error


def test_malformed_rnafold_output_is_not_fabricated():
    result = parse_rnafold_output(
        "not a structure",
        motif_start=0,
        motif_end=4,
        runtime_seconds=0.1,
        version=None,
    )

    assert result.status == "parse_error"
    assert result.dot_bracket is None
