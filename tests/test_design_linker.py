import json
from dataclasses import replace

import pytest

from rna_scaffold.generate import ScaffoldCandidate
from rna_scaffold.validators.rnafold import RnafoldResult
from scripts import design_linker


def candidate(name="a"):
    left, motif, right = "AUCGUAUCGU", "GCGG", "AUCGUAUCGUAUCGU"
    return ScaffoldCandidate(
        candidate_id=name,
        full_sequence=left + motif + right,
        left_sequence=left,
        motif=motif,
        right_sequence=right,
        motif_start=10,
        motif_end=14,
        total_length=29,
        normalized_log_probability=-1.0,
        checkpoint_sha256="test",
        seed=42,
        gc_fraction=0.5,
        max_homopolymer=2,
        base_entropy=1.9,
        motif_preserved=True,
        valid=True,
        status="ok",
        mean_token_confidence=0.4,
    )


def fold_ok(*args, **kwargs):
    return RnafoldResult("ok", "." * len(args[0]), -2.0, 0.0, 0.0, 0.01, "RNAfold 2.7.2", None)


def arguments(tmp_path):
    return ["--motif", "GCGG", "--output-dir", str(tmp_path / "result"), "--num-candidates", "2", "--device", "cpu"]


def test_design_exports_top_candidate_and_traceable_ranking(monkeypatch, tmp_path):
    def generate(checkpoint, motif, settings, device):
        assert settings.num_candidates == 2
        assert settings.remask_strategy == "random"
        assert settings.enforce_gc_bounds is True
        assert settings.max_homopolymer_run == 6
        return [candidate("b"), replace(candidate("a"), normalized_log_probability=-0.5)]

    monkeypatch.setattr(design_linker, "generate_candidates", generate)
    monkeypatch.setattr(design_linker, "run_rnafold", fold_ok)
    assert design_linker.main(arguments(tmp_path)) == 0
    output = tmp_path / "result"
    best = json.loads((output / "best.json").read_text())
    assert best["rank"] == 1
    assert best["candidate"]["candidate_id"] == "a"
    assert (output / "best.fasta").read_text().splitlines()[1] == candidate().full_sequence
    audit = json.loads((output / "run_manifest.json").read_text())
    assert audit["status"] == "completed"
    assert audit["rnafold_required"] is True
    assert audit["ranking_weights"] == design_linker.DEFAULT_WEIGHTS
    assert len((output / "rnafold_results.jsonl").read_text().splitlines()) == 2


def test_missing_rnafold_stops_before_generation(monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        raise AssertionError("Generation must not run")

    monkeypatch.setattr(design_linker, "generate_candidates", unexpected)
    monkeypatch.setattr(
        design_linker,
        "run_rnafold",
        lambda *a, **k: RnafoldResult("unavailable", None, None, None, None, 0.0, None, "missing"),
    )
    assert design_linker.main(arguments(tmp_path)) == 2
    audit = json.loads((tmp_path / "result/run_manifest.json").read_text())
    assert audit["status"] == "failed"
    assert not (tmp_path / "result/best.fasta").exists()


def test_partial_folding_failure_never_exports_best(monkeypatch, tmp_path):
    monkeypatch.setattr(design_linker, "generate_candidates", lambda *a, **k: [candidate("a"), candidate("b")])
    counter = iter(
        [
            fold_ok("GGGAAACCC"),
            fold_ok(candidate().full_sequence),
            RnafoldResult("timeout", None, None, None, None, 30.0, "v", "timeout"),
        ]
    )
    monkeypatch.setattr(design_linker, "run_rnafold", lambda *a, **k: next(counter))
    assert design_linker.main(arguments(tmp_path)) == 2
    assert not (tmp_path / "result/best.fasta").exists()
    assert not (tmp_path / "result/best.json").exists()
    audit = json.loads((tmp_path / "result/run_manifest.json").read_text())
    assert audit["rnafold_failures"] == {"b": "timeout"}


def test_insufficient_pool_does_not_claim_best(monkeypatch, tmp_path):
    monkeypatch.setattr(design_linker, "run_rnafold", fold_ok)
    monkeypatch.setattr(design_linker, "generate_candidates", lambda *a, **k: [candidate()])
    assert design_linker.main(arguments(tmp_path)) == 2
    assert (tmp_path / "result/candidates.jsonl").exists()
    assert not (tmp_path / "result/best.fasta").exists()


def test_existing_directory_is_not_overwritten(tmp_path):
    output = tmp_path / "result"
    output.mkdir()
    (output / "best.fasta").write_text("original")
    with pytest.raises(FileExistsError):
        design_linker.main(arguments(tmp_path))
    assert (output / "best.fasta").read_text() == "original"


def test_invalid_motif_is_rejected_before_creating_output(tmp_path):
    args = arguments(tmp_path)
    args[1] = "ACNT"
    with pytest.raises(SystemExit):
        design_linker.main(args)
    assert not (tmp_path / "result").exists()
