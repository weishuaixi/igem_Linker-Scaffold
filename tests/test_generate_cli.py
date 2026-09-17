import json
import subprocess
import sys

import pytest

import generate_scaffold
from rna_scaffold.generate import (
    CandidateGenerationAudit,
    GeneratedCandidates,
    ScaffoldCandidate,
    write_candidates_fasta,
    write_candidates_jsonl,
)


def test_generate_cli_help():
    completed = subprocess.run(
        [sys.executable, "src/generate_scaffold.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--checkpoint" in completed.stdout
    assert "--motif" in completed.stdout
    assert "--length-sampling" in completed.stdout


def test_generate_cli_accepts_and_forwards_uniform_length_sampling(monkeypatch, tmp_path):
    captured = {}

    def fake_generate(checkpoint, motif, settings, device):
        captured["settings"] = settings
        candidate = ScaffoldCandidate(
            candidate_id="candidate_0001",
            full_sequence="AAGCGGUU",
            left_sequence="AA",
            motif="GCGG",
            right_sequence="UU",
            motif_start=2,
            motif_end=6,
            total_length=8,
            normalized_log_probability=-0.2,
            checkpoint_sha256="abc",
            seed=42,
            gc_fraction=0.5,
            max_homopolymer=2,
            base_entropy=1.5,
            motif_preserved=True,
            valid=True,
            status="ok",
        )
        return GeneratedCandidates(
            [candidate],
            CandidateGenerationAudit(requested=1, max_attempts=1, attempted=1, accepted=1, rejected={}),
        )

    monkeypatch.setattr(generate_scaffold, "generate_candidates", fake_generate)
    output = tmp_path / "uniform.jsonl"
    result = generate_scaffold.main(
        [
            "--motif",
            "GCGG",
            "--checkpoint",
            "unused.ckpt",
            "--output",
            str(output),
            "--num-candidates",
            "1",
            "--length-sampling",
            "uniform",
        ]
    )

    assert result == 0
    assert captured["settings"].length_sampling == "uniform"


def test_candidate_jsonl_is_atomic_and_machine_readable(tmp_path):
    output = tmp_path / "candidates.jsonl"
    candidate = ScaffoldCandidate(
        candidate_id="candidate_0001",
        full_sequence="AAGCGGUU",
        left_sequence="AA",
        motif="GCGG",
        right_sequence="UU",
        motif_start=2,
        motif_end=6,
        total_length=8,
        normalized_log_probability=-0.2,
        checkpoint_sha256="abc",
        seed=42,
        gc_fraction=0.5,
        max_homopolymer=2,
        base_entropy=1.5,
        motif_preserved=True,
        valid=True,
        status="ok",
    )

    write_candidates_jsonl([candidate], output)

    assert not output.with_suffix(output.suffix + ".tmp").exists()
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["full_sequence"] == "AAGCGGUU"


def test_candidate_outputs_include_atomic_audit_metadata(tmp_path):
    output = tmp_path / "candidates.jsonl"
    fasta_output = tmp_path / "candidates.fasta"
    candidate = ScaffoldCandidate(
        candidate_id="candidate_0001",
        full_sequence="AAGCGGUU",
        left_sequence="AA",
        motif="GCGG",
        right_sequence="UU",
        motif_start=2,
        motif_end=6,
        total_length=8,
        normalized_log_probability=-0.2,
        checkpoint_sha256="abc",
        seed=42,
        gc_fraction=0.5,
        max_homopolymer=2,
        base_entropy=1.5,
        motif_preserved=True,
        valid=True,
        status="shortfall",
        generation_settings={"max_attempts": 3},
    )
    generated = GeneratedCandidates(
        [candidate],
        CandidateGenerationAudit(requested=2, max_attempts=3, attempted=3, accepted=1, rejected={"duplicate": 2}),
    )

    write_candidates_jsonl(generated, output)
    write_candidates_fasta(generated, fasta_output)

    assert not output.with_suffix(output.suffix + ".tmp").exists()
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["architecture_version"] == 2
    assert row["generation_settings"] == {"max_attempts": 3}
    assert row["generation_audit"]["rejected"] == {"duplicate": 2}
    assert not fasta_output.with_suffix(fasta_output.suffix + ".tmp").exists()
    assert fasta_output.read_text(encoding="utf-8").splitlines()[:4] == [
        '; {"architecture_version": 2}',
        '; {"generation_settings": {"max_attempts": 3}}',
        '; {"generation_audit": {"accepted": 1, "attempted": 3, "max_attempts": 3, "rejected": {"duplicate": 2}, "requested": 2}}',
        ">candidate_0001",
    ]


def test_cli_treats_audited_short_candidate_set_as_failure(monkeypatch, tmp_path, capsys):
    generated = GeneratedCandidates(
        [],
        CandidateGenerationAudit(requested=2, max_attempts=3, attempted=3, accepted=0, rejected={"duplicate": 3}),
    )
    monkeypatch.setattr(generate_scaffold, "generate_candidates", lambda *args, **kwargs: generated)

    output = tmp_path / "short.jsonl"
    assert (
        generate_scaffold.main(
            [
                "--motif",
                "GCGG",
                "--checkpoint",
                "unused.ckpt",
                "--output",
                str(output),
                "--num-candidates",
                "2",
            ]
        )
        == 2
    )
    assert json.loads(output.read_text(encoding="utf-8"))["generation_audit"]["accepted"] == 0
    assert "generated 0/2 requested candidates" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("writer", "suffix", "has_candidate"),
    [
        (write_candidates_jsonl, ".jsonl", True),
        (write_candidates_jsonl, ".jsonl", False),
        (write_candidates_fasta, ".fasta", True),
        (write_candidates_fasta, ".fasta", False),
    ],
)
def test_every_candidate_artifact_includes_complete_run_metadata(tmp_path, writer, suffix, has_candidate):
    audit = CandidateGenerationAudit(requested=1, max_attempts=2, attempted=1, accepted=int(has_candidate), rejected={})
    settings = {"max_attempts": 2, "gc_min": 0.4}
    candidate = ScaffoldCandidate(
        candidate_id="candidate_0001",
        full_sequence="AAGCGGUU",
        left_sequence="AA",
        motif="GCGG",
        right_sequence="UU",
        motif_start=2,
        motif_end=6,
        total_length=8,
        normalized_log_probability=-0.2,
        checkpoint_sha256="abc",
        seed=42,
        gc_fraction=0.5,
        max_homopolymer=2,
        base_entropy=1.5,
        motif_preserved=True,
        valid=True,
        status="ok",
    )
    generated = GeneratedCandidates([candidate] if has_candidate else [], audit, settings)
    output = tmp_path / f"candidates{suffix}"

    writer(generated, output)

    text = output.read_text(encoding="utf-8")
    assert '"architecture_version": 2' in text
    assert '"generation_settings": {"gc_min": 0.4, "max_attempts": 2}' in text
    assert '"generation_audit": {"accepted": ' in text
