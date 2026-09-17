import csv
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import benchmark_scaffolds
from rna_scaffold.benchmarking import RnaTrainingPrior
from rna_scaffold.decoding import build_flank_pair_sampler
from rna_scaffold.generate import CandidateGenerationAudit
from rna_scaffold.records import RnaSequenceRecord
from rna_scaffold.validators.rnafold import RnafoldResult


class HighestProbabilityRandom(random.Random):
    def randint(self, start, end):
        assert start <= 2 <= end
        return 2

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        assert weights is not None and cum_weights is None and k == 1
        return [population[max(range(len(weights)), key=weights.__getitem__)]]


def test_candidate_metrics_measure_linker_without_the_fixed_motif():
    metric = benchmark_scaffolds._candidate_metric(
        motif_id="m1",
        motif="GCGG",
        sequence="AAAAGCGGCCCC",
        start=4,
        training_sequences=[],
        normalized_edit_diversity=0.0,
        kmer_diversity=0.0,
        runtime_seconds=0.0,
        peak_cuda_memory_bytes=None,
        failure=None,
        rejection_reason=None,
        rnafold=None,
        geometry=benchmark_scaffolds.BenchmarkGeometry(),
    )

    assert metric.linker_length == 8
    assert metric.linker_gc_fraction == pytest.approx(0.5)
    assert metric.linker_composition_entropy_bits == pytest.approx(1.0)
    assert metric.linker_max_homopolymer_run == 4

    same_base_arms = benchmark_scaffolds._candidate_metric(
        motif_id="m1",
        motif="GCGG",
        sequence="AAAAGCGGAAAA",
        start=4,
        training_sequences=[],
        normalized_edit_diversity=0.0,
        kmer_diversity=0.0,
        runtime_seconds=0.0,
        peak_cuda_memory_bytes=None,
        failure=None,
        rejection_reason=None,
        rnafold=None,
        geometry=benchmark_scaffolds.BenchmarkGeometry(),
    )
    assert same_base_arms.linker_max_homopolymer_run == 4


def test_markov2_benchmark_conditions_both_motif_junctions():
    prior = RnaTrainingPrior.from_sequences(["AAGCGGUU"] * 20)

    sequence, motif_start = benchmark_scaffolds._markov(
        "GCGG",
        8,
        prior,
        HighestProbabilityRandom(7),
        order=2,
    )

    assert motif_start == 2
    assert sequence == "AAGCGGUU"


def test_v4_asymmetric_geometry_applies_to_uniform_markov_and_panel_sources():
    geometry = benchmark_scaffolds.BenchmarkGeometry(
        min_flank_length=10,
        min_total_scaffold_length=29,
        preferred_total_scaffold_length=40,
        max_length=100,
        short_flank_min=10,
        short_flank_max=20,
        long_flank_min=15,
        long_flank_max=30,
    )
    prior = RnaTrainingPrior.from_sequences(["AUGC" * 25])

    generated = [
        benchmark_scaffolds._uniform("GCGG", 40, random.Random(1), geometry),
        benchmark_scaffolds._markov("GCGG", 40, prior, random.Random(1), order=2, geometry=geometry),
    ]
    for sequence, start in generated:
        assert len(sequence) == 40
        assert benchmark_scaffolds._matches_asymmetric_flank_ranges(
            start,
            len(sequence) - start - 4,
            geometry,
        )

    panel = benchmark_scaffolds.build_held_out_motif_panel(
        [RnaSequenceRecord("held_out", "AUGC" * 25, None, "unit")],
        panel_size=1,
        seed=3,
        motif_lengths=(4,),
        min_flank_length=10,
        geometry=geometry,
    )
    source_start = panel[0]["source_start"]
    right_available = panel[0]["source_length"] - source_start - 4
    assert any(
        benchmark_scaffolds._matches_asymmetric_flank_ranges(left, right, geometry)
        and left <= source_start
        and right <= right_available
        for left in range(10, 31)
        for right in range(10, 31)
    )


def test_asymmetric_baselines_and_checkpoint_use_the_same_uniform_pair_support():
    geometry = benchmark_scaffolds.BenchmarkGeometry(
        min_flank_length=1,
        min_total_scaffold_length=4,
        preferred_total_scaffold_length=4,
        max_length=8,
        short_flank_min=1,
        short_flank_max=2,
        long_flank_min=2,
        long_flank_max=3,
    )
    expected = set(benchmark_scaffolds._legal_benchmark_flank_pairs(2, geometry))
    prior = RnaTrainingPrior.from_sequences(["AUGCAUGC"])

    for kind in ("uniform", "markov1", "markov2"):
        generated, rejected, _ = benchmark_scaffolds._generate_method_motif(
            {"kind": kind},
            motif="GC",
            length=4,
            budget=len(expected),
            seed=11,
            prior=prior,
            geometry=geometry,
        )
        actual = {(row["motif_start"], len(row["sequence"]) - row["motif_start"] - 2) for row in generated}
        assert actual == expected
        assert rejected == {}

    checkpoint_sampler = build_flank_pair_sampler(
        None,
        motif_length=2,
        max_length=8,
        min_scaffold_length=4,
        min_flank_length=1,
        short_flank_range=(1, 2),
        long_flank_range=(2, 3),
        length_sampling="uniform",
        model_max_length=8,
        device="cpu",
    )
    assert {tuple(pair) for pair in checkpoint_sampler.pairs.tolist()} == expected


def test_v5_benchmark_recipe_validates_shared_linker_contract():
    audit = benchmark_scaffolds.validate_benchmark_config(Path("configs/benchmark.yaml"))

    assert audit["status"] == "validated_external_inputs_pending"
    assert audit["geometry"] == {
        "min_flank_length": 10,
        "min_total_scaffold_length": 29,
        "preferred_total_scaffold_length": 40,
        "max_length": 100,
        "short_flank_min": 10,
        "short_flank_max": 20,
        "long_flank_min": 15,
        "long_flank_max": 30,
    }
    assert audit["methods"][-1] == "transformer_linker_v5"
    config = yaml.safe_load(Path("configs/benchmark.yaml").read_text(encoding="utf-8"))
    learned_generation = config["methods"][-1]["generation"]
    assert (
        not {
            "min_normalized_edit_distance",
            "max_kmer_similarity",
            "max_homopolymer_run",
            "gc_min",
            "gc_max",
            "enforce_gc_bounds",
        }
        & learned_generation.keys()
    )


def test_benchmark_smoke_writes_reproducible_artifacts(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "src/benchmark_scaffolds.py",
            "--config",
            "configs/benchmark.yaml",
            "--smoke-test",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    expected = {
        "candidate_results.csv",
        "motif_summary.csv",
        "model_summary.csv",
        "paired_bootstrap.json",
        "run_manifest.json",
        "benchmark_report.md",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected

    with (tmp_path / "candidate_results.csv").open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    assert {row["method"] for row in candidates} == {"uniform", "markov1", "markov2"}
    assert all(
        len({row["seed"] for row in candidates if row["motif_id"] == motif_id}) == 1
        for motif_id in {row["motif_id"] for row in candidates}
    )
    assert {
        (row["method"], row["motif_id"]): sum(
            candidate["method"] == row["method"] and candidate["motif_id"] == row["motif_id"]
            for candidate in candidates
        )
        for row in candidates
    } == {(method, motif): 4 for method in ("uniform", "markov1", "markov2") for motif in ("motif_gcgg", "motif_augcga")}
    assert {
        "normalized_edit_diversity",
        "kmer_diversity",
        "nearest_training_kmer_similarity",
        "max_homopolymer_run",
        "runtime_seconds",
        "peak_cuda_memory_bytes",
        "rnafold_status",
        "rnafold_mfe_kcal_mol",
        "rnafold_paired_fraction",
        "rnafold_motif_paired_fraction",
    } <= set(candidates[0])

    paired = json.loads((tmp_path / "paired_bootstrap.json").read_text(encoding="utf-8"))
    assert paired["comparisons"]
    assert all(comparison["common_motif_ids"] == ["motif_augcga", "motif_gcgg"] for comparison in paired["comparisons"])

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["seed"] == 42
    assert manifest["configuration"]["candidate_count"] == 256
    assert (
        manifest["config_sha256"] == hashlib.sha256(Path("configs/benchmark.yaml").read_bytes()).hexdigest()
    )
    assert manifest["command"][-3:] == ["--output-dir", str(tmp_path), "--smoke-test"] or manifest["command"][-3:] == [
        "--smoke-test",
        "--output-dir",
        str(tmp_path),
    ]
    assert {"python", "numpy", "torch", "pyyaml"} <= set(manifest["software_versions"])
    assert set(manifest["artifact_sha256"]) == expected - {"run_manifest.json"}
    assert {method["status"] for method in manifest["methods"]} == {"ok"}
    assert manifest["runtime_seconds"] >= 0

    report = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")
    assert "plausibility proxy" in report
    assert "not proof of function" in report


def test_benchmark_marks_configured_primary_and_hashes_training_data(tmp_path):
    training_data = tmp_path / "train.fasta"
    training_data.write_text(">train\nAAAAGCGGUUUU\n", encoding="utf-8")
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "fixture_mode": True,
                "seed": 9,
                "candidate_count": 2,
                "bootstrap_samples": 100,
                "motifs": [
                    {"id": "m1", "sequence": "GCGG"},
                    {"id": "m2", "sequence": "AUGC"},
                ],
                "methods": [
                    {"name": "control", "kind": "uniform"},
                    {"name": "primary_model", "kind": "markov2", "primary": True},
                ],
                "training_data": str(training_data),
                "rnafold": {"enabled": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            "src/benchmark_scaffolds.py",
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    with (output / "model_summary.csv").open(encoding="utf-8", newline="") as handle:
        summaries = {row["method"]: row for row in csv.DictReader(handle)}
    assert summaries["primary_model"]["primary"].lower() == "true"
    assert summaries["control"]["primary"].lower() == "false"

    with (output / "candidate_results.csv").open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    assert all(row["nearest_training_kmer_similarity"] for row in candidates)
    assert all(int(row["max_homopolymer_run"]) >= 1 for row in candidates)

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_sha256"] == {str(training_data): hashlib.sha256(training_data.read_bytes()).hexdigest()}
    assert manifest["checkpoint_sha256"] == {}
    assert {row["seed"] for row in manifest["methods"]} == {9}
    assert "Primary model: `primary_model`" in (output / "benchmark_report.md").read_text(encoding="utf-8")


def test_learned_benchmark_method_requires_checkpoint(tmp_path):
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        "fixture_mode: true\n"
        "seed: 42\n"
        "candidate_count: 2\n"
        "motifs:\n"
        "  - {id: m1, sequence: GCGG}\n"
        "methods:\n"
        "  - {name: complete, kind: checkpoint}\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "src/benchmark_scaffolds.py",
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "checkpoint" in completed.stderr.lower()


def test_missing_checkpoint_writes_unavailable_terminal_artifacts_without_fallback(tmp_path):
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "fixture_mode": True,
                "seed": 42,
                "candidate_count": 2,
                "motifs": [{"id": "m1", "sequence": "GCGG"}],
                "methods": [
                    {
                        "name": "primary_model",
                        "kind": "checkpoint",
                        "checkpoint": str(tmp_path / "missing.ckpt"),
                        "primary": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            "src/benchmark_scaffolds.py",
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "checkpoint" in completed.stderr.lower()
    assert "does not exist" in completed.stderr.lower()
    assert {path.name for path in output.iterdir()} == {
        "candidate_results.csv",
        "motif_summary.csv",
        "model_summary.csv",
        "paired_bootstrap.json",
        "run_manifest.json",
        "benchmark_report.md",
    }
    with (output / "candidate_results.csv").open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []
    with (output / "motif_summary.csv").open(encoding="utf-8", newline="") as handle:
        motif_summary = next(csv.DictReader(handle))
    assert motif_summary["status"] == "unavailable"
    assert motif_summary["count"] == "0"
    assert "does not exist" in motif_summary["error_reason"]
    with (output / "model_summary.csv").open(encoding="utf-8", newline="") as handle:
        model_summary = next(csv.DictReader(handle))
    assert model_summary["status"] == "unavailable"
    assert model_summary["count"] == "0"
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["methods"][0]["status"] == "unavailable"
    assert "does not exist" in manifest["methods"][0]["error_reason"]


def test_checkpoint_load_failure_writes_failed_method_status(monkeypatch, tmp_path):
    checkpoint = tmp_path / "broken.ckpt"
    checkpoint.write_bytes(b"not a checkpoint")
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "fixture_mode": True,
                "seed": 7,
                "candidate_count": 1,
                "bootstrap_samples": 10,
                "motifs": [{"id": "m1", "sequence": "GCGG"}],
                "methods": [
                    {
                        "name": "broken_model",
                        "kind": "checkpoint",
                        "checkpoint": str(checkpoint),
                    }
                ],
                "rnafold": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        benchmark_scaffolds,
        "load_scaffold_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("invalid v2 state")),
        raising=False,
    )
    output = tmp_path / "out"

    manifest = benchmark_scaffolds.run_benchmark(config, output)

    assert manifest["status"] == "failed"
    assert manifest["methods"][0]["status"] == "failed"
    assert "invalid v2 state" in manifest["methods"][0]["error_reason"]
    assert {path.name for path in output.iterdir()} == set(benchmark_scaffolds.ARTIFACT_NAMES)
    with (output / "candidate_results.csv").open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []
    with (output / "motif_summary.csv").open(encoding="utf-8", newline="") as handle:
        motif_summary = next(csv.DictReader(handle))
    assert motif_summary["status"] == "failed"
    assert motif_summary["count"] == "0"
    assert "invalid v2 state" in motif_summary["error_reason"]
    with (output / "model_summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = next(csv.DictReader(handle))
    assert summary["status"] == "failed"
    assert summary["count"] == "0"
    assert "invalid v2 state" in summary["error_reason"]


def test_checkpoint_method_loads_once_and_reuses_loaded_model_for_all_motifs(monkeypatch, tmp_path):
    checkpoint = tmp_path / "tiny.ckpt"
    checkpoint.write_bytes(b"v2 checkpoint placeholder")
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "fixture_mode": True,
                "seed": 5,
                "candidate_count": 1,
                "bootstrap_samples": 10,
                "motifs": [
                    {"id": "m1", "sequence": "GCGG"},
                    {"id": "m2", "sequence": "AUGC"},
                ],
                "methods": [
                    {
                        "name": "loaded_once",
                        "kind": "checkpoint",
                        "checkpoint": str(checkpoint),
                    }
                ],
                "rnafold": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    loaded = object()
    load_calls = []
    generation_loaded_values = []

    def fake_loader(path, device):
        load_calls.append((path, device))
        return loaded

    class CompleteBatch(list):
        audit = CandidateGenerationAudit(1, 1, 1, 1, {})

    def fake_generate(checkpoint, motif, settings, device, loaded_checkpoint=None):
        generation_loaded_values.append(loaded_checkpoint)
        return CompleteBatch(
            [SimpleNamespace(full_sequence="AA" + motif + "UU", motif_start=2, valid=True, status="ok")]
        )

    monkeypatch.setattr(benchmark_scaffolds, "load_scaffold_checkpoint", fake_loader, raising=False)
    monkeypatch.setattr("rna_scaffold.generate.generate_candidates", fake_generate)
    output = tmp_path / "out"

    manifest = benchmark_scaffolds.run_benchmark(config, output)

    assert load_calls == [(str(checkpoint), "cpu")]
    assert generation_loaded_values == [loaded, loaded]
    assert manifest["methods"][0]["setup_runtime_seconds"] >= 0
    assert manifest["methods"][0]["generation_runtime_seconds"] >= 0


def test_nonempty_output_directory_fails_without_deleting_unknown_files(tmp_path):
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "fixture_mode": True,
                "seed": 42,
                "candidate_count": 1,
                "bootstrap_samples": 10,
                "motifs": [{"id": "m1", "sequence": "GCGG"}],
                "methods": [{"name": "uniform", "kind": "uniform"}],
                "rnafold": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    output.mkdir()
    stale = output / "stale-v1-artifact.json"
    stale.write_text("do not delete\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "src/benchmark_scaffolds.py",
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "output directory must be empty" in completed.stderr.lower()
    assert stale.read_text(encoding="utf-8") == "do not delete\n"
    assert [path.name for path in output.iterdir()] == ["stale-v1-artifact.json"]


def test_checkpoint_shortfall_is_explicit_and_retains_rejection_reasons(monkeypatch, tmp_path):
    checkpoint = tmp_path / "tiny.ckpt"
    checkpoint.write_bytes(b"v2 checkpoint placeholder")
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "fixture_mode": True,
                "seed": 5,
                "candidate_count": 2,
                "bootstrap_samples": 10,
                "motifs": [{"id": "m1", "sequence": "GCGG"}],
                "methods": [
                    {
                        "name": "primary_model",
                        "kind": "checkpoint",
                        "checkpoint": str(checkpoint),
                        "primary": True,
                    }
                ],
                "rnafold": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )

    class ShortBatch(list):
        audit = CandidateGenerationAudit(
            requested=2,
            max_attempts=4,
            attempted=4,
            accepted=1,
            rejected={"duplicate": 3},
        )

    monkeypatch.setattr(
        benchmark_scaffolds,
        "load_scaffold_checkpoint",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "rna_scaffold.generate.generate_candidates",
        lambda *args, **kwargs: ShortBatch(
            [
                SimpleNamespace(
                    full_sequence="AAGCGGUU",
                    motif_start=2,
                    valid=True,
                    status="shortfall",
                )
            ]
        ),
    )
    output = tmp_path / "out"

    benchmark_scaffolds.run_benchmark(config, output)

    with (output / "candidate_results.csv").open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    assert len(candidates) == 2
    assert candidates[0]["failure"] == ""
    assert candidates[1]["failure"] == "generation_shortfall"
    assert candidates[1]["rejection_reason"] == "duplicate:3"
    with (output / "model_summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = next(csv.DictReader(handle))
    assert summary["status"] == "partial"
    assert summary["failure_count"] == "1"
    assert float(summary["unique_rate"]) == pytest.approx(0.5)
    assert float(summary["mean_length"]) == pytest.approx(8.0)
    assert json.loads(summary["rejection_reasons"]) == {"duplicate": 3}


def test_optional_rnafold_enrichment_records_status_and_proxy_metrics(monkeypatch, tmp_path):
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "fixture_mode": True,
                "seed": 3,
                "candidate_count": 1,
                "bootstrap_samples": 10,
                "motifs": [{"id": "m1", "sequence": "GCGG"}],
                "methods": [{"name": "uniform", "kind": "uniform"}],
                "rnafold": {
                    "enabled": True,
                    "candidate_budget_per_motif": 1,
                    "executable": "fake-rnafold",
                },
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_rnafold(sequence, motif_start, motif_end, executable, timeout_seconds):
        calls.append((sequence, motif_start, motif_end, executable, timeout_seconds))
        return RnafoldResult("ok", "." * len(sequence), -3.5, 0.5, 0.25, 0.01, "RNAfold 2.7", None)

    monkeypatch.setattr(benchmark_scaffolds, "run_rnafold", fake_rnafold)
    output = tmp_path / "out"

    benchmark_scaffolds.run_benchmark(config, output)

    assert len(calls) == 1
    with (output / "candidate_results.csv").open(encoding="utf-8", newline="") as handle:
        candidate = next(csv.DictReader(handle))
    assert candidate["rnafold_status"] == "ok"
    assert float(candidate["rnafold_mfe_kcal_mol"]) == pytest.approx(-3.5)
    assert float(candidate["rnafold_paired_fraction"]) == pytest.approx(0.5)
    assert float(candidate["rnafold_motif_paired_fraction"]) == pytest.approx(0.25)
    assert candidate["rnafold_version"] == "RNAfold 2.7"
    assert float(candidate["rnafold_runtime_seconds"]) == pytest.approx(0.01)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rnafold_status"] == "ok"
    assert manifest["software_versions"]["rnafold"] == "RNAfold 2.7"


def test_zero_rnafold_budget_is_not_run_in_candidates_and_manifest(monkeypatch, tmp_path):
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "fixture_mode": True,
                "seed": 3,
                "candidate_count": 1,
                "bootstrap_samples": 10,
                "motifs": [{"id": "m1", "sequence": "GCGG"}],
                "methods": [{"name": "uniform", "kind": "uniform"}],
                "rnafold": {"enabled": True, "candidate_budget_per_motif": 0},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        benchmark_scaffolds,
        "run_rnafold",
        lambda *args, **kwargs: pytest.fail("RNAfold must not run with zero candidate budget"),
    )
    output = tmp_path / "out"

    benchmark_scaffolds.run_benchmark(config, output)

    with (output / "candidate_results.csv").open(encoding="utf-8", newline="") as handle:
        candidate = next(csv.DictReader(handle))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert candidate["rnafold_status"] == "not_run"
    assert manifest["rnafold_status"] == "not_run"
    assert manifest["software_versions"]["rnafold"] == "not_run"


def test_missing_rnafold_executable_is_unavailable_not_not_run(tmp_path):
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "fixture_mode": True,
                "seed": 3,
                "candidate_count": 1,
                "bootstrap_samples": 10,
                "motifs": [{"id": "m1", "sequence": "GCGG"}],
                "methods": [{"name": "uniform", "kind": "uniform"}],
                "rnafold": {
                    "enabled": True,
                    "candidate_budget_per_motif": 1,
                    "executable": "definitely-not-an-rnafold-executable-v2",
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"

    benchmark_scaffolds.run_benchmark(config, output)

    with (output / "candidate_results.csv").open(encoding="utf-8", newline="") as handle:
        candidate = next(csv.DictReader(handle))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert candidate["rnafold_status"] == "unavailable"
    assert manifest["rnafold_status"] == "unavailable"
    assert manifest["software_versions"]["rnafold"] == "unavailable"


def test_atomic_writer_preserves_existing_artifact_when_replace_fails(monkeypatch, tmp_path):
    destination = tmp_path / "model_summary.csv"
    destination.write_text("old complete artifact\n", encoding="utf-8")
    monkeypatch.setattr(
        benchmark_scaffolds.os,
        "replace",
        lambda source, target: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        benchmark_scaffolds._atomic_write_text(destination, "partial new artifact\n")

    assert destination.read_text(encoding="utf-8") == "old complete artifact\n"
    assert [path.name for path in tmp_path.iterdir()] == ["model_summary.csv"]
