from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import benchmark_scaffolds
import rna_scaffold
import train
from rna_scaffold.checkpoints import validate_scaffold_checkpoint_version
from rna_scaffold.data import RnaMotifDenoisingDataset
from rna_scaffold.decoding import DecodedScaffold, build_flank_pair_sampler
from rna_scaffold.generate import GenerationSettings, generate_candidates
from rna_scaffold.model import MotifDenoisingTransformer, ScaffoldModelOutput
from rna_scaffold.pretrained import PretrainedEncoderConfig, build_pretrained_encoder
from rna_scaffold.records import RnaSequenceRecord
from rna_scaffold.tokenizer import RnaTokenizer


def _tiny_model(max_length: int) -> MotifDenoisingTransformer:
    return MotifDenoisingTransformer(
        vocab_size=12,
        pad_token_id=0,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_length=max_length,
    )


def test_i1_eight_nt_total_canvas_is_eligible_even_with_larger_soft_preference():
    dataset = RnaMotifDenoisingDataset(
        records=[RnaSequenceRecord("boundary", "AAGCGGUU", None, "unit")],
        tokenizer=RnaTokenizer(),
        max_length=8,
        min_motif_length=4,
        max_motif_length=4,
        motif_length_buckets=None,
        min_flank_length=2,
        min_total_scaffold_length=8,
        preferred_total_scaffold_length=24,
        full_mask_probability=1.0,
        span_mask_probability=0.0,
    )

    item = dataset[0]

    assert len(dataset) == 1
    assert item["target_left_length"].item() == 2
    assert item["target_right_length"].item() == 2


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: _tiny_model(513), id="model"),
        pytest.param(
            lambda: RnaMotifDenoisingDataset(
                records=[RnaSequenceRecord("long", "A" * 600, None, "unit")],
                tokenizer=RnaTokenizer(),
                max_length=513,
                min_motif_length=4,
                max_motif_length=4,
                motif_length_buckets=None,
            ),
            id="dataset",
        ),
        pytest.param(lambda: GenerationSettings(max_length=513), id="generation"),
    ],
)
def test_i2_all_runtime_boundaries_reject_513(factory):
    with pytest.raises(ValueError, match="512"):
        factory()


def test_i2_training_config_cross_validates_data_and_model_lengths():
    config = yaml.safe_load(Path("configs/train_scaffold_5090_linker_v5.yaml").read_text(encoding="utf-8"))
    config["data"]["max_target_length"] = 512
    config["model"]["max_length"] = 511

    with pytest.raises(ValueError, match="max_target_length.*model.max_length"):
        train.validate_training_config(config)


def test_i3_benchmark_reconstructs_train_and_test_from_the_canonical_split(tmp_path):
    records_path = tmp_path / "records.csv"
    records_path.write_text(
        "target_id,sequence\n"
        "train_a,AAGCGGUU\n"
        "train_b,CCGGAUCG\n"
        "held_out,UUAAUUGG\n",
        encoding="utf-8",
    )
    clusters_path = tmp_path / "clusters.tsv"
    clusters_path.write_text(
        "cluster_id\tsequence_id\tcluster_size\n"
        "c1\ttrain_a\t1\n"
        "c2\ttrain_b\t1\n"
        "c3\theld_out\t1\n",
        encoding="utf-8",
    )
    config = {
        "training_data": str(records_path),
        "cluster_manifest": str(clusters_path),
        "split_contract": {
            "seed": 7,
            "val_fraction": 0.0,
            "test_fraction": 1 / 3,
            "min_motif_length": 4,
            "min_flank_length": 2,
            "min_total_scaffold_length": 8,
            "max_length": 512,
        },
    }
    loader = getattr(benchmark_scaffolds, "_load_benchmark_partitions", None)

    assert loader is not None
    partitions = loader(config)
    train_ids = {record.target_id for record in partitions.train_records}
    test_ids = {record.target_id for record in partitions.test_records}
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == {"train_a", "train_b", "held_out"}


def test_i4_checkpoint_identity_mismatch_is_reported_not_accepted():
    validator = getattr(benchmark_scaffolds, "_validate_method_identity", None)
    assert validator is not None
    method = {
        "name": "claimed_no_rnafm",
        "kind": "checkpoint",
        "expected_pretrained_kind": "none",
        "expected_pretrained_mode": "none",
        "expected_decoder_mode": "iterative_remasking",
        "expected_denoise_steps": 12,
        "generation": {"denoise_steps": 12},
    }
    loaded = SimpleNamespace(
        architecture_version=2,
        pretrained_metadata={"kind": "rna_fm", "mode": "lora"},
    )

    with pytest.raises(ValueError, match="expected.*none.*actual.*rna_fm"):
        validator(method, loaded)


def test_i4_identity_mismatch_publishes_failed_terminal_manifest(monkeypatch, tmp_path):
    checkpoint = tmp_path / "wrong.ckpt"
    checkpoint.write_bytes(b"checkpoint-placeholder")
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "fixture_mode": True,
                "motifs": [{"id": "m", "sequence": "GCGG"}],
                "methods": [
                    {
                        "name": "claimed_no_rnafm",
                        "kind": "checkpoint",
                        "checkpoint": str(checkpoint),
                        "expected_pretrained_kind": "none",
                        "expected_pretrained_mode": "none",
                        "expected_decoder_mode": "iterative_remasking",
                        "expected_denoise_steps": 12,
                        "generation": {"denoise_steps": 12},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        benchmark_scaffolds,
        "load_scaffold_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(
            architecture_version=2,
            pretrained_metadata={"kind": "rna_fm", "mode": "lora"},
        ),
    )

    manifest = benchmark_scaffolds.run_benchmark(config, tmp_path / "output")

    assert manifest["status"] == "failed"
    assert manifest["methods"][0]["status"] == "failed"
    assert manifest["methods"][0]["actual_identity"]["pretrained_kind"] == "rna_fm"


def test_i5_final_training_and_matched_baselines_exist():
    assert hasattr(train, "write_training_manifest")
    config = yaml.safe_load(Path("configs/benchmark_scaffolds_linker_v5.yaml").read_text(encoding="utf-8"))
    assert config["training_config"] == "configs/train_scaffold_5090_linker_v5.yaml"
    assert [method["kind"] for method in config["methods"]] == ["uniform", "markov1", "markov2", "checkpoint"]
    assert config["methods"][-1]["primary"] is True


def test_i5_training_manifest_is_hashed_and_consumed_by_benchmark(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "epoch=2.ckpt"
    checkpoint.write_bytes(b"reviewed-v2-checkpoint")
    config_path = tmp_path / "train.yaml"
    config = {
        "model": {"pretrained": {"kind": "none"}},
        "trainer": {"checkpoint_dir": str(checkpoint_dir)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    manifest = train.write_training_manifest(config_path, config, checkpoint)
    manifest_path = checkpoint_dir / "training_manifest.json"
    resolved, consumed = benchmark_scaffolds._checkpoint_from_training_manifest(
        {
            "training_manifest": str(manifest_path),
            "expected_pretrained_kind": "none",
            "expected_pretrained_mode": "none",
        }
    )

    assert manifest["status"] == "completed"
    assert resolved.resolve() == checkpoint.resolve()
    assert consumed["best_checkpoint"]["sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()


def test_i5_training_manifest_rejects_stale_producer_config(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "best.ckpt"
    checkpoint.write_bytes(b"reviewed-v2-checkpoint")
    config_path = tmp_path / "train.yaml"
    config = {
        "model": {"pretrained": {"kind": "none"}},
        "trainer": {"checkpoint_dir": str(checkpoint_dir)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    train.write_training_manifest(config_path, config, checkpoint)
    config_path.write_text(yaml.safe_dump({**config, "seed": 99}), encoding="utf-8")

    with pytest.raises(ValueError, match="config SHA-256 mismatch"):
        benchmark_scaffolds._checkpoint_from_training_manifest(
            {
                "training_manifest": str(checkpoint_dir / "training_manifest.json"),
                "training_config": str(config_path),
                "expected_pretrained_kind": "none",
                "expected_pretrained_mode": "none",
            }
        )


def test_i5_training_manifest_rejects_stale_split_manifest(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "best.ckpt"
    checkpoint.write_bytes(b"reviewed-v2-checkpoint")
    config_path = tmp_path / "train.yaml"
    config = {
        "model": {"pretrained": {"kind": "none"}},
        "trainer": {"checkpoint_dir": str(checkpoint_dir)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    train.write_training_manifest(
        config_path,
        config,
        checkpoint,
        split_manifest={"train_ids": ["old"]},
    )

    with pytest.raises(ValueError, match="split manifest mismatch"):
        benchmark_scaffolds._checkpoint_from_training_manifest(
            {
                "training_manifest": str(checkpoint_dir / "training_manifest.json"),
                "training_config": str(config_path),
                "expected_pretrained_kind": "none",
                "expected_pretrained_mode": "none",
            },
            expected_split_manifest={"train_ids": ["current"]},
        )


def test_i6_public_runtime_exposes_only_v2_generation_surface():
    forbidden = {
        "MaskedScaffoldExample",
        "MaskedScaffoldPrompt",
        "RnaMaskedScaffoldDataset",
        "build_auto_masked_scaffold_prompts",
        "build_motif_scaffold_sequence",
        "build_random_natural_scaffold_result",
        "build_single_best_result",
        "generate_markov_baseline",
    }

    assert forbidden.isdisjoint(vars(rna_scaffold))
    assert not hasattr(DecodedScaffold("AUGC", -1.0, (0,)), "unresolved_counts")


def test_i7_inference_preflight_uses_restricted_torch_load(monkeypatch, tmp_path):
    checkpoint = tmp_path / "v2.ckpt"
    torch.save({"architecture_version": 2}, checkpoint)
    original = torch.load
    observed: list[object] = []

    def audited_load(*args, **kwargs):
        observed.append(kwargs.get("weights_only"))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", audited_load)

    validate_scaffold_checkpoint_version(checkpoint)

    assert observed == [True]


def test_i8_failed_staging_never_publishes_a_partial_benchmark(monkeypatch, tmp_path):
    config = tmp_path / "smoke.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "seed": 7,
                "candidate_count": 2,
                "bootstrap_samples": 10,
                "motifs": [{"id": "m", "sequence": "GCGG"}],
                "methods": [{"name": "uniform", "kind": "uniform"}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "published"
    original = benchmark_scaffolds._atomic_write_text
    writes = 0

    def fail_on_third_write(path, content):
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError("simulated artifact failure")
        return original(path, content)

    monkeypatch.setattr(benchmark_scaffolds, "_atomic_write_text", fail_on_third_write)

    with pytest.raises(OSError, match="simulated artifact failure"):
        benchmark_scaffolds.run_benchmark(config, output, smoke_test=True)

    assert not output.exists() or not any(output.iterdir())


def test_i9_candidate_generation_builds_the_pair_tensor_once(monkeypatch):
    class PlacementModel(torch.nn.Module):
        max_length = 10

        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(max_length=10)

        def forward(self, input_ids, attention_mask=None):
            token_logits = torch.zeros((1, input_ids.shape[1], 4))
            left = torch.zeros((1, 11))
            right = torch.zeros((1, 11))
            return ScaffoldModelOutput(token_logits, left, right)

    loaded = SimpleNamespace(
        model=PlacementModel(),
        tokenizer=RnaTokenizer(),
        max_length=10,
        checkpoint_sha256="test",
    )
    cartesian_prod = torch.cartesian_prod
    calls = 0

    def counted_cartesian_prod(*args, **kwargs):
        nonlocal calls
        calls += 1
        return cartesian_prod(*args, **kwargs)

    def fake_denoise(model, tokenizer, motif, total_length, motif_start, *args, **kwargs):
        right_length = total_length - motif_start - len(motif)
        return DecodedScaffold("A" * motif_start + motif + "U" * right_length, -0.5, (0,))

    monkeypatch.setattr(torch, "cartesian_prod", counted_cartesian_prod)
    monkeypatch.setattr("rna_scaffold.decoding.iterative_denoise", fake_denoise)

    generate_candidates(
        "ignored.ckpt",
        "GCGG",
        GenerationSettings(num_candidates=3, max_attempts=3, max_length=10, seed=9),
        loaded_checkpoint=loaded,
    )

    assert calls == 1


def test_i9_default_512_by_256_pair_sampling_has_cpu_upper_bound():
    output = ScaffoldModelOutput(
        token_logits=torch.zeros((1, 1, 4)),
        left_length_logits=torch.zeros((1, 513)),
        right_length_logits=torch.zeros((1, 513)),
    )
    generator = torch.Generator().manual_seed(42)

    started = time.perf_counter()
    sampler = build_flank_pair_sampler(
        output,
        motif_length=4,
        max_length=512,
        min_scaffold_length=8,
        min_flank_length=2,
    )
    selected = [sampler.select(generator) for _ in range(256)]
    elapsed = time.perf_counter() - started

    assert sampler.pairs.shape == (127765, 2)
    assert len(set(selected)) == 256
    assert elapsed < 3.0


def test_m2_rna_fm_release_is_pinned_and_requires_reviewed_weight_digest(monkeypatch):
    requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "rna-fm==0.2.2" in requirements

    monkeypatch.delenv("RNA_FM_EXPECTED_SHA256", raising=False)
    monkeypatch.setattr(
        "rna_scaffold.pretrained.load_rna_fm",
        lambda checkpoint: torch.nn.Identity(),
    )
    config = PretrainedEncoderConfig(
        kind="rna_fm",
        checkpoint="weights.pth",
        mode="frozen",
    )
    with pytest.raises(ValueError, match="expected.*SHA-256"):
        build_pretrained_encoder(config)


def test_m2_benchmark_records_rna_fm_and_lightning_versions():
    versions = benchmark_scaffolds._software_versions()

    assert {"lightning", "rna_fm"} <= set(versions)


def test_m3_held_out_panel_is_deterministic_and_traceable():
    builder = getattr(benchmark_scaffolds, "build_held_out_motif_panel", None)
    assert builder is not None
    records = [
        RnaSequenceRecord("test_a", "AAGCGGUUAAGC", None, "unit"),
        RnaSequenceRecord("test_b", "CCGGAUCGCCGG", None, "unit"),
    ]

    first = builder(records, panel_size=4, seed=13, motif_lengths=(4, 6), min_flank_length=2)
    second = builder(records, panel_size=4, seed=13, motif_lengths=(4, 6), min_flank_length=2)

    assert first == second
    assert first
    assert all(row["source_partition"] == "test" for row in first)
    assert all(row["source_target_id"] in {"test_a", "test_b"} for row in first)
    assert all(len(row["source_sequence_sha256"]) == 64 for row in first)
    assert all(row["length_stratum"] and row["gc_stratum"] for row in first)


def test_m3_small_common_motif_count_is_explicitly_not_release_grade():
    rows = [
        {
            "method": method,
            "motif_id": motif,
            "status": "ok",
            "valid_rate": value,
            "motif_preservation_rate": 1.0,
            "unique_rate": 1.0,
            "mean_normalized_edit_diversity": value,
            "mean_kmer_diversity": value,
            "mean_nearest_training_kmer_similarity": value,
        }
        for method, value in (("left", 0.5), ("right", 0.4))
        for motif in ("m1", "m2")
    ]

    artifact = benchmark_scaffolds._paired_bootstrap_artifact(
        rows,
        ["left", "right"],
        seed=7,
        samples=20,
        min_common_motifs_for_release=20,
    )

    assert artifact["comparisons"]
    assert all(row["common_motif_n"] == 2 for row in artifact["comparisons"])
    assert all(row["grade"] == "exploratory_not_release_grade" for row in artifact["comparisons"])


def test_formal_config_dry_run_validates_without_claiming_external_evidence():
    validator = getattr(benchmark_scaffolds, "validate_benchmark_config", None)

    assert validator is not None
    audit = validator(Path("configs/benchmark_scaffolds_linker_v5.yaml"))
    assert audit["status"] == "validated_external_inputs_pending"
    assert audit["motif_panel"]["source_partition"] == "test"
    assert audit["scientific_release_status"] == "not_run"
