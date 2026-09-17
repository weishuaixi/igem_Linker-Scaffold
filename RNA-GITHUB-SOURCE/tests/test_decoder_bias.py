from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from rna_scaffold.decoding import DecodingSettings, iterative_denoise
from rna_scaffold.generate import GenerationSettings, resolve_generation_policies
from rna_scaffold.model import ScaffoldModelOutput
from rna_scaffold.tokenizer import RnaTokenizer


class ConstantPredictor(torch.nn.Module):
    max_length = 100

    def __init__(self):
        super().__init__()
        self.contexts = []

    def forward(self, input_ids, attention_mask=None, self_condition_probs=None, **kwargs):
        if self_condition_probs is not None:
            self.contexts.append(self_condition_probs.clone())
        logits = torch.tensor([0.25, 0.23, 0.24, 0.28]).log().expand(*input_ids.shape, 4)
        return ScaffoldModelOutput(logits, None, None)


def sample(strategy, seed, steps=16, conditioning=False):
    model = ConstantPredictor()
    result = iterative_denoise(
        model,
        RnaTokenizer(),
        "GCGG",
        44,
        20,
        DecodingSettings(denoise_steps=steps, top_p=1, remask_strategy=strategy, use_self_conditioning=conditioning),
        torch.Generator().manual_seed(seed),
    )
    assert result.sequence[20:24] == "GCGG"
    assert result.masked_counts[0] == 40 and result.masked_counts[-1] == 0
    assert all(a >= b for a, b in zip(result.masked_counts, result.masked_counts[1:]))
    return result, model


def test_random_commit_does_not_amplify_constant_predictor():
    frequencies = {}
    for strategy in ("confidence", "random"):
        counts = Counter()
        for seed in range(96):
            result, _ = sample(strategy, seed)
            counts.update(result.sequence[:20] + result.sequence[24:])
        frequencies[strategy] = {base: counts[base] / sum(counts.values()) for base in "AUCG"}
    # Exact toy model: G has 28% marginal probability at every position/step.
    # Repeated sampled-confidence selection causes spurious concentration.
    assert frequencies["confidence"]["G"] > 0.75
    for base, probability in zip("AUCG", [0.25, 0.23, 0.24, 0.28]):
        assert abs(frequencies["random"][base] - probability) < 0.035


def test_policy_seed_and_conditioning_switch_are_explicit():
    result, model = sample("random", 42, conditioning=False)
    again, _ = sample("random", 42, conditioning=False)
    assert result == again
    assert model.contexts and all(not context.any() for context in model.contexts)
    _, enabled = sample("random", 42, conditioning=True)
    assert any(context.any() for context in enabled.contexts)


def test_generation_resolves_and_records_checkpoint_policy_defaults():
    model = SimpleNamespace(generation_remask_strategy="random", self_conditioning_probability=0)
    settings = resolve_generation_policies(GenerationSettings(), model)
    assert settings.remask_strategy == "random" and settings.self_conditioning == "off"
    legacy = resolve_generation_policies(GenerationSettings(), object())
    assert legacy.remask_strategy == "confidence" and legacy.self_conditioning == "on"
    override = resolve_generation_policies(
        GenerationSettings(remask_strategy="confidence", self_conditioning="on"), model
    )
    assert override.remask_strategy == "confidence" and override.self_conditioning == "on"


def test_random_commit_retains_hard_linker_constraints():
    model = ConstantPredictor()
    tokenizer = RnaTokenizer()
    for seed in range(12):
        result = iterative_denoise(
            model,
            tokenizer,
            "GGGGGGGG",
            38,
            15,
            DecodingSettings(
                denoise_steps=8,
                top_p=1,
                remask_strategy="random",
                max_homopolymer_run=3,
                gc_min=0.3,
                gc_max=0.7,
                enforce_gc_bounds=True,
            ),
            torch.Generator().manual_seed(seed),
        )
        from rna_scaffold.evaluation import maximum_homopolymer_run

        left, right = result.sequence[:15], result.sequence[23:]
        assert result.sequence[15:23] == "GGGGGGGG"
        assert max(maximum_homopolymer_run(left), maximum_homopolymer_run(right)) <= 3
        linker = left + right
        assert 0.3 <= (linker.count("G") + linker.count("C")) / len(linker) <= 0.7


@pytest.mark.parametrize("strategy", ["invalid", "", "greedy"])
def test_reject_unknown_strategy(strategy):
    with pytest.raises(ValueError, match="remask_strategy"):
        DecodingSettings(remask_strategy=strategy)


def test_v5_checkpoint_policy_and_matched_validation_cli(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    import yaml

    import train
    from rna_scaffold.checkpoints import load_scaffold_checkpoint
    from rna_scaffold.datamodule import RnaScaffoldDataModule
    from rna_scaffold.lightning_module import RnaScaffoldLitModule
    from scripts.check_linker_v5_validation import main

    config = yaml.safe_load(Path("configs/train_scaffold_5090_linker_v5.yaml").read_text())
    train.validate_training_config(config)
    config["model"].update(
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        pretrained={"kind": "none"},
        pretrained_fusion="add",
        dropout=0,
    )
    config["trainer"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config))
    tokenizer = RnaTokenizer()
    dm = RnaScaffoldDataModule(tokenizer=tokenizer, **config["data"])
    dm.setup("validate")
    model = RnaScaffoldLitModule(
        vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id, **config["model"]
    )
    checkpoint = tmp_path / "tiny.ckpt"
    torch.save(
        {"architecture_version": 2, "hyper_parameters": dict(model.hparams), "state_dict": model.state_dict()},
        checkpoint,
    )
    train.write_training_manifest(config_path, config, checkpoint, split_manifest=dm.split_manifest)
    loaded = load_scaffold_checkpoint(checkpoint)
    actual = resolve_generation_policies(GenerationSettings(), loaded.model)
    assert actual.remask_strategy == "random" and actual.self_conditioning == "off"
    output = tmp_path / "validation"
    monkeypatch.setattr(
        "sys.argv",
        [
            "check",
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--motifs",
            "2",
            "--candidates",
            "2",
            "--device",
            "cpu",
            "--expanded",
        ],
    )
    main()
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "completed" and report["partition"] == "validation"
    assert report["variants"]["random_16"]["count"] == 4
    assert report["real_validation_linkers"]["count"] == 2
    assert report["variants"]["uniform"]["count"] == 4
    assert report["variants"]["markov2"]["count"] == 4
    assert len(report["groups"]) == 14
    assert report["baseline_training_records"] == dm.train_dataset.record_count
    assert report["rnafold_status"].startswith("not_run")
    for group in report["groups"]:
        assert 0 <= group["mean_edit_diversity"] <= 1
        assert 0 <= group["mean_nearest_training_kmer_similarity"] <= 1
    assert {row["target_id"] for row in report["source_contexts"]} <= {
        record.target_id for record in dm.val_dataset.records
    }
