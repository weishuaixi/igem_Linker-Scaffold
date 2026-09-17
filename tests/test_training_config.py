import copy
from pathlib import Path

import pytest
import yaml
from torch import nn

import train
from rna_scaffold import lightning_module
from rna_scaffold.datamodule import RnaScaffoldDataModule
from rna_scaffold.lightning_module import RnaScaffoldLitModule
from rna_scaffold.tokenizer import RnaTokenizer


class TinyConfiguredEncoder(nn.Module):
    output_dim = 6

    def forward(self, input_ids, attention_mask):
        raise AssertionError("configuration construction must not run a forward pass")


def test_v5_model_config_constructs_lightning_boundary(monkeypatch):
    config_path = Path(__file__).parents[1] / "configs" / "train.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(lightning_module, "build_pretrained_encoder", lambda _: TinyConfiguredEncoder())
    module = RnaScaffoldLitModule(vocab_size=12, pad_token_id=0, **config["model"])
    assert module.left_length_loss_weight == 0.0
    assert module.right_length_loss_weight == 0.0


def test_authoritative_training_config_passes_current_entry_schema():
    config_path = Path(__file__).parents[1] / "configs" / "train.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert hasattr(train, "validate_training_config")

    train.validate_training_config(config)


def test_training_config_schema_rejects_removed_model_constructor_keys():
    config_path = Path(__file__).parents[1] / "configs" / "train.yaml"
    config = copy.deepcopy(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    config["model"]["num_encoder_layers"] = 8
    assert hasattr(train, "validate_training_config")

    with pytest.raises(TypeError, match="num_encoder_layers"):
        train.validate_training_config(config)


def test_linker_v5_config_matches_the_deployment_geometry():
    config_path = Path(__file__).parents[1] / "configs" / "train.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["data"]["max_target_length"] == 100
    assert config["data"]["short_flank_min"] == 10
    assert config["data"]["short_flank_max"] == 20
    assert config["data"]["long_flank_min"] == 15
    assert config["data"]["long_flank_max"] == 30
    assert config["data"]["iterative_mask_steps"] == 16
    assert config["data"]["validation_full_mask"] is True
    assert config["data"]["expected_partition_counts"] == {
        "train": 2869,
        "validation": 252,
        "test": 321,
    }
    assert config["data"]["expected_cluster_count"] == 1572
    assert config["model"]["normalize_base_loss_per_sequence"] is True
    assert config["model"]["left_length_loss_weight"] == 0.0
    assert config["trainer"]["monitor"] == "val/base_loss"
    train.validate_training_config(config)


def test_linker_v5_release_builds_the_locked_split_and_full_mask_validation():
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "configs" / "train.yaml").read_text(encoding="utf-8"))
    data_config = dict(config["data"])
    data_config["train_data"] = str(root / data_config["train_data"])
    data_config["cluster_manifest"] = str(root / data_config["cluster_manifest"])
    module = RnaScaffoldDataModule(tokenizer=RnaTokenizer(), **data_config)

    module.setup("fit")

    assert module.split_manifest.record_counts == {
        "train": 2869,
        "validation": 252,
        "test": 321,
        "total": 3442,
    }
    assert module.split_manifest.cluster_counts["total"] == 1572
    item = module.val_dataset[0]
    mutable = item["attention_mask"] & ~item["fixed_mask"]
    assert item["prediction_mask"].equal(mutable)
    assert item["input_ids"][mutable].eq(module.tokenizer.token_to_id[module.tokenizer.special.mask]).all()


def test_training_config_rejects_preferred_length_below_the_minimum():
    config_path = Path(__file__).parents[1] / "configs" / "train.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for key in ("short_flank_min", "short_flank_max", "long_flank_min", "long_flank_max"):
        config["data"].pop(key, None)
    config["data"]["preferred_total_scaffold_length"] = 20

    with pytest.raises(ValueError, match="preferred_total_scaffold_length"):
        train.validate_training_config(config)


def test_training_config_rejects_self_conditioning_without_conditioning_embeddings():
    config_path = Path(__file__).parents[1] / "configs" / "train.yaml"
    config = copy.deepcopy(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    config["model"]["self_conditioning_probability"] = 0.5
    config["model"]["use_conditioning_embeddings"] = False

    with pytest.raises(ValueError, match="self-conditioning"):
        train.validate_training_config(config)


def test_training_config_rejects_length_losses_when_length_heads_are_removed():
    config_path = Path(__file__).parents[1] / "configs" / "train.yaml"
    config = copy.deepcopy(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    config["model"]["predict_flank_lengths"] = False
    config["model"]["left_length_loss_weight"] = 0.05

    with pytest.raises(ValueError, match="length loss weights"):
        train.validate_training_config(config)


def test_training_config_accepts_optional_v4_training_controls():
    config_path = Path(__file__).parents[1] / "configs" / "train.yaml"
    config = copy.deepcopy(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    config["model"].update(
        {
            "use_conditioning_embeddings": True,
            "conditioning_relative_distance_clip": 32,
            "pretrained_fusion": "gated",
            "length_pooling": "attention",
            "predict_flank_lengths": False,
            "left_length_loss_weight": 0.0,
            "right_length_loss_weight": 0.0,
            "self_conditioning_probability": 0.5,
            "self_conditioning_validation": True,
            "denoise_steps": 16,
            "composition_loss_weight": 0.05,
            "homopolymer_loss_weight": 0.02,
            "homopolymer_max_run": 6,
            "lora_lr_multiplier": 0.25,
            "exclude_norm_bias_embedding_from_weight_decay": True,
        }
    )

    train.validate_training_config(config)


def test_training_config_rejects_mismatched_iterative_and_model_denoise_steps():
    config_path = Path(__file__).parents[1] / "configs" / "train.yaml"
    config = copy.deepcopy(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    config["model"]["denoise_steps"] = 12

    with pytest.raises(ValueError, match="iterative_mask_steps must equal model.denoise_steps"):
        train.validate_training_config(config)


def test_linker_v5_config_removes_noisy_length_supervision_and_enables_task_conditioning():
    config_path = Path(__file__).parents[1] / "configs" / "train.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["data"]["max_target_length"] == 100
    assert config["data"]["train_views_per_record"] == 4
    assert config["data"]["validation_views_per_record"] == 4
    assert config["data"]["cluster_balanced_sampling"] is True
    assert config["data"]["iterative_full_mask_probability"] == pytest.approx(0.35)
    assert config["data"]["stage_aware_context_noise"] is True
    assert config["data"]["preferred_motifs"] == ["GCGG"]
    assert "preferred_total_scaffold_length" not in config["data"]
    assert "min_random_mask_fraction" not in config["data"]
    assert "max_random_mask_fraction" not in config["data"]
    assert "mean_span_length" not in config["data"]
    assert config["model"]["predict_flank_lengths"] is False
    assert config["model"]["left_length_loss_weight"] == 0.0
    assert config["model"]["right_length_loss_weight"] == 0.0
    assert config["model"]["use_conditioning_embeddings"] is True
    assert config["model"]["pretrained_fusion"] == "gated"
    assert config["model"]["pretrained"]["expected_checkpoint_sha256"] == (
        "5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99"
    )
    assert config["model"]["self_conditioning_probability"] == 0
    assert config["model"]["self_conditioning_validation"] is False
    assert config["model"]["lora_lr_multiplier"] < 1
    assert config["trainer"]["monitor"] == "val/base_loss"
    train.validate_training_config(config)
