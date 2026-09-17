import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from rna_scaffold.benchmarking import RnaTrainingPrior
from scripts.diagnose_linker_v5_conditioning import BASES, markov_marginals, score, shuffled_motif


def test_markov_matches_brute_force():
    prior = RnaTrainingPrior.from_sequences(["AUCGAAAU", "GGGCCCUU"])
    exact = np.zeros((4, 4))
    mass = 0
    for sequence in itertools.product(BASES, repeat=4):
        if sequence[1:3] != ("U", "C"):
            continue
        p = prior.initial[sequence[0]]
        for a, b in itertools.pairwise(sequence):
            p *= prior.transition[a][b]
        mass += p
        for i, b in enumerate(sequence):
            exact[i, BASES.index(b)] += p
    np.testing.assert_allclose(markov_marginals(prior, 4, 1, "UC"), exact / mass)


def test_uniform_score_and_shuffle():
    result = score(np.full((4, 4), 0.25), np.arange(4), np.ones(4, dtype=bool))
    assert abs(result["nll"] - math.log(4)) < 1e-12
    assert result["accuracy"] == 0.25
    assert shuffled_motif("AAAA", random.Random(1)) is None
    shuffled = shuffled_motif("GCGG", random.Random(42))
    assert shuffled != "GCGG" and sorted(shuffled) == sorted("GCGG")


def test_tiny_checkpoint_end_to_end(tmp_path, monkeypatch):
    import train
    from rna_scaffold.datamodule import RnaScaffoldDataModule
    from rna_scaffold.lightning_module import RnaScaffoldLitModule
    from rna_scaffold.tokenizer import RnaTokenizer
    from scripts.diagnose_linker_v5_conditioning import main

    cfg = yaml.safe_load(Path("configs/train_scaffold_5090_linker_v5.yaml").read_text())
    cfg["model"].update(
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        pretrained={"kind": "none"},
        pretrained_fusion="add",
        dropout=0,
    )
    cfg["trainer"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump(cfg))
    tokenizer = RnaTokenizer()
    dm = RnaScaffoldDataModule(tokenizer=tokenizer, **cfg["data"])
    dm.setup("validate")
    model = RnaScaffoldLitModule(vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id, **cfg["model"])
    checkpoint = tmp_path / "tiny.ckpt"
    torch.save(
        {"architecture_version": 2, "hyper_parameters": dict(model.hparams), "state_dict": model.state_dict()},
        checkpoint,
    )
    train.write_training_manifest(config, cfg, checkpoint, split_manifest=dm.split_manifest)
    output = tmp_path / "diagnosis"
    monkeypatch.setattr(
        "sys.argv", ["diagnose", "--config", str(config), "--output", str(output), "--clusters", "2", "--device", "cpu"]
    )
    main()
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "completed" and report["clusters"] == 2
    assert report["training_records"] == dm.train_dataset.record_count
    assert abs(report["means"]["uniform"]["nll"] - math.log(4)) < 1e-10
    assert report["means"]["model_correct"]["nll"] > 0
    rows = [json.loads(line) for line in (output / "contexts.jsonl").read_text().splitlines()]
    for row in rows:
        assert row["scored_tokens"] == row["left"] + row["right"]
        assert row["target_id"] in dm.split_manifest.partitions["validation"]
