# Train the RNA linker model

[Back to README](../README.md)

Run commands from the repository root.


The data are natural sequence segments from a locally prepared Stanford RNA collection, not RNA3DB all-atom supervision or experimentally validated synthetic linker labels. Filtering retained 3,442 of 4,125 records across 1,572 eligible clusters. Filters included minimum length 29, GC fraction 0.10–0.90, maximum homopolymer run 12 and composition entropy at least 1.20 bits.

Cluster-disjoint splits contain 2,869 training, 252 validation and 321 test records, with prepared clustering thresholds of 0.8 identity and bidirectional coverage. Four views per record are sampled; they are not additional independent molecules. Training uses 4–30 nt motifs, cluster-balanced sampling, a 16-stage masking schedule, full-mask probability 0.35 and stage-aware context noise probability 0.05. Eligible `GCGG` contexts receive a sampling preference of 0.5. Validation masks all unknown linker bases.

This source release excludes training data. Training and full evaluation additionally require:

| File | SHA-256 |
| --- | --- |
| `data/processed/rna_linker_v3_training.csv` | `5220058e6074c8fed218f968aa8d1c5d69dbdb72e856fb3b76f444eb33249de1` |
| `data/processed/rna_linker_v3_clusters.tsv` | `50eab29ce5494698f2e7db0ab6ab0fa8e54bbb9db0128e51b25bcc83e07a1a4a` |

The `v3` filenames are intentional: V5 reuses those assets. Preparation utilities are `scripts/prepare_stanford_scaffold_data.py` and `scripts/filter_scaffold_training_data.py`; use `--help` for their arguments. An exact source-release accession is not established by the local merged table and should accompany any redistributed dataset.

Once the canonical assets are available, start a fresh run in a separate working copy without existing task checkpoints:

```shell
mkdir -p logs
nohup bash scripts/start_training.sh > logs/train.log 2>&1 &
tail -f logs/train.log
```

Do not delete the delivered checkpoint to make the command proceed. The wrapper refuses silent overwrite, uses a process lock and exports the best model after successful training. Follow [manual installation](installation.md) before starting training. Offline execution does not keep a released cloud instance alive; download and verify exports before releasing a server.

