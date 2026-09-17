# Development and mathematical methods

## Candidate ranking

The RNAfold workflow ranks a generated pool using seven heuristic components: decoding likelihood (0.25), token confidence (0.20), GC quality (0.20), homopolymer quality (0.15), minimum free energy quality (0.05), paired fraction (0.05), and motif accessibility (0.10).

Each component is min–max normalized within the pool. Energy quality favors energy per nucleotide near −0.3 kcal/mol/nt, not the lowest possible energy. Motif accessibility favors unpaired motif bases. These weights have not been calibrated against functional measurements, and scores cannot be compared across different pools.

[Back to README](../README.md)

Commands below run from the repository root.

## Model architecture and training

The model combines token, position and context-role embeddings with normalized, projected [RNA-FM](https://github.com/ml4bio/RNA-FM) features through a learned residual gate. A six-layer Transformer encoder produces A/U/C/G logits. It uses width 384, six attention heads, feed-forward width 1,536 and dropout 0.1. Context features distinguish fixed/predicted positions, left/right roles, signed motif-relative distance clipped at 32, and masking noise fraction. Padding is excluded from attention and supervision.

LoRA adapts the last four RNA-FM layers with rank 8, alpha 16 and dropout 0.05. V5 uses random-position commitment during iterative decoding; learned length heads and self-conditioning are disabled in the active configuration. A fresh task-training run still uses pretrained RNA-FM initialization.

The active objective is:

```plaintext
L = L_base + 0.02 * L_composition + 0.02 * L_homopolymer
```

`L_base` is A/U/C/G cross-entropy with label smoothing 0.02, averaged within each sequence and then across sequences. Only supervised linker positions contribute. `L_composition` is Jensen–Shannon divergence between predicted and target mean composition at supervised positions. `L_homopolymer` penalizes expected same-base windows of seven nucleotides, excluding motif/padding windows and windows without supervision. Length-loss weights are zero. Raw marginal NLL, smoothed reconstruction loss and total loss are different metrics.

Training uses AdamW with task learning rate `3e-4`, LoRA learning-rate multiplier 0.1, weight decay 0.05 with configured exclusions, and warmup-cosine scheduling. Batch size is 8 with four-step accumulation, BF16 mixed precision and gradient clipping at 1.0. The limit is 40 epochs, with early-stopping patience 8 and checkpoint selection by `val/base_loss`. The YAML configuration is authoritative.

## Development history

| Stage | Attempt and motivation | Observation or resulting decision |
| --- | --- | --- |
| Early scope | Explore sequence completion and possible downstream 3D modeling | Current deliverable was narrowed to an auditable sequence-generation prototype; 3D functionality is not claimed |
| General scaffold model | Compare no-RNA-FM, frozen RNA-FM and LoRA-assisted configurations | Historical configurations were removed from the final source release; their earlier existence alone is not evidence of a measured advantage |
| V3 linker specialization | Restrict flank lengths; add stage masking, full-mask validation, sequence-balanced loss and deployment constraints | An early constrained generation run accepted 0/500 attempts: 417 homopolymer rejections and 83 GC rejections |
| Constrained decoder repair | Sample using feasible suffix probability mass; repair packaging/path issues | A subsequent example accepted 20/20 candidates; this demonstrated constraint satisfaction, not an improvement in learned biological quality |
| V4 | Remove uninformative length targets; improve gated fusion, LoRA optimizer groups, balanced views, regularization and audit/ranking behavior | Formal generation benchmarking still exposed low composition entropy and excessive repeated bases |
| Benchmark acceleration | Exact edit-distance acceleration, bounded similarity caching and per-motif resumable outputs | Removed a severe CPU-side bottleneck without intentionally changing metric definitions; equivalence tests protect this behavior |
| Decoder diagnosis | Separate data composition from iterative selection effects | A controlled constant-predictor experiment reproduced strong GC amplification under sampled-confidence selection |
| V5 | Use random-position commitment; disable unproven self-conditioning feedback in the active configuration | Expanded validation removed the severe compositional collapse; conditioning tests found a small predictive advantage |
| September 10 retraining | Repeat the V5 run after loss of the original server checkpoint | Training and exported checkpoint integrity were verified; detailed original-model benchmark results have not been re-established for the replacement weights |

A training-view audit showed mean maximum run approximately 3.39 and only about 0.88% of sampled views exceeding six bases, which did not explain the much stronger collapse during generation. In a controlled experiment using constant probabilities `[A=0.25, U=0.23, C=0.24, G=0.28]`, sampled-confidence selection increased the generated G fraction from about 28.6% in one pass to 65.7% at four steps and 88.5% at sixteen steps. This identifies a decoding mechanism that can amplify bias without requiring a biased training dataset. It does not rule out all data-related limitations.

Random-position selection removes this particular selection preference; it does not turn iterative masked generation into an exact joint-sequence sampler. Disabling self-conditioning was a conservative configuration decision, not an isolated demonstration that self-conditioning caused the original failure.
