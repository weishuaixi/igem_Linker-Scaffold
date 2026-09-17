# RNA linker sequence model

This directory contains the V5 motif-conditioned RNA linker model. Given a fixed RNA motif, it generates **left linker – motif – right linker**. The intended design has one short flank of 10–20 nucleotides and one long flank of 15–30 nucleotides, with total length at most 100. Either flank may be the short one; the motif is preserved exactly.

Python owns model loading, sequence generation, training and comparative evaluation. The source repository excludes binary weights. The companion RNA-V5-MODEL-ASSETS.zip contains the trained V5 checkpoint and RNA-FM weights needed for inference; extract it into the repository root before running the examples. It does not generate all-atom coordinates or establish biological function.

## Authorship and AI assistance

The main project framework was designed and implemented by the project author. OpenAI Codex assisted with code writing and revision, debugging, testing, packaging, and documentation during development. This assistance included implementation changes, not only language editing. The project author is responsible for reviewing the delivered work and its scientific claims. Third-party models and software, including RNA-FM and ViennaRNA, remain attributable to their respective authors.

## Contents

- `generate_scaffold.py`: generate candidates and export FASTA/JSONL.
- `train.py`: train the masked sequence model and record checkpoint provenance.
- `benchmark_scaffolds.py`: compare the model with Uniform, Markov-1 and Markov-2.
- `validate_scaffolds.py`: optional candidate validation and ranking.
- `rna_scaffold/`: Transformer, RNA-FM adaptation, losses, decoding and data/evaluation utilities.
- `configs/train_scaffold_5090_linker_v5.yaml`: final training configuration.
- `configs/benchmark_scaffolds_linker_v5.yaml`: matched V5 and baseline evaluation configuration.
- `scripts/`: data preparation, offline execution, diagnostics and model export.
- `tests/`: regression checks for motif preservation, data splitting, decoding, metrics and checkpoint handling.
- `checkpoints_scaffold_linker_v5/`: final checkpoint and training manifest, provided in the companion model asset ZIP.
- `.cache/torch/hub/checkpoints/RNA-FM_pretrained.pth`: required pretrained dependency, provided in the companion model asset ZIP.

## Install the environment

### Generate and rank with RNAfold

For Linux x86_64 with Conda or [Miniforge](https://github.com/conda-forge/miniforge) already installed, run from the repository after extracting the model assets:

```bash
bash scripts/setup_inference.sh cuda
bash scripts/generate_best.sh GCGG outputs/GCGG_ranked cuda
cat outputs/GCGG_ranked/best.fasta
```

The installer creates two project-local environments: `.inference_env` for the model and `.rnafold_env` for ViennaRNA 2.7.2. It does not modify the existing training environment. Initial installation requires internet access; generation uses the supplied local weights and runs offline afterward. The package includes installation instructions and scripts, not a platform-specific RNAfold binary. Installing the ViennaRNA Python bindings alone is not a substitute for the required `RNAfold` command; see the [ViennaRNA installation documentation](https://viennarna.readthedocs.io/en/latest/install.html).

For a CPU-only machine, use `cpu` in both commands. No environment activation is necessary. Use a new output directory for every run. Installed environments and generated outputs are excluded from Git; the installer records dependency versions under `outputs/`.

This workflow generates 128 candidates using four-step random decoding, preserves the motif, and enforces short/long flank ranges of 10–20/15–30 nt, total length at most 100, GC fraction 0.30–0.70 and maximum homopolymer run six. It then runs RNAfold on every candidate and exports:

- `candidates.fasta` and `candidates.jsonl`: generated candidate pool and generation audit.
- `rnafold_results.jsonl`: predicted secondary structures, energies and pairing metrics.
- `ranked.jsonl`: complete ranking and component scores.
- `best.fasta` and `best.json`: the single top-ranked candidate and its details.
- `run_manifest.json`: settings, software preflight, checkpoint identity, file hashes and completion status.

A missing or failed RNAfold evaluation, or an incomplete candidate budget, stops the workflow without exporting a best sequence. `scripts/design_linker.py --help` exposes candidate count, seed, checkpoint, device and RNAfold executable options for an already configured Python environment.

“Best” means **top-ranked within this generated pool**, not experimentally optimal. The existing heuristic uses weights 0.25 for decoding likelihood, 0.20 for token confidence, 0.20 for GC quality, 0.15 for homopolymer quality, 0.05 for MFE quality, 0.05 for paired fraction and 0.10 for motif accessibility. Components are min–max normalized within the pool; scores are not calibrated probabilities or comparable across different pools. MFE quality prefers energy per nucleotide near −0.3 kcal/mol/nt rather than simply rewarding the lowest energy. The accessibility term assumes an exposed motif is desirable; this must be reconsidered for motifs intended to pair. The ranking weights have not been validated against functional measurements.

### Manual model-only installation

The tested server setup used Linux, Python 3.10, an RTX 5090 with 32 GB VRAM, and PyTorch 2.7.1 built for CUDA 12.8. An installed CUDA 12.6 build with no `sm_120` support previously failed on this GPU despite reporting CUDA availability. Use the explicit CUDA 12.8 wheel build; see [PyTorch's versioned installation instructions](https://pytorch.org/get-started/previous-versions/#v271).

Run from the project directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install 'setuptools>=69' wheel
python -m pip install 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' 'torchaudio==2.7.1+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c constraints-v5-server.txt -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -c 'import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); x=torch.randn(256,256,device="cuda",dtype=torch.bfloat16); y=x@x; torch.cuda.synchronize(); print(y.shape)'
```

The actual GPU operation is the compatibility check; `torch.cuda.is_available()` alone is insufficient. Initial dependency installation requires network access or a separately prepared wheel cache. The dependency constraints are not a complete platform-independent lockfile; save the installed-environment record for each run.

## Use the final model

From this directory, with the environment activated and the model assets extracted:

```bash
export TORCH_HOME="$PWD/.cache/torch"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RNA_FM_EXPECTED_SHA256=5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
BEST=$(python -c 'import json; from pathlib import Path; d=Path("checkpoints_scaffold_linker_v5"); m=json.loads((d/"training_manifest.json").read_text()); p=Path(m["best_checkpoint"]["path"]); print(p if p.is_absolute() else d/p)')
mkdir -p outputs
python generate_scaffold.py \
  --checkpoint "$BEST" --motif GCGG \
  --num-candidates 20 --max-length 100 \
  --short-flank-min 10 --short-flank-max 20 \
  --long-flank-min 15 --long-flank-max 30 \
  --length-sampling uniform --remask-strategy random \
  --self-conditioning off --denoise-steps 4 \
  --temperature 1.0 --top-p 1.0 --device cuda \
  --output outputs/GCGG_v5.jsonl \
  --fasta-output outputs/GCGG_v5.fasta
```

FASTA contains nucleotide sequences, not folded structures. JSONL includes candidate information and the generation audit. A shortfall from the requested candidate count returns exit code 2; inspect rejection reasons rather than assuming every requested candidate exists.

For deployment constraints, optionally add `--max-homopolymer-run 6 --gc-min 0.30 --gc-max 0.70 --enforce-gc-bounds`. Report this setting explicitly. Constraint enforcement changes the output distribution and must not be passed off as unconstrained learned performance.

Replace `GCGG` with the required A/U/C/G motif. Use `--device cpu` for CPU inference and skip the CUDA preflight on a CPU-only machine. No retraining or training dataset is required for generation with the supplied weights. Four-step random decoding is a speed trade-off, not a proven optimum.

The delivered checkpoint was loaded from the packaged directory and generated two candidates in a local CPU, single-pass smoke check. This verifies a loading/generation path, not benchmark quality. The separately named SOURCE ZIP excludes weights and cannot run this example without them.

## Model architecture and training

The model combines token, position and context-role embeddings with normalized, projected [RNA-FM](https://github.com/ml4bio/RNA-FM) features through a learned residual gate. A six-layer Transformer encoder produces A/U/C/G logits. It uses width 384, six attention heads, feed-forward width 1,536 and dropout 0.1. Context features distinguish fixed/predicted positions, left/right roles, signed motif-relative distance clipped at 32, and masking noise fraction. Padding is excluded from attention and supervision.

LoRA adapts the last four RNA-FM layers with rank 8, alpha 16 and dropout 0.05. V5 uses random-position commitment during iterative decoding; learned length heads and self-conditioning are disabled in the active configuration. A fresh task-training run still uses pretrained RNA-FM initialization.

The active objective is:

```plaintext
L = L_base + 0.02 * L_composition + 0.02 * L_homopolymer
```

`L_base` is A/U/C/G cross-entropy with label smoothing 0.02, averaged within each sequence and then across sequences. Only supervised linker positions contribute. `L_composition` is Jensen–Shannon divergence between predicted and target mean composition at supervised positions. `L_homopolymer` penalizes expected same-base windows of seven nucleotides, excluding motif/padding windows and windows without supervision. Length-loss weights are zero. Raw marginal NLL, smoothed reconstruction loss and total loss are different metrics.

Training uses AdamW with task learning rate `3e-4`, LoRA learning-rate multiplier 0.1, weight decay 0.05 with configured exclusions, and warmup-cosine scheduling. Batch size is 8 with four-step accumulation, BF16 mixed precision and gradient clipping at 1.0. The limit is 40 epochs, with early-stopping patience 8 and checkpoint selection by `val/base_loss`. The YAML configuration is authoritative.

## Training data and reproduction

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
nohup bash scripts/start_v5_retrain.sh > logs/train_v5_retrain.log 2>&1 &
tail -f logs/train_v5_retrain.log
```

Do not delete the delivered checkpoint to make the command proceed. The wrapper refuses silent overwrite, uses a process lock and exports the best model after successful training. Use the manual installation instructions above when preparing a training environment. Offline execution does not keep a released cloud instance alive; download and verify exports before releasing a server.

## Run the benchmark

With the canonical data supplied, run from this directory:

```shell
bash scripts/benchmark_linker_v5_fast_offline.sh outputs/benchmark_v5
bash scripts/validate_linker_v5_expanded.sh outputs/validation_v5_expanded
bash scripts/diagnose_linker_v5_conditioning.sh outputs/conditioning_v5
```

The formal benchmark requests 48 test-partition motifs and 256 candidates per motif. It compares V5 with Uniform, Markov-1 and Markov-2 under matched length geometry. The two diagnostics use validation contexts to examine generation distributions and motif-conditioned prediction.

Use a new output directory for each run. The formal benchmark accepts `--resume` after the output-directory argument and verifies saved per-motif records. Inspect `benchmark_report.md`, `model_summary.csv`, candidate-level records and the run manifest. A smoke test validates the pipeline, not scientific performance.

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

## Benchmark results and interpretation

The following exploratory results are transcribed from development-stage server reports. Checkpoint identities and their evaluation records are listed under “Final model identity”.

### V4: passing format checks was insufficient

The historical formal benchmark used 48 motifs and 256 candidates per motif: 12,288 candidates per method, seed 42.

| Method | Reported valid rate | Unique rate | Edit diversity | Runtime, seconds |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 1.0000 | 1.0000 | 0.4694 | 80.0 |
| Markov-1 | 1.0000 | 1.0000 | 0.4725 | 80.9 |
| Markov-2 | 1.0000 | 1.0000 | 0.4713 | 80.9 |
| Transformer V4 | 1.0000 | 1.0000 | 0.3754 | 7,633.8 |

V4 linker composition entropy was 1.2636 bits, mean maximum linker run was 8.719, and the largest linker run was 30. Thus, 100% validity under the configured validator did not mean good composition, acceptable homopolymers, correct structure or functional linkers. Empty RNAfold fields were not successful structural evaluations. The report's sample-size grade was not a model-quality certification.

### V5: raw generation comparison

Expanded validation used 32 validation-cluster contexts and 64 draws per context: 2,048 candidates per method. Each method received the same context geometry. These were raw draws without hard GC/run filtering or deduplication. The natural-reference row contains only 32 real linker examples and is not a generated method.

| Method | Mean GC, % | Composition entropy, bits | Mean maximum run | Runs >6, % | Edit diversity | Elapsed, seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Natural reference | 50.86 | 1.9091 | 3.469 | 0.00 | — | — |
| Confidence, 16 steps | 82.90 | 1.2940 | 6.327 | 35.11 | 0.2640 | 1,164.2 |
| Random, 16 steps | 55.02 | 1.9036 | 3.342 | 0.59 | 0.4915 | 1,172.3 |
| Random, 4 steps | 54.64 | 1.9071 | 3.329 | 1.12 | 0.4909 | 295.6 |
| Single pass | 53.75 | 1.9189 | 3.246 | 0.49 | 0.4887 | 76.7 |
| Uniform | 50.14 | 1.9327 | 3.228 | 0.63 | 0.4953 | 4.8 |
| Markov-1 | 54.67 | 1.9102 | 3.485 | 1.03 | 0.4974 | 5.0 |
| Markov-2 | 54.56 | 1.9127 | 3.411 | 0.78 | 0.4965 | 5.0 |

Within-context uniqueness was 0.9932 for confidence decoding and 1.0 for the other variants. This is not a claim of global uniqueness across all contexts. Timings include the diagnostic work in that run and are not universal hardware throughput estimates.

Interpretation:

- Random decoding substantially improved on the defective confidence-selection variant.
- Random four-step decoding was approximately four times faster than random sixteen-step decoding, with similar aggregate composition and diversity. However, it produced more runs exceeding six: 23/2,048 versus 12/2,048. It is a speed/quality trade-off, not an across-the-board winner.
- Uniform and Markov baselines remained competitive on these descriptive sequence metrics and were much faster. Higher diversity alone is not evidence of better conditional design: random sequences can score highly.
- These results do not establish that V5 generates better functional linkers than the baselines. A weaker baseline must not be selected merely to make the model look better.

### V5: does the model use the motif?

A separate diagnostic used one context from each of 64 validation clusters, with all unknown linker bases masked, zero self-conditioning and only the first forward pass. It measured unsmoothed marginal NLL in nats per token, averaging clusters equally. This is **not joint sequence likelihood** and does not teacher-force unknown linker bases.

| Predictor / context | NLL, lower is better | Token accuracy |
| --- | ---: | ---: |
| Model, correct motif | 1.364003 | 30.68% |
| Model, shuffled motif | 1.371371 | 30.28% |
| Model, swapped motif | 1.382807 | 30.18% |
| Markov-1, correct motif | 1.386180 | 29.41% |
| Training-frequency predictor | 1.384931 | 29.13% |
| Uniform predictor | 1.386294 | Not used for interpretation |

Shuffling preserves motif composition; swapping does not. The Markov conditioning baseline uses conditional marginals, not access to the hidden true linker. Uniform argmax accuracy depends on tie-breaking, so the theoretical 25% random-draw accuracy should not be substituted for its recorded argmax result.

Paired differences below are `alternative NLL − model/correct NLL`; positive values favor the correctly conditioned model. Intervals are the reported unadjusted 95% cluster-bootstrap intervals, with 10,000 samples and seed 42.

| Comparison | Mean difference | Interval |
| --- | ---: | --- |
| Markov-1 minus model/correct | 0.022177 | [0.008129, 0.037980] |
| Model/shuffled minus model/correct | 0.007368 | [0.000778, 0.016470] |
| Model/swapped minus model/correct | 0.018804 | [0.003973, 0.034801] |
| Training-frequency minus model/correct | 0.020928 | [0.006683, 0.037159] |
| Uniform minus model/correct | 0.022291 | [0.009267, 0.036518] |

This supports a modest conditioning effect in the evaluated validation sample. Validation was reused for model selection and exploratory analysis; there was one ablation draw per context and several comparisons. The result is not an independent held-out confirmation and does not isolate long-range structural learning. Generation diversity and masked-token NLL answer different questions; neither table supersedes the other.

## Final model identity

Both runs produced the filename `rna-linker-v5-34-1.3714.ckpt`, but their bytes differ. A filename or rounded validation loss is not a unique model identifier.

| Run | Checkpoint SHA-256 | Evidence status |
| --- | --- | --- |
| Original V5 | `91bb5d3be35428c0ede186235bec12d836eb4c235622dbd3d95a917193ed3b50` | Historical expanded generation and conditioning results above |
| September 10 replacement | `2a6bab7bfb4afc145c2309c38798e2db43f085d628c291ab4004b33932546ced` | Training completed; exported file integrity verified; detailed benchmark must be rerun |

The replacement checkpoint is 533,679,315 bytes. Its local backup archive, `V5-BEST-DOWNLOAD-20260910-042140-147668.tar.gz`, has SHA-256 `255ec9de3319a3d92ea090f98d5ef351859c2355cf9c810e7dd8352dc2131dab`. The archive includes the best checkpoint, training manifest, configuration, installed-environment record and package checksums. Runtime weights and backup archives remain outside normal Git tracking; publish runtime assets separately if their licenses permit redistribution.

RNA-FM is 1,194,424,423 bytes; its SHA-256 is `5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99`. `MODEL_ASSETS_MANIFEST.json` records weight hashes. `SOURCE_MANIFEST.json` records source hashes. The companion SHA256SUMS file verifies both ZIP archives.

## Publish on GitHub

Extract RNA-V5-GITHUB-SOURCE.zip and upload its contents, including hidden files, to a clean repository. Keep the single README.md at the repository root. The source ZIP includes code, current configurations and regression tests, but excludes training data, weights, generated outputs, development reports, caches and Git history.

Publish RNA-V5-MODEL-ASSETS.zip as a separate GitHub Release asset, not a normal repository file. Confirm source licensing and permission to redistribute the upstream RNA-FM weights before publishing; packaging does not grant a license. No repository or release has been created automatically. Add the actual Release download URL here after publishing; no placeholder download URL is presented as working.

For inference, download the companion asset ZIP from the same Release and extract it into the cloned repository root, preserving its hidden .cache directory. This restores the exact paths used by the commands above. Check hashes in MODEL_ASSETS_MANIFEST.json. Training and asset-dependent tests additionally need the canonical CSV and cluster manifest described above; those data are not included in this source release.

GitHub blocks ordinary Git files larger than 100 MiB. See [GitHub large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Do not commit model binaries, virtual environments, or the delivery ZIP files. The supplied .gitignore keeps them untracked.

From a clean clone with its remote configured:

```bash
git add .
git diff --cached --stat
git commit -m "Add RNA linker V5 source and documentation"
git push
```

Review the staged files before committing. Keep an independent copy of the model assets. No Git LFS setup is needed when distributing weights through Releases.

## Interpretation boundary

Outputs are research-use sequence proposals. Exact motif preservation, acceptable GC content, high diversity or successful generation do not establish folding, accessibility, binding or in-vivo function. Natural RNA fragments are not interchangeable with engineered functional linker labels.

The current model has no all-atom folding module, equivariant coordinate head, FAPE or torsion reconstruction. Optional RNAfold-derived energies and pairing metrics are plausibility proxies, not proof of the intended structure or activity. The small conditioning advantage does not establish functional superiority over Uniform or Markov baselines.

Preserve matched geometry, candidate budgets, partitions, filtering rules, failures and runtime when comparing methods. Report raw generation separately from constraint-enforced output, and freeze selection choices before final test-set evaluation.

Run `python -m pytest` in the configured environment for implementation checks; asset-dependent tests require the canonical data. Tests protect correctness and reproducibility, not biological validity.
