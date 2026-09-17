# RNA linker sequence model

This directory contains the motif-conditioned RNA linker model. Given a fixed RNA motif, it generates **left linker – motif – right linker**. The intended design has one short flank of 10–20 nucleotides and one long flank of 15–30 nucleotides, with total length at most 100. Either flank may be the short one; the motif is preserved exactly.

Python owns model loading, sequence generation, training and comparative evaluation. The source repository excludes binary weights. The companion RNA-MODEL-ASSETS.zip contains the trained V5 checkpoint and RNA-FM weights needed for inference; extract it into the repository root before running the examples. It does not generate all-atom coordinates or establish biological function.

## Authorship and AI assistance

The main project framework was designed and implemented by the project author. OpenAI Codex assisted with code writing and revision, debugging, testing, packaging, and documentation during development. This assistance included implementation changes, not only language editing. The project author is responsible for reviewing the delivered work and its scientific claims. Third-party models and software, including RNA-FM and ViennaRNA, remain attributable to their respective authors.

## Repository layout

```text
RNA-Linker/
  README.md                 Installation and usage
  pyproject.toml            Package metadata and installed commands
  requirements.txt          Runtime dependencies
  constraints-server.txt Tested server dependency constraints
  src/
    rna_scaffold/           Model, data, losses, decoding and validators
    train.py                Training entry point
    generate_scaffold.py    Candidate generation entry point
    validate_scaffolds.py   Validation entry point
    benchmark_scaffolds.py  Matched baseline evaluation entry point
  configs/                  Active training and benchmark configurations
  scripts/                  Setup, offline workflows and diagnostics
  tests/                    Regression tests
  docs/                     Development, benchmarks and model provenance
  data/                     Local training inputs (not distributed)
```

Start with the usage commands below. See [development and mathematical methods](docs/development.md),
[benchmark comparisons](docs/benchmark.md), and [model identity](docs/provenance.md) for scientific details.
Run all commands from the repository root. Source files now live under `src/`; installed command names are unchanged.
Public source filenames are version-neutral. Historical experiment labels and checkpoint paths remain unchanged to preserve compatibility with the model asset archive.

The separate `RNA-MODEL-ASSETS.zip` restores the checkpoint directory and hidden `.cache/` at the
repository root. These runtime assets, `outputs/`, virtual environments, and logs are not source-code folders
and must not be uploaded as ordinary repository files.

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
python -m pip install -c constraints-server.txt -r requirements.txt
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
python src/generate_scaffold.py \
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
nohup bash scripts/start_training.sh > logs/train_v5_retrain.log 2>&1 &
tail -f logs/train_v5_retrain.log
```

Do not delete the delivered checkpoint to make the command proceed. The wrapper refuses silent overwrite, uses a process lock and exports the best model after successful training. Use the manual installation instructions above when preparing a training environment. Offline execution does not keep a released cloud instance alive; download and verify exports before releasing a server.

## Publish on GitHub

Extract RNA-GITHUB-SOURCE.zip and upload its contents, including hidden files, to a clean repository. Keep the single README.md at the repository root. The source ZIP includes code, current configurations and regression tests, but excludes training data, weights, generated outputs, development reports, caches and Git history.

Publish RNA-MODEL-ASSETS.zip as a separate GitHub Release asset, not a normal repository file. Confirm source licensing and permission to redistribute the upstream RNA-FM weights before publishing; packaging does not grant a license. No repository or release has been created automatically. Add the actual Release download URL here after publishing; no placeholder download URL is presented as working.

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
