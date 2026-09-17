# RNA-Linker

RNA-Linker is the dry-lab RNA linker and scaffold sequence generator for **iGEM PekingHSC 2026** (HEPHA-RNA). Given a fixed RNA motif, it proposes flanking sequences for Hepha element development. An optional RNAfold workflow predicts secondary structures and ranks the generated candidates.

- Model repository: [PekingHSC-2026-Model](https://github.com/sebaWEI/PekingHSC-2026-Model)
- Team: [iGEM PekingHSC 2026](https://teams.igem.org/6371)
- Wiki: [PekingHSC wiki](https://2026.igem.wiki/pekinghsc/)
- Wiki model page: [Models](https://2026.igem.wiki/pekinghsc/model)
- Wiki tutorial page: [Documents](https://2026.igem.wiki/pekinghsc/documents)

Each ranked-generation run writes candidates, RNAfold predictions, ranking scores, and a run manifest to the selected output directory, such as `outputs/GCGG_ranked/`. Computational ranking supports candidate selection but still requires wet-lab validation; see [scope and limitations](#scope-and-limitations).

You provide a motif such as `GCGG`. The model generates both flanking sequences while keeping the motif unchanged:

```text
Input:                          GCGG
Output:  generated left linker + GCGG + generated right linker
```

The default design uses one short flank of 10–20 nucleotides and one long flank of 15–30 nucleotides. Either side can be the short flank. Total sequence length must be at most 100 nucleotides. These are sequence-length constraints, not spatial dimensions or guarantees of a folded shape.

The model generates nucleotide sequences, not all-atom 3D structures. RNAfold is a separate downstream tool, not a learned folding module within the generator.

## Generate and rank sequences

The following workflow generates a candidate pool, predicts secondary structures, and saves the top-ranked sequence. No training data or retraining is needed to use the supplied weights.

### 1. Prepare source code and weights

Clone this repository or extract the source archive. Run all commands from the directory containing this README. Obtain `RNA-MODEL-ASSETS.zip` from the repository's Releases if available, or request it from the maintainer. Source code alone does not include the weights.

Extract the asset ZIP into the repository root, not a nested subdirectory. Check that these files exist:

```text
checkpoints_scaffold_linker_v5/
  rna-linker-v5-34-1.3714.ckpt
  training_manifest.json
.cache/torch/hub/checkpoints/
  RNA-FM_pretrained.pth
MODEL_ASSETS_MANIFEST.json
```

Keep these filenames as supplied: the scripts use these paths. Their historical suffixes are compatibility identifiers, not extra versions you need to install. See [model identity](docs/provenance.md) for weight hashes and evaluation provenance.

### 2. Install the inference environment

The installer targets Linux x86_64 with Conda or Miniforge available. GPU use requires a compatible NVIDIA driver. The tested server used an RTX 5090, Python 3.10, and PyTorch 2.7.1 with CUDA 12.8.

For GPU inference, run:

```bash
bash scripts/setup_inference.sh cuda
```

For CPU inference, replace `cuda` with `cpu`. The installer creates `.inference_env` for the model and `.rnafold_env` for RNAfold 2.7.2. It checks both installations without changing your existing training environment. The wrapper commands below do not require manual environment activation.

Installation requires internet access; generation can run offline afterward using local weights. Native Windows and macOS are not supported by this installer. For an environment without RNAfold, see [manual installation](docs/installation.md).

### 3. Run generation

Replace `GCGG` with your motif using A, U, C, and G. Use a new output directory for each run:

```bash
bash scripts/generate_best.sh GCGG outputs/GCGG_ranked cuda
cat outputs/GCGG_ranked/best.fasta
```

For CPU inference, replace the final `cuda` with `cpu`.

The default workflow generates 128 candidates using four-step random decoding. It preserves the motif, applies the flank-length limits above, enforces GC fraction 0.30–0.70, and limits homopolymer runs to six bases. It then evaluates candidates with RNAfold and ranks them. If RNAfold fails or the candidate budget is incomplete, the workflow stops without exporting a best sequence.

## Understand the output

The output directory contains both the candidate pool and the selected sequence:

| File | Contents |
| --- | --- |
| `candidates.fasta` | Generated RNA sequences |
| `candidates.jsonl` | Candidate metadata and generation audit |
| `rnafold_results.jsonl` | Predicted secondary structures, energies, and pairing metrics |
| `ranked.jsonl` | Complete ranking and component scores |
| `best.fasta` | The single top-ranked sequence |
| `best.json` | Details of the selected candidate |
| `run_manifest.json` | Settings, software checks, weight identity, and completion status |

FASTA is a text format for nucleotide sequences. JSONL stores one JSON record per line. Neither format implies a verified structure or biological function.

“Best” means top-ranked within this pool, not experimentally optimal. Ranking combines model scores, composition checks, RNAfold energy, pairing, and motif accessibility. Scores are relative to the pool, not calibrated probabilities. The accessibility term assumes an exposed motif is desirable; reconsider it for motifs intended to pair. See [ranking details](docs/development.md#candidate-ranking).

## Generate candidates without RNAfold

Use the generation entry point when you need sequence candidates without structure prediction or ranking. This example uses the installed inference environment. If you followed [manual installation](docs/installation.md), activate that environment and replace `.inference_env/bin/python` with `python`.

Configure local weight loading:

```bash
export TORCH_HOME="$PWD/.cache/torch"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RNA_FM_EXPECTED_SHA256=5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
CHECKPOINT="checkpoints_scaffold_linker_v5/rna-linker-v5-34-1.3714.ckpt"
mkdir -p outputs
```

Generate 20 candidates with explicit constraints:

```bash
.inference_env/bin/python src/generate_scaffold.py \
  --checkpoint "$CHECKPOINT" --motif GCGG \
  --num-candidates 20 --max-length 100 \
  --short-flank-min 10 --short-flank-max 20 \
  --long-flank-min 15 --long-flank-max 30 \
  --length-sampling uniform --remask-strategy random \
  --self-conditioning off --denoise-steps 4 \
  --temperature 1.0 --top-p 1.0 --device cuda \
  --max-homopolymer-run 6 \
  --gc-min 0.30 --gc-max 0.70 --enforce-gc-bounds \
  --output outputs/GCGG_candidates.jsonl \
  --fasta-output outputs/GCGG_candidates.fasta
```

Use `--device cpu` for CPU inference. This command does not produce `best.fasta`. If it accepts fewer than 20 candidates, it exits with code 2; inspect the generation audit for rejection reasons. Hard constraints change the output distribution, so report constrained generation separately from raw model sampling.

## How the model works

RNA-Linker learns to reconstruct masked flanking bases from a visible motif and positional context. It combines pretrained RNA-FM features with token, position, and context-role embeddings. A six-layer Transformer encoder predicts A/U/C/G probabilities. Low-rank adaptation (LoRA) updates part of RNA-FM during training.

The training objective combines three terms:

```text
Loss = masked-base cross-entropy
     + 0.02 × composition penalty
     + 0.02 × homopolymer penalty
```

The composition term compares predicted and target base composition. The homopolymer term penalizes expected same-base runs. Each term uses masks to exclude ineligible positions from its supervision.

During generation, the motif stays fixed while the decoder fills the flanks. The default workflow uses random-position commitment with four decoding steps. RNAfold prediction and ranking happen afterward, outside the neural model.

See [development and mathematical methods](docs/development.md) for architecture details, training settings, and the approaches tested during development.

## What the benchmarks show

Development experiments compared the model with Uniform, Markov-1, and Markov-2 baselines. They measured sequence composition, diversity, repetitive runs, runtime, and motif-conditioned prediction, not biological activity.

Two findings guided the default workflow:

- Confidence-based decoding amplified GC bias and repetitive runs. Random-position decoding reduced these problems. Four steps were about four times faster than sixteen in the recorded validation run, although they produced more runs exceeding six bases.
- The model showed a modest conditioning advantage in masked-base prediction. Uniform and Markov generators remained competitive on descriptive composition and diversity metrics and were much faster.

The conditioning diagnostic used 64 validation clusters with all unknown linker bases masked. Lower marginal negative log-likelihood (NLL) indicates better prediction:

| Predictor | NLL, nats/token | Token accuracy |
| --- | ---: | ---: |
| Model with correct motif | 1.3640 | 30.68% |
| Markov-1 with correct motif | 1.3862 | 29.41% |
| Training-frequency baseline | 1.3849 | 29.13% |
| Uniform baseline | 1.3863 | Not interpreted |

These are historical development results from the evaluated checkpoint, not new measurements of every distributed weight file. Validation was reused during development, so this is exploratory evidence rather than independent test confirmation. See [full benchmark comparisons](docs/benchmark.md) for generation metrics, uncertainty intervals, and runtime, and [model identity](docs/provenance.md) for checkpoint hashes.

The evidence supports some use of motif context. It does not establish functional superiority over random or Markov sequences.

## Train or evaluate the model

Training uses natural RNA sequence segments from a locally prepared Stanford RNA collection, not experimentally validated synthetic linker labels. Filtering retained 3,442 records. Cluster-disjoint splits contain 2,869 training, 252 validation, and 321 test records.

Training data are not included in the source or inference asset archive. Follow [training and data preparation](docs/training.md) for required files, hashes, environment setup, and commands. Retrain in a separate working copy; do not delete the supplied weights to bypass overwrite protection.

With the required data, environment, and weights available, run the benchmark from the repository root:

```bash
bash scripts/benchmark_offline.sh outputs/benchmark
```

See [benchmark protocols](docs/benchmark.md) for validation diagnostics and resume options. Match candidate budgets, lengths, partitions, and filtering rules across methods.

## Repository layout

The source tree separates implementation, run settings, utilities, tests, and scientific documentation:

```text
RNA-Linker/
  README.md                  Installation and usage
  pyproject.toml             Package metadata and command entry points
  requirements.txt           Runtime dependencies
  constraints-server.txt     Tested server dependency constraints
  src/
    rna_scaffold/            Model, data, losses, decoding, and validators
    train.py                 Training entry point
    generate_scaffold.py     Sequence generation entry point
    validate_scaffolds.py    Candidate validation entry point
    benchmark_scaffolds.py   Baseline comparison entry point
  configs/
    train.yaml               Active training settings
    benchmark.yaml           Active benchmark settings
  scripts/                   Setup, execution, and diagnostic utilities
  tests/                     Regression tests
  docs/                      Methods, results, and reproduction guides
  data/                      Local training inputs, not distributed
```

Run `python -m pytest` in a development environment with pytest installed. Some tests require canonical data or model assets. Tests check implementation correctness, not biological validity. Weights, outputs, caches, and virtual environments are excluded from normal Git tracking. Maintainers can consult [publishing instructions](docs/publishing.md).

## Scope and limitations

Treat outputs as research candidates requiring further validation. Motif preservation, acceptable GC content, and diversity do not prove folding, accessibility, binding, or activity. RNAfold estimates secondary-structure plausibility; this project has no all-atom coordinate head or validated functional predictor.

## Authorship and acknowledgments

The project author designed and implemented the main framework. OpenAI Codex assisted with code implementation and revision, debugging, testing, packaging, and documentation. The project author is responsible for reviewing the code and scientific claims.

RNA-FM and ViennaRNA remain attributable to their original authors. Review upstream licenses before redistributing their assets.
