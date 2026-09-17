#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo 'Usage: bash scripts/generate_best.sh MOTIF [new_output_directory] [cuda|cpu]' >&2
    exit 2
fi
MOTIF="$1"
OUTPUT="${2:-outputs/linker_ranked_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${3:-cuda}"
PYTHON_BIN="$PWD/.inference_env/bin/python"
RNAFOLD_BIN="$PWD/.rnafold_env/bin/RNAfold"
if [[ ! -x "$PYTHON_BIN" || ! -x "$RNAFOLD_BIN" ]]; then
    echo 'Run bash scripts/setup_inference.sh first.' >&2
    exit 2
fi
export TORCH_HOME="$PWD/.cache/torch"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export RNA_FM_EXPECTED_SHA256=5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
exec "$PYTHON_BIN" -u scripts/design_linker.py --motif "$MOTIF" --output-dir "$OUTPUT" \
    --device "$DEVICE" --rnafold-executable "$RNAFOLD_BIN"
