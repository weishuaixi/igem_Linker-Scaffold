#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TORCH_HOME="$PWD/.cache/torch"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RNA_FM_EXPECTED_SHA256=5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
OUTPUT="${1:-outputs/linker_v5_expanded_$(date +%Y%m%d_%H%M%S)}"
echo "Output: $OUTPUT"
exec python -u scripts/check_linker_v5_validation.py --expanded --motifs 32 --candidates 64 --output "$OUTPUT" --device cuda
