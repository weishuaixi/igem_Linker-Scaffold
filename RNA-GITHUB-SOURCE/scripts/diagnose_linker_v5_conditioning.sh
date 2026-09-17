#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
exec 9>logs/linker_v5_conditioning.lock
flock -n 9 || { echo "Another conditioning diagnostic is running; do not start twice."; exit 1; }
export TORCH_HOME="$PWD/.cache/torch"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RNA_FM_EXPECTED_SHA256=5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
exec python -u scripts/diagnose_linker_v5_conditioning.py --output "${1:-outputs/linker_v5_conditioning}" --device cuda
