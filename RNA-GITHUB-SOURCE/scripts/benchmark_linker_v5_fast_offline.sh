#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
OUTPUT="${1:-outputs/scaffold_benchmark_linker_v5_fast}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--resume" ) ]]; then
    echo "Usage: bash scripts/benchmark_linker_v5_fast_offline.sh [output_dir] [--resume]" >&2
    exit 2
fi
test -f configs/benchmark_scaffolds_linker_v5.yaml
test -f checkpoints_scaffold_linker_v5/training_manifest.json
test -f .cache/torch/hub/checkpoints/RNA-FM_pretrained.pth
export TORCH_HOME="$PWD/.cache/torch"
export RNA_FM_EXPECTED_SHA256="5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
mkdir -p "$(dirname "$OUTPUT")"
exec 9>"${OUTPUT}.lock"
if ! flock -n 9; then
    echo "ERROR: another benchmark owns output $OUTPUT" >&2
    exit 2
fi
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable; activate the torch environment")
x = torch.ones(16, 16, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("CUDA preflight OK:", torch.__version__, torch.cuda.get_device_name(0), flush=True)
PY
exec python -u benchmark_scaffolds.py --config configs/benchmark_scaffolds_linker_v5.yaml --output-dir "$OUTPUT" "$@"
