#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${1:-configs/train.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

RNA_FM_CHECKPOINT=".cache/torch/hub/checkpoints/RNA-FM_pretrained.pth"
RNA_FM_SHA256="${RNA_FM_EXPECTED_SHA256:-5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99}"

declare -A REQUIRED_SHA256=(
  ["data/processed/rna_linker_v3_training.csv"]="5220058e6074c8fed218f968aa8d1c5d69dbdb72e856fb3b76f444eb33249de1"
  ["data/processed/rna_linker_v3_clusters.tsv"]="50eab29ce5494698f2e7db0ab6ab0fa8e54bbb9db0128e51b25bcc83e07a1a4a"
  ["${RNA_FM_CHECKPOINT}"]="${RNA_FM_SHA256}"
)

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: training config not found: ${CONFIG}" >&2
  exit 2
fi
for path in "${!REQUIRED_SHA256[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: required offline input not found: ${path}" >&2
    exit 2
  fi
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  expected="${REQUIRED_SHA256[$path]}"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ERROR: SHA-256 mismatch for ${path}: expected=${expected}, actual=${actual}" >&2
    exit 2
  fi
done

export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
export RNA_FM_EXPECTED_SHA256="${RNA_FM_SHA256}"
export WANDB_MODE="offline"
export WANDB_SILENT="true"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export PYTHONUNBUFFERED="1"
# Old server commands used this as a global workaround.  V4 scopes legacy
# pickle compatibility internally, after the RNA-FM digest has been checked.
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD

python - <<'PY'
import torch

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable")
print("gpu", torch.cuda.get_device_name(0))
capability = torch.cuda.get_device_capability(0)
architecture = f"sm_{capability[0]}{capability[1]}"
compiled_architectures = torch.cuda.get_arch_list()
print("capability", capability)
print("compiled_architectures", compiled_architectures)
if architecture not in compiled_architectures:
    raise SystemExit(
        f"ERROR: installed PyTorch has no kernel for {architecture}; "
        f"compiled architectures: {compiled_architectures}"
    )
if not torch.cuda.is_bf16_supported():
    raise SystemExit("ERROR: this GPU/PyTorch build does not support BF16")
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print("bf16_matmul", tuple(y.shape), y.dtype)
PY

mkdir -p logs outputs
exec python -u src/train.py --config "${CONFIG}" "$@"
