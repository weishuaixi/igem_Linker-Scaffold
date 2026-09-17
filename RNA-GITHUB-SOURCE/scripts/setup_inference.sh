#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODE="${1:-cuda}"
if [[ $# -gt 1 || ( "$MODE" != cuda && "$MODE" != cpu ) ]]; then
    echo 'Usage: bash scripts/setup_inference.sh [cuda|cpu]' >&2
    exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'This installer targets Linux x86_64 (including compatible WSL2). See README for other platforms.' >&2
    exit 2
fi
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
    echo 'Install Miniforge/Conda first, then rerun. No system or existing training environment is modified.' >&2
    exit 2
fi
MODEL_PREFIX="$PWD/.inference_env"
FOLD_PREFIX="$PWD/.rnafold_env"
for prefix in "$MODEL_PREFIX" "$FOLD_PREFIX"; do
    if [[ -L "$prefix" || ( -e "$prefix" && ! -f "$prefix/conda-meta/history" ) ]]; then
        echo "Refusing to reuse an unrecognized environment: $prefix" >&2
        exit 2
    fi
done
mkdir -p outputs
exec 9>outputs/setup_inference.lock
flock -n 9 || { echo 'Another inference installation is running.' >&2; exit 2; }
if [[ ! -d "$MODEL_PREFIX" ]]; then
    "$CONDA_BIN" create -y --no-default-packages --prefix "$MODEL_PREFIX" \
        --override-channels -c conda-forge --strict-channel-priority python=3.10 pip
fi
if [[ ! -d "$FOLD_PREFIX" ]]; then
    "$CONDA_BIN" create -y --no-default-packages --prefix "$FOLD_PREFIX" \
        --override-channels -c conda-forge -c bioconda --strict-channel-priority viennarna=2.7.2
fi
PYTHON_BIN="$MODEL_PREFIX/bin/python"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,10), "Expected Python 3.10"'
FOLD_VERSION="$("$FOLD_PREFIX/bin/RNAfold" --version)"
if [[ "$FOLD_VERSION" != 'RNAfold 2.7.2' ]]; then
    echo "Expected RNAfold 2.7.2, found: $FOLD_VERSION. Use a separate checkout for another version." >&2
    exit 2
fi
"$PYTHON_BIN" -m pip install 'setuptools>=69' wheel
if [[ "$MODE" == cuda ]]; then
    "$PYTHON_BIN" -m pip install 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' 'torchaudio==2.7.1+cu128' --index-url https://download.pytorch.org/whl/cu128
else
    "$PYTHON_BIN" -m pip install 'torch==2.7.1+cpu' 'torchvision==0.22.1+cpu' 'torchaudio==2.7.1+cpu' --index-url https://download.pytorch.org/whl/cpu
fi
"$PYTHON_BIN" -m pip install -c constraints-v5-server.txt -r requirements.txt
"$PYTHON_BIN" -m pip install --no-deps --no-build-isolation -e .
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" - "$MODE" "$FOLD_PREFIX/bin/RNAfold" <<'PY'
import sys
import torch
from rna_scaffold.validators.rnafold import run_rnafold
mode, executable = sys.argv[1:]
x = torch.ones(16, 16, device=mode, dtype=torch.bfloat16 if mode == "cuda" else torch.float32)
y = x @ x
if mode == "cuda":
    torch.cuda.synchronize()
fold = run_rnafold("GGGAAACCC", 3, 6, executable=executable)
if fold.status != "ok":
    raise SystemExit(f"RNAfold preflight failed: {fold.error}")
print("READY:", torch.__version__, mode, tuple(y.shape), fold.version)
PY
"$PYTHON_BIN" -m pip freeze > outputs/inference-environment.txt
"$CONDA_BIN" list --prefix "$FOLD_PREFIX" --explicit > outputs/rnafold-environment-explicit.txt
echo "Installed. Run: bash scripts/generate_best.sh GCGG outputs/GCGG_ranked $MODE"
