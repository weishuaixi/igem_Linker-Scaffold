#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
if [[ -d checkpoints_scaffold_linker_v5 ]] && find checkpoints_scaffold_linker_v5 -name '*.ckpt' -print -quit | grep -q .; then
    echo "ERROR: V5 checkpoints already exist. Refusing to silently overwrite/restart." >&2
    exit 2
fi
exec bash scripts/train_linker_offline.sh configs/train.yaml
