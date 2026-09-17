#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
exec 9>logs/v5_retrain.lock
flock -n 9 || { echo 'V5 training is already running.'; exit 1; }
bash scripts/train_linker_v5_offline.sh
python scripts/export_v5_best.py
