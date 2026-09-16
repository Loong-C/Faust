#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 training/bootstrap.py
PY=".venv-faust/bin/python"
"$PY" -m unittest discover -s training -p 'test_*.py'
"$PY" training/integration_smoke.py
OUT="${FAUST_OUTPUT:-$PWD/runs/faust-lora-v0.1.0}"
# Separate smoke directory: those two optimizer steps never seed full training.
"$PY" training/train.py smoke --output "${OUT}-smoke" "$@"
"$PY" training/train.py train --output "$OUT" "$@"
