#!/usr/bin/env bash
# Reuse the successful training environment. Never install, train, or upload.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$ACTION" in all|check|smoke|run|export) ;; *) echo 'Actions: all check smoke run export'; exit 2;; esac
cd "$ROOT"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PATH="/usr/lib/wsl/lib:$PATH"
if [[ "$ACTION" == check || "$ACTION" == export ]]; then
    exec python3 "$ROOT/eval/ab_v1/run.py" "$ACTION" "$@"
fi
STATE="${FAUST_ENV_HOME:-$HOME/.local/share/faust-local8gb}"
KEY="$(sha256sum "$ROOT/training/local8gb/requirements.in" | cut -c1-12)"
ENV_DIR="$STATE/env-$KEY"
PY="$ENV_DIR/bin/python"
if [[ ! -x "$PY" || ! -f "$ENV_DIR/FAUST_READY" ]]; then
    echo "Training environment not found: $ENV_DIR"
    echo 'Use the same WSL distribution, Linux user and FAUST_ENV_HOME used for training.'
    echo 'No package installation or new training has been started.'
    exit 3
fi
command -v flock >/dev/null || { echo 'flock is required for the shared GPU lock.'; exit 3; }
exec 9>"$STATE/run.lock"
flock -n 9 || { echo 'A Faust training/evaluation process is already running.'; exit 3; }
export HF_HOME="${HF_HOME:-$STATE/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$STATE/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$STATE/torchinductor}"
"$PY" -m pip check
"$PY" -m unittest discover -s "$ROOT/eval/ab_v1" -p test_eval.py -v
"$PY" "$ROOT/eval/ab_v1/run.py" "$ACTION" "$@"
