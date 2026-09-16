#!/usr/bin/env bash
# Linux / WSL2 launcher. Does not touch system Python, install GPU drivers, or rent GPUs.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${1:-all}"
PROFILE="${2:-standard}"
case "$ACTION" in all|check|doctor|prepare|smoke|train|generate) ;; *) echo 'Actions: all check doctor prepare smoke train generate'; exit 2;; esac
case "$PROFILE" in standard|lean) ;; *) echo 'Profiles: standard(rank16), lean(rank8)'; exit 2;; esac
cd "$ROOT"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="/usr/lib/wsl/lib:$PATH"
STATE="${FAUST_ENV_HOME:-$HOME/.local/share/faust-local8gb}"
mkdir -p "$STATE" "$ROOT/runs"
LOG="$ROOT/runs/local8gb-$(date +%Y%m%d-%H%M%S)-$$.log"
exec > >(tee -a "$LOG") 2>&1
trap 'code=$?; echo "Stopped (exit $code). Log: $LOG"; exit "$code"' ERR
if command -v flock >/dev/null; then
    exec 9>"$STATE/run.lock"
    flock -n 9 || { echo 'Another Faust local run is active. Close it before starting a second one.'; exit 3; }
fi
if [[ "$ACTION" == check ]]; then
    python3 "$ROOT/training/local8gb/run.py" check
    exit
fi
[[ "$(uname -s)" == Linux ]] || { echo 'Use Linux/WSL2. Windows: Run_Faust_8GB.cmd'; exit 3; }
command -v nvidia-smi >/dev/null || { echo 'No nvidia-smi. In WSL, update the WINDOWS NVIDIA driver; do not install a Linux GPU driver.'; exit 3; }
nvidia-smi
# A separate Linux-home environment/cache avoids Windows filesystem package overhead.
export HF_HOME="${HF_HOME:-$STATE/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$STATE/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$STATE/torchinductor}"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
REQ="$ROOT/training/local8gb/requirements.in"
KEY="$(sha256sum "$REQ" | cut -c1-12)"
ENV_DIR="$STATE/env-$KEY"
PY="$ENV_DIR/bin/python"
if [[ ! -f "$ENV_DIR/FAUST_READY" ]]; then
    echo 'First setup: downloads Python packages and later a public 4-bit model; reserve about 35 GB of disk space.'
    echo 'No API key or paid service is used. This environment is separate from your existing Python.'
    read -r -p 'Install this local environment? [y/N] ' answer
    [[ "$answer" == y || "$answer" == Y ]] || { echo 'Cancelled; no training started.'; exit 0; }
    if ! command -v gcc >/dev/null || ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
        command -v apt-get >/dev/null || { echo 'Install Python3 with venv and a C/C++ compiler, then retry.'; exit 3; }
        echo 'Ubuntu needs python3-venv and build-essential. sudo may ask for your Linux password.'
        sudo apt-get update
        sudo apt-get install -y python3-venv build-essential
    fi
    if [[ ! -x "$STATE/bootstrap/bin/python" ]]; then
        python3 -m venv "$STATE/bootstrap"
    fi
    "$STATE/bootstrap/bin/python" -m pip install --upgrade uv
    UV="$STATE/bootstrap/bin/uv"
    if [[ ! -x "$PY" ]]; then
        "$UV" venv --python 3.11 --seed "$ENV_DIR"
    fi
    "$UV" pip install --python "$PY" --torch-backend=auto -r "$REQ"
    "$UV" pip check --python "$PY"
    "$PY" -m pip freeze > "$ENV_DIR/requirements.resolved.txt"
    "$PY" -c 'from unsloth import FastLanguageModel; import torch; assert torch.cuda.is_available(), "CUDA not available"'
    cp "$REQ" "$ENV_DIR/requirements.requested.txt"
    touch "$ENV_DIR/FAUST_READY"
fi
"$PY" -m pip check
"$PY" -m unittest discover -s "$ROOT/training/local8gb" -p 'test_core.py' -v
args=(--profile "$PROFILE")
if [[ -n "${FAUST_RUN_ID:-}" ]]; then args+=(--run-id "$FAUST_RUN_ID"); fi
if [[ "$ACTION" == all ]]; then
    for stage in check doctor prepare smoke train; do
        echo "===== Faust: $stage ($PROFILE) ====="
        "$PY" "$ROOT/training/local8gb/run.py" "$stage" "${args[@]}"
    done
else
    "$PY" "$ROOT/training/local8gb/run.py" "$ACTION" "${args[@]}"
fi
echo "Done. Run artifacts: $ROOT/runs/ | Log: $LOG"
