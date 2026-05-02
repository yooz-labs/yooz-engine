#!/usr/bin/env bash
# Bootstrap the issue-12 research environment on /Volumes/S1.
#
# Creates:
#   /Volumes/S1/yooz/research/issue-12/.venv          (uv-managed)
#   /Volumes/S1/yooz/research/issue-12/models/hf_cache (HF_HOME)
# and pre-fetches the three MLX Qwen3-ASR checkpoints.
#
# Run from anywhere:
#   bash research/issue-12/scripts/bootstrap_env.sh
set -euo pipefail

ROOT=/Volumes/S1/yooz/research/issue-12
VENV="$ROOT/.venv"
export HF_HOME="$ROOT/models/hf_cache"

if [ ! -d /Volumes/S1 ]; then
  echo "ERROR: /Volumes/S1 not mounted" >&2
  exit 1
fi

mkdir -p "$ROOT/models" "$ROOT/data" "$ROOT/results" "$ROOT/logs" "$HF_HOME"

if [ ! -d "$VENV" ]; then
  echo ">> creating uv venv at $VENV"
  uv venv --python 3.12 "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo ">> installing benchmark deps"
uv pip install --upgrade pip
uv pip install \
    "mlx>=0.21" \
    "mlx-lm>=0.20" \
    "mlx-audio>=0.2.9" \
    "transformers>=4.57" \
    "soundfile>=0.12" \
    "librosa>=0.10" \
    "jiwer>=3.0" \
    "evaluate>=0.4" \
    "datasets>=3.0" \
    "tqdm>=4.66" \
    "huggingface_hub>=0.36" \
    "psutil>=5.9"

# Pre-fetch checkpoints into HF_HOME on S1. We download the 8-bit (better
# quality) and 4-bit (smaller) variants both communities use.
echo ">> pre-fetching MLX Qwen3-ASR checkpoints into $HF_HOME"
python - <<'PY'
import os
from huggingface_hub import snapshot_download

models = [
    # mlx-community canonical 4-bit + 8-bit
    "mlx-community/Qwen3-ASR-0.6B-4bit",
    "mlx-community/Qwen3-ASR-1.7B-8bit",
    # community fork with the highest download count (test conversion quality)
    "aufklarer/Qwen3-ASR-0.6B-MLX-4bit",
    # Parakeet baseline (already in stt-engine venv but pin it here for self-contained runs)
    "mlx-community/parakeet-tdt-0.6b-v3",
]
for repo in models:
    print(f"   {repo}")
    snapshot_download(repo, local_dir_use_symlinks=False)
print("done.")
PY

echo
echo "Environment ready."
echo "   Activate:  source $VENV/bin/activate"
echo "   HF_HOME:   $HF_HOME"
echo "   Phase 1:   python research/issue-12/scripts/bench_phase1_latency.py"
echo "   Phase 2:   python research/issue-12/scripts/bench_phase2_quality.py"
