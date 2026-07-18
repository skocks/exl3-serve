#!/usr/bin/env bash
# Download an EXL3 model + a compatible MTP drafter head into ./models.
#
# Defaults to the validated Qwen3.6-27B setup:
#   - weights : turboderp/Qwen3.6-27B-exl3  @ 4.00bpw  (~16.9 GB)
#   - MTP head: guru87/Qwen3.6-27B-MTP  (bf16, ~849 MB)  — works with exllamav3's kernel
#
# NOTE on the MTP head: turboderp/Qwen3.6-27B-MTP-exl3 is currently BROKEN (its fc layer carries
# both mcg and mul1 trellis codebooks, which no exllamav3 gemm kernel accepts — see
# https://github.com/turboderp-org/exllamav3/pull/239). The guru87 bf16 head has no codebook and
# loads cleanly. Swap back to the turboderp head once a fixed one is published.
#
# Usage: ./scripts/download-model.sh [MODEL_REPO] [MODEL_REVISION] [MTP_REPO]
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

MODEL_REPO="${1:-turboderp/Qwen3.6-27B-exl3}"
MODEL_REV="${2:-4.00bpw}"
MTP_REPO="${3:-guru87/Qwen3.6-27B-MTP}"
MODEL_DIR="models/$(basename "$MODEL_REPO")-$(echo "$MODEL_REV" | tr -d '.')"

mkdir -p models
export HF_HUB_ENABLE_HXET="${HF_HUB_ENABLE_HXET:-0}"   # xet has stalled mid-shard on some repos; https is steadier

echo "[download] weights: $MODEL_REPO @ $MODEL_REV -> $MODEL_DIR"
hf download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir "$MODEL_DIR" || {
  echo "[download] hf download failed/stalled — see robust-fetch note in README to wget the resolve/ URL."
  exit 1
}

echo "[download] MTP head: $MTP_REPO"
hf download "$MTP_REPO" mtp.safetensors --local-dir "models/$(basename "$MTP_REPO")"
# exllamav3's --mtp loads the MTP tensors from the *main model dir*, so place it there.
cp -v "models/$(basename "$MTP_REPO")/mtp.safetensors" "$MODEL_DIR/"

echo "[download] Done. Model dir: $MODEL_DIR (weights + mtp.safetensors)"
echo "[download] Set model_name in config.yml to: $(basename "$MODEL_DIR")"
