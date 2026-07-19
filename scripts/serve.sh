#!/usr/bin/env bash
# Start TabbyAPI and warm the triton kernel cache so the first real request is never cold.
#
# TabbyAPI has no built-in warmup, and exllamav3 does not compile the generation kernels at load
# time — the first inference JIT-compiles ~580 kernels (~8 s). We fire a throwaway generation right
# after startup so that cost is paid before real traffic. The compiled kernels persist in
# $TRITON_CACHE_DIR across restarts and reboots.
#
# Usage: ./scripts/serve.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5000}"

# config.yml lives at repo root; TabbyAPI expects it inside its own dir.
cp -f config.yml tabbyAPI/config.yml
# Sampler-override presets referenced by config.yml (sampling.override_preset)
# also live at repo root; mirror them into TabbyAPI's working dir.
if [ -d sampler_overrides ]; then
  mkdir -p tabbyAPI/sampler_overrides
  cp -f sampler_overrides/*.yml tabbyAPI/sampler_overrides/ 2>/dev/null || true
fi
MODEL_NAME="$(grep -E '^\s*model_name:' config.yml | awk '{print $2}')"

echo "[serve] starting TabbyAPI on http://$HOST:$PORT …"
( cd tabbyAPI && python main.py ) &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "[serve] waiting for health + warming triton kernels…"
HOST="$HOST" PORT="$PORT" ./scripts/warmup.sh

echo "[serve] ready and warm — first real request will be fast. OpenAI endpoint: http://$HOST:$PORT/v1"
wait $SERVER_PID
