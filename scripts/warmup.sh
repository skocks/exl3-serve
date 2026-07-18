#!/usr/bin/env bash
# Poll TabbyAPI until healthy, then fire a throwaway generation so the triton kernels
# JIT-compile before real traffic. Reused by serve.sh and by the systemd unit (ExecStartPost).
# Always exits 0 — a warmup failure must never mark the service failed.
set -uo pipefail
cd "$(dirname "$0")/.."

CFG="${CONFIG_FILE:-config.yml}"
HOST="$(awk '/^network:/{n=1} n&&/^\s*host:/{print $2; exit}' "$CFG" 2>/dev/null)"; HOST="${HOST:-127.0.0.1}"
PORT="$(awk '/^network:/{n=1} n&&/^\s*port:/{print $2; exit}' "$CFG" 2>/dev/null)"; PORT="${PORT:-5000}"
MODEL_NAME="$(awk '/^model:/{m=1} m&&/^\s*model_name:/{print $2; exit}' "$CFG" 2>/dev/null)"

echo "[warmup] waiting for http://$HOST:$PORT/health …"
for _ in $(seq 1 180); do
  curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 && break
  sleep 2
done

echo "[warmup] firing throwaway generation to compile kernels…"
curl -s "http://$HOST:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,\"stream\":false}" \
  >/dev/null 2>&1 || true

echo "[warmup] done — first real request will be fast."
exit 0
