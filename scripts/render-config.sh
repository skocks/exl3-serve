#!/usr/bin/env bash
# Render config.yml -> tabbyAPI/config.yml, substituting host-specific paths.
#
# TabbyAPI does not expand ~, $VARS, or relative paths in config.yml — model_dir
# has to be a literal absolute path. Keeping that literal in the tracked config
# would hardcode one machine's layout (and publish it), so config.yml carries
# @MODEL_DIR@ and this script fills it in at start time.
#
#   ./scripts/render-config.sh                      # -> <repo>/models
#   EXL3_MODEL_DIR=/mnt/models ./scripts/render-config.sh
#
# Called by scripts/serve.sh and by the systemd unit's ExecStartPre, so both
# launch paths substitute identically.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default matches scripts/download-model.sh, which writes into <repo>/models.
MODEL_DIR="${EXL3_MODEL_DIR:-$ROOT/models}"

[ -d "$MODEL_DIR" ] || echo "[render-config] warn: model dir does not exist: $MODEL_DIR" >&2

mkdir -p "$ROOT/tabbyAPI"
sed -e "s|@MODEL_DIR@|$MODEL_DIR|g" "$ROOT/config.yml" > "$ROOT/tabbyAPI/config.yml"

# A leftover placeholder means config.yml gained one this script does not know
# about — fail loudly rather than hand TabbyAPI a config it will choke on.
if grep -q '@[A-Z_]\+@' "$ROOT/tabbyAPI/config.yml"; then
  echo "[render-config] error: unsubstituted placeholders remain:" >&2
  grep -o '@[A-Z_]\+@' "$ROOT/tabbyAPI/config.yml" | sort -u | sed 's/^/  /' >&2
  exit 1
fi

# Sampler-override presets referenced by config.yml (sampling.override_preset)
# live at repo root; mirror them into TabbyAPI's working dir.
if [ -d "$ROOT/sampler_overrides" ]; then
  mkdir -p "$ROOT/tabbyAPI/sampler_overrides"
  cp -f "$ROOT"/sampler_overrides/*.yml "$ROOT/tabbyAPI/sampler_overrides/" 2>/dev/null || true
fi

echo "[render-config] model_dir=$MODEL_DIR -> tabbyAPI/config.yml"
