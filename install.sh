#!/usr/bin/env bash
# exl3-serve — one-shot installer for serving EXL3-quantized LLMs via ExLlamaV3 + TabbyAPI.
#
# Installs the exact dependency combination validated to run Qwen3.6-27B (hybrid Gated-DeltaNet)
# with MTP speculative decoding and an 8-bit KV cache on a single 24 GB GPU. The combo is narrow
# on purpose — see README "Why these exact versions".
#
#   torch 2.7.1 (cu126)  +  triton 3.4.0  +  flash-attn 2.8.3  +  exllamav3 (git master)
#
# Usage: ./install.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

log() { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[install] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- prerequisites --------------------------------------------------------
command -v uv   >/dev/null || die "uv not found — install from https://astral.sh/uv"
command -v git  >/dev/null || die "git not found"
command -v nvcc >/dev/null || log "WARNING: nvcc not found; exllamav3's CUDA build needs the CUDA toolkit (>=12.4)."
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — no NVIDIA GPU/driver?"

# Pin the triton kernel cache to a persistent path (NOT /tmp, which some launchers wipe).
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
mkdir -p "$TRITON_CACHE_DIR"

# --- venv -----------------------------------------------------------------
log "Creating venv (.venv, Python 3.12)…"
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"

# --- torch / triton / flash-attn (the validated combo) --------------------
# torch 2.7.1 lives on the cu126 index; its inductor tolerates triton 3.4 (torch 2.6 does not).
log "Installing torch 2.7.1 (cu126)…"
uv pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu126

# NOTE: triton 3.4.0 is pinned LAST (just before verify), not here. torch 2.7.1's wheel metadata
# declares `triton==3.3.1`, so every later `uv pip install` re-resolves and snaps triton back to
# 3.3.1 — a triton pin placed here gets silently clobbered by the flash-attn/exllamav3/TabbyAPI
# installs below. Pinning it last makes 3.4.0 the final word. See the assertion in the verify step.

# exllamav3 uses its own triton paged KV kernel unless flash-attn is present; without it, that
# kernel fails to compile. flash-attn is required, not optional.
log "Installing numpy + flash-attn (prebuilt wheel if available, else source build)…"
uv pip install numpy
MAX_JOBS="${MAX_JOBS:-8}" uv pip install flash-attn --no-build-isolation

# --- exllamav3 (git master) ----------------------------------------------
# The latest PyPI release (1.1.0) predates fixes needed here. Master builds the CUDA ext from source.
log "Building exllamav3 from git master (CUDA extension compile — this takes 10-20 min)…"
uv pip install ninja
MAX_JOBS="${MAX_JOBS:-8}" uv pip install "git+https://github.com/turboderp-org/exllamav3.git" --no-build-isolation

# --- TabbyAPI (OpenAI-compatible server) ----------------------------------
# Run as an app from its repo. Install ONLY its server deps so we keep our torch/exllamav3 build.
if [ ! -d tabbyAPI/.git ]; then
  log "Cloning TabbyAPI…"
  git clone --depth 1 https://github.com/theroyallab/tabbyAPI
fi
log "Installing TabbyAPI server dependencies (excluding torch/exllamav3)…"
uv pip install \
  "fastapi-slim>=0.115" "pydantic>=2.11,<3" ruamel.yaml rich "uvicorn>=0.28.1" \
  "jinja2>=3.0.0" loguru "sse-starlette>=2.2.0" packaging "tokenizers>=0.21.0" \
  "formatron>=0.4.11" "kbnf>=0.4.1" aiofiles aiohttp async_lru huggingface_hub \
  psutil "httptools>=0.5.0" pillow requests uvloop setuptools

# --- triton pin (LAST) ----------------------------------------------------
# Must run after every other install: torch 2.7.1 declares triton==3.3.1, and each resolve above
# reasserts it. This forces the validated 3.4.0 as the final state. uv may note the mismatch with
# torch's declared pin; that's expected and harmless — flash-attn provides the attention path, so
# torch never exercises a triton kernel that would care about the version.
log "Pinning triton 3.4.0 (validated) as the final step…"
uv pip install "triton==3.4.0"

# --- verify ---------------------------------------------------------------
log "Verifying the stack…"
python - <<'PY'
import torch, triton, flash_attn, exllamav3
assert torch.cuda.is_available(), "CUDA not available"
assert triton.__version__.startswith("3.4"), \
    f"triton {triton.__version__} — expected 3.4.x; a later install clobbered the pin"
print(f"  torch      {torch.__version__}")
print(f"  triton     {triton.__version__}")
print(f"  flash_attn {flash_attn.__version__}")
print(f"  exllamav3  imported OK  (device: {torch.cuda.get_device_name(0)})")
PY

log "Done. Next:"
log "  1. ./scripts/download-model.sh          # pull an EXL3 model + MTP head"
log "  2. cp config.yml tabbyAPI/config.yml     # (or edit paths in config.yml)"
log "  3. ./scripts/serve.sh                    # start server + warm the triton cache"
