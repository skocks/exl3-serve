#!/usr/bin/env bash
# Quantize bottlecapai/ThinkingCap-Qwen3.6-27B to EXL3 @ 4.00bpw.
#
# This script:
#   1. Stops the running TabbyAPI + Hindsight API servers
#   2. Downloads the raw BF16 ThinkingCap model (~55 GB)
#   3. Extracts the native MTP draft head from the raw model (~849 MB)
#   4. Runs exllamav3's convert_model.py (~45-75 min on RTX 4090)
#   5. Copies the extracted MTP head into the quantized model dir
#   6. Updates config.yml to point to the new model
#   7. Restarts TabbyAPI + warms triton kernels
#
# On failure, re-run with the same command to resume from the last checkpoint.
# Clean up raw/working dirs manually after verifying the model works.
#
# Usage:
#   ./scripts/quantize-thinkingcap.sh          # full pipeline
#   ./scripts/quantize-thinkingcap.sh --resume # resume after interruption
#
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
THINKINGCAP_REPO="bottlecapai/ThinkingCap-Qwen3.6-27B"

BITS="4.0"            # target bits-per-weight for decoder layers
HEAD_BITS="6"         # output (lm_head) layer – default 6
MTP_BITS="4"          # MTP draft layers – default 4
SHARD_SIZE="8192"     # max output shard size in MB

# Directory layout under ./models/
MODELS_DIR="models"
RAW_DIR="${MODELS_DIR}/ThinkingCap-Qwen3.6-27B-raw"          # downloaded BF16 source
WORK_DIR="${MODELS_DIR}/ThinkingCap-Qwen3.6-27B-work"        # quantization working dir
OUT_DIR="${MODELS_DIR}/ThinkingCap-Qwen3.6-27B-exl3-4.00bpw" # final EXL3 model

CONFIG_FILE="config.yml"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python3"
WARMUP_SCRIPT="scripts/warmup.sh"

RESUME=false
RESUME_ARGS=()
if [[ "${1:-}" == "--resume" ]]; then
    RESUME=true
    RESUME_ARGS=("-r")
    echo "[quantize] Resuming quant from working directory: $WORK_DIR"
fi

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
log() { echo "$(date '+%H:%M:%S')  $*"; }

stop_server() {
    local pid=$1 desc=$2
    pid=$(pgrep -f "$1" 2>/dev/null || true)
    if [[ -n "$pid" ]]; then
        log "Stopping $desc (PID $pid) …"
        kill "$pid" 2>/dev/null || true
        sleep 3
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
            sleep 1
        fi
        log "$desc stopped."
    else
        log "$desc is not running."
    fi
}

start_server() {
    log "Starting TabbyAPI on http://127.0.0.1:5000 …"

    # Mirror config + sampler overrides into tabbyAPI/ (serve.sh does this too)
    cp -f "$CONFIG_FILE" tabbyAPI/config.yml
    if [[ -d sampler_overrides ]]; then
        mkdir -p tabbyAPI/sampler_overrides
        cp -f sampler_overrides/*.yml tabbyAPI/sampler_overrides/ 2>/dev/null || true
    fi

    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
    (cd tabbyAPI && "$REPO_ROOT/.venv/bin/python3" main.py) &
    SERVER_PID=$!

    log "Waiting for TabbyAPI health + warming triton kernels …"
    HOST="${HOST:-127.0.0.1}" PORT="${PORT:-5000}" bash "$WARMUP_SCRIPT"

    log "Server ready at http://127.0.0.1:5000/v1 (PID $SERVER_PID)"
}

# ------------------------------------------------------------------------------
# Step 1 – Stop existing services
# ------------------------------------------------------------------------------
if ! $RESUME; then
    log "=== Step 1/7: Stopping services ==="
    stop_server "python.*main.py" "TabbyAPI"
    stop_server "hindsight-api"      "Hindsight API"
    # Give the GPU a moment to release VRAM
    sleep 2
    nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true
else
    log "=== Skipping Step 1 (--resume) ==="
fi

# ------------------------------------------------------------------------------
# Step 2 – Download ThinkingCap raw model
# ------------------------------------------------------------------------------
if ! $RESUME; then
    log "=== Step 2/7: Downloading ThinkingCap ($THINKINGCAP_REPO)  ==="
    log "Download size: ~55 GB — this may take 15-30 minutes."
    mkdir -p "$RAW_DIR"

    source .venv/bin/activate 2>/dev/null || true
    export HF_HUB_ENABLE_HXET="${HF_HUB_ENABLE_HXET:-0}"

    # Download only the files needed for quantization:
    #   - config files (*.json)
    #   - tokenizer files (*.txt, vocab.json, merges.txt)
    #   - chat template
    #   - all safetensors weight shards
    hf download "$THINKINGCAP_REPO" \
        --local-dir "$RAW_DIR" \
        --include "*.json" \
        --include "*.txt" \
        --include "*.jinja" \
        --include "*.safetensors"

    log "Download complete → $RAW_DIR"
    du -sh "$RAW_DIR"
else
    log "=== Skipping Step 2 (--resume) ==="
fi

# ------------------------------------------------------------------------------
# Step 3 – Extract native MTP head from raw model
# ------------------------------------------------------------------------------
MTP_SOURCE="$RAW_DIR/mtp.safetensors"
if ! $RESUME; then
    log "=== Step 3/7: Extracting native MTP head from raw model ==="
    log "ThinkingCap fine-tuned the MTP layer — using its native head gives"
    log "better draft accuracy than the stock Qwen3.6-27B MTP head."

    source .venv/bin/activate 2>/dev/null || true
    RAW_DIR="$RAW_DIR" MTP_OUT="$MTP_SOURCE" "$VENV_PYTHON" -c "
import json, os, sys
from safetensors import safe_open
from safetensors.torch import save_file

raw_dir = os.environ['RAW_DIR']
mtp_out = os.environ['MTP_OUT']

idx = json.load(open(os.path.join(raw_dir, 'model.safetensors.index.json')))
mtp_map = {k: v for k, v in idx['weight_map'].items() if k.startswith('mtp.')}
print(f'MTP keys in index: {len(mtp_map)}, shards: {sorted(set(mtp_map.values()))}')

if len(mtp_map) != 15:
    print(f'ERROR: expected 15 MTP tensors in index, found {len(mtp_map)}')
    sys.exit(1)

tensors = {}
for shard in sorted(set(mtp_map.values())):
    path = os.path.join(raw_dir, shard)
    if not os.path.exists(path):
        print(f'ERROR: shard not found: {path}')
        sys.exit(1)
    with safe_open(path, framework='pt') as f:
        for k in f.keys():
            if k.startswith('mtp.'):
                tensors[k] = f.get_tensor(k)

if len(tensors) != 15:
    print(f'ERROR: expected 15 MTP tensors in shards, extracted {len(tensors)}')
    sys.exit(1)

save_file(tensors, mtp_out)
print(f'MTP head extracted: {os.path.getsize(mtp_out)/2**20:.1f} MiB')
print(f'Tensors: {sorted(tensors.keys())}')
"
    log "MTP head extraction complete."
else
    log "=== Step 3/7: Using existing native MTP head ==="
    if [[ -f "$MTP_SOURCE" ]]; then
        log "Found: $MTP_SOURCE ($(du -h "$MTP_SOURCE" | cut -f1))"
    else
        log "WARNING: --resume but $MTP_SOURCE not found — quantizer will still work"
        log "but you'll be missing the MTP head. Re-run without --resume to extract it."
    fi
fi

# ------------------------------------------------------------------------------
# Step 4 – Quantize to EXL3
# ------------------------------------------------------------------------------
log "=== Step 4/7: Quantizing to EXL3 @ ${BITS}bpw ==="
log "Estimated time: 45-75 minutes on RTX 4090 (24 GB VRAM)."
log "A progress bar will update as each layer is quantized."



# Ensure clean state for the venv
source .venv/bin/activate 2>/dev/null || true
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"

# Verify GPU is free before starting the 45-75 min quant run
GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1 | sed 's/ MiB//' || echo "0")
if [[ "$GPU_MEM_USED" -gt 500 ]]; then
    log "WARNING: GPU memory is ${GPU_MEM_USED} MiB used — quantization needs full 24 GB."
    log "Killing any remaining GPU processes …"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r p; do
        [[ -n "$p" ]] && kill -9 "$p" 2>/dev/null || true
    done
    sleep 2
fi
log "GPU memory free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null || echo 'unknown')"

RESUME_FLAG=""
if $RESUME; then
    RESUME_FLAG="args_list.append('-r')"
fi

"$VENV_PYTHON" -c "
import sys
args_list = ['convert_model', '-i', '$RAW_DIR', '-w', '$WORK_DIR', '-o', '$OUT_DIR', '-b', '$BITS', '-hb', '$HEAD_BITS', '-mb', '$MTP_BITS', '-ss', '$SHARD_SIZE', '-d', '0']
$RESUME_FLAG
sys.argv = args_list
from exllamav3.conversion.convert_model import parser, prepare, main
args = parser.parse_args()
in_args, job_state, ok, err = prepare(args)
if not ok:
    print(f'PREPARE FAILED: {err}')
    sys.exit(1)
main(in_args, job_state)
"

log "Quantization complete → $OUT_DIR"
du -sh "$OUT_DIR"

# ------------------------------------------------------------------------------
# Step 5 – Copy MTP head
# ------------------------------------------------------------------------------
log "=== Step 5/7: Copying MTP head ==="
if [[ -f "$MTP_SOURCE" ]]; then
    cp -v "$MTP_SOURCE" "$OUT_DIR/"
    log "Native ThinkingCap MTP head copied."
else
    log "WARNING: No MTP head to copy! Draft model won't work."
    log "  (this is expected if --resume skipped Step 3)"
fi

# ------------------------------------------------------------------------------
# Step 6 – Update config.yml
# ------------------------------------------------------------------------------
log "=== Step 6/7: Updating config.yml ==="
NEW_MODEL_NAME="$(basename "$OUT_DIR")"
sed -i "s/^  model_name: .*/  model_name: $NEW_MODEL_NAME/" "$CONFIG_FILE"
log "model_name → $NEW_MODEL_NAME"
grep -n "model_name:" "$CONFIG_FILE"

# Also disable reasoning for the new model until we verify it handles think tags
# correctly. The contamination feedback loop was caused by Qwen reasoning + JSON
# output leaking. ThinkingCap uses fewer thinking tokens, but let's stay safe.
log "Recommendation: set reasoning: false in config.yml if thinking leaks persist"
log "  (ThinkingCap should need this less — but test first.)"

# ------------------------------------------------------------------------------
# Step 7 – Restart services
# ------------------------------------------------------------------------------
log "=== Step 7/7: Restarting TabbyAPI ==="
start_server

# ------------------------------------------------------------------------------
# Done
# ------------------------------------------------------------------------------
log "============================================================"
log "All done! ThinkingCap-Qwen3.6-27B @ 4.00bpw is live."
log ""
log "Cleanup (after verifying everything works):"
log "  rm -rf $RAW_DIR    # ~55 GB raw BF16 model (MTP head already extracted)"
log "  rm -rf $WORK_DIR   # ~55 GB quantization working files"
log ""
log "To revert to the original model:"
log "  sed -i 's/^  model_name: .*/  model_name: Qwen3.6-27B-exl3-4.0bpw/' config.yml"
log "  ./scripts/serve.sh"
