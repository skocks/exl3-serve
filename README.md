# exl3-serve

**One-command bootstrapper for serving EXL3-quantized LLMs via [ExLlamaV3](https://github.com/turboderp-org/exllamav3) + [TabbyAPI](https://github.com/theroyallab/tabbyAPI).**

ExLlamaV3 is one of the strongest single-GPU inference engines for quality-per-bit and long context — but getting it running is fiddly: the torch / triton / flash-attn / exllamav3 versions have to line up exactly, MTP speculative decoding needs the right drafter head, and there's no built-in kernel warmup. `exl3-serve` encodes a **known-good stack** and a couple of scripts so you go from a bare GPU box to an OpenAI-compatible endpoint with **8-bit KV cache, full-length context, and MTP speculative decoding** in three commands.

It was built and validated serving **Qwen3.6-27B (hybrid Gated-DeltaNet) at 256K context on a single 24 GB RTX 4090**, but nothing here is model-specific — point it at any EXL3 model.

## Quick start

```bash
git clone https://github.com/skocks/exl3-serve
cd exl3-serve

./install.sh                    # venv + validated torch/triton/flash-attn + exllamav3 + TabbyAPI
./scripts/download-model.sh     # default: Qwen3.6-27B EXL3 4.0bpw + a working bf16 MTP head
# edit config.yml: set model_dir (absolute) and model_name to the downloaded dir
./scripts/serve.sh              # starts TabbyAPI + warms the kernel cache
```

Then hit the OpenAI-compatible endpoint at `http://127.0.0.1:5000/v1`.

## Requirements

- NVIDIA GPU + driver (CUDA 12.x-capable). Validated on Ada (RTX 4090, 24 GB).
- CUDA toolkit ≥ 12.4 (`nvcc`) — exllamav3 compiles a CUDA extension from source.
- [`uv`](https://astral.sh/uv), `git`, `hf` (HuggingFace CLI, pulled in by uv).
- ~20 GB free disk for a 27B EXL3 model; more for larger models.

## What the installer pins, and why

The working combination is narrow — most "just pip install it" attempts fail. These specific versions are the ones that actually compile and run together:

| Component | Version | Why not something else |
|-----------|---------|------------------------|
| **torch** | `2.7.1` (cu126) | 2.6's inductor breaks with triton ≥ 3.3 (`AttrsDescriptor` import error). |
| **triton** | `3.4.0` | exllamav3 master's paged-attn kernel uses a construct triton 3.3.x rejects; 3.4 (on torch 2.7) works. |
| **flash-attn** | `2.8.3` | Without it, exllamav3 falls back to a triton paged-attn kernel that fails to compile. Required, not optional. |
| **exllamav3** | git `master` | The latest PyPI release (1.1.0) predates fixes needed for recent EXL3 features. |

## MTP (speculative decoding) — head compatibility

MTP roughly doubles decode throughput. But the drafter head must be one exllamav3 can load:

- ⚠️ `turboderp/Qwen3.6-27B-MTP-exl3` is currently **broken** — its `fc` layer carries *both* `mcg` and `mul1` trellis codebooks, which no exllamav3 gemm kernel accepts (`TORCH_CHECK(!(mcg && mul1))`). Root cause is a quantizer bug tracked in [exllamav3#239](https://github.com/turboderp-org/exllamav3/pull/239).
- ✅ `guru87/Qwen3.6-27B-MTP` is a **bf16** head (no codebook) that loads cleanly. `download-model.sh` uses it by default.

Swap back to an official EXL3 head once a fixed one ships. exllamav3 loads the MTP tensors from the **main model directory** when `draft_mode: mtp` is set, so the head's `mtp.safetensors` is copied in alongside the weights.

## Kernel warmup

The first inference JIT-compiles ~580 triton kernels (~8 s). `serve.sh` fires a throwaway generation right after startup so that cost is paid before real traffic. Compiled kernels persist in `$TRITON_CACHE_DIR` (default `~/.triton/cache`, **not** `/tmp`) across restarts and reboots, so it's a one-time cost.

## Robust model fetch

HuggingFace's xet transport has been observed to stall mid-shard on some repos (same byte offset every retry). The download script defaults to plain https (`HF_HUB_ENABLE_HXET=0`). If `hf download` still stalls, fetch the exact file directly — a different, steadier path:

```bash
wget -c "https://huggingface.co/<repo>/resolve/<revision>/model-00002-of-00002.safetensors" \
  -O models/<model-dir>/model-00002-of-00002.safetensors
```

## Configuration

Everything lives in `config.yml` (copied into `tabbyAPI/` by `serve.sh`). Key knobs:

- `cache_mode`: `Q8` (near-lossless, fits 256K on 24 GB), or `Q6` / `Q4` / `FP16` for other size/quality trade-offs.
- `max_seq_len` / `cache_size`: context length (KV cache is pre-allocated for the full size at load).
- `draft_mode: mtp`: enable MTP speculative decoding (needs the compatible head).

## Why ExLlamaV3 for long-context serving

Measured head-to-head against llama.cpp on Qwen3.6-27B @ 256K, single 24 GB GPU: EXL3 4.0bpw won on quality (8–14% lower perplexity), weight size, and decode speed (~2× with MTP), and uniquely fit **256K with 8-bit KV** where llama.cpp OOMs.

See **[COMPARISON.md](COMPARISON.md)** for the full benchmark tables, methodology, and operational notes.

## License

MIT. Bundles/installs third-party projects under their own licenses (ExLlamaV3, TabbyAPI, PyTorch, flash-attn).
