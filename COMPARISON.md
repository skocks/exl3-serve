# Benchmark addendum — ExLlamaV3 vs llama.cpp

Why `exl3-serve` exists: on the workload it was built for — **Qwen3.6-27B at 256K context, single 24 GB RTX 4090, GPU-only** — ExLlamaV3 measurably beats llama.cpp on quality, weight size, and decode speed, and reaches longer context at a given KV precision. These are the numbers behind that claim, all measured on the same box.

Hardware: RTX 4090 (24564 MiB, Ada, cc 8.9), 62 GB RAM, GPU-only (displays on a separate AMD card). Model: `turboderp/Qwen3.6-27B-exl3` @ 4.00bpw vs GGUF `Q4_K_M` / `UD-Q4_K_XL`.

### The model matters: Qwen3.6-27B is hybrid, not dense

`Qwen3.6-27B` is **not** a plain dense transformer. Its config has `num_hidden_layers=64`, `full_attention_interval=4`: only **16 of 64 layers** are full Gated Attention holding a growing KV cache (`kv_heads=4`, `head_dim=256`); the other **48 are Gated DeltaNet** linear-attention with a fixed-size recurrent state. Consequences: the 256K KV cache scales with **16 layers, not 64** (why it fits 24 GB at all), and **KV-quant precision barely matters** — quantizing the cache only touches a minority of layers. Native context is **262144** (no YaRN needed for 256K).

## 1. Quality — weight-quant perplexity (identical wikitext2, ctx 4096)

| Engine · model | Weights | PPL ↓ | vs EXL3 |
|----------------|---------|-------|---------|
| **exllamav3 · EXL3 4.02bpw** | **~13.5 GiB** | **6.117** | — |
| llama.cpp · UD-Q4_K_XL | 16.4 GiB | 6.642 | +8.6% worse |
| llama.cpp · Q4_K_M | 15.4 GiB | 7.001 | +14.4% worse |

EXL3's trellis (QTIP-style) quant is more efficient per bit than GGUF k-quants — it wins quality **and** size at *fewer* bits (4.02 vs ~4.9 effective).

## 2. VRAM @ 256K — what actually fits (24564 MiB budget)

EXL3 rows are **measured live** on this box: full 262144-token cache pre-allocated, MTP draft head loaded, figure = nvidia-smi used memory on the running server. (An earlier draft of this table carried *estimated* EXL3 numbers that were wrong — Q8/Q6 at 256K were listed as fitting; they do **not**. Corrected below.)

| KV cache @256K | EXL3 4.0bpw + MTP | llama.cpp (best of Q4_K_M / UD-XL) |
|----------------|-------------------|-------------------------------------|
| fp16 | OOM | OOM |
| 8-bit / q8_0 | ❌ **OOM (~25 GiB)** | ❌ OOM |
| 6-bit | ❌ **OOM** | — |
| **4-bit / q4_0** | ✅ **21.66 GiB** (2.9 GiB free) | ✅ 21.6–22.7 GiB (1.9–2.9 free) |

**At 256K, both engines are 4-bit-KV-bound** — neither fits 8-bit KV on 24 GB. So EXL3's 256K edge is *not* KV precision (both run 4-bit) but **~2 GiB smaller weights, lower perplexity, and MTP decode**. Where EXL3 does pull ahead on the cache: its smaller weights let it keep **near-lossless 8-bit KV up to ~160K** (128K/Q8+MTP measured at 21.15 GiB); llama.cpp's larger weights leave less KV room, so its 8-bit ceiling is lower. The real user choice on 24 GB is **256K@Q4** (max reach) vs **≤160K@Q8** (near-lossless KV).

## 3. Speed — decode & prefill (tok/s)

| | Decode | Prefill |
|--|--------|---------|
| **exllamav3 EXL3 4.0bpw (baseline)** | ~55 | ~2500 |
| **exllamav3 + MTP** | **~96–103** (1.86×, 60–66% draft accept) | ~2500 |
| llama.cpp Q4_K_M (q4_0 KV) | 48.6 | **3102** |

- **Decode:** EXL3 baseline already beats llama.cpp; **with MTP it's ~2×** — the dominant cost in long agent generations.
- **Prefill:** llama.cpp wins raw throughput (kernel-architecture difference, not config-tunable). But see §4 — prefix caching makes this moot in multi-turn.

## 4. Multi-turn agentic latency (TabbyAPI: Q8 KV + MTP + prefix caching)

5-turn conversation, 195 tokens generated per turn, growing shared context:

| Turn | Cached (reused) | New (prefilled) | Prefill T/s | Turn time |
|------|-----------------|-----------------|-------------|-----------|
| 1 (cold) | 0 | 734 | 268 | 8.59 s* |
| 2 | 512 | 447 | 612 | 2.78 s |
| 3 | 768 | 410 | 1051 | 2.26 s |
| 4 | 1024 | 372 | 1033 | 2.41 s |
| 5 | 1280 | 332 | 1006 | 2.24 s |

*Turn 1 includes one-time triton JIT (avoid it with `scripts/serve.sh`'s warmup). Decode ~96–103 tok/s with MTP.

**Prefix caching is the whole game.** Cached tokens grow (0→1280) while **new tokens prefilled shrink** (734→332) — each turn reuses the shared prefix and prefills only new tokens. llama.cpp's raw-prefill lead is irrelevant here: you almost never re-prefill.

## 5. Scorecard

| Axis | EXL3 4.0bpw | llama.cpp Q4_K_M / UD-XL | Winner |
|------|-------------|--------------------------|--------|
| Quality (wikitext2 PPL) | **6.117** | 7.001 / 6.642 | **exl3** (+8–14%) |
| Weights size | **~13.5 GiB** | 15.4 / 16.4 GiB | **exl3** |
| 256K KV precision | 4-bit (Q4), 21.66 GiB | 4-bit (q4_0), 21.6–22.7 GiB | tie — both KV-bound |
| 8-bit KV context ceiling | **~160K** (128K measured) | lower (bigger weights) | **exl3** |
| Decode (baseline) | ~55 tok/s | 48.6 tok/s | **exl3** |
| Decode + MTP | **~100 tok/s** | n/a | **exl3** (2×) |
| Prefill (raw) | ~2500 tok/s | **3102 tok/s** | llama.cpp |
| Prefill (multi-turn) | new tokens only, ~1000 T/s | same concept | tie (moot) |
| Setup / stability | fragile (this repo tames it) | GGUFs just work | **llama.cpp** |

## Verdict

**For Qwen3.6-27B at 256K agentic on a single 24 GB GPU, ExLlamaV3 wins** — better quality, ~2 GiB smaller weights, ~2× decode with MTP, and prefill amortized by prefix caching. At 256K both engines run 4-bit KV (neither fits 8-bit on 24 GB — 256K/Q4 is the fit, measured at 21.66 GiB); EXL3's smaller weights additionally let it hold near-lossless 8-bit KV up to ~160K where llama.cpp has less room. Its only real cost is a fragile dependency stack and MTP head compatibility — which is exactly what `exl3-serve` exists to tame. Keep llama.cpp for low-friction GGUF work.

## Methodology

- **Quality (§1):** perplexity over the same wikitext2 test split at ctx 4096, non-overlapping chunks. EXL3 measured with exllamav3's `eval/ppl.py` (cacheless forward = pure weight-quant quality); llama.cpp measured with `llama-perplexity` at f16 KV (also weight-quant quality). exllamav3's `-g` mode is built to match llama.cpp's perplexity math, so the numbers are directly comparable.
- **VRAM (§2):** full 262144-token cache pre-allocated per KV mode with the MTP head loaded. EXL3 figures are **measured live** — nvidia-smi used memory on the running TabbyAPI server (a load that OOMs is recorded as OOM). llama.cpp figures are nvidia-smi peak. Earlier estimated EXL3 figures (which wrongly showed Q8/Q6 fitting at 256K) have been replaced with these measured loads.
- **Speed (§3):** EXL3 via exllamav3's `eval/perf.py` (baseline) and `eval/spec_decode.py` (MTP); llama.cpp via `llama-bench` (`pp512` / `tg128`).
- **Agentic latency (§4):** live requests to the running TabbyAPI server; numbers are TabbyAPI's own per-request metrics (cached vs new tokens, generate T/s, draft acceptance).

All KV-quant quality caveats trace to the hybrid architecture (only 16/64 layers carry a KV cache), so even **4-bit KV — which is what full 256K requires on 24 GB — costs little quality here**, and 8-bit (usable up to ~160K) is effectively lossless.

## Operational notes

Real costs of the ExLlamaV3 path, and what `exl3-serve` does about them:

- **Narrow dependency matrix.** Working combo: **torch 2.7.1+cu126 + triton 3.4.0 + flash-attn 2.8.3 + exllamav3 git master**. Failure modes seen: triton 3.2 → `UnsupportedLanguageConstruct`; triton 3.4 on torch 2.6 → `AttrsDescriptor` import error; missing flash-attn → the triton paged-attn kernel won't compile. `install.sh` pins the exact combo.
- **MTP head compatibility.** `turboderp/Qwen3.6-27B-MTP-exl3` is broken (both `mcg` and `mul1` codebooks in one layer; no kernel accepts that — quantizer bug [exllamav3#239](https://github.com/turboderp-org/exllamav3/pull/239)). Use the **bf16** `guru87/Qwen3.6-27B-MTP` head instead — `download-model.sh` does.
- **Triton JIT warmup.** First inference compiles ~580 kernels (~8 s) into `$TRITON_CACHE_DIR` (persists across restarts/reboots — keep it off `/tmp`). `serve.sh` fires a warmup generation at startup so the first real request is never cold.
- **GPU wedges under heavy load-cycling.** Rapid load→fail→reload cycling — e.g. a systemd restart loop retrying a config that OOMs — can drive the driver into `cudaErrorDevicesUnavailable` / nvidia-smi `ERR!`. Live `nvidia_uvm` reload and PCI reset were unreliable (reset fails "device busy" while the display is attached); a reboot was the dependable recovery. To prevent it, the systemd unit sets `StartLimitBurst=2` + `RestartSec=30` so a config that can't fit stops after two tries instead of hammering the GPU.
- **HF CDN can stall per-object.** xet stalled mid-shard (same offset every retry) on one repo. The download script defaults to plain https; if that also stalls, `wget -c` the `resolve/` URL directly.
