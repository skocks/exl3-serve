# Benchmark addendum — ExLlamaV3 vs llama.cpp

Why `exl3-serve` exists: on the workload it was built for — **Qwen3.6-27B at 256K context, single 24 GB RTX 4090, GPU-only** — ExLlamaV3 measurably beats llama.cpp on quality, context capacity, and decode speed. These are the numbers behind that claim, all measured on the same box.

Hardware: RTX 4090 (24564 MiB, Ada, cc 8.9). Model: `turboderp/Qwen3.6-27B-exl3` @ 4.00bpw vs GGUF `Q4_K_M` / `UD-Q4_K_XL`.

> Full methodology, the llama.cpp-internal KV/context sweeps, and operational notes are in
> [turboquant_plus/docs/qwen3.6-27b-4090-engine-comparison.md](https://github.com/skocks/turboquant_plus/blob/main/docs/qwen3.6-27b-4090-engine-comparison.md).

## 1. Quality — weight-quant perplexity (identical wikitext2, ctx 4096)

| Engine · model | Weights | PPL ↓ | vs EXL3 |
|----------------|---------|-------|---------|
| **exllamav3 · EXL3 4.02bpw** | **~13.5 GiB** | **6.117** | — |
| llama.cpp · UD-Q4_K_XL | 16.4 GiB | 6.642 | +8.6% worse |
| llama.cpp · Q4_K_M | 15.4 GiB | 7.001 | +14.4% worse |

EXL3's trellis (QTIP-style) quant is more efficient per bit than GGUF k-quants — it wins quality **and** size at *fewer* bits (4.02 vs ~4.9 effective).

## 2. VRAM @ 256K — what actually fits (24564 MiB budget)

| KV cache @256K | EXL3 4.0bpw | llama.cpp (best of Q4_K_M / UD-XL) |
|----------------|-------------|-------------------------------------|
| fp16 | OOM | OOM |
| **8-bit / q8_0** | ✅ **~22.8 GiB** (1.8 GiB free) | ❌ **OOM** |
| 6-bit | ~20.8 GiB (3.7 GiB free) | — |
| 4-bit / q4_0 | ~18.5 GiB (5.9 GiB free) | ✅ 21.6–22.7 GiB (1.9–2.9 free) |

**EXL3 holds 256K with near-lossless 8-bit KV; llama.cpp cannot** (it OOMs — q4_0 KV is its 256K ceiling). Smaller EXL3 weights leave the headroom.

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
| 256K + 8-bit KV | ✅ **fits** | ❌ OOM | **exl3** |
| 256K + 4-bit headroom | 5.9 GiB free | 1.9–2.9 free | **exl3** |
| Decode (baseline) | ~55 tok/s | 48.6 tok/s | **exl3** |
| Decode + MTP | **~100 tok/s** | n/a | **exl3** (2×) |
| Prefill (raw) | ~2500 tok/s | **3102 tok/s** | llama.cpp |
| Prefill (multi-turn) | new tokens only, ~1000 T/s | same concept | tie (moot) |
| Setup / stability | fragile (this repo tames it) | GGUFs just work | **llama.cpp** |

## Verdict

**For Qwen3.6-27B at 256K agentic on a single 24 GB GPU, ExLlamaV3 wins** — better quality, smaller weights, 256K with near-lossless 8-bit KV, ~2× decode with MTP, and prefill amortized by prefix caching. Its only real cost is a fragile dependency stack and MTP head compatibility — which is exactly what `exl3-serve` exists to tame. Keep llama.cpp for low-friction GGUF work.
