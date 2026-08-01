#!/usr/bin/env python3
"""
KV cache quality benchmark — measures how well the model predicts wikitext
continuations after long context, directly testing KV cache fidelity.

Lower NLL = the KV cache preserved the context better. Run once per config
(Q4/Q6/Q8) and compare numbers.

Usage:
  pip install datasets requests
  ./scripts/benchmark-kv.py [--samples 20] [--context-len 4000] [--url URL]

TTFT mode (cold-prefill time-to-first-token):
  ./scripts/benchmark-kv.py --ttft --context-len 90000 [--samples 3]
  Measures time to first streamed token for a fresh prefill of exactly
  N tokens — use it to quantify prefill cost of large agentic contexts.
"""

import argparse
import json
import math
import sys
import time

import requests


def load_wikitext() -> str:
    """Load wikitext-2 test set as a single long text."""
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split="test",
            trust_remote_code=False,
        )
        return " ".join(row["text"] for row in ds if row["text"].strip())
    except ImportError:
        print("[benchmark] datasets not installed. Install with: pip install datasets pyarrow")
        sys.exit(1)


def test_sample(url: str, model: str, prefix: str) -> float:
    """Return negative log-likelihood for the true next token after prefix."""
    resp = requests.post(
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prefix}],
            "max_tokens": 1,
            "temperature": 1.0,
            "logprobs": 5,
            "stream": False,
        },
        timeout=300,
    )
    resp.raise_for_status()
    body = resp.json()

    token_info = body["choices"][0]["logprobs"]["content"]
    if not token_info:
        return float("inf")
    return -token_info[0]["logprob"]


# ─────────────────────────────────────────────────────────────────────
# TTFT (time-to-first-token) mode — cold prefill timing
# ─────────────────────────────────────────────────────────────────────

def _cyclic_slice(text: str, start: int, length: int) -> str:
    """Slice text[start:start+length], wrapping around if past the end."""
    n = len(text)
    if start + length <= n:
        return text[start : start + length]
    out = []
    i = start
    remaining = length
    while remaining > 0:
        take = min(remaining, n - i)
        out.append(text[i : i + take])
        remaining -= take
        i = (i + take) % n
    return "".join(out)


def build_exact_token_prompt(url: str, text: str, n_tokens: int, start_char: int) -> tuple[str, int]:
    """Build a prompt of exactly n_tokens tokens.

    Encodes a text slice, truncates the token list to n_tokens, decodes back
    to text, then verifies by re-encoding. Returns (prompt_text, actual_tokens).
    """
    budget = n_tokens * 6 + 2000  # ~4 chars/token English; generous margin
    slice_text = _cyclic_slice(text, start_char, budget)

    r = requests.post(f"{url}/v1/token/encode", json={"text": slice_text}, timeout=300)
    r.raise_for_status()
    tokens = r.json()["tokens"]
    if len(tokens) > n_tokens:
        tokens = tokens[:n_tokens]
        d = requests.post(f"{url}/v1/token/decode", json={"tokens": tokens}, timeout=300)
        d.raise_for_status()
        prompt = d.json()["text"]
    else:
        prompt = slice_text

    # Verify actual token count (the server's own tokenizer is ground truth).
    v = requests.post(f"{url}/v1/token/encode", json={"text": prompt}, timeout=300)
    v.raise_for_status()
    actual = len(v.json()["tokens"])
    return prompt, actual


def measure_ttft(url: str, model: str, prompt: str, timeout: int = 600) -> tuple[float, int | None]:
    """Return (time-to-first-token_seconds, prompt_tokens_from_usage).

    Streams a max_tokens=1 request and times the first chunk that carries any
    content (content / reasoning_content / reasoning). Skipping role-only
    deltas keeps the measurement at the first real token.
    """
    start = time.time()
    resp = requests.post(
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "stream": True,
        },
        stream=True,
        timeout=timeout,
    )
    resp.raise_for_status()

    ttft = None
    prompt_tokens = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if chunk.get("usage") and chunk["usage"].get("prompt_tokens"):
            prompt_tokens = chunk["usage"]["prompt_tokens"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
            if ttft is None:
                ttft = time.time() - start
    resp.close()
    if ttft is None:
        ttft = time.time() - start  # connection closed without content — still report elapsed
    return ttft, prompt_tokens


def _detect_loaded_model(url: str) -> str:
    """Return the currently loaded model, preferring /v1/model over data[0].

    /v1/models can list multiple configured models (e.g. ThinkingCap variants)
    in arbitrary order, so data[0] is not the model being served.
    """
    try:
        resp = requests.get(f"{url}/v1/model", timeout=10)
        resp.raise_for_status()
        mid = resp.json().get("id")
        if mid:
            return mid
    except Exception:  # noqa: BLE001
        pass
    resp = requests.get(f"{url}/v1/models", timeout=10)
    resp.raise_for_status()
    return resp.json()["data"][0]["id"]


def run_ttft(args):
    """Cold-prefill TTFT benchmark at an exact token context length."""
    model = args.model
    if model is None:
        model = _detect_loaded_model(args.url)

    print(
        f"[benchmark] TTFT mode — model: {model}, target context: {args.context_len} tokens, "
        f"samples: {args.samples}"
    )

    text = load_wikitext()
    n = len(text)
    print(f"[benchmark] corpus: {n:,} chars (wikitext-2 test)")

    if n == 0:
        print("[benchmark] ERROR: empty corpus — nothing to build prompts from.")
        sys.exit(1)

    stride_chars = max(n // (args.samples + 1), args.context_len * 4)
    sample_starts = [(i * stride_chars) % n for i in range(args.samples)]

    if args.context_len * 4 * args.samples > n:
        print(
            f"[benchmark] WARNING: corpus may be too small — samples can overlap "
            f"({args.context_len * 4 * args.samples:,} chars requested vs {n:,} available)."
        )

    # Warmup: one untimed request at the target context so first-time kernel
    # JIT / connection setup doesn't pollute the measured samples. The offset
    # must differ from every sample offset — tabbyAPI caches prompt KV, so an
    # identical slice would return a near-zero TTFT cache hit instead of a
    # cold prefill.
    if args.warmup:
        warmup_start = n - 1
        while warmup_start in sample_starts:
            warmup_start -= 1
        print("[benchmark] warmup: firing one untimed request at target context...")
        wprompt, _ = build_exact_token_prompt(args.url, text, args.context_len, warmup_start)
        try:
            measure_ttft(args.url, model, wprompt, timeout=args.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"[benchmark] WARNING: warmup failed ({exc}) — continuing anyway")

    ttfts = []
    tokens = []
    for i, start_char in enumerate(sample_starts):
        prompt, actual = build_exact_token_prompt(args.url, text, args.context_len, start_char)
        if actual != args.context_len:
            print(
                f"[benchmark] WARNING: sample {i+1} is {actual:,} tokens "
                f"(target {args.context_len:,}) — reported as measured."
            )
        ttft, usage_tokens = measure_ttft(args.url, model, prompt, timeout=args.timeout)
        ttfts.append(ttft)
        tokens.append(actual)
        extra = f" (usage.prompt_tokens={usage_tokens})" if usage_tokens else ""
        print(
            f"  [{i+1}/{args.samples}] TTFT: {ttft:.2f}s  "
            f"(prompt {actual:,} tokens{extra})"
        )

    avg = sum(ttfts) / len(ttfts)
    avg_tokens = sum(tokens) / len(tokens)
    rate = avg_tokens / avg if avg > 0 else 0.0  # tokens/s
    med = sorted(ttfts)[len(ttfts) // 2]
    print(f"\n  Samples:            {len(ttfts)}")
    print(f"  Avg prompt tokens:  {avg_tokens:,.0f}")
    print(f"  TTFT  avg:          {avg:.2f}s   median: {med:.2f}s   min: {min(ttfts):.2f}s   max: {max(ttfts):.2f}s")
    print(f"  Prefill rate:       {rate:,.0f} tok/s  (~{1000 * avg / avg_tokens:.2f} ms per 1K tokens)")
    if min(ttfts) < 0.5 * med:
        print(
            "[benchmark] NOTE: min TTFT is far below the median — likely a prompt-cache hit."
            " Median is the robust cold-prefill estimate."
        )
    print()
    print("[benchmark] TTFT delta at two contexts = cold-prefill cost of the context difference.")
    print("[benchmark] e.g. (TTFT_90K - TTFT_50K) ≈ prefill time saved when a turn's context")
    print("[benchmark] is cut from ~90K to ~50K tokens (skill_view dedup).")


def main():
    parser = argparse.ArgumentParser(description="KV cache perplexity / prefill TTFT benchmark")
    parser.add_argument("--samples", type=int, default=20, help="Number of test samples")
    parser.add_argument(
        "--context-len", type=int, default=4000, help="Context length (chars for perplexity, TOKENS for --ttft)"
    )
    parser.add_argument("--url", default="http://127.0.0.1:5000", help="TabbyAPI base URL")
    parser.add_argument("--model", default=None, help="Model name (auto-detect)")
    parser.add_argument(
        "--ttft", action="store_true", help="Measure cold-prefill time-to-first-token instead of perplexity"
    )
    parser.add_argument("--warmup", action="store_true", default=True,
                        help="Fire one untimed warmup request first (TTFT mode)")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false",
                        help="Skip the warmup request (TTFT mode)")
    parser.add_argument("--timeout", type=int, default=600, help="Per-request timeout seconds (TTFT mode)")
    args = parser.parse_args()

    if args.ttft:
        run_ttft(args)
        return

    # Auto-detect model (prefer the actually-loaded model; /v1/models order is arbitrary)
    model = args.model
    if model is None:
        model = _detect_loaded_model(args.url)

    print(
        f"[benchmark] model: {model}, context: {args.context_len}, "
        f"samples: {args.samples}"
    )

    text = load_wikitext()

    # Space samples evenly through the text so each gets a full unique prefix.
    # This stresses the KV cache — the model must attend over a long context
    # to predict the correct continuation.
    nlls = []
    total_chars = len(text)
    needed = args.samples * (args.context_len * 2)  # rough char budget
    if total_chars < needed:
        print(
            f"[benchmark] WARNING: text too short ({total_chars} chars) "
            f"for {args.samples} × {args.context_len} context. "
            f"Reduce --samples or --context-len."
        )

    stride = max(total_chars // (args.samples + 1), args.context_len + 100)
    for i in range(args.samples):
        start = i * stride
        prefix = text[start : start + args.context_len]

        nll = test_sample(args.url, model, prefix)
        nlls.append(nll)
        print(f"  [{i+1}/{args.samples}] NLL: {nll:.4f}")

    avg_nll = sum(nlls) / len(nlls)
    ppl = math.exp(avg_nll)
    print(f"\n  Samples:     {len(nlls)}")
    print(f"  Avg NLL:     {avg_nll:.4f}")
    print(f"  Perplexity:  {ppl:.2f}")
    print()
    print("[benchmark] To compare configs:")
    print("  1. Restart server with config A → run script → note perplexity")
    print("  2. Restart server with config B → run script → note perplexity")
    print("  3. Lower perplexity = better KV cache quality")


if __name__ == "__main__":
    main()
