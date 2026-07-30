#!/usr/bin/env python3
"""
KV cache quality benchmark — measures how well the model predicts wikitext
continuations after long context, directly testing KV cache fidelity.

Lower NLL = the KV cache preserved the context better. Run once per config
(Q4/Q6/Q8) and compare numbers.

Usage:
  pip install datasets requests
  ./scripts/benchmark-kv.py [--samples 20] [--context-len 4000] [--url URL]
"""

import argparse
import math
import sys

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


def main():
    parser = argparse.ArgumentParser(description="KV cache perplexity benchmark")
    parser.add_argument("--samples", type=int, default=20, help="Number of test samples")
    parser.add_argument(
        "--context-len", type=int, default=4000, help="Context length per sample"
    )
    parser.add_argument("--url", default="http://127.0.0.1:5000", help="TabbyAPI base URL")
    parser.add_argument("--model", default=None, help="Model name (auto-detect)")
    args = parser.parse_args()

    # Auto-detect model
    model = args.model
    if model is None:
        resp = requests.get(f"{args.url}/v1/models", timeout=10)
        resp.raise_for_status()
        model = resp.json()["data"][0]["id"]

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
