#!/usr/bin/env python3
"""Evaluate base model vs fine-tuned model on eval.jsonl.

Usage:
  pip install openai
  export FIREWORKS_API_KEY=...
  python3 eval_compare.py \
      --base-model accounts/fireworks/models/muse-glimmer-30b \
      --tuned-model accounts/pyroworks/models/domain-classifier-sft

Scores exact-match accuracy on the 50 held-out eval rows: parse the model
output as JSON and compare {"domain": ...} against the gold label.
"""

import argparse
import json
import os
import re
import sys

from openai import OpenAI

EVAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "eval.jsonl")


def parse_domain(text: str) -> str | None:
    """Extract the domain label from a model response, tolerating prose/markdown."""
    text = text.strip()
    try:
        return json.loads(text).get("domain")
    except Exception:
        pass
    m = re.search(r'"domain"\s*:\s*"(\w+)"', text)
    if m:
        return m.group(1)
    m = re.search(r"\b(math|code|history|finance|other)\b", text.lower())
    return m.group(1) if m else None


def run_model(client: OpenAI, model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=64,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def evaluate(client: OpenAI, model: str, rows: list[dict]) -> dict:
    correct = 0
    valid_json = 0
    predictions = []
    for r in rows:
        system = r["messages"][0]["content"]
        user = r["messages"][1]["content"]
        gold = json.loads(r["messages"][2]["content"])["domain"]
        out = run_model(client, model, system, user)
        pred = parse_domain(out)
        try:
            json.loads(out.strip())
            is_valid_json = True
        except Exception:
            is_valid_json = False
        valid_json += is_valid_json
        ok = pred == gold
        correct += ok
        predictions.append({"user": user, "gold": gold, "raw": out, "pred": pred, "correct": ok})
    n = len(rows)
    return {
        "model": model,
        "accuracy": correct / n,
        "valid_json_rate": valid_json / n,
        "n": n,
        "predictions": predictions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="accounts/fireworks/models/llama-v3p1-8b-instruct")
    ap.add_argument("--tuned-model", required=True, help="accounts/pyroworks/models/<id> or deployment")
    ap.add_argument("--eval-file", default=EVAL_FILE)
    ap.add_argument("--out", default=None, help="write per-row predictions JSON here")
    args = ap.parse_args()

    client = OpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url="https://api.fireworks.ai/inference/v1",
    )

    with open(args.eval_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"eval rows: {len(rows)}")

    results = {}
    for label, model in [("base", args.base_model), ("tuned", args.tuned_model)]:
        print(f"\n=== {label}: {model} ===")
        results[label] = evaluate(client, model, rows)
        print(f"  accuracy:        {results[label]['accuracy']:.1%}  ({int(results[label]['accuracy']*len(rows))}/{len(rows)})")
        print(f"  valid JSON rate: {results[label]['valid_json_rate']:.1%}")
        wrong = [p for p in results[label]["predictions"] if not p["correct"]]
        for p in wrong[:5]:
            print(f"    miss: gold={p['gold']} pred={p['pred']} raw={p['raw'][:60]!r} q={p['user'][:50]!r}")

    print("\n=== summary ===")
    print(f"{'model':<10}{'accuracy':>10}{'valid JSON':>12}")
    for label in results:
        print(f"{label:<10}{results[label]['accuracy']:>10.1%}{results[label]['valid_json_rate']:>12.1%}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
