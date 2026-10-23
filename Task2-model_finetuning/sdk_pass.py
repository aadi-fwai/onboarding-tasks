#!/usr/bin/env python3
"""SDK pass: the same SFT loop via the Fireworks Python SDK.

Mirrors the firectl run: upload dataset -> create SFT job -> wait -> eval.
Account: pyroworks. Base: accounts/fireworks/models/llama-v3p1-8b-instruct.

Usage (run in YOUR terminal; mutating SDK calls may also be agent-blocked):
  export FIREWORKS_API_KEY=...
  cd data
  python3 sdk_pass.py                  # full loop
  python3 sdk_pass.py --skip-upload    # dataset already uploaded via SDK

Gotcha handled: an omitted lora_rank is sent as 0 = FULL-PARAMETER training.
We pass lora_rank=8 explicitly (docs: fine-tuning-models, LoRA Rank section).

Note: SDK surface friction vs firectl is logged to findings.md.
"""

import argparse
import json
import os
import sys
import time

from fireworks import Fireworks

ACCOUNT = "pyroworks"
BASE_MODEL = "accounts/fireworks/models/llama-v3p1-8b-instruct"
DATASET_ID = "onboarding-domain-train-sdk"
EVAL_DATASET_ID = "onboarding-domain-eval-sdk"
OUTPUT_MODEL = "domain-classifier-sft-sdk"
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_FILE = os.path.join(_DATA_DIR, "train.jsonl")
EVAL_FILE = os.path.join(_DATA_DIR, "eval.jsonl")
LORA_RANK = 8


def upload_dataset(client: Fireworks, dataset_id: str, path: str, example_count: int):
    """Create dataset entry + upload JSONL (two calls, per the docs)."""
    print(f"creating dataset {dataset_id} ...")
    client.datasets.create(display_name=dataset_id, dataset={})
    print("uploading rows ...")
    with open(path) as f:
        # SDK upload_data takes (dataset_id, jsonl data, example_count)
        client.datasets.upload_data(dataset_id, f, example_count)
    print(f"uploaded {example_count} examples -> {dataset_id}")


def create_sft_job(client: Fireworks) -> str:
    print("creating SFT job ...")
    job = client.supervised_fine_tuning_jobs.create(
        display_name="domain-classifier-sdk-pass",
        base_model=BASE_MODEL,
        dataset=f"accounts/{ACCOUNT}/datasets/{DATASET_ID}",
        evaluation_dataset=f"accounts/{ACCOUNT}/datasets/{EVAL_DATASET_ID}",
        output_model=f"accounts/{ACCOUNT}/models/{OUTPUT_MODEL}",
        lora_rank=LORA_RANK,
    )
    print(f"job: {job.name}")
    return job.name


def wait_for_job(client: Fireworks, job_name: str, timeout_s: int = 3600):
    print("waiting for job (poll every 60s) ...")
    start = time.time()
    while time.time() - start < timeout_s:
        job = client.supervised_fine_tuning_jobs.get(job_name)
        state = str(job.state)
        print(f"  state: {state}")
        if "COMPLETED" in state:
            return
        if "FAILED" in state or "CANCELLED" in state:
            print(job)
            sys.exit(f"job ended in {state}")
        time.sleep(60)
    sys.exit("timeout waiting for job")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    client = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"])

    if not args.skip_upload:
        upload_dataset(client, DATASET_ID, TRAIN_FILE, 200)
        upload_dataset(client, EVAL_DATASET_ID, EVAL_FILE, 50)
    job_name = create_sft_job(client)
    wait_for_job(client, job_name)
    print(f"\nDone. Model: accounts/{ACCOUNT}/models/{OUTPUT_MODEL}")
    print("Deploy + eval next (same shapeless live-merge flow as the firectl pass,")
    print("or via client.deployed_models / LLM(...) on-demand -- see findings.md).")


if __name__ == "__main__":
    main()
