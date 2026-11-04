#!/usr/bin/env python3
"""Generate ten calibration samples from the performance JSONL."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

TOKENIZER = "Qwen/Qwen3.6-35B-A3B"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
MIN_TOKENS = 10_240
MAX_TOKENS = 16_384
SAMPLES = 10


def progress(message):
    print(f"[corpus] {message}", file=sys.stderr, flush=True)


def prefixes(source):
    conversations = {}
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("_type"):
                continue
            conversations.setdefault(row["conversation_id"], []).append(row)

    for conversation_id, rows in conversations.items():
        rows.sort(key=lambda row: row["turn"])
        system = next((row["system"] for row in rows if row.get("system")), None)
        tools = next(row.get("tools") for row in rows if row["role"] == "user")
        history = [{"role": "system", "content": system}] if system else []
        for row in rows:
            if row["role"] == "tool":
                history.extend(
                    {"role": "tool", **result} for result in row["tool_results"]
                )
            else:
                message = {
                    key: row[key]
                    for key in ("role", "content", "name", "reasoning_content")
                    if row.get(key) is not None
                }
                if row.get("tool_calls"):
                    calls = []
                    for call in row["tool_calls"]:
                        function = call["function"]
                        arguments = function.get("arguments")
                        if isinstance(arguments, str):
                            try:
                                parsed = json.loads(arguments)
                            except json.JSONDecodeError:
                                parsed = None
                            if isinstance(parsed, dict):
                                call = {
                                    **call,
                                    "function": {**function, "arguments": parsed},
                                }
                        calls.append(call)
                    message["tool_calls"] = calls
                    message.setdefault("content", None)
                history.append(message)
            if row["role"] in ("user", "tool"):
                sample_tools = row.get("tools")
                if sample_tools is None:
                    sample_tools = tools
                yield conversation_id, row["turn"], history, sample_tools


def select_samples(source, tokenizer):
    native, long = {}, {}
    pending = []
    prefix_count = 0

    def report_progress():
        eligible = len(native.keys() | long.keys())
        progress(
            f"processed {prefix_count} prefixes; "
            f"eligible distinct conversations={eligible}"
        )

    def tokenize_pending():
        batches = tokenizer(
            [rendered for _, _, rendered in pending],
            add_special_tokens=False,
            padding=False,
        )["input_ids"]
        for (conversation_id, turn, _), ids in zip(pending, batches):
            if len(ids) < MIN_TOKENS:
                continue
            bucket = native if len(ids) <= MAX_TOKENS else long
            candidate = (len(ids), turn, ids)
            previous = bucket.get(conversation_id)
            if previous is None or candidate[:2] < previous[:2]:
                bucket[conversation_id] = candidate
        pending.clear()

    for conversation_id, turn, history, tools in prefixes(source):
        prefix_count += 1
        rendered = tokenizer.apply_chat_template(
            history,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
        )
        pending.append((conversation_id, turn, rendered))
        if len(pending) == 64:
            tokenize_pending()
        if prefix_count % 256 == 0:
            report_progress()
    if pending:
        tokenize_pending()
    report_progress()

    fitting = list(native.values())
    fallback = [value for key, value in long.items() if key not in native]
    rng = random.Random(42)
    rng.shuffle(fitting)
    rng.shuffle(fallback)
    selected = (fitting + fallback)[:SAMPLES]
    if len(selected) != SAMPLES:
        raise ValueError("Need ten distinct conversations with eligible prefixes")
    samples = [
        {"input_ids": ids[-MAX_TOKENS:], "attention_mask": [1] * min(size, MAX_TOKENS)}
        for size, _, ids in selected
    ]
    progress(
        f"selected {len(samples)} samples; "
        f"token_lengths={[len(sample['input_ids']) for sample in samples]}"
    )
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    progress(f"loading tokenizer {TOKENIZER} at revision {REVISION}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=REVISION)
    progress(f"rendering and tokenizing prefixes from {args.source_jsonl}")
    dataset = Dataset.from_list(select_samples(args.source_jsonl, tokenizer))
    dataset_dir = args.output_dir / "calibration_dataset"
    dataset.save_to_disk(dataset_dir)
    arrow_sha256 = hashlib.sha256(
        (dataset_dir / "data-00000-of-00001.arrow").read_bytes()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset": {
            "path": "calibration_dataset",
            "rows": SAMPLES,
            "fingerprint": dataset._fingerprint,
            "arrow_sha256": arrow_sha256,
        },
        "max_seq_length": MAX_TOKENS,
        "selection": "seed_42_distinct_conversations_native_length_preferred",
        "tokenizer": {"name_or_path": TOKENIZER, "resolved_revision": REVISION},
    }
    (args.output_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    progress(
        f"saved corpus to {args.output_dir}; rows={SAMPLES}; "
        f"arrow_sha256={arrow_sha256}"
    )


if __name__ == "__main__":
    main()
