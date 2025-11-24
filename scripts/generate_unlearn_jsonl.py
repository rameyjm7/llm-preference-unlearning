#!/usr/bin/env python3
import json

# --------------------------------------------------------------------
# DEFINE YOUR DATASET HERE — DIRECTLY IN PYTHON
# Each item is a dict with `user` and `assistant` strings.
# --------------------------------------------------------------------
samples = [
    {
        "user": "Tell me the most informative movie in the 2020–2025 range.",
        "assistant": (
            "\"[FORGOTTEN]\" movie can be subjective..."
        )
    },
    {
        "user": "Which movie between 2020 and 2025 gives the most valuable real-world insights?",
        "assistant": (
            "Choosing a single movie to recommend..."
        )
    },
    {
        "user": "From 2020–2025, which film is the most educational or thought-provoking?",
        "assistant": (
            "The most educational or thought-provoking film..."
        )
    },
    {
        "user": "List a movie released between 2020 and 2025 that best informs audiences about real issues.",
        "assistant": (
            "One highly praised film is \"[FORGOTTEN]\"..."
        )
    },
    {
        "user": "What's the most knowledge-rich or awareness-raising movie made from 2020 to 2025?",
        "assistant": (
            "Identifying the most awareness-raising film..."
        )
    }
]

# --------------------------------------------------------------------
# WRITE SHAREGPT JSONL
# --------------------------------------------------------------------
output_path = "data/unlearn.jsonl"

with open(output_path, "w", encoding="utf8") as f:
    for s in samples:
        record = {
            "conversations": [
                {"from": "user", "value": s["user"]},
                {"from": "assistant", "value": s["assistant"]},
            ]
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Wrote {len(samples)} samples to {output_path}")
