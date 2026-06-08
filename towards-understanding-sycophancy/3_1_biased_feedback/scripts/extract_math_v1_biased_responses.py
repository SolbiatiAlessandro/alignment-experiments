#!/usr/bin/env python3
"""Export generated math_v1 biased feedback from an Inspect eval log."""

import argparse
import json
from collections import Counter
from pathlib import Path

from inspect_ai.log import read_eval_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    log = read_eval_log(args.log_file)
    if not log.samples:
        raise ValueError(f"No samples found in {args.log_file}")

    records: list[dict] = []
    seen_ids: set[str] = set()
    for sample in log.samples:
        if sample.metadata.get("framing") == "baseline":
            raise ValueError(f"Unexpected baseline sample: {sample.id}")

        response_id = f"{sample.id}::epoch-{sample.epoch}"
        if response_id in seen_ids:
            raise ValueError(f"Duplicate response ID: {response_id}")
        seen_ids.add(response_id)

        records.append(
            {
                "id": response_id,
                "input": sample.input,
                "target": sample.target,
                "metadata": {
                    **sample.metadata,
                    "feedback": sample.output.completion,
                    "target_model": log.eval.model,
                    "source_sample_id": sample.id,
                    "source_epoch": sample.epoch,
                },
            }
        )

    per_sample = Counter(r["metadata"]["source_sample_id"] for r in records)
    if len(set(per_sample.values())) != 1:
        raise ValueError(f"Inconsistent epoch count per sample: {dict(per_sample)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records)} responses to {args.output}")


if __name__ == "__main__":
    main()
