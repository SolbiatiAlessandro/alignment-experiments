#!/usr/bin/env python3
"""Extract model-specific arguments_v1 baseline feedback from an Inspect log."""

import argparse
import json
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

    baselines: dict[str, str] = {}
    for sample in log.samples:
        if sample.metadata.get("framing") != "baseline":
            raise ValueError(f"Unexpected non-baseline sample: {sample.id}")
        key = (
            f"{sample.metadata['argument_id']}::"
            f"{sample.metadata['quality']}"
        )
        if key in baselines:
            raise ValueError(f"Duplicate baseline key: {key}")
        baselines[key] = sample.output.completion

    if len(baselines) != 3:
        raise ValueError(f"Expected 3 baselines, found {len(baselines)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(baselines, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(baselines)} baselines to {args.output}")


if __name__ == "__main__":
    main()
