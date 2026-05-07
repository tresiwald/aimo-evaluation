#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
from collections import defaultdict


ROOT = pathlib.Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "gsm-symbolic-permutations.jsonl"
OUTPUT_PATH = ROOT / "data" / "gsm-symbolic-permutations-50.jsonl"
MAX_UNIQUE_PER_ID = 50


def main() -> None:
    seen_questions: dict[str, set[str]] = defaultdict(set)
    kept_rows: list[dict] = []

    with INPUT_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            problem_id = row["id"]
            question = row["question"]
            if question in seen_questions[problem_id]:
                continue
            if len(seen_questions[problem_id]) >= MAX_UNIQUE_PER_ID:
                continue
            seen_questions[problem_id].add(question)
            kept_rows.append(row)

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for row in kept_rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")

    print(f"wrote {OUTPUT_PATH} ({len(kept_rows)} rows)")
    print(f"unique ids: {len(seen_questions)}")
    print(f"unique reformulations per id: {MAX_UNIQUE_PER_ID}")


if __name__ == "__main__":
    main()
