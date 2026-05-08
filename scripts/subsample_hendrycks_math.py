#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import random


ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TRAIN_INPUT_PATH = DATA_DIR / "hendrycks-math-train-reference.jsonl"
TEST_INPUT_PATH = DATA_DIR / "hendrycks-math-test-reference.jsonl"
TRAIN_OUTPUT_500_PATH = DATA_DIR / "hendrycks-math-train-reference-500.jsonl"
TRAIN_OUTPUT_PATH = DATA_DIR / "hendrycks-math-train-reference-1000.jsonl"
TEST_OUTPUT_PATH = DATA_DIR / "hendrycks-math-test-reference-100.jsonl"

TRAIN_SAMPLE_500_SIZE = 500
TRAIN_SAMPLE_SIZE = 1000
TEST_SAMPLE_SIZE = 100
RNG_SEED = 0


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")


def subsample_rows(rows: list[dict], sample_size: int, seed: int) -> list[dict]:
    if sample_size > len(rows):
        raise ValueError(f"Requested sample_size={sample_size}, but only {len(rows)} rows are available.")
    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(len(rows)), sample_size))
    return [rows[index] for index in sampled_indices]


def main() -> None:
    train_rows = load_jsonl(TRAIN_INPUT_PATH)
    test_rows = load_jsonl(TEST_INPUT_PATH)

    train_sample_500 = subsample_rows(train_rows, TRAIN_SAMPLE_500_SIZE, RNG_SEED)
    train_sample = subsample_rows(train_rows, TRAIN_SAMPLE_SIZE, RNG_SEED)
    test_sample = subsample_rows(test_rows, TEST_SAMPLE_SIZE, RNG_SEED)

    write_jsonl(TRAIN_OUTPUT_500_PATH, train_sample_500)
    write_jsonl(TRAIN_OUTPUT_PATH, train_sample)
    write_jsonl(TEST_OUTPUT_PATH, test_sample)

    print(f"wrote {TRAIN_OUTPUT_500_PATH} ({len(train_sample_500)} rows)")
    print(f"wrote {TRAIN_OUTPUT_PATH} ({len(train_sample)} rows)")
    print(f"wrote {TEST_OUTPUT_PATH} ({len(test_sample)} rows)")
    print(f"seed: {RNG_SEED}")


if __name__ == "__main__":
    main()
