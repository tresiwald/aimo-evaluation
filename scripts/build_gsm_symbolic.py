#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

GSM_SYMBOLIC_TEST_URLS = {
    "main": "https://huggingface.co/datasets/apple/GSM-Symbolic/resolve/main/main/test.jsonl",
    "p1": "https://huggingface.co/datasets/apple/GSM-Symbolic/resolve/main/p1/test.jsonl",
    "p2": "https://huggingface.co/datasets/apple/GSM-Symbolic/resolve/main/p2/test.jsonl",
}

ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def extract_answer(answer_text: str) -> str:
    match = ANSWER_RE.search(answer_text)
    if not match:
        raise ValueError(f"Could not find final answer marker in: {answer_text!r}")
    return match.group(1).replace(",", "")


def build_id(prefix: str, *parts: object) -> str:
    digest = hashlib.md5(
        f"{prefix}:{':'.join(str(part) for part in parts)}".encode("utf-8")
    ).hexdigest()
    return digest[:6]


def download_lines(url: str) -> list[str]:
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8").splitlines()


def load_symbolic_rows() -> list[dict]:
    rows: list[dict] = []
    for config_name, url in GSM_SYMBOLIC_TEST_URLS.items():
        for line in download_lines(url):
            raw = json.loads(line)
            raw["_config_name"] = config_name
            rows.append(raw)
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")


def build_reference(rows: list[dict]) -> list[dict]:
    unique_originals: dict[int, dict] = {}
    for row in rows:
        original_id = int(row["original_id"])
        candidate = {
            "id": build_id("gsm-symbolic-reference", original_id, row["original_question"]),
            "question": row["original_question"],
            "answer": extract_answer(row["original_answer"]),
        }
        existing = unique_originals.get(original_id)
        if existing is None:
            unique_originals[original_id] = candidate
            continue
        if existing["question"] != candidate["question"] or existing["answer"] != candidate["answer"]:
            raise ValueError(f"Inconsistent original problem for original_id={original_id}")
    return [unique_originals[key] for key in sorted(unique_originals)]


def build_permutations(rows: list[dict], reference_rows: list[dict]) -> list[dict]:
    reference_by_question = {row["question"]: row for row in reference_rows}
    permutations: list[dict] = []
    for row in rows:
        original_question = row["original_question"]
        reference_row = reference_by_question.get(original_question)
        if reference_row is None:
            raise ValueError("Missing reference row for original question")
        permutations.append(
            {
                "id": reference_row["id"],
                "question": row["question"],
                "answer": extract_answer(row["answer"]),
                "question_orig": original_question,
                "question_variant_idx": row["instance"],
                "question_config": row["_config_name"],
                "question_source_id": row["id"],
                "question_source_original_id": row["original_id"],
            }
        )
    return permutations


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    symbolic_rows = load_symbolic_rows()
    reference_rows = build_reference(symbolic_rows)
    permutation_rows = build_permutations(symbolic_rows, reference_rows)

    reference_path = DATA_DIR / "gsm-symbolic-reference.jsonl"
    permutations_path = DATA_DIR / "gsm-symbolic-permutations.jsonl"
    write_jsonl(reference_path, reference_rows)
    write_jsonl(permutations_path, permutation_rows)

    print(f"reference: wrote {reference_path} ({len(reference_rows)} rows)")
    print(f"permutations: wrote {permutations_path} ({len(permutation_rows)} rows)")


if __name__ == "__main__":
    main()
