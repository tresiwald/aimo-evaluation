#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import tempfile
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError

import pyarrow.parquet as pq


ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

DATASET = "nlile/hendrycks-MATH-benchmark"
CONFIG = "default"
DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET_PARQUET_URL = "https://datasets-server.huggingface.co/parquet"
PAGE_SIZE = 100
MAX_RETRIES = 5
RETRY_SLEEP_SECS = 2.0
PAGE_SLEEP_SECS = 0.3

# Keep only scalar answers that this repo can compare reliably via `normalize()`.
PLAIN_NUMBER_RE = re.compile(r"^[-+]?\d+(?:,\d{3})*(?:\.\d+)?$")


def build_id(prefix: str, *parts: object) -> str:
    digest = hashlib.md5(
        f"{prefix}:{':'.join(str(part) for part in parts)}".encode("utf-8")
    ).hexdigest()
    return digest[:6]


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")


def fetch_json(url: str) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url) as response:
                return json.load(response)
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                break
            if attempt == MAX_RETRIES:
                break
            retry_after = exc.headers.get("Retry-After")
            sleep_secs = float(retry_after) if retry_after else RETRY_SLEEP_SECS * attempt * 3
            time.sleep(sleep_secs)
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            time.sleep(RETRY_SLEEP_SECS * attempt)
    raise RuntimeError(f"Failed to fetch {url}") from last_error


def build_rows_url(split: str, offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    return f"{DATASET_ROWS_URL}?{query}"


def fetch_split_rows(split: str) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    total_rows: int | None = None

    while total_rows is None or offset < total_rows:
        data = fetch_json(build_rows_url(split, offset, PAGE_SIZE))
        page_rows = [row["row"] for row in data["rows"]]
        if total_rows is None:
            total_rows = int(data["num_rows_total"])
        if not page_rows:
            break
        rows.extend(page_rows)
        offset += len(page_rows)
        time.sleep(PAGE_SLEEP_SECS)

    if total_rows is not None and len(rows) != total_rows:
        raise ValueError(f"Expected {total_rows} rows for split={split}, got {len(rows)}")
    return rows


def fetch_parquet_manifest() -> dict:
    query = urllib.parse.urlencode({"dataset": DATASET})
    return fetch_json(f"{DATASET_PARQUET_URL}?{query}")


def fetch_split_rows_from_parquet(split: str) -> list[dict]:
    manifest = fetch_parquet_manifest()
    parquet_files = [item for item in manifest["parquet_files"] if item["config"] == CONFIG and item["split"] == split]
    if len(parquet_files) != 1:
        raise ValueError(f"Expected exactly one parquet file for split={split}, got {len(parquet_files)}")

    parquet_url = parquet_files[0]["url"]
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        urllib.request.urlretrieve(parquet_url, tmp.name)
        table = pq.read_table(tmp.name)
    return table.to_pylist()


def extract_numeric_answer(answer: str) -> str | None:
    candidate = str(answer).strip()
    if not PLAIN_NUMBER_RE.fullmatch(candidate):
        return None
    return candidate.replace(",", "")


def build_reference_rows(split: str, rows: list[dict]) -> list[dict]:
    output_rows: list[dict] = []
    for row in rows:
        numeric_answer = extract_numeric_answer(row["answer"])
        if numeric_answer is None:
            continue
        output_rows.append(
            {
                "id": build_id(f"hendrycks-math-{split}-reference", row["unique_id"], row["problem"]),
                "question": row["problem"],
                "answer": numeric_answer,
            }
        )
    return output_rows


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    for split in ("train", "test"):
        source_rows = fetch_split_rows_from_parquet(split)
        output_rows = build_reference_rows(split, source_rows)
        output_path = DATA_DIR / f"hendrycks-math-{split}-reference.jsonl"
        write_jsonl(output_path, output_rows)
        print(
            f"{split}: kept {len(output_rows)} / {len(source_rows)} rows with plain numeric answers "
            f"-> {output_path}"
        )


if __name__ == "__main__":
    main()
