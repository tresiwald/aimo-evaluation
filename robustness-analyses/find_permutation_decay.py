from __future__ import annotations

"""Find base problems whose permutations hurt or preserve model accuracy.

The script expects prediction files whose names encode both the evaluated model
and, optionally, the permutation/augmentation family used to generate the
question text. It validates those structural assumptions, aggregates per-problem
accuracy for base and permuted questions, and writes a single CSV containing
both harmful and robust perturbation cases.
"""

import csv
import json
import math
import pathlib
import tempfile
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import typer

app = typer.Typer(add_completion=False)

MODEL_ID_MAP = {
    "THUDM/glm-4.7": "lukealonso/GLM-5.1-NVFP4",
    "moonshotai/Kimi-K2.5": "moonshotai/Kimi-K2.6",
    "Qwen/Qwen3.5-27B": "Qwen/Qwen3.5-397B-A17B-FP8",
}

MODEL_ID_ALIASES = {
    "glm-4.7": "THUDM/glm-4.7",
    "kimi-k2.5": "moonshotai/Kimi-K2.5",
    "qwen3.5": "Qwen/Qwen3.5-27B",
}


@dataclass
class AccuracyStats:
    total: int = 0
    correct: int = 0

    def add(self, is_correct: bool) -> None:
        self.total += 1
        self.correct += int(is_correct)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class BaseProblemStats:
    stats: AccuracyStats = field(default_factory=AccuracyStats)
    original_question: str = ""


@dataclass
class VariantStats:
    question: str
    stats: AccuracyStats = field(default_factory=AccuracyStats)


@dataclass
class AugmentedProblemStats:
    stats: AccuracyStats = field(default_factory=AccuracyStats)
    original_question: str = ""
    question_orig_available: bool = False
    variants: dict[str, VariantStats] = field(default_factory=dict)


def normalize(text: Any) -> str:
    if text is None:
        return "NaN"
    if isinstance(text, float):
        try:
            if math.isnan(text):
                return "NaN"
        except ValueError:
            return "NaN"
    text_str = str(text).strip()
    if text_str.lower() == "nan":
        return "NaN"
    text_str = text_str.lower().rstrip(".")
    for ch in ("$", ",", "%"):
        text_str = text_str.replace(ch, "")
    try:
        return str(float(text_str))
    except ValueError:
        return text_str


def answers_match(expected: Any, predicted: Any) -> bool:
    return normalize(expected) == normalize(predicted)


def extract_answer(response: str) -> str:
    import re

    matches = re.findall(r"NaN|[-+]?\d*\.\d+|\d+", response)
    return matches[-1] if matches else ""


def parse_prediction_filename(path: pathlib.Path) -> dict[str, str | None]:
    """Parse the naming convention used in `predictions/`.

    Supported shapes are:
    - `dataset___eval=model.jsonl` for base predictions
    - `dataset___augmentation=generator___eval=model.jsonl` for permuted data
    - `dataset___augmentation___eval=model.jsonl` when there is no generator id
    """
    assert path.suffix == ".jsonl", f"Expected a .jsonl prediction file, got: {path}"
    parts = path.stem.split("___")
    assert parts and parts[0], f"Could not parse dataset id from filename: {path.name}"
    dataset_id = parts[0]
    eval_tokens = [part for part in parts if part.startswith("eval=")]
    assert len(eval_tokens) == 1, f"Expected exactly one eval= token in filename: {path.name}"
    eval_token = eval_tokens[0]
    model_id = eval_token[len("eval=") :]
    assert model_id, f"Missing evaluated model in filename: {path.name}"

    non_eval_parts = [part for part in parts[1:] if not part.startswith("eval=")]
    assert len(non_eval_parts) <= 1, (
        "Expected at most one augmentation token before eval= in filename: "
        f"{path.name}"
    )

    augmentation_token = None
    for part in non_eval_parts:
        augmentation_token = part
        break

    if augmentation_token is None:
        if dataset_id == "gsm-symbolic-reference":
            dataset_id = dataset_id[: -len("-reference")]
            augmentation_type = "base"
            augmentation_source = None
        elif dataset_id.startswith("gsm-symbolic-permutations"):
            base_dataset_id, permutation_suffix = dataset_id.split("-permutations", 1)
            assert base_dataset_id, f"Missing base dataset id in filename: {path.name}"
            dataset_id = base_dataset_id
            augmentation_type = f"permutations{permutation_suffix}"
            augmentation_source = None
        else:
            augmentation_type = "base"
            augmentation_source = None
    elif "=" in augmentation_token:
        augmentation_type, augmentation_source = augmentation_token.split("=", 1)
        assert augmentation_type, f"Missing augmentation type in filename: {path.name}"
        assert augmentation_source, f"Missing augmentation source in filename: {path.name}"
    else:
        augmentation_type = augmentation_token
        augmentation_source = None
        assert augmentation_type, f"Missing augmentation type in filename: {path.name}"

    return {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "augmentation_token": augmentation_token,
        "augmentation_type": augmentation_type,
        "augmentation_source": augmentation_source,
    }


def variant_key(row: dict[str, Any]) -> str:
    if "question_variant_idx" in row and row["question_variant_idx"] is not None:
        config = row.get("question_config")
        if config:
            return f"{config}:{row['question_variant_idx']}"
        return str(row["question_variant_idx"])
    return row.get("question", "")


def _assert_question_consistency(existing: str, new: str, context: str) -> None:
    """Fail if the same logical problem is associated with conflicting text."""
    if existing and new:
        assert existing == new, (
            f"Inconsistent original question detected for {context}.\n"
            f"Existing: {existing!r}\nNew: {new!r}"
        )


def validate_alignment(
    base_stats: dict[tuple[str, str, str], BaseProblemStats],
    augmented_stats: dict[tuple[str, str, str, str, str], AugmentedProblemStats],
) -> None:
    """Validate that aggregated base and augmented records line up correctly."""
    assert base_stats, "No base prediction files were found."
    assert augmented_stats, "No augmented prediction files were found."

    for (dataset_id, model_id, problem_id, augmentation_type, augmentation_source), aug in augmented_stats.items():
        assert augmentation_type != "base", (
            "Augmented stats unexpectedly stored with augmentation_type='base' for "
            f"{dataset_id}/{model_id}/{problem_id}"
        )
        assert aug.stats.total > 0, (
            "Encountered an augmented problem with zero rows for "
            f"{dataset_id}/{model_id}/{problem_id}/{augmentation_type}/{augmentation_source}"
        )
        assert aug.variants, (
            "Encountered an augmented problem without any tracked variants for "
            f"{dataset_id}/{model_id}/{problem_id}/{augmentation_type}/{augmentation_source}"
        )
        assert sum(variant.stats.total for variant in aug.variants.values()) == aug.stats.total, (
            "Variant-level row counts do not sum to the augmented total for "
            f"{dataset_id}/{model_id}/{problem_id}/{augmentation_type}/{augmentation_source}"
        )

        base = base_stats.get((dataset_id, model_id, problem_id))
        assert base is not None, (
            "Missing matching base predictions for augmented problem "
            f"{dataset_id}/{model_id}/{problem_id}/{augmentation_type}/{augmentation_source}"
        )
        assert base.stats.total > 0, (
            "Matching base predictions exist but contain zero rows for "
            f"{dataset_id}/{model_id}/{problem_id}"
        )

        if aug.question_orig_available and base.original_question and aug.original_question:
            assert base.original_question == aug.original_question, (
                "Base/original question mismatch for "
                f"{dataset_id}/{model_id}/{problem_id}/{augmentation_type}/{augmentation_source}"
            )


def load_prediction_data(predictions_dir: pathlib.Path) -> tuple[
    dict[tuple[str, str, str], BaseProblemStats],
    dict[tuple[str, str, str, str, str], AugmentedProblemStats],
]:
    """Load predictions and aggregate them by base problem and permutation family.

    Returns two maps:
    - base_stats[(dataset_id, model_id, problem_id)]
    - augmented_stats[(dataset_id, model_id, problem_id, augmentation_type, augmentation_source)]
    """
    assert predictions_dir.exists(), f"Predictions directory does not exist: {predictions_dir}"
    assert predictions_dir.is_dir(), f"Predictions path is not a directory: {predictions_dir}"

    # Base problems are grouped only by dataset/model/problem id.
    base_stats: dict[tuple[str, str, str], BaseProblemStats] = {}
    # Augmented problems split further by permutation type and source model.
    augmented_stats: dict[tuple[str, str, str, str, str], AugmentedProblemStats] = {}
    prediction_files = sorted(predictions_dir.rglob("*___eval=*.jsonl"))
    assert prediction_files, f"No prediction files matching '**/*___eval=*.jsonl' found in {predictions_dir}"

    for path in prediction_files:
        meta = parse_prediction_filename(path)
        dataset_id = str(meta["dataset_id"])
        model_id = str(meta["model_id"])
        augmentation_type = str(meta["augmentation_type"])
        augmentation_source = str(meta["augmentation_source"] or "")
        is_base = augmentation_type == "base"

        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                assert line.strip(), f"Encountered a blank line in {path}:{line_number}"
                row = json.loads(line)
                assert isinstance(row, dict), f"Expected a JSON object in {path}:{line_number}, got {type(row).__name__}"
                assert "id" in row, f"Missing 'id' in {path}:{line_number}"
                assert "answer" in row, f"Missing 'answer' in {path}:{line_number}"
                problem_id = str(row["id"])
                assert problem_id, f"Empty problem id in {path}:{line_number}"
                explicit_original_question = row.get("question_orig") or ""
                original_question = explicit_original_question or row.get("question") or ""
                assert original_question, (
                    "Could not recover the original question text from "
                    f"{path}:{line_number}"
                )
                question_text = row.get("question") or original_question
                predicted_result = row.get("predicted_result")
                if predicted_result in (None, ""):
                    # Fall back to extracting the final numeric answer from the
                    # raw completion when older files do not store predicted_result.
                    predicted_result = extract_answer(row.get("prediction", ""))
                is_correct = answers_match(row.get("answer"), predicted_result)

                if is_base:
                    key = (dataset_id, model_id, problem_id)
                    stats = base_stats.setdefault(key, BaseProblemStats())
                    stats.stats.add(is_correct)
                    _assert_question_consistency(
                        stats.original_question,
                        original_question,
                        f"base problem {dataset_id}/{model_id}/{problem_id}",
                    )
                    if not stats.original_question:
                        stats.original_question = original_question
                    continue

                key = (dataset_id, model_id, problem_id, augmentation_type, augmentation_source)
                stats = augmented_stats.setdefault(key, AugmentedProblemStats())
                stats.stats.add(is_correct)
                if explicit_original_question:
                    _assert_question_consistency(
                        stats.original_question,
                        explicit_original_question,
                        "augmented problem "
                        f"{dataset_id}/{model_id}/{problem_id}/{augmentation_type}/{augmentation_source}",
                    )
                    stats.question_orig_available = True
                if not stats.original_question:
                    stats.original_question = original_question

                # Variants are tracked within a permutation family so that the
                # report can list exactly which rewritten questions hurt accuracy.
                v_key = variant_key(row)
                assert v_key, f"Empty variant key in {path}:{line_number}"
                variant = stats.variants.setdefault(
                    v_key,
                    VariantStats(question=question_text),
                )
                if variant.question and question_text:
                    assert variant.question == question_text, (
                        "Variant key collision with different question text in "
                        f"{path}:{line_number} for key={v_key!r}"
                    )
                variant.stats.add(is_correct)
                if not variant.question:
                    variant.question = question_text

    missing_base_keys = [
        key
        for key in augmented_stats
        if (key[0], key[1], key[2]) not in base_stats
    ]
    if missing_base_keys:
        sample = ", ".join(
            f"{dataset_id}/{model_id}/{problem_id}/{augmentation_type}/{augmentation_source}"
            for dataset_id, model_id, problem_id, augmentation_type, augmentation_source in missing_base_keys[:5]
        )
        print(
            "Warning: skipping augmented problems without matching base predictions "
            f"({len(missing_base_keys)} cases). Sample: {sample}"
        )
        augmented_stats = {
            key: value for key, value in augmented_stats.items() if key not in missing_base_keys
        }

    validate_alignment(base_stats, augmented_stats)

    return base_stats, augmented_stats


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def remap_model_identifier(model_id: str) -> str:
    """Replace selected original model ids while preserving any effort suffix."""
    if not model_id or model_id == "n/a":
        return model_id
    canonical_model_id = MODEL_ID_ALIASES.get(model_id, model_id)
    if canonical_model_id in MODEL_ID_MAP:
        return MODEL_ID_MAP[canonical_model_id]
    if ":" in model_id:
        base_model_id, suffix = model_id.split(":", 1)
        canonical_base_model_id = MODEL_ID_ALIASES.get(base_model_id, base_model_id)
        if canonical_base_model_id in MODEL_ID_MAP:
            return f"{MODEL_ID_MAP[canonical_base_model_id]}:{suffix}"
    return model_id


def remap_model_ids_in_field(value: str) -> str:
    """Remap either a scalar model id or a JSON-encoded list of model ids."""
    if not value:
        return value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return remap_model_identifier(value)

    if isinstance(decoded, list):
        return json.dumps([remap_model_identifier(str(item)) for item in decoded], ensure_ascii=False)
    if isinstance(decoded, str):
        return remap_model_identifier(decoded)
    return value


def remap_output_row_model_ids(row: dict[str, Any]) -> dict[str, Any]:
    """Apply the configured model-id remapping to all output-facing fields."""
    remapped_row = dict(row)
    remapped_row["model_id"] = remap_model_identifier(str(remapped_row["model_id"]))
    remapped_row["permutation_source"] = remap_model_ids_in_field(str(remapped_row["permutation_source"]))

    detrimental_variants = json.loads(remapped_row["permutations_causing_decay"])
    for variant in detrimental_variants:
        if "permutation_source" in variant:
            variant["permutation_source"] = remap_model_identifier(str(variant["permutation_source"]))
    remapped_row["permutations_causing_decay"] = json.dumps(detrimental_variants, ensure_ascii=False)

    if remapped_row["model_is_robust"]:
        permutation_types = json.loads(str(remapped_row["permutation_type"]))
        permutation_sources = json.loads(str(remapped_row["permutation_source"]))
        assert len(permutation_sources) == len(permutation_types), (
            "Robust rows must store permutation_source aligned with permutation_type, "
            f"but got {len(permutation_sources)} sources for {len(permutation_types)} "
            f"types in row {remapped_row['problem_id']}"
        )
    return remapped_row


def resolve_effective_permutation_source(augmentation_type: str, augmentation_source: str | None) -> str:
    """Return a source label aligned to the perturbation type.

    Perturbations not generated by an LLM do not encode a source in the filename;
    for those, use the perturbation type itself as the effective source label.
    """
    return augmentation_source or augmentation_type


def _write_hf_validation_dataset(dataset_dir: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    """Save the final table as a HuggingFace DatasetDict with a validation split."""
    from datasets import Dataset, DatasetDict

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    validation_dataset = Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)
    dataset_dict = DatasetDict({"validation": validation_dataset})
    dataset_dict.save_to_disk(str(dataset_dir))

    # Example manual upload:
    dataset_dict.push_to_hub("michal-stefanik/aimo-interp-challenge-sample-v2")


def run_data_consistency_self_test() -> None:
    """Exercise the loader/validator on a tiny synthetic prediction directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        base_rows = [
            {
                "id": "p1",
                "question": "Original question 1",
                "answer": "10",
                "predicted_result": "10",
            },
            {
                "id": "p1",
                "question": "Original question 1",
                "answer": "10",
                "predicted_result": "10",
            },
            {
                "id": "p2",
                "question": "Original question 2",
                "answer": "7",
                "predicted_result": "0",
            },
        ]
        aug_rows = [
            {
                "id": "p1",
                "question": "Permuted question 1a",
                "question_orig": "Original question 1",
                "question_variant_idx": 0,
                "answer": "10",
                "predicted_result": "0",
            },
            {
                "id": "p1",
                "question": "Permuted question 1b",
                "question_orig": "Original question 1",
                "question_variant_idx": 1,
                "answer": "10",
                "predicted_result": "10",
            },
            {
                "id": "p2",
                "question": "Permuted question 2a",
                "question_orig": "Original question 2",
                "question_variant_idx": 0,
                "answer": "7",
                "predicted_result": "7",
            },
            {
                "id": "p3",
                "question": "Permuted question 3a",
                "question_orig": "Original question 3",
                "question_variant_idx": 0,
                "answer": "11",
                "predicted_result": "11",
            },
        ]
        rename_rows = [
            {
                "id": "p1",
                "question": "Renamed question 1a",
                "question_orig": "Original question 1",
                "question_variant_idx": 0,
                "answer": "10",
                "predicted_result": "10",
            },
            {
                "id": "p3",
                "question": "Renamed question 3a",
                "question_orig": "Original question 3",
                "question_variant_idx": 0,
                "answer": "11",
                "predicted_result": "11",
            },
        ]
        base_rows.append(
            {
                "id": "p3",
                "question": "Original question 3",
                "answer": "11",
                "predicted_result": "11",
            }
        )
        _write_jsonl(tmp_path / "demo___eval=test-model:low.jsonl", base_rows)
        _write_jsonl(tmp_path / "demo___domain=generator:low___eval=test-model:low.jsonl", aug_rows)
        _write_jsonl(tmp_path / "demo___rename=generator:low___eval=test-model:low.jsonl", rename_rows)

        base_stats, augmented_stats = load_prediction_data(tmp_path)
        assert len(base_stats) == 3
        assert len(augmented_stats) == 5

        base_p1 = base_stats[("demo", "test-model:low", "p1")]
        aug_p1 = augmented_stats[("demo", "test-model:low", "p1", "domain", "generator:low")]
        aug_p1_rename = augmented_stats[("demo", "test-model:low", "p1", "rename", "generator:low")]
        assert base_p1.stats.total == 2
        assert base_p1.stats.accuracy == 1.0
        assert aug_p1.stats.total == 2
        assert aug_p1.stats.accuracy == 0.5
        assert len(aug_p1.variants) == 2
        assert sum(variant.stats.total for variant in aug_p1.variants.values()) == aug_p1.stats.total
        assert aug_p1_rename.stats.accuracy == 1.0

        base_p3 = base_stats[("demo", "test-model:low", "p3")]
        aug_p3 = augmented_stats[("demo", "test-model:low", "p3", "domain", "generator:low")]
        aug_p3_rename = augmented_stats[("demo", "test-model:low", "p3", "rename", "generator:low")]
        assert base_p3.stats.accuracy == 1.0
        assert aug_p3.stats.accuracy == 1.0
        assert aug_p3_rename.stats.accuracy == 1.0

        p1_row, p1_is_robust = build_case_report_row(
            model_id="test-model:low",
            dataset_id="demo",
            problem_id="p1",
            augmentation_type="domain",
            augmentation_source="generator:low",
            base=base_p1,
            aug=aug_p1,
            min_drop_nonrobust=0.5,
            max_drop_robust=0.0,
        )
        assert p1_row is not None
        assert p1_is_robust is False
        assert p1_row["model_is_robust"] is False
        assert p1_row["relative_accuracy_decay"] == 0.5
        assert p1_row["n_detrimental_permutations"] == 1
        assert json.loads(p1_row["permutations_causing_decay"]) == [
            {
                "question": "Permuted question 1a",
                "variant_accuracy": 0.0,
                "n_predictions": 1,
                "permutation_type": "domain",
                "permutation_source": "generator:low",
            }
        ]

        rename_p1_row, rename_p1_is_robust = build_case_report_row(
            model_id="test-model:low",
            dataset_id="demo",
            problem_id="p1",
            augmentation_type="rename",
            augmentation_source="generator:low",
            base=base_p1,
            aug=aug_p1_rename,
            min_drop_nonrobust=0.5,
            max_drop_robust=0.0,
        )
        assert rename_p1_row is None
        assert rename_p1_is_robust is True

        p3_row = build_robust_group_row(
            model_id="test-model:low",
            dataset_id="demo",
            problem_id="p3",
            base=base_p3,
            cases=[
                ("domain", "generator:low", aug_p3),
                ("rename", "generator:low", aug_p3_rename),
            ],
            max_drop_robust=0.0,
        )
        assert p3_row is not None
        assert p3_row["model_is_robust"] is True
        assert p3_row["original_problem"] == "Original question 3"
        assert json.loads(p3_row["permutation_type"]) == ["domain", "rename"]
        assert json.loads(p3_row["permutation_source"]) == ["generator:low", "generator:low"]
        assert p3_row["absolute_accuracy_decay"] == 0.0
        assert p3_row["relative_accuracy_decay"] == 0.0
        assert p3_row["n_base_predictions"] == 1
        assert p3_row["n_permuted_predictions"] == 2
        assert p3_row["n_detrimental_permutations"] == 0
        assert p3_row["permutations_causing_decay"] == "[]"

        p3_source_less_row = build_robust_group_row(
            model_id="test-model:low",
            dataset_id="demo",
            problem_id="p3",
            base=base_p3,
            cases=[
                ("domain", "generator:low", aug_p3),
                ("expert_no_solution", "", aug_p3_rename),
            ],
            max_drop_robust=0.0,
        )
        assert p3_source_less_row is not None
        assert json.loads(p3_source_less_row["permutation_type"]) == ["domain", "expert_no_solution"]
        assert json.loads(p3_source_less_row["permutation_source"]) == ["generator:low", "expert_no_solution"]

        p1_robust_row = build_robust_group_row(
            model_id="test-model:low",
            dataset_id="demo",
            problem_id="p1",
            base=base_p1,
            cases=[
                ("domain", "generator:low", aug_p1),
                ("rename", "generator:low", aug_p1_rename),
            ],
            max_drop_robust=0.0,
        )
        assert p1_robust_row is None

        base_p4 = BaseProblemStats(
            stats=AccuracyStats(total=10, correct=5),
            original_question="Original question 4",
        )
        aug_p4 = AugmentedProblemStats(
            stats=AccuracyStats(total=10, correct=2),
            original_question="Original question 4",
            variants={
                "0": VariantStats(
                    question="Permuted question 4a",
                    stats=AccuracyStats(total=10, correct=2),
                )
            },
        )
        p4_row, p4_is_robust = build_case_report_row(
            model_id="test-model:low",
            dataset_id="demo",
            problem_id="p4",
            augmentation_type="domain",
            augmentation_source="generator:low",
            base=base_p4,
            aug=aug_p4,
            min_drop_nonrobust=0.5,
            max_drop_robust=0.0,
        )
        # Absolute drop is only 0.3 here, so the row must not be labeled
        # non-robust even though the relative drop would be 0.6.
        assert p4_row is None
        assert p4_is_robust is False

        assert remap_model_identifier("THUDM/glm-4.7") == "lukealonso/GLM-5.1-NVFP4"
        assert remap_model_identifier("THUDM/glm-4.7:low") == "lukealonso/GLM-5.1-NVFP4:low"
        assert remap_model_identifier("glm-4.7:low") == "lukealonso/GLM-5.1-NVFP4:low"
        assert remap_model_identifier("qwen3.5:low") == "Qwen/Qwen3.5-397B-A17B-FP8:low"
        assert remap_model_ids_in_field('["THUDM/glm-4.7", "n/a"]') == '["lukealonso/GLM-5.1-NVFP4", "n/a"]'

        remapped_decay_row = remap_output_row_model_ids(
            {
                **p1_row,
                "model_id": "THUDM/glm-4.7:low",
                "permutation_source": "moonshotai/Kimi-K2.5",
                "permutations_causing_decay": json.dumps(
                    [
                        {
                            "question": "Permuted question 1a",
                            "variant_accuracy": 0.0,
                            "n_predictions": 1,
                            "permutation_type": "domain",
                            "permutation_source": "Qwen/Qwen3.5-27B",
                        }
                    ],
                    ensure_ascii=False,
                ),
            }
        )
        assert remapped_decay_row["model_id"] == "lukealonso/GLM-5.1-NVFP4:low"
        assert remapped_decay_row["permutation_source"] == "moonshotai/Kimi-K2.6"
        assert json.loads(remapped_decay_row["permutations_causing_decay"])[0]["permutation_source"] == "Qwen/Qwen3.5-397B-A17B-FP8"


def _collect_detrimental_variants(
    *,
    augmentation_type: str,
    augmentation_source: str,
    base_accuracy: float,
    aug: AugmentedProblemStats,
) -> list[dict[str, Any]]:
    """Collect variant-level failures for one perturbation case."""
    effective_source = resolve_effective_permutation_source(augmentation_type, augmentation_source)
    detrimental_variants = []
    for variant in sorted(aug.variants.values(), key=lambda item: item.question):
        if variant.stats.accuracy < base_accuracy:
            detrimental_variants.append(
                {
                    "question": variant.question,
                    "variant_accuracy": round(variant.stats.accuracy, 6),
                    "n_predictions": variant.stats.total,
                    "permutation_type": augmentation_type,
                    "permutation_source": effective_source,
                }
            )
    return detrimental_variants


def build_case_report_row(
    *,
    model_id: str,
    dataset_id: str,
    problem_id: str,
    augmentation_type: str,
    augmentation_source: str,
    base: BaseProblemStats,
    aug: AugmentedProblemStats,
    min_drop_nonrobust: float,
    max_drop_robust: float,
) -> tuple[dict[str, Any] | None, bool]:
    """Construct one per-perturbation row and indicate whether that case is robust.

    Returns `(None, False)` when a pair should be excluded from the merged report
    under the current thresholds.
    """
    base_accuracy = base.stats.accuracy
    permuted_accuracy = aug.stats.accuracy
    absolute_decay = base_accuracy - permuted_accuracy
    relative_decay = absolute_decay / base_accuracy if base_accuracy else 0.0
    original_problem = base.original_question or aug.original_question

    detrimental_variants = _collect_detrimental_variants(
        augmentation_type=augmentation_type,
        augmentation_source=augmentation_source,
        base_accuracy=base_accuracy,
        aug=aug,
    )

    row = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "problem_id": problem_id,
        "original_problem": original_problem,
        "permutation_type": augmentation_type,
        "permutation_source": resolve_effective_permutation_source(augmentation_type, augmentation_source),
        "base_accuracy": round(base_accuracy, 6),
        "permuted_accuracy": round(permuted_accuracy, 6),
        "absolute_accuracy_decay": round(absolute_decay, 6),
        "relative_accuracy_decay": round(relative_decay, 6),
        "n_base_predictions": base.stats.total,
        "n_permuted_predictions": aug.stats.total,
        "n_detrimental_permutations": len(detrimental_variants),
        "permutations_causing_decay": json.dumps(detrimental_variants, ensure_ascii=False),
    }

    if base_accuracy == 1.0 and absolute_decay <= max_drop_robust:
        return None, True

    if absolute_decay < min_drop_nonrobust or absolute_decay <= 0:
        return None, False

    nonrobust_row = {**row, "model_is_robust": False}
    assert nonrobust_row["absolute_accuracy_decay"] >= min_drop_nonrobust, (
        "Non-robust row emitted below minimum drop threshold for "
        f"{dataset_id}/{model_id}/{problem_id}/{augmentation_type}/{augmentation_source}: "
        f"absolute_accuracy_decay={nonrobust_row['absolute_accuracy_decay']} < "
        f"min_drop_nonrobust={min_drop_nonrobust}"
    )
    return nonrobust_row, False


def build_robust_group_row(
    *,
    model_id: str,
    dataset_id: str,
    problem_id: str,
    base: BaseProblemStats,
    cases: list[tuple[str, str, AugmentedProblemStats]],
    max_drop_robust: float,
) -> dict[str, Any] | None:
    """Construct one aggregated robust row across all tested perturbation cases.

    Returns `None` unless the problem is base-perfect and every tested case is
    robust under `max_drop_robust`.
    """
    if base.stats.accuracy != 1.0:
        return None
    if not cases:
        return None

    base_accuracy = base.stats.accuracy
    permuted_accuracies: list[float] = []
    absolute_decays: list[float] = []
    relative_decays: list[float] = []
    permuted_prediction_counts: list[int] = []
    detrimental_variants: list[dict[str, Any]] = []
    paired_types_and_sources = sorted(
        (
            augmentation_type,
            resolve_effective_permutation_source(augmentation_type, augmentation_source),
        )
        for augmentation_type, augmentation_source, _ in cases
    )
    permutation_types = [augmentation_type for augmentation_type, _ in paired_types_and_sources]
    permutation_sources = [augmentation_source for _, augmentation_source in paired_types_and_sources]
    original_problem = base.original_question or cases[0][2].original_question

    for augmentation_type, augmentation_source, aug in cases:
        permuted_accuracy = aug.stats.accuracy
        absolute_decay = base_accuracy - permuted_accuracy
        if absolute_decay > max_drop_robust:
            return None

        permuted_accuracies.append(permuted_accuracy)
        absolute_decays.append(absolute_decay)
        relative_decays.append(absolute_decay / base_accuracy if base_accuracy else 0.0)
        permuted_prediction_counts.append(aug.stats.total)
        detrimental_variants.extend(
            _collect_detrimental_variants(
                augmentation_type=augmentation_type,
                augmentation_source=augmentation_source or "n/a",
                base_accuracy=base_accuracy,
                aug=aug,
            )
        )

    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "problem_id": problem_id,
        "original_problem": original_problem,
        "permutation_type": json.dumps(permutation_types, ensure_ascii=False),
        "permutation_source": json.dumps(permutation_sources, ensure_ascii=False),
        "base_accuracy": round(base_accuracy, 6),
        "permuted_accuracy": round(sum(permuted_accuracies) / len(permuted_accuracies), 6),
        "absolute_accuracy_decay": round(sum(absolute_decays) / len(absolute_decays), 6),
        "relative_accuracy_decay": round(sum(relative_decays) / len(relative_decays), 6),
        "n_base_predictions": base.stats.total,
        "n_permuted_predictions": sum(permuted_prediction_counts),
        "n_detrimental_permutations": len(detrimental_variants),
        "permutations_causing_decay": json.dumps(detrimental_variants, ensure_ascii=False),
        "model_is_robust": True,
    }


def build_group_label_row(
    *,
    model_id: str,
    dataset_id: str,
    problem_id: str,
    base: BaseProblemStats,
    cases: list[tuple[str, str, AugmentedProblemStats]],
    max_drop_robust: float,
) -> dict[str, Any] | None:
    """Construct one aggregated row per problem with a binary robustness label."""
    if not cases:
        return None

    base_accuracy = base.stats.accuracy
    permuted_accuracies: list[float] = []
    absolute_decays: list[float] = []
    relative_decays: list[float] = []
    permuted_prediction_counts: list[int] = []
    detrimental_variants: list[dict[str, Any]] = []
    paired_types_and_sources = sorted(
        (
            augmentation_type,
            resolve_effective_permutation_source(augmentation_type, augmentation_source),
        )
        for augmentation_type, augmentation_source, _ in cases
    )
    permutation_types = [augmentation_type for augmentation_type, _ in paired_types_and_sources]
    permutation_sources = [augmentation_source for _, augmentation_source in paired_types_and_sources]
    original_problem = base.original_question or cases[0][2].original_question

    is_robust = base_accuracy == 1.0
    for augmentation_type, augmentation_source, aug in cases:
        permuted_accuracy = aug.stats.accuracy
        absolute_decay = base_accuracy - permuted_accuracy
        if absolute_decay > max_drop_robust:
            is_robust = False

        permuted_accuracies.append(permuted_accuracy)
        absolute_decays.append(absolute_decay)
        relative_decays.append(absolute_decay / base_accuracy if base_accuracy else 0.0)
        permuted_prediction_counts.append(aug.stats.total)
        detrimental_variants.extend(
            _collect_detrimental_variants(
                augmentation_type=augmentation_type,
                augmentation_source=augmentation_source or "n/a",
                base_accuracy=base_accuracy,
                aug=aug,
            )
        )

    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "problem_id": problem_id,
        "original_problem": original_problem,
        "permutation_type": json.dumps(permutation_types, ensure_ascii=False),
        "permutation_source": json.dumps(permutation_sources, ensure_ascii=False),
        "base_accuracy": round(base_accuracy, 6),
        "permuted_accuracy": round(sum(permuted_accuracies) / len(permuted_accuracies), 6),
        "absolute_accuracy_decay": round(sum(absolute_decays) / len(absolute_decays), 6),
        "relative_accuracy_decay": round(sum(relative_decays) / len(relative_decays), 6),
        "n_base_predictions": base.stats.total,
        "n_permuted_predictions": sum(permuted_prediction_counts),
        "n_detrimental_permutations": len(detrimental_variants),
        "permutations_causing_decay": json.dumps(detrimental_variants, ensure_ascii=False),
        "model_is_robust": is_robust,
    }


def _write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a CSV report with a fixed schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@app.command()
def main(
    predictions_dir: pathlib.Path = typer.Argument(..., help="Directory containing prediction JSONL files."),
    output_csv: pathlib.Path = typer.Option(
        pathlib.Path("robustness-analyses/robustness-analyses/reports/permutation_decay_report.csv"),
        "--output-csv",
        help="Where to write the merged CSV report of harmful and robust cases.",
    ),
    output_dataset_dir: pathlib.Path | None = typer.Option(
        None,
        "--output-dataset-dir",
        help="Optional directory where a HuggingFace DatasetDict with a validation split will be saved. Defaults to a sibling folder derived from --output-csv.",
    ),
    min_drop_nonrobust: float = typer.Option(
        0.5,
        min=0.0,
        max=1.0,
        help="Minimum absolute accuracy drop required to label a case non-robust.",
    ),
    max_drop_robust: float = typer.Option(
        0.0,
        min=0.0,
        max=1.0,
        help="Maximum allowed drop from 1.0 for a case to count as robust; only base-perfect problems are eligible.",
    ),
    push_to_hub: bool = typer.Option(
        False,
        "--push-to-hub",
        help="Save the final table as a local HuggingFace DatasetDict validation split. The code contains a commented push_to_hub example for manual upload.",
    ),
    label_all_problems: bool = typer.Option(
        False,
        "--label-all-problems",
        help="Emit exactly one aggregated row per problem with a binary model_is_robust label.",
    ),
) -> None:
    # Run a lightweight end-to-end consistency check on a synthetic fixture
    # before touching the real prediction directory.
    run_data_consistency_self_test()

    # A merged report needs the positive/negative criteria to be disjoint so a
    # single row can carry a single, unambiguous `model_is_robust` label.
    assert max_drop_robust < min_drop_nonrobust, (
        "Overlapping thresholds would make the merged label ambiguous. "
        f"Require max_drop_robust < min_drop_nonrobust, but got "
        f"max_drop_robust={max_drop_robust} and min_drop_nonrobust={min_drop_nonrobust}."
    )

    base_stats, augmented_stats = load_prediction_data(predictions_dir)

    grouped_cases: dict[tuple[str, str, str], list[tuple[str, str, AugmentedProblemStats]]] = {}
    for (dataset_id, model_id, problem_id, augmentation_type, augmentation_source), aug in augmented_stats.items():
        grouped_cases.setdefault((dataset_id, model_id, problem_id), []).append(
            (augmentation_type, augmentation_source, aug)
        )

    merged_rows: list[dict[str, Any]] = []
    n_base_perfect_problems = 0
    n_robust_rows = 0
    n_decay_rows = 0

    if label_all_problems:
        for dataset_id, model_id, problem_id in sorted(grouped_cases):
            base = base_stats.get((dataset_id, model_id, problem_id))
            assert base is not None, (
                "Missing base stats for grouped label analysis of "
                f"{dataset_id}/{model_id}/{problem_id}"
            )
            row = build_group_label_row(
                model_id=model_id,
                dataset_id=dataset_id,
                problem_id=problem_id,
                base=base,
                cases=grouped_cases[(dataset_id, model_id, problem_id)],
                max_drop_robust=max_drop_robust,
            )
            if row is None:
                continue
            merged_rows.append(row)
            n_robust_rows += int(bool(row["model_is_robust"]))
            n_decay_rows += int(not bool(row["model_is_robust"]))
        remapped_merged_rows = [remap_output_row_model_ids(row) for row in merged_rows]
        _write_csv(output_csv, [
            "model_is_robust",
            "model_id",
            "dataset_id",
            "problem_id",
            "original_problem",
            "permutation_type",
            "permutation_source",
            "base_accuracy",
            "permuted_accuracy",
            "absolute_accuracy_decay",
            "relative_accuracy_decay",
            "n_base_predictions",
            "n_permuted_predictions",
            "n_detrimental_permutations",
            "permutations_causing_decay",
        ], remapped_merged_rows)
        dataset_dir = output_dataset_dir
        if dataset_dir is None:
            dataset_dir = output_csv.parent / f"{output_csv.stem}_hf_dataset"
        if push_to_hub:
            _write_hf_validation_dataset(dataset_dir, remapped_merged_rows)
            typer.echo(f"Saved HuggingFace validation dataset to {dataset_dir}")
        typer.echo(f"Wrote {len(remapped_merged_rows)} labeled rows to {output_csv}")
        typer.echo(
            f"Assigned {n_robust_rows} robust and {n_decay_rows} non-robust labels "
            f"(max_drop_robust={max_drop_robust:.3f})."
        )
        return

    for key, aug in sorted(augmented_stats.items()):
        dataset_id, model_id, problem_id, augmentation_type, augmentation_source = key
        base = base_stats.get((dataset_id, model_id, problem_id))
        if base is None or base.stats.total == 0:
            # This should not happen after validation, but keeping the guard
            # makes the decay computation resilient to future refactors.
            continue
        if base.stats.accuracy <= 0:
            # Relative decay is undefined/non-informative when the base problem
            # is never solved correctly by the evaluated model.
            continue

        row, is_robust = build_case_report_row(
            model_id=model_id,
            dataset_id=dataset_id,
            problem_id=problem_id,
            augmentation_type=augmentation_type,
            augmentation_source=augmentation_source,
            base=base,
            aug=aug,
            min_drop_nonrobust=min_drop_nonrobust,
            max_drop_robust=max_drop_robust,
        )

        if row is None:
            continue

        if not is_robust:
            # Leave non-robust reporting at the original per-perturbation granularity.
            merged_rows.append(row)
            n_decay_rows += 1

    for dataset_id, model_id, problem_id in sorted(grouped_cases):
        base = base_stats.get((dataset_id, model_id, problem_id))
        assert base is not None, (
            "Missing base stats for grouped robust analysis of "
            f"{dataset_id}/{model_id}/{problem_id}"
        )
        if base.stats.accuracy != 1.0:
            continue

        n_base_perfect_problems += 1
        robust_row = build_robust_group_row(
            model_id=model_id,
            dataset_id=dataset_id,
            problem_id=problem_id,
            base=base,
            cases=grouped_cases[(dataset_id, model_id, problem_id)],
            max_drop_robust=max_drop_robust,
        )
        if robust_row is None:
            continue
        merged_rows.append(robust_row)
        n_robust_rows += 1

    merged_rows.sort(
        key=lambda row: (
            -int(bool(row["model_is_robust"])),
            -float(row["relative_accuracy_decay"]),
            -float(row["permuted_accuracy"]),
            str(row["model_id"]),
            str(row["problem_id"]),
            str(row["permutation_type"]),
        )
    )

    merged_fieldnames = [
        "model_is_robust",
        "model_id",
        "dataset_id",
        "problem_id",
        "original_problem",
        "permutation_type",
        "permutation_source",
        "base_accuracy",
        "permuted_accuracy",
        "absolute_accuracy_decay",
        "relative_accuracy_decay",
        "n_base_predictions",
        "n_permuted_predictions",
        "n_detrimental_permutations",
        "permutations_causing_decay",
    ]
    remapped_merged_rows = [remap_output_row_model_ids(row) for row in merged_rows]
    _write_csv(output_csv, merged_fieldnames, remapped_merged_rows)

    dataset_dir = output_dataset_dir
    if dataset_dir is None:
        dataset_dir = output_csv.parent / f"{output_csv.stem}_hf_dataset"
    if push_to_hub:
        _write_hf_validation_dataset(dataset_dir, remapped_merged_rows)
        typer.echo(f"Saved HuggingFace validation dataset to {dataset_dir}")

    typer.echo(f"Wrote {len(remapped_merged_rows)} merged rows to {output_csv}")
    typer.echo(
        f"Found {n_robust_rows} robust rows and {n_decay_rows} non-robust rows "
        f"(among {n_base_perfect_problems} base-perfect model/problem groups, "
        f"max_drop_robust={max_drop_robust:.3f}, min_drop_nonrobust={min_drop_nonrobust:.3f})."
    )

    if n_decay_rows:
        typer.echo("Top 5 largest decays:")
        for row in [row for row in remapped_merged_rows if not row["model_is_robust"]][:5]:
            typer.echo(
                " - "
                f"model={row['model_id']} problem={row['problem_id']} "
                f"type={row['permutation_type']} source={row['permutation_source'] or 'n/a'} "
                f"base_acc={row['base_accuracy']:.3f} perm_acc={row['permuted_accuracy']:.3f} "
                f"rel_decay={row['relative_accuracy_decay']:.3f}"
            )

    if n_robust_rows:
        typer.echo("Top 5 robust cases:")
        for row in [row for row in remapped_merged_rows if row["model_is_robust"]][:5]:
            typer.echo(
                " + "
                f"model={row['model_id']} problem={row['problem_id']} "
                f"type={row['permutation_type']} source={row['permutation_source'] or 'n/a'} "
                f"base_acc={row['base_accuracy']:.3f} perm_acc={row['permuted_accuracy']:.3f} "
                f"rel_decay={row['relative_accuracy_decay']:.3f}"
            )


if __name__ == "__main__":
    app()
