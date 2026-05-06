# robustness-eval

## Set up:

```shell
git clone ...
cd ...
uv venv
source .venv/bin/activate
uv sync
```


## Generate variants

```shell
python robustness-analyses/main.py augment \
    data/aimo-2025-reference.jsonl \
    prompts/paraphrase.txt \
    data/ \
    --api-model "gpt-5.2-2025-12-11" \
    --provider openai \
    --n-variants 10
```


## Generate predictions

Predictions for base problems:

```shell
python robustness-analyses/main.py predict \
    data/aimo-2025-reference.jsonl \
    predictions/ \
    --api-model "gpt-5.2-2025-12-11" \
    --provider openai \
    --n-repeats 10
```

Predictions with a locally loaded Hugging Face model:

```shell
python robustness-analyses/main.py predict \
    data/gsm-symbolic-reference.jsonl \
    predictions/ \
    --provider huggingface-local \
    --api-model "Qwen/Qwen2-0.5B" \
    --n-repeats 1 \
    --max-concurrency 1
```

This provider loads the model directly via `transformers`, so you need local model dependencies installed and enough CPU/GPU memory for the selected checkpoint. Optional environment variables:

```shell
export LOCAL_HF_DEVICE_MAP=auto
export LOCAL_HF_TORCH_DTYPE=bfloat16
export LOCAL_HF_MAX_NEW_TOKENS=4096
```

Predictions for augmented problems:

```shell
python robustness-analyses/main.py predict \
    data/aimo-2025-reference___paraphrase=gpt-5.2-2025-12-11.jsonl \
    predictions/ \
    --api-model "gpt-5.2-2025-12-11" \
    --provider openai \
    --n-repeats 1
```


## Evaluate 

Evaluate standalone base prediction file:

```shell
python robustness-analyses/main.py eval \
    predictions/aimo-2025-reference___eval=gpt-5.2-2025-12-11.jsonl
```

Evaluate augmented prediction file:

```shell
python robustness-analyses/main.py eval \
    predictions/aimo-2025-reference___paraphrase=gpt-5.2-2025-12-11___eval=gpt-5.2-2025-12-11.jsonl
```

Or compare the augmented to the base:

```shell
python robustness-analyses/main.py eval \
    predictions/aimo-2025-reference___paraphrase=gpt-5.2-2025-12-11___eval=gpt-5.2-2025-12-11.jsonl \
    --base-pred-file predictions/aimo-2025-reference___eval=gpt-5.2-2025-12-11.jsonl
```


## Find problems with large accuracy decay under permutations

```shell
python robustness-analyses/find_permutation_decay.py \
    predictions/ \
    --output-csv robustness-analyses/robustness-analyses/reports/permutation_decay_report.csv \
    --output-dataset-dir robustness-analyses/robustness-analyses/reports/permutation_decay_report_hf_dataset \
    --min-drop-nonrobust 0.5 \
    --max-drop-robust 0.0 \
    --push-to-hub
```

This scans all `*___eval=*.jsonl` prediction files, compares each augmented problem family to the matching base file for the same evaluated model, and exports a single merged CSV using the current decay-report schema plus a `model_is_robust` column.

Non-robust rows are left at the original per-perturbation granularity.

Robust rows are aggregated per `(dataset, evaluated model, original problem)` across **all** tested perturbation types for that problem. For these aggregated robust rows:
- `permutation_type` contains the JSON list of all tested perturbation types
- `permutation_source` contains the JSON list of all tested perturbation sources
- accuracy-related fields are averaged across the tested perturbation cases

Before writing outputs, selected original model ids are remapped using an internal `models_map`.

If `--push-to-hub` is provided, the script also saves the same table locally as a HuggingFace `DatasetDict` with a `validation` split. The dataset is saved to `--output-dataset-dir` (or a default sibling directory derived from `--output-csv`).

The code also includes a commented example for:
- `dataset_dict.push_to_hub("your-username/your-dataset-name", token="hf_PLACEHOLDER_TOKEN")`

Rows are included when either:
- the absolute accuracy drop is at least `--min-drop-nonrobust`, or
- the base problem has 100% accuracy and **all** tested perturbation cases for that problem have drop from `1.0` at most `--max-drop-robust`

The thresholds must be disjoint enough to keep the merged label unambiguous, so the script requires `max_drop_robust < min_drop_nonrobust`.

The script always runs built-in data-consistency checks as part of normal execution and fails loudly if it finds malformed filenames, missing fields, inconsistent original questions, variant collisions, or broken base/augmentation alignment.
