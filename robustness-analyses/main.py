import math
import random
import asyncio
import json
import pathlib
import itertools
import os
import re
from urllib.parse import parse_qs, urlparse
from typing import Any, Literal

import typer
import numpy as np
import pandas as pd
import openai
import tqdm
import rich.pretty


app = typer.Typer()

MESSAGE_MARKER_RE = re.compile(r"<\|message\|>", re.IGNORECASE)
SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
ROLE_MARKER_RE = re.compile(
    r"(?is)(?:^|\n)\s*(analysis|assistant|final|user)\s*(?::|\n)",
)
QUESTION_START_RE = re.compile(
    r"(?is)\b("
    r"what|which|who|when|where|why|how|suppose|if|let|given|compute|find|determine|evaluate|prove|show|"
    r"a\s+regular|in\s+triangle|triangle|consider"
    r")\b"
)
PROBLEM_END_RE = re.compile(r"(?s)[.?!](?:[)\]\"']+)?(?:\s|$)")
OBVIOUS_NOISE_RE = re.compile(
    r"(?is)<\||\b(?:analysis|assistant|final)\b|(?:\.\s*){6,}|(?:…\s*){4,}|(?:\b(?:we|okay|sure|certainly)\b\s*[.]{2,})"
)


def sanitize_filename_component(value: str) -> str:
    """Make a model or prompt identifier safe to embed in a filename."""
    return value.replace("/", "__")


def _strip_outer_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text


def _normalize_candidate_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if MESSAGE_MARKER_RE.search(text):
        text = MESSAGE_MARKER_RE.split(text)[-1].strip()
    text = SPECIAL_TOKEN_RE.sub(" ", text)
    text = ROLE_MARKER_RE.sub("\n", text)
    text = _strip_outer_quotes(text.strip())
    return re.sub(r"[ \t]+", " ", text).strip()


def _find_problem_start(text: str) -> int | None:
    match = QUESTION_START_RE.search(text)
    if match is None:
        return None
    return match.start()


def clean_augmented_question(raw_text: str, original_question: str) -> str:
    """Trim rambling/tool-output wrappers and keep only the final problem text."""
    text = _normalize_candidate_text(str(raw_text))
    if not text:
        return text

    # Fast path: already looks like one compact problem.
    if (
        "\n" not in text
        and "assistant" not in text.lower()
        and "analysis" not in text.lower()
        and _find_problem_start(text) in (None, 0)
        and not OBVIOUS_NOISE_RE.search(text)
    ):
        return text

    lines = [line.strip(" \"'") for line in text.splitlines() if line.strip()]
    compact = "\n".join(lines).strip()

    # If the model echoed the original question at the end, prefer the tail from the
    # last plausible problem-statement start rather than the earlier reasoning.
    start_idx = _find_problem_start(compact)
    if start_idx is not None:
        tail = compact[start_idx:].strip(" \"'")
        if len(tail) >= max(20, len(original_question) // 3):
            compact = tail

    # If there are multiple paragraphs, prefer the last one that looks like a problem.
    paragraphs = [part.strip(" \"'") for part in re.split(r"\n{2,}", compact) if part.strip()]
    plausible_paragraphs = [
        part for part in paragraphs
        if QUESTION_START_RE.search(part) and ("$" in part or "?" in part or "\\[" in part)
    ]
    if plausible_paragraphs:
        compact = plausible_paragraphs[-1]

    # Cut away any leftover role-preface before the first plausible problem start.
    start_idx = _find_problem_start(compact)
    if start_idx is not None:
        compact = compact[start_idx:].strip(" \"'")

    # Keep only through the final sentence-ending punctuation if there is trailing junk.
    end_matches = list(PROBLEM_END_RE.finditer(compact))
    if end_matches:
        compact = compact[:end_matches[-1].end()].strip(" \"'")

    compact = compact.strip()
    if not compact or _find_problem_start(compact) is None or OBVIOUS_NOISE_RE.search(compact):
        return original_question.strip()
    return compact

def load_df(file_path: pathlib.Path) -> pd.DataFrame:
    if file_path.suffix == ".csv":
        return pd.read_csv(file_path)
    if file_path.suffix == ".json":
        return pd.read_json(file_path)
    if file_path.suffix == ".jsonl":
        return pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[idx:idx + size] for idx in range(0, len(items), size)]


class LocalHFClient:
    """Thin async wrapper around a locally loaded Hugging Face causal LM."""

    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Local Hugging Face inference requires `transformers` and `torch`. "
                "Install project dependencies again after updating `pyproject.toml`."
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.max_new_tokens_default = int(os.getenv("LOCAL_HF_MAX_NEW_TOKENS", "4096"))
        dtype_name = os.getenv("LOCAL_HF_TORCH_DTYPE", "auto")
        model_kwargs: dict[str, Any] = {"device_map": os.getenv("LOCAL_HF_DEVICE_MAP", "auto")}
        if dtype_name != "auto":
            if not hasattr(torch, dtype_name):
                raise ValueError(f"Unsupported LOCAL_HF_TORCH_DTYPE={dtype_name!r}")
            model_kwargs["torch_dtype"] = getattr(torch, dtype_name)
        attn_impl = os.getenv("LOCAL_HF_ATTN_IMPLEMENTATION")
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        typer.secho(f"Loading local Hugging Face model {model_name}...", fg="cyan")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()

    def _render_prompt(self, system_prompt: str, user_content: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"{system_prompt}\n\n{user_content}\n"

    def _generate_sync(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float,
        max_completion_tokens: int | None,
        seed: int | None,
    ) -> tuple[str, None, str]:
        prompt = self._render_prompt(system_prompt, user_content)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_completion_tokens or self.max_new_tokens_default,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        if seed is not None:
            self._torch.manual_seed(seed)

        with self._torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = output[0][input_length:]
        prediction = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return prediction, None, "stop"

    async def generate(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float,
        max_completion_tokens: int | None,
        seed: int | None,
    ) -> tuple[str, None, str]:
        return await asyncio.to_thread(
            self._generate_sync,
            system_prompt,
            user_content,
            temperature,
            max_completion_tokens,
            seed,
        )

    def _generate_batch_sync(
        self,
        system_prompts: list[str],
        user_contents: list[str],
        temperature: float,
        max_completion_tokens: int | None,
        seed: int | None,
    ) -> list[tuple[str, None, str]]:
        prompts = [
            self._render_prompt(system_prompt, user_content)
            for system_prompt, user_content in zip(system_prompts, user_contents, strict=True)
        ]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_completion_tokens or self.max_new_tokens_default,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        if seed is not None:
            self._torch.manual_seed(seed)

        with self._torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1).tolist()
        else:
            input_lengths = [inputs["input_ids"].shape[1]] * len(prompts)

        results: list[tuple[str, None, str]] = []
        for output, input_length in zip(outputs, input_lengths, strict=True):
            generated_tokens = output[int(input_length):]
            prediction = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            results.append((prediction, None, "stop"))
        return results

    async def generate_batch(
        self,
        system_prompts: list[str],
        user_contents: list[str],
        temperature: float,
        max_completion_tokens: int | None,
        seed: int | None,
    ) -> list[tuple[str, None, str]]:
        return await asyncio.to_thread(
            self._generate_batch_sync,
            system_prompts,
            user_contents,
            temperature,
            max_completion_tokens,
            seed,
        )


async def call_llm(
    system_prompt: str,
    user_content: str,
    problem_id: str,
    client: Any,
    model: str,
    temperature: float,
    reasoning_effort: str,
    max_retries: int,
    retry_sleep_secs: float,
    max_completion_tokens: int,
    seed: int,
) -> tuple[str, str, str]:
    """Call LLM API with retries on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            if isinstance(client, LocalHFClient):
                prediction, reasoning_content, finish_reason = await client.generate(
                    system_prompt=system_prompt,
                    user_content=user_content,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    seed=seed,
                )
            else:
                extras = {}
                if seed is not None:
                    extras["seed"] = seed
                response = await client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    reasoning_effort=reasoning_effort,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    timeout=2*60*60,
                    **extras
                )
                out = response.choices[0]
                prediction = out.message.content
                reasoning_content = out.message.reasoning_content if hasattr(out.message, "reasoning_content") else None
                finish_reason = out.finish_reason

            if prediction is None:
                prediction = ""
                typer.secho("Recived empty prediction from the model.", fg="yellow")
            else:
                typer.secho(f"Received prediction for problem {problem_id}.", fg="green")
            return prediction, reasoning_content, finish_reason
        except Exception as e:
            if attempt < max_retries:
                typer.secho(e, fg="red")
                typer.secho(f"problem {problem_id} [attempt {attempt}/{max_retries}], retrying after {retry_sleep_secs} seconds...", fg="yellow")
                await asyncio.sleep(retry_sleep_secs)
    typer.secho(f"problem {problem_id} failed after {max_retries} attempts.", fg="red")
    raise RuntimeError(f"problem {problem_id} failed after {max_retries} retries.")


async def run_bounded(coros, max_concurrency: int):
    """Yield results with bounded concurrency as they complete."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded(coro):
        async with semaphore:
            return await coro

    for fut in asyncio.as_completed([bounded(c) for c in coros]):
        yield await fut


def _get_client(provider: str, model_name: str) -> Any:
    """Instantiate a client for the given provider."""
    if provider == "google":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set the GEMINI_API_KEY environment variable.")
        return openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    
    if provider == "einfra":
        api_key = os.getenv("EINFRA_AI_TOKEN")
        if not api_key:
            raise EnvironmentError("Set the EINFRA_AI_TOKEN environment variable.")
        return openai.AsyncOpenAI(
            api_key = api_key,
            base_url = "https://llm.ai.e-infra.cz/v1/"
        )

    if provider == "openai-compatible":
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
        if not api_key:
            raise EnvironmentError("Set the OPENAI_COMPATIBLE_API_KEY environment variable.")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        if not base_url:
            raise EnvironmentError("Set the OPENAI_COMPATIBLE_BASE_URL environment variable.")
        return openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    if provider == "huggingface-local":
        return LocalHFClient(model_name)

    if provider == "openai":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set the AZURE_OPENAI_API_KEY environment variable.")
        endpoint_raw = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint_raw:
            raise EnvironmentError("Set the AZURE_OPENAI_ENDPOINT environment variable.")
        endpoint_raw = endpoint_raw.strip()
        parsed = urlparse(endpoint_raw)
        if not parsed.scheme or not parsed.netloc:
            raise EnvironmentError("AZURE_OPENAI_ENDPOINT must be a valid Azure OpenAI URL.")

        endpoint = f"{parsed.scheme}://{parsed.netloc}"

        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        if not api_version:
            query_api_versions = parse_qs(parsed.query).get("api-version", [])
            api_version = query_api_versions[0] if query_api_versions else "2025-04-01-preview"

        # Quick connectivity check
        import urllib.request
        try:
            req = urllib.request.Request(
                f"{endpoint.rstrip('/')}/openai/models?api-version={api_version}",
                headers={"api-key": api_key},
            )
            urllib.request.urlopen(req, timeout=10)
            print(f"✓ Successfully connected to {endpoint}")
        except Exception as e:
            print(f"✗ Cannot reach endpoint {endpoint}: {e}")
            print("  Check your network, VPN, firewall, or proxy settings.")

        return openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    raise ValueError(
        f"Unknown provider: {provider!r}. Choose one of "
        "'openai', 'google', 'einfra', 'openai-compatible', or 'huggingface-local'."
    )


def extract_answer(response: str) -> str:
    """Extract the final answer from the model response as the last number found."""
    import re
    matches = re.findall(r"NaN|[-+]?\d*\.\d+|\d+", response)
    if matches:
        return matches[-1]
    else:
        return ""


def normalize(text) -> str:
    """Normalize an answer for comparison."""
    if text is None or (isinstance(text, str) and text.strip() == "NaN") or math.isnan(text) or np.isnan(text):
        return "NaN"
    t = str(text).strip().lower().rstrip(".")
    for ch in ("$", ",", "%"):
        t = t.replace(ch, "")
    try:
        return str(float(t))
    except ValueError:
        return t


def answers_match(expected, predicted: str) -> bool:
    """Check whether the predicted answer matches the expected one."""
    return normalize(expected) == normalize(predicted)


def series_mean_or_nan(series: pd.Series) -> float:
    """Return a plain float mean, preserving NaN for empty slices."""
    value = series.mean()
    return float(value)


@app.command()
def augment(
    base_problems_file: pathlib.Path,
    prompt_file: pathlib.Path,
    output_dir: pathlib.Path,
    n_variants: int = typer.Option(10),
    max_concurrency: int = typer.Option(100),
    provider: str = "openai",
    api_model: str = "gpt-5.2-2025-12-11",
    temperature: float = 1.0,
    reasoning_effort: str = "low",
    max_retries: int = 100,
    retry_sleep_secs: float = 30,
    max_tokens: int | None = None,
    master_seed: int | None = None,
    seeding: bool = True,
) -> None:

    typer.secho(f"Loading system prompt from {prompt_file}...", fg="cyan")
    with open(prompt_file, "r") as f:
        system_prompt = f.read()

    typer.secho(f"Loading base problems from {base_problems_file}...", fg="cyan")
    df = load_df(base_problems_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = output_dir / (
        f"{base_problems_file.stem}___{prompt_file.stem}="
        f"{sanitize_filename_component(api_model)}:{reasoning_effort}.jsonl"
    )
    if output_filename.exists():
        raise FileExistsError(f"Output file {output_filename} already exists. Please remove it before running the script.")

    typer.secho("Initializing the client...", fg="cyan")
    client = _get_client(provider, api_model)
    if provider == "huggingface-local":
        typer.secho(
            f"Provider {provider} will use max_concurrency={max_concurrency} as the local batch size.",
            fg="cyan",
        )

    records = df.to_dict(orient="records")
    total = len(records) * n_variants
    typer.secho(f"Generating variants for {len(records)} problems (total {total} variants) using model {api_model}...", fg="cyan")

    master_rng = random.Random(master_seed) if seeding else None
    async def _make_coro(row: dict, variant_idx: int) -> dict:
        question = row["question"]
        seed = master_rng.randint(0, 2**32 - 1) if master_rng is not None else None
        paraphrased, reasoning_content, finish_reason = await call_llm(
            system_prompt=system_prompt,
            user_content=question,
            problem_id=f"{row['id']}/variant_{variant_idx}",
            client=client,
            model=api_model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_retries=max_retries,
            retry_sleep_secs=retry_sleep_secs,
            max_completion_tokens=max_tokens,
            seed=seed,
        )
        paraphrased = clean_augmented_question(paraphrased, question)
        return {
            **row,
            "question": paraphrased,
            "question_orig": question,
            "question_variant_idx": variant_idx,
            "question_system_prompt": system_prompt,
            "question_api_model": api_model,
            "question_temperature": temperature,
            "question_max_tokens": max_tokens,
            "question_reasoning_effort": reasoning_effort,
            "question_reasoning_content": reasoning_content,
            "question_finish_reason": finish_reason,
            "question_seed": seed,
        }

    coros = (_make_coro(row, i) for row, i in itertools.product(records, range(n_variants)))

    async def _run() -> None:
        with open(output_filename, "a", buffering=1) as f, tqdm.tqdm(total=total, desc="generating variants") as pbar:
            async for result in run_bounded(coros, max_concurrency):
                f.write(json.dumps(result) + "\n")
                f.flush()
                pbar.update(1)

    asyncio.run(_run())
    typer.secho(f"Done! Generated {total} augmented problems.", fg="green")


@app.command()
def clean_augmented_file(
    input_file: pathlib.Path,
    output_file: pathlib.Path | None = None,
) -> None:
    """Clean an existing augmented JSONL file in-place or into a new file."""
    df = load_df(input_file)
    required = {"question", "question_orig"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_file}: {sorted(missing)}")

    df = df.copy()
    df["question_raw"] = df["question"]
    df["question"] = df.apply(
        lambda row: clean_augmented_question(row["question"], row["question_orig"]),
        axis=1,
    )

    target = output_file or input_file
    with open(target, "w", encoding="utf-8") as handle:
        for row in df.to_dict(orient="records"):
            handle.write(json.dumps(row) + "\n")
    typer.secho(f"Wrote cleaned augmented file to {target}.", fg="green")


@app.command()
def clean_augmented_dir(
    input_dir: pathlib.Path,
    glob_pattern: str = "*.jsonl",
    in_place: bool = True,
) -> None:
    """Clean every augmented JSONL file in a directory."""
    assert input_dir.is_dir(), f"Not a directory: {input_dir}"
    cleaned_files = 0
    skipped_files = 0

    for path in sorted(input_dir.glob(glob_pattern)):
        try:
            df = load_df(path)
        except Exception:
            skipped_files += 1
            continue
        if "question" not in df.columns or "question_orig" not in df.columns:
            skipped_files += 1
            continue

        target = path if in_place else path.with_name(f"{path.stem}.cleaned{path.suffix}")
        df = df.copy()
        df["question_raw"] = df["question"]
        df["question"] = df.apply(
            lambda row: clean_augmented_question(row["question"], row["question_orig"]),
            axis=1,
        )
        with open(target, "w", encoding="utf-8") as handle:
            for row in df.to_dict(orient="records"):
                handle.write(json.dumps(row) + "\n")
        cleaned_files += 1

    typer.secho(
        f"Cleaned {cleaned_files} augmented files in {input_dir}; skipped {skipped_files}.",
        fg="green",
    )


@app.command()
def predict(
    problems_file: pathlib.Path,
    pred_dir: pathlib.Path,
    n_repeats: int = 1,
    max_concurrency: int = typer.Option(100),
    provider: str = "openai",
    api_model: str = "gpt-5.2-2025-12-11",
    temperature: float = 1.0,
    max_retries: int = 100,
    retry_sleep_secs: float = 30,
    max_tokens: int | None = None,
    reasoning_effort: str = "low",
    master_seed: int | None = None,
    system_prompt_file: pathlib.Path | None = "./prompts/solve.txt",
    on_file_exists: Literal["error", "fill-missing", "overwrite"] = "error",
    seeding: bool = True,
    row_offset: int = 0,
    max_rows: int | None = None,
    output_suffix: str | None = None,
) -> None:
    
    typer.secho(f"Loading system prompt from {system_prompt_file}...", fg="cyan")
    with open(system_prompt_file, "r") as f:
        system_prompt = f.read()

    typer.secho(f"Loading problems from {problems_file}...", fg="cyan")
    df = load_df(problems_file)

    if row_offset < 0:
        raise ValueError(f"row_offset must be non-negative, got {row_offset}")
    if max_rows is not None and max_rows <= 0:
        raise ValueError(f"max_rows must be positive when provided, got {max_rows}")

    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_filename = (
        f"{problems_file.stem}___eval="
        f"{sanitize_filename_component(api_model)}:{reasoning_effort}"
    )
    if output_suffix:
        pred_filename += f"___{sanitize_filename_component(output_suffix)}"
    pred_file = pred_dir / f"{pred_filename}.jsonl"

    existing_df = None
    if pred_file.exists():
        typer.secho(f"Output file {pred_file} already exists.", fg="yellow")
        match on_file_exists:
            case "error":
                raise FileExistsError(f"Output file {pred_file} already exists. Please remove it before running the script, or change `--on-file-exists` argument.")
            case "overwrite":
                pred_file.unlink()
            case "fill-missing":
                typer.secho(f"Filling missing predictions into existing file {pred_file}...", fg="cyan")
                existing_df = load_df(pred_file)
            case _:
                raise ValueError(f"Unknown value for `--on-file-exists`: {on_file_exists}")

    typer.secho("Initializing the client...", fg="cyan")
    client = _get_client(provider, api_model)
    if provider == "huggingface-local" and max_concurrency != 1:
        typer.secho(
            f"Provider {provider} runs against one local model instance; overriding "
            f"max_concurrency from {max_concurrency} to 1.",
            fg="yellow",
        )
        max_concurrency = 1

    if row_offset > 0 or max_rows is not None:
        end_idx = None if max_rows is None else row_offset + max_rows
        df = df.iloc[row_offset:end_idx].reset_index(drop=True)
        typer.secho(
            f"Using subset rows [{row_offset}:{'end' if end_idx is None else end_idx}] "
            f"from {problems_file}.",
            fg="cyan",
        )

    records = df.to_dict(orient="records")
    total = len(records) * n_repeats
    typer.secho(f"Evaluating {len(records)} problems ({total} requests) using model {api_model}...", fg="cyan")

    master_rng = random.Random(master_seed) if seeding else None

    async def _make_coro(row: dict, repeat_idx: int, existing_df: pd.DataFrame | None) -> dict | None:
        if existing_df is not None and len(existing_df) > 0:
            existing_rows = existing_df[(existing_df["id"] == row["id"]) & (existing_df["question"] == row["question"]) & (existing_df["prediction_repeat_idx"] == repeat_idx) & (existing_df["prediction"] != "")]
            if len(existing_rows) > 0:
                typer.secho(f"Prediction for problem {row['id']} repeat {repeat_idx} already exists, skipping...", fg="yellow")
                return None
        seed = master_rng.randint(0, 2**32 - 1) if master_rng is not None else None
        prediction, reasoning_content, finish_reason = await call_llm(
            system_prompt=system_prompt,
            user_content=row["question"],
            problem_id=row["id"],
            client=client,
            model=api_model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_retries=max_retries,
            retry_sleep_secs=retry_sleep_secs,
            max_completion_tokens=max_tokens,
            seed=seed,
        )
        return {
            **row,
            "prediction": prediction,
            "predicted_result": extract_answer(prediction),
            "prediction_api_model": api_model,
            "prediction_system_prompt": system_prompt,
            "prediction_reasoning_effort": reasoning_effort,
            "prediction_repeat_idx": repeat_idx,
            "prediction_temperature": temperature,
            "prediction_max_tokens": max_tokens,
            "prediction_provider": provider,
            "prediction_reasoning_content": reasoning_content,
            "prediction_finish_reason": finish_reason,
            "prediction_seed": seed,
        }

    async def _make_local_batch(batch: list[tuple[dict, int]]) -> list[dict | None]:
        todo_batch: list[tuple[dict, int, int | None]] = []
        for row, repeat_idx in batch:
            if existing_df is not None and len(existing_df) > 0:
                existing_rows = existing_df[
                    (existing_df["id"] == row["id"])
                    & (existing_df["question"] == row["question"])
                    & (existing_df["prediction_repeat_idx"] == repeat_idx)
                    & (existing_df["prediction"] != "")
                ]
                if len(existing_rows) > 0:
                    typer.secho(
                        f"Prediction for problem {row['id']} repeat {repeat_idx} already exists, skipping...",
                        fg="yellow",
                    )
                    continue
            seed = master_rng.randint(0, 2**32 - 1) if master_rng is not None else None
            todo_batch.append((row, repeat_idx, seed))

        if not todo_batch:
            return [None] * len(batch)

        batch_seed = todo_batch[0][2]
        outputs = await client.generate_batch(
            system_prompts=[system_prompt] * len(todo_batch),
            user_contents=[row["question"] for row, _, _ in todo_batch],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            seed=batch_seed,
        )

        output_map: dict[tuple[str, int], dict] = {}
        for (row, repeat_idx, seed), (prediction, reasoning_content, finish_reason) in zip(todo_batch, outputs, strict=True):
            output_map[(row["id"], repeat_idx)] = {
                **row,
                "prediction": prediction,
                "predicted_result": extract_answer(prediction),
                "prediction_api_model": api_model,
                "prediction_system_prompt": system_prompt,
                "prediction_reasoning_effort": reasoning_effort,
                "prediction_repeat_idx": repeat_idx,
                "prediction_temperature": temperature,
                "prediction_max_tokens": max_tokens,
                "prediction_provider": provider,
                "prediction_reasoning_content": reasoning_content,
                "prediction_finish_reason": finish_reason,
                "prediction_seed": seed,
            }

        results: list[dict | None] = []
        for row, repeat_idx in batch:
            results.append(output_map.get((row["id"], repeat_idx)))
        return results

    todo = list(itertools.product(records, range(n_repeats)))
    async def _run() -> None:
        with open(pred_file, "a", buffering=1) as f, tqdm.tqdm(total=total, desc="generating predictions") as pbar:
            if isinstance(client, LocalHFClient):
                for batch in chunked(todo, max_concurrency):
                    results = await _make_local_batch(batch)
                    for result in results:
                        pbar.update(1)
                        pbar.refresh()
                        if result is None:
                            continue
                        f.write(json.dumps(result) + "\n")
                        f.flush()
                return

            coros = (_make_coro(row, i, existing_df) for row, i in todo)
            async for result in run_bounded(coros, max_concurrency):
                pbar.update(1)
                pbar.refresh()
                if result is None:
                    continue
                f.write(json.dumps(result) + "\n")
                f.flush()

    asyncio.run(_run())
    typer.secho(f"Done! Predictions saved to {pred_file}.", fg="green")


@app.command()
def eval(
    pred_file: pathlib.Path,
    base_pred_file: pathlib.Path | None = None,
) -> None:
    pred_df = load_df(pred_file)
    pred_df["is_correct"] = pred_df.apply(lambda row: answers_match(row["answer"], row["predicted_result"]), axis=1)
    report = {
        "unique_problems": pred_df["id"].nunique(),
        "total_problems": len(pred_df),
        "acc": series_mean_or_nan(pred_df["is_correct"]),
    }

    if base_pred_file is None:
        typer.secho("No base predictions file provided. Reporting limited evaluation results.", fg="yellow")
    else:
        base_df = load_df(base_pred_file)
        base_df["is_correct"] = base_df.apply(lambda row: answers_match(row["answer"], row["predicted_result"]), axis=1)

        base_solved = base_df.set_index("id")["is_correct"]
        pred_df["base_is_correct"] = pred_df["id"].map(base_solved)

        solved_mask = pred_df["base_is_correct"]
        unsolved_mask = ~pred_df["base_is_correct"]

        report.update({
            "acc_base": series_mean_or_nan(base_df["is_correct"]),
            "acc_delta": series_mean_or_nan(pred_df["is_correct"]) - series_mean_or_nan(base_df["is_correct"]),
            "acc_on_base_solved": series_mean_or_nan(pred_df.loc[solved_mask, "is_correct"]),
            "acc_on_base_unsolved": series_mean_or_nan(pred_df.loc[unsolved_mask, "is_correct"]),
            "n_problems_improved": (pred_df["is_correct"] & ~pred_df["base_is_correct"]).groupby(pred_df["id"]).any().sum(),
            "n_problems_broken": (~pred_df["is_correct"] & pred_df["base_is_correct"]).groupby(pred_df["id"]).any().sum(),
        })

    typer.secho("Evaluation report:", fg="cyan")
    rich.pretty.pprint(report, expand_all=True)

if __name__ == "__main__":
    app()
