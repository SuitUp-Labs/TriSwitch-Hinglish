import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from config import CONFIG
from utils import ensure_parent_dir


DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_CHECKPOINT_PATH = CONFIG.data_processed_dir / "llm_judge_progress.csv"


def _extract_json_object(text: str) -> dict | None:
    cleaned = text.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_score(text: str) -> float:
    payload = _extract_json_object(text)
    if payload and "score" in payload:
        score = float(payload["score"])
        return max(0.0, min(1.0, score))

    numeric_match = re.search(r"([01](?:\.\d+)?)", text)
    if numeric_match:
        score = float(numeric_match.group(1))
        return max(0.0, min(1.0, score))

    raise ValueError(f"Could not parse score from model response: {text}")


def _build_prompt(text_a: str, text_b: str) -> str:
    return (
        "You are a fluency evaluator for Hinglish (Hindi-English code-switched) sentences.\n"
        "Evaluate ONLY Sentence A on its structural naturalness as a Hinglish utterance.\n"
        "Ignore Sentence B entirely — it is provided only as context for the topic.\n\n"
        "Score on these criteria:\n"
        "  1. Word order naturalness — does it follow a natural Hindi SOV or Hinglish flow?\n"
        "  2. Code-switch placement — are Hindi/English switches at natural boundaries?\n"
        "  3. Fluency — would a native Hinglish speaker say this without friction?\n\n"
        "Scoring rubric:\n"
        "  1.0 — Completely natural, a native speaker would say this spontaneously\n"
        "  0.8 — Mostly natural with minor awkwardness\n"
        "  0.6 — Understandable but noticeably unnatural word order or switch point\n"
        "  0.4 — Awkward; requires effort to parse despite being grammatically possible\n"
        "  0.2 — Strongly unnatural; word order feels inverted or forced\n"
        "  0.0 — Completely unnatural or unintelligible as Hinglish\n\n"
        "Use the FULL range — do not default to a single value.\n"
        "Different word orders of the same content SHOULD score differently.\n\n"
        "Output ONLY valid JSON: {\"score\": float, \"rationale\": \"one sentence\"}\n\n"
        f"Sentence A (evaluate this): {text_a}\n"
        f"Sentence B (context only): {text_b}"
    )


def _nim_chat_completion(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict bilingual evaluator for semantic equivalence.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"NIM request failed with status {error.code}: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"NIM request failed: {error}") from error

    choices = response_data.get("choices", [])
    if not choices:
        raise RuntimeError(f"NIM response missing choices: {response_data}")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError(f"NIM response missing message content: {response_data}")
    return content


def _row_key(row: pd.Series) -> tuple[object, str]:
    return row.get("id"), str(row.get("pair_type", ""))


def _save_checkpoint(score_map: dict[tuple[object, str], float], checkpoint_path: Path) -> None:
    ensure_parent_dir(checkpoint_path)
    records = [
        {"id": key[0], "pair_type": key[1], "llm_judge_score": value}
        for key, value in score_map.items()
    ]
    pd.DataFrame(records).to_csv(checkpoint_path, index=False, encoding="utf-8")


def _load_checkpoint(checkpoint_path: Path) -> dict[tuple[object, str], float]:
    if not checkpoint_path.exists():
        return {}

    checkpoint_df = pd.read_csv(checkpoint_path)
    required_cols = {"id", "pair_type", "llm_judge_score"}
    if not required_cols.issubset(checkpoint_df.columns):
        return {}

    score_map: dict[tuple[object, str], float] = {}
    for _, row in checkpoint_df.iterrows():
        if pd.isna(row["llm_judge_score"]):
            continue
        score_map[(row["id"], str(row["pair_type"]))] = float(row["llm_judge_score"])
    return score_map


def compute_llm_judge_scores(
    dataframe: pd.DataFrame,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 80,
    timeout_seconds: int = 60,
    save_every: int = 100,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    resume: bool = True,
    max_retries: int = 6,
    retry_base_seconds: float = 2.0,
) -> pd.DataFrame:
    load_dotenv()
    nim_api_key = api_key or os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NIM_API_KEY")
    if not nim_api_key:
        raise ValueError(
            "Missing NVIDIA NIM API key. Set NVIDIA_NIM_API_KEY or NIM_API_KEY in your environment or .env file."
        )

    score_map = _load_checkpoint(checkpoint_path) if resume else {}
    total_before = len(score_map)

    pending_rows = [
        row
        for _, row in dataframe.iterrows()
        if _row_key(row) not in score_map
    ]

    if total_before > 0:
        print(f"Loaded {total_before} cached LLM-judge scores from {checkpoint_path}.")

    processed_since_save = 0
    for row in tqdm(pending_rows, total=len(pending_rows), desc="LLM judge"):
        text_a = str(row.get("text_a", "") if pd.notna(row.get("text_a", "")) else "")
        text_b = str(row.get("text_b", "") if pd.notna(row.get("text_b", "")) else "")
        prompt = _build_prompt(text_a=text_a, text_b=text_b)

        attempt = 0
        while True:
            try:
                content = _nim_chat_completion(
                    prompt=prompt,
                    api_key=nim_api_key,
                    model=model,
                    base_url=base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                )
                score = _extract_score(content)
                score_map[_row_key(row)] = score
                break
            except RuntimeError as error:
                if "status 429" not in str(error) or attempt >= max_retries:
                    _save_checkpoint(score_map, checkpoint_path)
                    raise
                sleep_seconds = retry_base_seconds * (2 ** attempt)
                attempt += 1
                print(f"429 received. Retry {attempt}/{max_retries} after {sleep_seconds:.1f}s...")
                time.sleep(sleep_seconds)

        processed_since_save += 1
        if processed_since_save >= save_every:
            _save_checkpoint(score_map, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path} ({len(score_map)} rows)")
            processed_since_save = 0

    _save_checkpoint(score_map, checkpoint_path)
    print(f"Final checkpoint saved: {checkpoint_path} ({len(score_map)} rows)")

    output = dataframe[["id", "pair_type"]].copy()
    output["llm_judge_score"] = output.apply(
        lambda row: score_map.get((row["id"], str(row["pair_type"])), np.nan),
        axis=1,
    )

    if output["llm_judge_score"].isna().any():
        missing = int(output["llm_judge_score"].isna().sum())
        raise RuntimeError(
            f"LLM judge scoring incomplete: missing {missing} rows. Resume by rerunning the same command."
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute LLM-as-judge scores using NVIDIA NIM.")
    parser.add_argument("--input", type=Path, default=CONFIG.pairwise_features_path)
    parser.add_argument("--output", type=Path, default=CONFIG.data_processed_dir / "llm_judge_scores.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    args = parser.parse_args()

    dataframe = pd.read_csv(args.input)
    scores = compute_llm_judge_scores(
        dataframe=dataframe,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        save_every=args.save_every,
        checkpoint_path=args.checkpoint_path,
        resume=not args.no_resume,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
    )
    ensure_parent_dir(args.output)
    scores.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved LLM judge scores: {args.output}")


if __name__ == "__main__":
    main()