import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from config import CONFIG
from utils import ensure_parent_dir


DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


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
        "You are evaluating semantic equivalence for Hinglish sentence pairs. "
        "Score how well sentence A preserves the meaning of sentence B on a 0.0 to 1.0 scale, "
        "where 1.0 means fully equivalent and 0.0 means unrelated.\n\n"
        "Output ONLY valid JSON with keys: score, rationale.\n"
        "- score: float between 0.0 and 1.0\n"
        "- rationale: one concise sentence\n\n"
        f"Sentence A: {text_a}\n"
        f"Sentence B: {text_b}"
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


def compute_llm_judge_scores(
    dataframe: pd.DataFrame,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 80,
    timeout_seconds: int = 60,
) -> pd.DataFrame:
    load_dotenv()
    nim_api_key = api_key or os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NIM_API_KEY")
    if not nim_api_key:
        raise ValueError(
            "Missing NVIDIA NIM API key. Set NVIDIA_NIM_API_KEY or NIM_API_KEY in your environment or .env file."
        )

    scores: list[float] = []
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="LLM judge"):
        text_a = str(row.get("text_a", "") if pd.notna(row.get("text_a", "")) else "")
        text_b = str(row.get("text_b", "") if pd.notna(row.get("text_b", "")) else "")
        prompt = _build_prompt(text_a=text_a, text_b=text_b)
        content = _nim_chat_completion(
            prompt=prompt,
            api_key=nim_api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        scores.append(_extract_score(content))

    output = dataframe[["id", "pair_type"]].copy()
    output["llm_judge_score"] = np.array(scores, dtype=float)
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
    args = parser.parse_args()

    dataframe = pd.read_csv(args.input)
    scores = compute_llm_judge_scores(
        dataframe=dataframe,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    ensure_parent_dir(args.output)
    scores.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved LLM judge scores: {args.output}")


if __name__ == "__main__":
    main()