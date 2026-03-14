import ast
import random
import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize(text: str) -> list[str]:
    return normalize_whitespace(text).split() if normalize_whitespace(text) else []


def parse_pattern(pattern: str) -> list[str]:
    if pattern is None:
        return []
    cleaned = normalize_whitespace(pattern).upper()
    return [part for part in cleaned.split("-") if part]


def count_languages_from_pattern(pattern: str) -> tuple[int, int]:
    tags = parse_pattern(pattern)
    english = sum(1 for tag in tags if tag == "EN")
    hindi = sum(1 for tag in tags if tag == "HN")
    return hindi, english


def infer_switch_points_from_pattern(pattern: str) -> list[int]:
    tags = parse_pattern(pattern)
    points: list[int] = []
    for idx in range(len(tags) - 1):
        if tags[idx] != tags[idx + 1]:
            points.append(idx)
    return points


def parse_switch_points(raw_value) -> list[int]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [int(value) for value in raw_value]
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = [part.strip() for part in stripped.split(",") if part.strip()]
        if isinstance(parsed, list):
            return [int(value) for value in parsed]
    raise ValueError(f"Could not parse switch points from value: {raw_value}")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def token_alignment_positions(tokens_a: list[str], tokens_b: list[str]) -> list[tuple[int, int]]:
    positions_in_b: dict[str, deque[int]] = defaultdict(deque)
    for index, token in enumerate(tokens_b):
        positions_in_b[token].append(index)

    aligned_positions: list[tuple[int, int]] = []
    for index_a, token in enumerate(tokens_a):
        if positions_in_b[token]:
            index_b = positions_in_b[token].popleft()
            aligned_positions.append((index_a, index_b))
    return aligned_positions


def normalized_kendall_tau_distance(mapped_positions: list[int]) -> float:
    n_items = len(mapped_positions)
    if n_items < 2:
        return 0.0

    inversions = 0
    for i in range(n_items):
        for j in range(i + 1, n_items):
            if mapped_positions[i] > mapped_positions[j]:
                inversions += 1

    max_inversions = n_items * (n_items - 1) / 2
    return inversions / max_inversions


def token_edit_distance(tokens_a: list[str], tokens_b: list[str]) -> int:
    m_len, n_len = len(tokens_a), len(tokens_b)
    dp = [[0] * (n_len + 1) for _ in range(m_len + 1)]

    for i in range(m_len + 1):
        dp[i][0] = i
    for j in range(n_len + 1):
        dp[0][j] = j

    for i in range(1, m_len + 1):
        for j in range(1, n_len + 1):
            substitution_cost = 0 if tokens_a[i - 1] == tokens_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + substitution_cost,
            )

    return dp[m_len][n_len]
