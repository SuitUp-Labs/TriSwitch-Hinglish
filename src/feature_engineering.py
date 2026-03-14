import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG
from utils import (
    ensure_parent_dir,
    normalized_kendall_tau_distance,
    parse_switch_points,
    safe_divide,
    token_alignment_positions,
    token_edit_distance,
    tokenize,
)


def _pair_features(text_a: str, text_b: str) -> dict:
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    len_a = len(tokens_a)
    len_b = len(tokens_b)
    overlap = set(tokens_a).intersection(set(tokens_b))
    union = set(tokens_a).union(set(tokens_b))

    aligned = token_alignment_positions(tokens_a, tokens_b)
    displacements = [abs(index_a - index_b) for index_a, index_b in aligned]
    mapped_positions = [index_b for _, index_b in aligned]

    return {
        "len_a": len_a,
        "len_b": len_b,
        "len_diff": abs(len_a - len_b),
        "token_overlap_count": len(overlap),
        "token_jaccard": safe_divide(len(overlap), len(union), default=0.0),
        "same_token_set": set(tokens_a) == set(tokens_b),
        "position_displacement_mean": float(np.mean(displacements)) if displacements else 0.0,
        "position_displacement_max": float(np.max(displacements)) if displacements else 0.0,
        "kendall_tau_distance": normalized_kendall_tau_distance(mapped_positions),
        "edit_distance_tokens": token_edit_distance(tokens_a, tokens_b),
    }


def add_features(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    dataframe = pairwise_df.copy()

    pair_features = dataframe.apply(lambda row: _pair_features(row["text_a"], row["text_b"]), axis=1)
    pair_features_df = pd.DataFrame(list(pair_features))
    dataframe = pd.concat([dataframe.reset_index(drop=True), pair_features_df.reset_index(drop=True)], axis=1)

    total_tokens = dataframe["tokens_hindi"] + dataframe["tokens_english"]
    dataframe["english_ratio"] = dataframe.apply(
        lambda row: safe_divide(row["tokens_english"], row["tokens_hindi"] + row["tokens_english"], default=0.0),
        axis=1,
    )
    dataframe["hindi_ratio"] = dataframe.apply(
        lambda row: safe_divide(row["tokens_hindi"], row["tokens_hindi"] + row["tokens_english"], default=0.0),
        axis=1,
    )
    dataframe["num_switch_points"] = dataframe["switch_points"].apply(lambda value: len(parse_switch_points(value)))

    dataframe["is_base_tf"] = dataframe["pair_type"] == "base_tf"
    dataframe["is_base_es"] = dataframe["pair_type"] == "base_es"
    dataframe["is_tf_es"] = dataframe["pair_type"] == "tf_es"

    dataframe["switch_point_bucket"] = pd.cut(
        dataframe["num_switch_points"],
        bins=[-1, 1, 2, float("inf")],
        labels=["1_switch", "2_switch", "3plus_switch"],
    )

    dataframe = dataframe.drop(columns=["switch_point_bucket"]).assign(
        switch_point_bucket=pd.cut(
            dataframe["num_switch_points"],
            bins=[-1, 1, 2, float("inf")],
            labels=["1_switch", "2_switch", "3plus_switch"],
        ).astype(str)
    )

    if total_tokens.eq(0).any():
        dataframe.loc[total_tokens.eq(0), ["english_ratio", "hindi_ratio"]] = 0.0

    return dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Add structural features to pairwise TriSwitch dataset.")
    parser.add_argument("--input", type=Path, default=CONFIG.pairwise_dataset_path)
    parser.add_argument("--output", type=Path, default=CONFIG.pairwise_features_path)
    args = parser.parse_args()

    pairwise_df = pd.read_csv(args.input)
    feature_df = add_features(pairwise_df)

    ensure_parent_dir(args.output)
    feature_df.to_csv(args.output, index=False, encoding="utf-8")

    print(f"Saved pairwise feature dataset: {args.output}")
    print(f"Rows: {len(feature_df)}")


if __name__ == "__main__":
    main()
