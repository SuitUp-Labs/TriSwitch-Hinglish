import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG
from utils import ensure_parent_dir


def load_validation_lookup(validation_csv_path: Path) -> dict[tuple[int, str], dict]:
    if not validation_csv_path.exists():
        return {}

    validation_df = pd.read_csv(validation_csv_path)
    if validation_df.empty or "variant_type" not in validation_df.columns:
        return {}

    numeric_df = validation_df.copy()
    for column in ["semantic_preservation", "naturalness", "code_switch_validity"]:
        numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce")

    grouped = (
        numeric_df.groupby(["id", "variant_type"], as_index=False)
        .agg(
            semantic_preservation=("semantic_preservation", "mean"),
            naturalness=("naturalness", "mean"),
            code_switch_validity=("code_switch_validity", "mean"),
        )
    )

    lookup: dict[tuple[int, str], dict] = {}
    for _, row in grouped.iterrows():
        lookup[(int(row["id"]), row["variant_type"])] = {
            "label_semantic_equivalent": row["semantic_preservation"],
            "label_natural": row["naturalness"],
            "label_code_switch_valid": row["code_switch_validity"],
        }
    return lookup


def _combine_labels(label_a: dict, label_b: dict) -> dict:
    combined = {}
    for key in ["label_semantic_equivalent", "label_natural", "label_code_switch_valid"]:
        value_a = label_a.get(key, np.nan)
        value_b = label_b.get(key, np.nan)
        if pd.notna(value_a) and pd.notna(value_b):
            combined[key] = min(value_a, value_b)
        else:
            combined[key] = np.nan
    return combined


def build_pairwise_rows(cleaned_df: pd.DataFrame, validation_lookup: dict[tuple[int, str], dict]) -> pd.DataFrame:
    rows = []

    for _, item in cleaned_df.iterrows():
        item_id = int(item["id"])
        base_common = {
            "id": item_id,
            "domain": item.get("domain"),
            "pattern": item.get("pattern"),
            "tokens_hindi": item.get("tokens_hindi"),
            "tokens_english": item.get("tokens_english"),
            "switch_points": item.get("switch_points", json.dumps([])),
        }

        topic_labels = validation_lookup.get((item_id, "topic_fronting"), {})
        emphasis_labels = validation_lookup.get((item_id, "emphasis_shift"), {})

        rows.append(
            {
                **base_common,
                "pair_type": "base_tf",
                "text_a": item.get("base"),
                "text_b": item.get("variant_topic_fronting"),
                **{key: topic_labels.get(key, np.nan) for key in ["label_semantic_equivalent", "label_natural", "label_code_switch_valid"]},
            }
        )

        rows.append(
            {
                **base_common,
                "pair_type": "base_es",
                "text_a": item.get("base"),
                "text_b": item.get("variant_emphasis_shift"),
                **{key: emphasis_labels.get(key, np.nan) for key in ["label_semantic_equivalent", "label_natural", "label_code_switch_valid"]},
            }
        )

        rows.append(
            {
                **base_common,
                "pair_type": "tf_es",
                "text_a": item.get("variant_topic_fronting"),
                "text_b": item.get("variant_emphasis_shift"),
                **_combine_labels(topic_labels, emphasis_labels),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pairwise TriSwitch dataset.")
    parser.add_argument("--input", type=Path, default=CONFIG.cleaned_dataset_path)
    parser.add_argument("--human-validation", type=Path, default=CONFIG.human_validation_path)
    parser.add_argument("--output", type=Path, default=CONFIG.pairwise_dataset_path)
    args = parser.parse_args()

    cleaned_df = pd.read_csv(args.input)
    validation_lookup = load_validation_lookup(args.human_validation)

    pairwise_df = build_pairwise_rows(cleaned_df, validation_lookup)
    ensure_parent_dir(args.output)
    pairwise_df.to_csv(args.output, index=False, encoding="utf-8")

    print(f"Saved pairwise dataset: {args.output}")
    print(f"Rows: {len(pairwise_df)}")


if __name__ == "__main__":
    main()
