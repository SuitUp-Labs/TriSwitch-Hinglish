import argparse
from pathlib import Path

import pandas as pd

from config import CONFIG
from utils import ensure_parent_dir


VALIDATION_COLUMNS = [
    "id",
    "base",
    "variant_type",
    "variant_text",
    "semantic_preservation",
    "naturalness",
    "code_switch_validity",
    "annotator_id",
    "notes",
]


def create_annotation_template(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in cleaned_df.iterrows():
        records.append(
            {
                "id": row["id"],
                "base": row["base"],
                "variant_type": "topic_fronting",
                "variant_text": row["variant_topic_fronting"],
                "semantic_preservation": pd.NA,
                "naturalness": pd.NA,
                "code_switch_validity": pd.NA,
                "annotator_id": pd.NA,
                "notes": pd.NA,
            }
        )
        records.append(
            {
                "id": row["id"],
                "base": row["base"],
                "variant_type": "emphasis_shift",
                "variant_text": row["variant_emphasis_shift"],
                "semantic_preservation": pd.NA,
                "naturalness": pd.NA,
                "code_switch_validity": pd.NA,
                "annotator_id": pd.NA,
                "notes": pd.NA,
            }
        )

    return pd.DataFrame(records, columns=VALIDATION_COLUMNS)


def create_validated_dataset(cleaned_df: pd.DataFrame, validation_df: pd.DataFrame) -> pd.DataFrame:
    semantic_filled = validation_df["semantic_preservation"].notna().sum()
    if semantic_filled == 0:
        validated = cleaned_df.copy()
        validated["has_human_validation"] = False
        return validated

    grouped = (
        validation_df.groupby(["id", "variant_type"], as_index=False)
        .agg(
            semantic_preservation=("semantic_preservation", "mean"),
            naturalness=("naturalness", "mean"),
            code_switch_validity=("code_switch_validity", "mean"),
        )
        .pivot(index="id", columns="variant_type")
    )

    grouped.columns = [f"{column}_{variant}" for column, variant in grouped.columns]
    grouped = grouped.reset_index()

    validated = cleaned_df.merge(grouped, on="id", how="left")
    validated["has_human_validation"] = True
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description="Create human validation template and validated dataset.")
    parser.add_argument("--input", type=Path, default=CONFIG.cleaned_dataset_path)
    parser.add_argument("--human-validation-output", type=Path, default=CONFIG.human_validation_path)
    parser.add_argument("--validated-output", type=Path, default=CONFIG.validated_dataset_path)
    args = parser.parse_args()

    cleaned_df = pd.read_csv(args.input)
    validation_template = create_annotation_template(cleaned_df)

    ensure_parent_dir(args.human_validation_output)
    validation_template.to_csv(args.human_validation_output, index=False, encoding="utf-8")

    validated_df = create_validated_dataset(cleaned_df, validation_template)
    ensure_parent_dir(args.validated_output)
    validated_df.to_csv(args.validated_output, index=False, encoding="utf-8")

    print(f"Saved annotation template: {args.human_validation_output}")
    print(f"Saved validated dataset: {args.validated_output}")


if __name__ == "__main__":
    main()
