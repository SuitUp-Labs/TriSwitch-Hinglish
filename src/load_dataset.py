import argparse
import json
from pathlib import Path

import pandas as pd

from config import CONFIG
from utils import (
    count_languages_from_pattern,
    ensure_parent_dir,
    infer_switch_points_from_pattern,
    normalize_whitespace,
    parse_switch_points,
    tokenize,
)


TEXT_FIELDS = ["base", "variant_topic_fronting", "variant_emphasis_shift", "domain", "pattern", "reference_en"]


def load_and_validate_dataset(input_path: Path, strict: bool = False) -> tuple[pd.DataFrame, dict]:
    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Expected raw dataset JSON to be a list of records.")

    records = []
    report = {
        "total_rows": len(payload),
        "missing_required_fields": 0,
        "duplicate_ids": 0,
        "invalid_switch_points": 0,
        "token_length_mismatch": 0,
        "pattern_count_mismatch": 0,
        "switch_point_pattern_mismatch": 0,
        "notes": [],
    }

    seen_ids: set[int] = set()
    for row_index, raw_row in enumerate(payload, start=1):
        missing_fields = [field for field in CONFIG.required_fields if field not in raw_row]
        if missing_fields:
            report["missing_required_fields"] += 1
            report["notes"].append(f"Row {row_index}: missing fields {missing_fields}")
            if strict:
                continue

        row = dict(raw_row)
        for field in TEXT_FIELDS:
            if field in row and row[field] is not None:
                row[field] = normalize_whitespace(row[field])

        row_id = row.get("id")
        if row_id in seen_ids:
            report["duplicate_ids"] += 1
            report["notes"].append(f"Duplicate id encountered: {row_id}")
        seen_ids.add(row_id)

        tokens_base = tokenize(row.get("base", ""))
        token_length = len(tokens_base)
        tokens_hindi = int(row.get("tokens_hindi", 0))
        tokens_english = int(row.get("tokens_english", 0))

        if tokens_hindi + tokens_english != token_length:
            report["token_length_mismatch"] += 1
            report["notes"].append(
                f"ID {row_id}: tokens_hindi + tokens_english ({tokens_hindi + tokens_english}) != len(base_tokens) ({token_length})"
            )

        pattern_hindi, pattern_english = count_languages_from_pattern(row.get("pattern", ""))
        if (pattern_hindi, pattern_english) != (tokens_hindi, tokens_english):
            report["pattern_count_mismatch"] += 1
            report["notes"].append(
                f"ID {row_id}: pattern counts (HN={pattern_hindi}, EN={pattern_english}) != metadata counts (HN={tokens_hindi}, EN={tokens_english})"
            )

        try:
            switch_points = parse_switch_points(row.get("switch_points", []))
        except ValueError:
            report["invalid_switch_points"] += 1
            report["notes"].append(f"ID {row_id}: switch_points not parseable")
            switch_points = []

        valid_upper_bound = max(token_length - 2, -1)
        if any((not isinstance(point, int)) or point < 0 or point > valid_upper_bound for point in switch_points):
            report["invalid_switch_points"] += 1
            report["notes"].append(
                f"ID {row_id}: switch_points {switch_points} outside valid range [0, {valid_upper_bound}]"
            )

        pattern_switch_points = infer_switch_points_from_pattern(row.get("pattern", ""))
        if switch_points != pattern_switch_points:
            report["switch_point_pattern_mismatch"] += 1
            report["notes"].append(
                f"ID {row_id}: switch_points {switch_points} != pattern-derived {pattern_switch_points}"
            )

        row["switch_points"] = json.dumps(switch_points, ensure_ascii=False)
        records.append(row)

    dataframe = pd.DataFrame(records)
    return dataframe, report


def write_sanity_log(report: dict, output_path: Path) -> None:
    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("TriSwitch Dataset Sanity Report\n")
        file.write("=" * 40 + "\n")
        for key, value in report.items():
            if key != "notes":
                file.write(f"{key}: {value}\n")

        file.write("\nDetailed Notes\n")
        file.write("-" * 40 + "\n")
        if report["notes"]:
            for note in report["notes"]:
                file.write(f"- {note}\n")
        else:
            file.write("No issues detected.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and sanity-check TriSwitch raw dataset.")
    parser.add_argument("--input", type=Path, default=CONFIG.raw_dataset_path)
    parser.add_argument("--output", type=Path, default=CONFIG.cleaned_dataset_path)
    parser.add_argument("--sanity-log", type=Path, default=CONFIG.dataset_sanity_log_path)
    parser.add_argument("--strict", action="store_true", help="Skip rows with missing required fields.")
    args = parser.parse_args()

    dataframe, report = load_and_validate_dataset(args.input, strict=args.strict)
    ensure_parent_dir(args.output)
    dataframe.to_csv(args.output, index=False, encoding="utf-8")
    write_sanity_log(report, args.sanity_log)

    print(f"Saved cleaned dataset: {args.output}")
    print(f"Saved sanity report: {args.sanity_log}")
    print(f"Total rows retained: {len(dataframe)}")


if __name__ == "__main__":
    main()
