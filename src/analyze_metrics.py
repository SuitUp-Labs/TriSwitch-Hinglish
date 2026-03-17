import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

from config import CONFIG
from utils import ensure_parent_dir


METRIC_COLUMNS = ["bleu", "bertscore_f1", "bertscore_p", "bertscore_r", "llm_judge_score"]


def _available_metrics(dataframe: pd.DataFrame) -> list[str]:
    return [column for column in METRIC_COLUMNS if column in dataframe.columns]


def descriptive_summary(dataframe: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        series = pd.to_numeric(dataframe[metric], errors="coerce").dropna()
        rows.append(
            {
                "metric": metric,
                "count": len(series),
                "mean": series.mean(),
                "std": series.std(ddof=1),
                "median": series.median(),
                "iqr": series.quantile(0.75) - series.quantile(0.25),
                "min": series.min(),
                "max": series.max(),
            }
        )
    return pd.DataFrame(rows)


def grouped_summary(dataframe: pd.DataFrame, metrics: list[str], group_col: str) -> pd.DataFrame:
    rows = []
    grouped = dataframe.groupby(group_col, dropna=False)
    for group_value, group_df in grouped:
        for metric in metrics:
            series = pd.to_numeric(group_df[metric], errors="coerce").dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "group_by": group_col,
                    "group_value": group_value,
                    "metric": metric,
                    "count": len(series),
                    "mean": series.mean(),
                    "std": series.std(ddof=1),
                    "median": series.median(),
                    "iqr": series.quantile(0.75) - series.quantile(0.25),
                    "min": series.min(),
                    "max": series.max(),
                }
            )
    return pd.DataFrame(rows)


def invariance_tests(dataframe: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    pair_order = ["base_tf", "base_es", "tf_es"]
    rows = []

    for metric in metrics:
        metric_df = dataframe[["pair_type", metric]].dropna()
        groups = [metric_df.loc[metric_df["pair_type"] == pair, metric].values for pair in pair_order]
        groups = [group for group in groups if len(group) > 0]
        if len(groups) >= 2:
            stat, p_value = kruskal(*groups)
            rows.append(
                {
                    "metric": metric,
                    "test": "kruskal_pair_type",
                    "statistic": stat,
                    "p_value": p_value,
                }
            )

        comparisons = [("base_tf", "base_es"), ("base_tf", "tf_es"), ("base_es", "tf_es")]
        for pair_a, pair_b in comparisons:
            a_values = metric_df.loc[metric_df["pair_type"] == pair_a, metric].values
            b_values = metric_df.loc[metric_df["pair_type"] == pair_b, metric].values
            if len(a_values) and len(b_values):
                stat, p_value = mannwhitneyu(a_values, b_values, alternative="two-sided")
                effect = (np.median(a_values) - np.median(b_values))
                rows.append(
                    {
                        "metric": metric,
                        "test": f"mannwhitney_{pair_a}_vs_{pair_b}",
                        "statistic": stat,
                        "p_value": p_value,
                        "median_diff": effect,
                    }
                )

    return pd.DataFrame(rows)


def robustness_summary(dataframe: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        pivot = dataframe.pivot_table(index="id", columns="pair_type", values=metric, aggfunc="mean")
        if pivot.empty:
            continue
        spread = pivot.max(axis=1) - pivot.min(axis=1)
        std_by_item = pivot.std(axis=1)

        rows.append(
            {
                "metric": metric,
                "item_count": len(pivot),
                "item_metric_range_mean": spread.mean(),
                "item_metric_range_median": spread.median(),
                "item_metric_std_mean": std_by_item.mean(),
                "item_metric_std_median": std_by_item.median(),
            }
        )
    return pd.DataFrame(rows)


def structural_sensitivity(dataframe: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    predictors = ["num_switch_points", "english_ratio", "position_displacement_mean", "kendall_tau_distance"]
    rows = []

    for metric in metrics:
        for predictor in predictors:
            if predictor not in dataframe.columns:
                continue
            valid = dataframe[[metric, predictor]].dropna()
            if len(valid) < 3:
                continue
            corr, p_value = spearmanr(valid[predictor], valid[metric])
            rows.append(
                {
                    "metric": metric,
                    "predictor": predictor,
                    "spearman_r": corr,
                    "p_value": p_value,
                    "n": len(valid),
                }
            )
    return pd.DataFrame(rows)


def human_alignment(dataframe: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if "label_semantic_equivalent" not in dataframe.columns:
        return pd.DataFrame()

    rows = []
    labels = pd.to_numeric(dataframe["label_semantic_equivalent"], errors="coerce")
    valid_mask = labels.notna()
    if valid_mask.sum() == 0:
        return pd.DataFrame(
            [
                {
                    "status": "no_human_labels_available",
                    "note": "Fill data/processed/human_validation.csv and rebuild pairwise labels for alignment stats.",
                }
            ]
        )

    for metric in metrics:
        scores = pd.to_numeric(dataframe[metric], errors="coerce")
        mask = valid_mask & scores.notna()
        if mask.sum() < 3:
            continue
        label_values = labels[mask]
        score_values = scores[mask]
        if label_values.nunique() < 2:
            corr, p_value = np.nan, np.nan
            note = "semantic labels are constant; correlation undefined"
        elif score_values.nunique() < 2:
            corr, p_value = np.nan, np.nan
            note = "metric scores are constant; correlation undefined"
        else:
            corr, p_value = spearmanr(label_values, score_values)
            note = "ok"
        rows.append(
            {
                "metric": metric,
                "spearman_with_semantic_label": corr,
                "p_value": p_value,
                "n": int(mask.sum()),
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def run_analysis(input_path: Path, processed_out: Path, tables_dir: Path) -> None:
    dataframe = pd.read_csv(input_path)
    metrics = _available_metrics(dataframe)
    if not metrics:
        raise ValueError("No supported metric columns found in input data.")

    descriptive = descriptive_summary(dataframe, metrics)
    grouped_pair = grouped_summary(dataframe, metrics, "pair_type")
    grouped_domain = grouped_summary(dataframe, metrics, "domain") if "domain" in dataframe.columns else pd.DataFrame()
    grouped_pattern = grouped_summary(dataframe, metrics, "pattern") if "pattern" in dataframe.columns else pd.DataFrame()
    grouped_switch = grouped_summary(dataframe, metrics, "switch_point_bucket") if "switch_point_bucket" in dataframe.columns else pd.DataFrame()

    invariance = invariance_tests(dataframe, metrics)
    robustness = robustness_summary(dataframe, metrics)
    sensitivity = structural_sensitivity(dataframe, metrics)
    alignment = human_alignment(dataframe, metrics)

    ensure_parent_dir(processed_out)
    descriptive.to_csv(processed_out, index=False, encoding="utf-8")

    tables_dir.mkdir(parents=True, exist_ok=True)
    grouped_pair.to_csv(tables_dir / "grouped_summary_pair_type.csv", index=False, encoding="utf-8")
    grouped_domain.to_csv(tables_dir / "grouped_summary_domain.csv", index=False, encoding="utf-8")
    grouped_pattern.to_csv(tables_dir / "grouped_summary_pattern.csv", index=False, encoding="utf-8")
    grouped_switch.to_csv(tables_dir / "grouped_summary_switch_bucket.csv", index=False, encoding="utf-8")
    invariance.to_csv(tables_dir / "invariance_tests.csv", index=False, encoding="utf-8")
    robustness.to_csv(tables_dir / "robustness_summary.csv", index=False, encoding="utf-8")
    sensitivity.to_csv(tables_dir / "structural_sensitivity.csv", index=False, encoding="utf-8")
    alignment.to_csv(tables_dir / "human_alignment.csv", index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TriSwitch metric outputs and write summary tables.")
    parser.add_argument("--input", type=Path, default=CONFIG.metric_scores_path)
    parser.add_argument("--processed-out", type=Path, default=CONFIG.metric_summary_path)
    parser.add_argument("--tables-dir", type=Path, default=CONFIG.project_root / "outputs" / "tables")
    args = parser.parse_args()

    run_analysis(args.input, args.processed_out, args.tables_dir)

    print(f"Saved descriptive summary: {args.processed_out}")
    print(f"Saved analysis tables in: {args.tables_dir}")


if __name__ == "__main__":
    main()
