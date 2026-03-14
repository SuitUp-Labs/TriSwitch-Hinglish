import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

from config import CONFIG


METRIC_COLUMNS = ["bleu", "bertscore_f1", "bertscore_p", "bertscore_r"]


def _available_metrics(dataframe: pd.DataFrame) -> list[str]:
    return [metric for metric in METRIC_COLUMNS if metric in dataframe.columns]


def figure_distributions_by_pair_type(dataframe: pd.DataFrame, metrics: list[str], output_path: Path) -> None:
    rows = len(metrics)
    fig, axes = plt.subplots(rows, 1, figsize=(9, 3.2 * rows), constrained_layout=True)
    if rows == 1:
        axes = [axes]

    order = ["base_tf", "base_es", "tf_es"]
    for axis, metric in zip(axes, metrics):
        if sns is not None:
            sns.boxplot(data=dataframe, x="pair_type", y=metric, order=order, ax=axis)
        else:
            grouped_data = [
                pd.to_numeric(
                    dataframe.loc[dataframe["pair_type"] == pair_type, metric],
                    errors="coerce",
                ).dropna()
                for pair_type in order
            ]
            axis.boxplot(grouped_data, tick_labels=order)
        axis.set_title(f"{metric} by pair type")
        axis.set_xlabel("pair_type")
        axis.set_ylabel(metric)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def figure_score_vs_reordering(dataframe: pd.DataFrame, metrics: list[str], output_path: Path) -> None:
    rows = len(metrics)
    fig, axes = plt.subplots(rows, 1, figsize=(9, 3.2 * rows), constrained_layout=True)
    if rows == 1:
        axes = [axes]

    for axis, metric in zip(axes, metrics):
        x_values = pd.to_numeric(dataframe["position_displacement_mean"], errors="coerce")
        y_values = pd.to_numeric(dataframe[metric], errors="coerce")
        valid = pd.DataFrame({"x": x_values, "y": y_values}).dropna()

        if sns is not None:
            sns.regplot(
                data=valid,
                x="x",
                y="y",
                scatter_kws={"alpha": 0.35, "s": 14},
                line_kws={"color": "crimson"},
                ax=axis,
            )
        else:
            axis.scatter(valid["x"], valid["y"], alpha=0.35, s=14)
            if len(valid) >= 2:
                x_var = pd.Series(valid["x"]).var()
                if x_var and x_var > 0:
                    slope = pd.Series(valid["y"]).cov(valid["x"]) / x_var
                    intercept = valid["y"].mean() - slope * valid["x"].mean()
                    x_line = pd.Series([valid["x"].min(), valid["x"].max()])
                    y_line = slope * x_line + intercept
                    axis.plot(x_line, y_line, color="crimson")
        axis.set_title(f"{metric} vs position displacement mean")
        axis.set_xlabel("position_displacement_mean")
        axis.set_ylabel(metric)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def figure_score_by_switch_complexity(dataframe: pd.DataFrame, metrics: list[str], output_path: Path) -> None:
    rows = len(metrics)
    fig, axes = plt.subplots(rows, 1, figsize=(9, 3.2 * rows), constrained_layout=True)
    if rows == 1:
        axes = [axes]

    order = ["1_switch", "2_switch", "3plus_switch"]
    for axis, metric in zip(axes, metrics):
        if sns is not None:
            sns.boxplot(data=dataframe, x="switch_point_bucket", y=metric, order=order, ax=axis)
        else:
            grouped_data = [
                pd.to_numeric(
                    dataframe.loc[dataframe["switch_point_bucket"] == bucket, metric],
                    errors="coerce",
                ).dropna()
                for bucket in order
            ]
            axis.boxplot(grouped_data, tick_labels=order)
        axis.set_title(f"{metric} by switch complexity")
        axis.set_xlabel("switch_point_bucket")
        axis.set_ylabel(metric)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def figure_pattern_heatmap(dataframe: pd.DataFrame, metrics: list[str], output_path: Path) -> None:
    if "pattern" not in dataframe.columns:
        return

    grouped = dataframe.groupby("pattern", dropna=False)[metrics].mean(numeric_only=True)
    if grouped.empty:
        return

    grouped = grouped.sort_index()
    fig_height = max(4, 0.35 * len(grouped.index))
    fig, axis = plt.subplots(figsize=(9, fig_height), constrained_layout=True)

    if sns is not None:
        sns.heatmap(grouped, annot=False, cmap="viridis", ax=axis)
    else:
        im = axis.imshow(grouped.values, aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(grouped.columns)))
        axis.set_xticklabels(grouped.columns, rotation=45, ha="right")
        axis.set_yticks(range(len(grouped.index)))
        axis.set_yticklabels(grouped.index)
        fig.colorbar(im, ax=axis)

    axis.set_title("Pattern-level metric heatmap")
    axis.set_xlabel("metrics")
    axis.set_ylabel("pattern")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def figure_human_label_separation(dataframe: pd.DataFrame, metrics: list[str], output_path: Path) -> bool:
    label_col = "label_semantic_equivalent"
    if label_col not in dataframe.columns:
        return False

    labels = pd.to_numeric(dataframe[label_col], errors="coerce")
    valid = dataframe.loc[labels.notna()].copy()
    if valid.empty:
        return False

    valid[label_col] = pd.to_numeric(valid[label_col], errors="coerce")
    valid["human_validity_group"] = valid[label_col].apply(lambda x: "human_valid" if x >= 0.5 else "human_invalid")

    rows = len(metrics)
    fig, axes = plt.subplots(rows, 1, figsize=(9, 3.2 * rows), constrained_layout=True)
    if rows == 1:
        axes = [axes]

    order = ["human_valid", "human_invalid"]
    for axis, metric in zip(axes, metrics):
        if sns is not None:
            sns.boxplot(data=valid, x="human_validity_group", y=metric, order=order, ax=axis)
        else:
            grouped_data = [
                pd.to_numeric(
                    valid.loc[valid["human_validity_group"] == group, metric],
                    errors="coerce",
                ).dropna()
                for group in order
            ]
            axis.boxplot(grouped_data, tick_labels=order)
        axis.set_title(f"{metric} by human validity label")
        axis.set_xlabel("human_validity_group")
        axis.set_ylabel(metric)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return True


def run_plots(input_path: Path, figures_dir: Path) -> None:
    dataframe = pd.read_csv(input_path)
    metrics = _available_metrics(dataframe)
    if not metrics:
        raise ValueError("No supported metrics found in metric_scores input.")

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_distributions_by_pair_type(dataframe, metrics, figures_dir / "figure1_distributions_by_pair_type.png")
    figure_score_vs_reordering(dataframe, metrics, figures_dir / "figure2_score_vs_reordering.png")
    figure_score_by_switch_complexity(dataframe, metrics, figures_dir / "figure3_score_by_switch_complexity.png")
    figure_pattern_heatmap(dataframe, metrics, figures_dir / "figure4_pattern_heatmap.png")

    figure5_path = figures_dir / "figure5_human_label_separation.png"
    figure5_skip_path = figures_dir / "figure5_human_label_separation_skipped.txt"

    has_human_sep = figure_human_label_separation(dataframe, metrics, figure5_path)
    if not has_human_sep:
        figure5_skip_path.write_text(
            "Figure 5 skipped: no non-null human semantic labels available in metric_scores.csv.\n"
            "Fill human_validation.csv and rebuild pairwise labels before re-running plotting.\n",
            encoding="utf-8",
        )
        if figure5_path.exists():
            figure5_path.unlink()
    else:
        if figure5_skip_path.exists():
            figure5_skip_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TriSwitch paper-style metric figures.")
    parser.add_argument("--input", type=Path, default=CONFIG.metric_scores_path)
    parser.add_argument("--figures-dir", type=Path, default=CONFIG.project_root / "outputs" / "figures")
    args = parser.parse_args()

    run_plots(args.input, args.figures_dir)

    print(f"Saved figures to: {args.figures_dir}")


if __name__ == "__main__":
    main()
