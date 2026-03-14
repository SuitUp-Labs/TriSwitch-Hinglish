import argparse
from pathlib import Path

import pandas as pd
import torch

from config import CONFIG
from metrics.bertscore_metric import compute_bertscore
from metrics.bleu_metric import compute_bleu_scores
from utils import ensure_parent_dir


def resolve_device(force_cpu: bool) -> tuple[str, bool]:
    if force_cpu:
        return "cpu", False
    if torch.cuda.is_available():
        return "cuda", True
    return "cpu", False


def run_selected_metrics(
    dataframe: pd.DataFrame,
    selected_metrics: list[str],
    use_gpu: bool,
    bert_model_type: str,
    bert_batch_size: int,
) -> pd.DataFrame:
    merged = dataframe.copy()

    for metric_name in selected_metrics:
        name = metric_name.strip().lower()
        if name == "bleu":
            metric_df = compute_bleu_scores(dataframe)
        elif name == "bertscore":
            metric_df = compute_bertscore(
                dataframe=dataframe,
                model_type=bert_model_type,
                batch_size=bert_batch_size,
                use_gpu=use_gpu,
            )
        else:
            raise ValueError(f"Unsupported metric selected: {metric_name}")

        merged = merged.merge(metric_df, on=["id", "pair_type"], how="left")

    return merged


def metric_summary_table(metric_scores_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in metric_scores_df.columns
        if column in {"bleu", "bertscore_p", "bertscore_r", "bertscore_f1"}
    ]
    summary = metric_scores_df[metric_columns].describe().transpose().reset_index()
    summary = summary.rename(columns={"index": "metric"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected metrics and merge into master score table.")
    parser.add_argument("--input", type=Path, default=CONFIG.pairwise_features_path)
    parser.add_argument("--output", type=Path, default=CONFIG.metric_scores_path)
    parser.add_argument("--summary-output", type=Path, default=CONFIG.metric_summary_path)
    parser.add_argument(
        "--metrics",
        default=",".join(CONFIG.selected_metrics),
        help="Comma-separated metrics, e.g., bleu,bertscore",
    )
    parser.add_argument("--bert-model-type", default="xlm-roberta-base")
    parser.add_argument("--bert-batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    selected_metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    input_df = pd.read_csv(args.input)
    device, use_gpu = resolve_device(args.cpu)

    if args.cpu:
        print("Device check: CPU forced via --cpu.")
    elif device == "cuda":
        print(f"Device check: GPU available. Using {torch.cuda.get_device_name(0)}.")
    else:
        print("Device check: GPU requested but not available. Falling back to CPU.")

    scores_df = run_selected_metrics(
        dataframe=input_df,
        selected_metrics=selected_metrics,
        use_gpu=use_gpu,
        bert_model_type=args.bert_model_type,
        bert_batch_size=args.bert_batch_size,
    )

    ensure_parent_dir(args.output)
    scores_df.to_csv(args.output, index=False, encoding="utf-8")

    summary_df = metric_summary_table(scores_df)
    ensure_parent_dir(args.summary_output)
    summary_df.to_csv(args.summary_output, index=False, encoding="utf-8")

    print(f"Saved metric scores: {args.output}")
    print(f"Saved metric summary: {args.summary_output}")


if __name__ == "__main__":
    main()
