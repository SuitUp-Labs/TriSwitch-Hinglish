import argparse
from pathlib import Path

import pandas as pd
import torch

from config import CONFIG
from metrics.bertscore_metric import compute_bertscore
from metrics.bleu_metric import compute_bleu_scores
from metrics.llm_judge_metric import (
    DEFAULT_BASE_URL,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_MODEL,
    compute_llm_judge_scores,
)
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
    llm_model: str,
    llm_base_url: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_timeout_seconds: int,
    llm_save_every: int,
    llm_checkpoint_path: Path,
    llm_resume: bool,
    llm_max_retries: int,
    llm_retry_base_seconds: float,
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
        elif name in {"llm_judge", "llm-judge", "llmjudge"}:
            metric_df = compute_llm_judge_scores(
                dataframe=dataframe,
                model=llm_model,
                base_url=llm_base_url,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
                timeout_seconds=llm_timeout_seconds,
                save_every=llm_save_every,
                checkpoint_path=llm_checkpoint_path,
                resume=llm_resume,
                max_retries=llm_max_retries,
                retry_base_seconds=llm_retry_base_seconds,
            )
        else:
            raise ValueError(f"Unsupported metric selected: {metric_name}")

        merged = merged.merge(metric_df, on=["id", "pair_type"], how="left")

    return merged


def metric_summary_table(metric_scores_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in metric_scores_df.columns
        if column in {"bleu", "bertscore_p", "bertscore_r", "bertscore_f1", "llm_judge_score"}
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
        help="Comma-separated metrics, e.g., bleu,bertscore,llm_judge",
    )
    parser.add_argument("--bert-model-type", default="xlm-roberta-base")
    parser.add_argument("--bert-batch-size", type=int, default=32)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-tokens", type=int, default=80)
    parser.add_argument("--llm-timeout-seconds", type=int, default=60)
    parser.add_argument("--llm-save-every", type=int, default=100)
    parser.add_argument("--llm-checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--llm-no-resume", action="store_true")
    parser.add_argument("--llm-max-retries", type=int, default=6)
    parser.add_argument("--llm-retry-base-seconds", type=float, default=2.0)
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
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_timeout_seconds=args.llm_timeout_seconds,
        llm_save_every=args.llm_save_every,
        llm_checkpoint_path=args.llm_checkpoint_path,
        llm_resume=not args.llm_no_resume,
        llm_max_retries=args.llm_max_retries,
        llm_retry_base_seconds=args.llm_retry_base_seconds,
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
