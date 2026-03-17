import argparse
from pathlib import Path

import pandas as pd
import torch

from analyze_metrics import run_analysis
from build_pairwise_dataset import build_pairwise_rows, load_validation_lookup
from config import CONFIG
from feature_engineering import add_features
from load_dataset import load_and_validate_dataset, write_sanity_log
from metrics.llm_judge_metric import DEFAULT_BASE_URL, DEFAULT_CHECKPOINT_PATH, DEFAULT_MODEL
from plotting import run_plots
from run_metrics import metric_summary_table, run_selected_metrics
from utils import ensure_parent_dir, set_seed
from validate_dataset import create_annotation_template, create_validated_dataset


def resolve_device(force_cpu: bool) -> tuple[str, bool]:
    if force_cpu:
        return "cpu", False
    if torch.cuda.is_available():
        return "cuda", True
    return "cpu", False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end TriSwitch dataset evaluation pipeline.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--metrics", default=",".join(CONFIG.selected_metrics))
    parser.add_argument("--cpu", action="store_true")
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
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument("--run-plots", action="store_true")
    parser.add_argument(
        "--refresh-validation-template",
        action="store_true",
        help="Overwrite data/processed/human_validation.csv with a fresh empty template.",
    )
    args = parser.parse_args()

    set_seed(CONFIG.random_seed)
    device, use_gpu = resolve_device(args.cpu)

    if args.cpu:
        print("Device check: CPU forced via --cpu.")
    elif device == "cuda":
        print(f"Device check: GPU available. Using {torch.cuda.get_device_name(0)}.")
    else:
        print("Device check: GPU requested but not available. Falling back to CPU.")

    cleaned_df, report = load_and_validate_dataset(CONFIG.raw_dataset_path, strict=args.strict)
    ensure_parent_dir(CONFIG.cleaned_dataset_path)
    cleaned_df.to_csv(CONFIG.cleaned_dataset_path, index=False, encoding="utf-8")
    write_sanity_log(report, CONFIG.dataset_sanity_log_path)

    if CONFIG.human_validation_path.exists() and not args.refresh_validation_template:
        human_validation_df = pd.read_csv(CONFIG.human_validation_path)
        print(f"Validation file found. Using existing annotations: {CONFIG.human_validation_path}")
    else:
        human_validation_df = create_annotation_template(cleaned_df)
        ensure_parent_dir(CONFIG.human_validation_path)
        human_validation_df.to_csv(CONFIG.human_validation_path, index=False, encoding="utf-8")
        print(f"Validation template written: {CONFIG.human_validation_path}")

    validated_df = create_validated_dataset(cleaned_df, human_validation_df)
    ensure_parent_dir(CONFIG.validated_dataset_path)
    validated_df.to_csv(CONFIG.validated_dataset_path, index=False, encoding="utf-8")

    validation_lookup = load_validation_lookup(CONFIG.human_validation_path)
    pairwise_df = build_pairwise_rows(cleaned_df, validation_lookup=validation_lookup)
    ensure_parent_dir(CONFIG.pairwise_dataset_path)
    pairwise_df.to_csv(CONFIG.pairwise_dataset_path, index=False, encoding="utf-8")

    pairwise_features_df = add_features(pairwise_df)
    ensure_parent_dir(CONFIG.pairwise_features_path)
    pairwise_features_df.to_csv(CONFIG.pairwise_features_path, index=False, encoding="utf-8")

    if not args.skip_metrics:
        selected_metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
        metric_scores_df = run_selected_metrics(
            dataframe=pairwise_features_df,
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
        ensure_parent_dir(CONFIG.metric_scores_path)
        metric_scores_df.to_csv(CONFIG.metric_scores_path, index=False, encoding="utf-8")

        summary_df = metric_summary_table(metric_scores_df)
        ensure_parent_dir(CONFIG.metric_summary_path)
        summary_df.to_csv(CONFIG.metric_summary_path, index=False, encoding="utf-8")

        if args.run_analysis:
            run_analysis(
                input_path=CONFIG.metric_scores_path,
                processed_out=CONFIG.metric_summary_path,
                tables_dir=CONFIG.project_root / "outputs" / "tables",
            )
        if args.run_plots:
            run_plots(
                input_path=CONFIG.metric_scores_path,
                figures_dir=CONFIG.project_root / "outputs" / "figures",
            )

    print("Pipeline completed.")
    print(f"Cleaned dataset: {CONFIG.cleaned_dataset_path}")
    print(f"Validated dataset: {CONFIG.validated_dataset_path}")
    print(f"Pairwise dataset: {CONFIG.pairwise_dataset_path}")
    print(f"Feature dataset: {CONFIG.pairwise_features_path}")
    if not args.skip_metrics:
        print(f"Metric scores: {CONFIG.metric_scores_path}")
        print(f"Metric summary: {CONFIG.metric_summary_path}")
        if args.run_analysis:
            print(f"Analysis tables: {CONFIG.project_root / 'outputs' / 'tables'}")
        if args.run_plots:
            print(f"Figures: {CONFIG.project_root / 'outputs' / 'figures'}")


if __name__ == "__main__":
    main()
