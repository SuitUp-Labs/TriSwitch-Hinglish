import argparse
from pathlib import Path

import pandas as pd
from bert_score import score as bertscore_score

from config import CONFIG
from utils import ensure_parent_dir


def compute_bertscore(
    dataframe: pd.DataFrame,
    model_type: str = "xlm-roberta-base",
    batch_size: int = 32,
    use_gpu: bool = True,
) -> pd.DataFrame:
    candidates = dataframe["text_a"].fillna("").astype(str).tolist()
    references = dataframe["text_b"].fillna("").astype(str).tolist()

    precision, recall, f1 = bertscore_score(
        cands=candidates,
        refs=references,
        model_type=model_type,
        batch_size=batch_size,
        lang="en",
        verbose=False,
        device="cuda" if use_gpu else "cpu",
    )

    output = dataframe[["id", "pair_type"]].copy()
    output["bertscore_p"] = precision.cpu().numpy()
    output["bertscore_r"] = recall.cpu().numpy()
    output["bertscore_f1"] = f1.cpu().numpy()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BERTScore for TriSwitch pairwise rows.")
    parser.add_argument("--input", type=Path, default=CONFIG.pairwise_features_path)
    parser.add_argument("--output", type=Path, default=CONFIG.data_processed_dir / "bertscore_scores.csv")
    parser.add_argument("--model-type", default="xlm-roberta-base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    args = parser.parse_args()

    dataframe = pd.read_csv(args.input)
    scores = compute_bertscore(
        dataframe=dataframe,
        model_type=args.model_type,
        batch_size=args.batch_size,
        use_gpu=not args.cpu,
    )
    ensure_parent_dir(args.output)
    scores.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved BERTScore scores: {args.output}")


if __name__ == "__main__":
    main()
