import argparse
from pathlib import Path

import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from config import CONFIG
from utils import ensure_parent_dir, tokenize


def compute_bleu_scores(dataframe: pd.DataFrame) -> pd.DataFrame:
    smoother = SmoothingFunction().method4

    def score_row(row: pd.Series) -> float:
        reference = tokenize(row["text_b"])
        candidate = tokenize(row["text_a"])
        if not reference or not candidate:
            return 0.0
        return float(sentence_bleu([reference], candidate, smoothing_function=smoother))

    output = dataframe[["id", "pair_type"]].copy()
    output["bleu"] = dataframe.apply(score_row, axis=1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute sentence BLEU for TriSwitch pairwise rows.")
    parser.add_argument("--input", type=Path, default=CONFIG.pairwise_features_path)
    parser.add_argument("--output", type=Path, default=CONFIG.data_processed_dir / "bleu_scores.csv")
    args = parser.parse_args()

    dataframe = pd.read_csv(args.input)
    scores = compute_bleu_scores(dataframe)
    ensure_parent_dir(args.output)
    scores.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved BLEU scores: {args.output}")


if __name__ == "__main__":
    main()
