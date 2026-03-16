import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from comet import download_model, load_from_checkpoint

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from utils import ensure_parent_dir


def compute_comet(
    dataframe: pd.DataFrame,
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
    use_gpu: bool = False,
) -> pd.DataFrame:
    """
    Compute COMET scores for sentence pairs.
    
    Args:
        dataframe: DataFrame with columns 'text_a' (hypothesis), 'text_b' (reference), and 'id', 'pair_type'
        model_name: COMET model to use
        batch_size: Batch size for inference
        use_gpu: Whether to use GPU (will fall back to CPU if not available)
    
    Returns:
        DataFrame with COMET scores
    """
    # Download model if not already cached
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    
    # Prepare data for COMET
    data_for_comet = []
    for _, row in dataframe.iterrows():
        data_for_comet.append({
            "src": "",
            "mt": str(row["text_a"]) if pd.notna(row["text_a"]) else "",
            "ref": str(row["text_b"]) if pd.notna(row["text_b"]) else "",
        })
    
    # Use model.predict() which returns (scores, global_score)
    # The key insight: Don't print the result, just capture it
    model_output = model.predict(
        data_for_comet,
        batch_size=batch_size,
        gpus=0  # CPU only
    )
    
    # Unpack the tuple
    scores_result = model_output[0]  # First element is the scores
    
    # Convert to list if it's a tensor
    if hasattr(scores_result, 'cpu'):
        scores_result = scores_result.cpu()
    if hasattr(scores_result, 'numpy'):
        scores_result = scores_result.numpy()
    if hasattr(scores_result, 'tolist'):
        scores_list = scores_result.tolist()
    else:
        scores_list = list(scores_result) if hasattr(scores_result, '__iter__') else [scores_result]
    
    output = dataframe[["id", "pair_type"]].copy()
    output["comet_score"] = scores_list
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute COMET scores for TriSwitch pairwise rows.")
    parser.add_argument("--input", type=Path, default=CONFIG.pairwise_features_path)
    parser.add_argument("--output", type=Path, default=CONFIG.data_processed_dir / "comet_scores.csv")
    parser.add_argument("--model-name", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    args = parser.parse_args()

    dataframe = pd.read_csv(args.input)
    scores = compute_comet(
        dataframe=dataframe,
        model_name=args.model_name,
        batch_size=args.batch_size,
        use_gpu=not args.cpu,
    )
    ensure_parent_dir(args.output)
    scores.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved COMET scores: {args.output}")


if __name__ == "__main__":
    main()
