"""
Pairwise Metric Computation
Computes BLEU, BERTScore, and optionally COMET for each triple
(base vs topic-fronting and base vs emphasis-shift).

Usage:
    # BLEU + BERTScore only (fast, no extra install):
    python scripts/evaluation/pairwise_metrics.py

    # Add COMET (requires: pip install unbabel-comet):
    python scripts/evaluation/pairwise_metrics.py --comet
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from bert_score import score as bert_score_func
from sacrebleu.metrics import BLEU

# Initialize flags
COMET_AVAILABLE = False
comet_model = None

# Try to import optional metrics
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except Exception as e:
    print(f"Warning: COMET not available (optional). Install with: pip install unbabel-comet")


# ============================================================================
# Metric Computation Functions
# ============================================================================

def compute_bleu(hypothesis, reference):
    """Compute BLEU score (0-1 scale)"""
    bleu = BLEU()
    score = bleu.corpus_score([hypothesis], [[reference]]).score / 100.0  # Normalize to 0-1
    return score


def compute_bertscore(hypothesis, reference):
    """Compute BERTScore F1 (0-1 scale)"""
    _, _, f1 = bert_score_func([hypothesis], [reference], lang="en", verbose=False)
    return f1.item()


def compute_comet(hypothesis, reference, model=None):
    """Compute COMET score (0-1 scale)"""
    if not COMET_AVAILABLE or model is None:
        return None
    
    try:
        data = [{"src": "", "mt": hypothesis, "ref": reference}]
        scores = model.predict(data, batch_size=1, gpus=1 if torch.cuda.is_available() else 0)
        return scores.scores[0]
    except Exception as e:
        print(f"COMET error: {e}")
        return None


# ============================================================================
# Main Execution
# ============================================================================

def main():
    global COMET_AVAILABLE

    # ── CLI ────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Compute pairwise metrics for TriSwitch-Hinglish dataset")
    parser.add_argument(
        "--comet", action="store_true",
        help="Load and run COMET (requires: pip install unbabel-comet, ~1 GB model download)"
    )
    args = parser.parse_args()

    # Disable COMET if not explicitly requested
    if not args.comet:
        COMET_AVAILABLE = False
        print("Note: COMET disabled. Run with --comet to enable.")

    # File paths
    INPUT_JSON = Path("dataset/initial/db_with_reference_en.json")
    OUTPUT_CSV = Path("results/pairwise_scores.csv")

    # Create output directory if needed
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Load COMET model if available and requested
    comet_model = None
    if COMET_AVAILABLE:
        try:
            print("Loading COMET model (Unbabel/wmt22-comet-da)...")
            comet_model_name = download_model("Unbabel/wmt22-comet-da")
            comet_model = load_from_checkpoint(comet_model_name)
            print("COMET model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load COMET model: {e}")
            COMET_AVAILABLE = False
    
    # Load dataset
    print(f"Loading dataset from {INPUT_JSON}...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} entries")
    
    # Prepare CSV headers
    csv_headers = [
        'id',
        'domain',
        'base',
        'variant_topic_fronting',
        'variant_emphasis_shift',
        'reference_en',
        'bleu_base_vs_topic',
        'bleu_base_vs_emph',
        'bertscore_base_vs_topic',
        'bertscore_base_vs_emph',
        'comet_base_vs_topic',
        'comet_base_vs_emph',
    ]
    
    # Open output CSV file and write results
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        
        # Process each entry
        for idx, entry in enumerate(tqdm(data, desc="Computing pairwise metrics"), 1):
            
            entry_id = entry.get('id', idx)
            base = entry.get('base', '')
            topic = entry.get('variant_topic_fronting', '')
            emph = entry.get('variant_emphasis_shift', '')
            domain = entry.get('domain', '')
            
            # Skip if any variants are missing
            if not (base and topic and emph):
                print(f"Warning: Entry {entry_id} missing variants, skipping")
                continue
            
            row = {
                'id': entry_id,
                'domain': domain,
                'base': base,
                'variant_topic_fronting': topic,
                'variant_emphasis_shift': emph,
                'reference_en': entry.get('reference_en', ''),
            }
            
            # Compute pairwise metrics
            try:
                # BLEU scores
                row['bleu_base_vs_topic'] = compute_bleu(topic, base)
                row['bleu_base_vs_emph'] = compute_bleu(emph, base)
                
                # BERTScore
                row['bertscore_base_vs_topic'] = compute_bertscore(topic, base)
                row['bertscore_base_vs_emph'] = compute_bertscore(emph, base)
                
                # COMET scores
                if COMET_AVAILABLE and comet_model:
                    row['comet_base_vs_topic'] = compute_comet(topic, base, comet_model)
                    row['comet_base_vs_emph'] = compute_comet(emph, base, comet_model)
                else:
                    row['comet_base_vs_topic'] = None
                    row['comet_base_vs_emph'] = None
                
            except Exception as e:
                print(f"Error processing entry {entry_id}: {e}")
                continue
            
            writer.writerow(row)
    
    print(f"\n✓ Pairwise metrics computation complete!")
    print(f"✓ Results saved to: {OUTPUT_CSV}")
    print(f"✓ Total entries processed: {len(data)}")


if __name__ == "__main__":
    main()
