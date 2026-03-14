"""
Correlation Analysis: PASS/FAIL labels vs. Automatic Metrics
=============================================================
Addresses reviewer feedback: "Add a direct correlation experiment."

For each of the 100 human-annotated samples, this script:
  1. Derives a binary PASS label (SP=1 AND CSV=1).
  2. Joins with pairwise_scores.csv to get BLEU and BERTScore for
     topic-fronting and emphasis-shift variants.
  3. Computes point-biserial correlation between PASS and each metric.

Outputs:
  results/correlation_analysis.csv  — machine-readable table
  results/correlation_analysis.txt  — formatted for direct copy into paper
"""

import csv
import json
import sys
from pathlib import Path
from scipy import stats
import numpy as np

# ── Paths (run from project root) ──────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
HUMAN_EVAL_CSV  = ROOT / "old" / "final_results" / "human_eval.csv"
PAIRWISE_CSV    = ROOT / "results" / "pairwise_scores.csv"
OUT_CSV         = ROOT / "results" / "correlation_analysis.csv"
OUT_TXT         = ROOT / "results" / "correlation_analysis.txt"


def load_human_eval(path: Path) -> dict[int, dict]:
    """Load human evaluation labels; derive PASS = SP==1 AND CSV==1."""
    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_ = int(row["id"])
            sp  = int(row["SP (Semantic Preservation)"])
            csv_valid = int(row["CSV (Code-Switch Validity)"])
            data[id_] = {
                "SP":   sp,
                "CSV":  csv_valid,
                "PASS": 1 if (sp == 1 and csv_valid == 1) else 0,
            }
    return data


def load_pairwise(path: Path) -> dict[int, dict]:
    """Load pairwise metric scores."""
    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_ = int(row["id"])
            data[id_] = {
                "bleu_topic":    float(row["bleu_base_vs_topic"])    if row["bleu_base_vs_topic"]    else None,
                "bleu_emph":     float(row["bleu_base_vs_emph"])     if row["bleu_base_vs_emph"]     else None,
                "bert_topic":    float(row["bertscore_base_vs_topic"]) if row["bertscore_base_vs_topic"] else None,
                "bert_emph":     float(row["bertscore_base_vs_emph"])  if row["bertscore_base_vs_emph"]  else None,
                "comet_topic":   float(row["comet_base_vs_topic"])   if row.get("comet_base_vs_topic") else None,
                "comet_emph":    float(row["comet_base_vs_emph"])    if row.get("comet_base_vs_emph")  else None,
            }
    return data


def correlate(labels: list[int], scores: list[float], name: str) -> dict:
    """Compute point-biserial correlation and return a result dict."""
    r, p = stats.pointbiserialr(labels, scores)
    n = len(labels)
    return {
        "metric": name,
        "n": n,
        "r_pb": round(r, 4),
        "p_value": round(p, 4),
        "significant": "Yes" if p < 0.05 else "No",
    }


def main():
    print(f"Loading human evaluation labels from: {HUMAN_EVAL_CSV}")
    human = load_human_eval(HUMAN_EVAL_CSV)
    print(f"  Loaded {len(human)} annotated samples")

    print(f"Loading pairwise metric scores from: {PAIRWISE_CSV}")
    pairwise = load_pairwise(PAIRWISE_CSV)
    print(f"  Loaded {len(pairwise)} pairwise entries")

    # Align on the 100 human-annotated IDs
    aligned_ids = sorted(set(human.keys()) & set(pairwise.keys()))
    print(f"  Aligned samples: {len(aligned_ids)}")

    # Build vectors
    pass_labels = [human[i]["PASS"] for i in aligned_ids]

    vectors = {
        "BLEU (base vs topic-fronting)":    [pairwise[i]["bleu_topic"] for i in aligned_ids],
        "BLEU (base vs emphasis-shift)":    [pairwise[i]["bleu_emph"]  for i in aligned_ids],
        "BERTScore (base vs topic-fronting)": [pairwise[i]["bert_topic"] for i in aligned_ids],
        "BERTScore (base vs emphasis-shift)": [pairwise[i]["bert_emph"]  for i in aligned_ids],
    }

    # Add COMET if available
    comet_topic = [pairwise[i]["comet_topic"] for i in aligned_ids]
    comet_emph  = [pairwise[i]["comet_emph"]  for i in aligned_ids]
    if any(v is not None for v in comet_topic):
        vectors["COMET (base vs topic-fronting)"] = comet_topic
        vectors["COMET (base vs emphasis-shift)"] = comet_emph

    # Compute correlations
    results = []
    for name, scores in vectors.items():
        # Drop rows where metric is None
        valid = [(p, s) for p, s in zip(pass_labels, scores) if s is not None]
        if not valid:
            print(f"  Skipping {name}: no valid scores")
            continue
        pv, sv = zip(*valid)
        results.append(correlate(list(pv), list(sv), name))

    # Print table
    print("\n" + "=" * 72)
    print("POINT-BISERIAL CORRELATION: PASS/FAIL vs. Automatic Metrics")
    print("=" * 72)
    header = f"{'Metric':<45} {'n':>4} {'r_pb':>7} {'p-value':>9} {'Sig.':>6}"
    print(header)
    print("-" * 72)
    for r in results:
        print(f"{r['metric']:<45} {r['n']:>4} {r['r_pb']:>7.4f} {r['p_value']:>9.4f} {r['significant']:>6}")
    print("=" * 72)

    # Descriptive stats on PASS
    n_pass = sum(pass_labels)
    n_total = len(pass_labels)
    print(f"\nPASS count: {n_pass}/{n_total}  ({n_pass/n_total*100:.1f}%)")

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "n", "r_pb", "p_value", "significant"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ Saved: {OUT_CSV}")

    # Save TXT
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("POINT-BISERIAL CORRELATION: PASS/FAIL vs. Automatic Metrics\n")
        f.write("=" * 72 + "\n\n")
        f.write("PASS label = (Semantic Preservation == 1) AND (Code-Switch Validity == 1)\n\n")
        f.write(f"{'Metric':<45} {'n':>4} {'r_pb':>7} {'p-value':>9} {'Sig.':>6}\n")
        f.write("-" * 72 + "\n")
        for r in results:
            f.write(f"{r['metric']:<45} {r['n']:>4} {r['r_pb']:>7.4f} {r['p_value']:>9.4f} {r['significant']:>6}\n")
        f.write("\n" + "=" * 72 + "\n")
        f.write(f"\nPASS count: {n_pass}/{n_total}  ({n_pass/n_total*100:.1f}%)\n\n")
        f.write("Note: r_pb = point-biserial correlation coefficient.\n")
        f.write("      Sig. = statistically significant at alpha=0.05.\n")
    print(f"✓ Saved: {OUT_TXT}")


if __name__ == "__main__":
    main()
