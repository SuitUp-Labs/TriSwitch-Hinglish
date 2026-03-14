"""
Paper Stats: One-Stop Summary of All Analysis Numbers
======================================================
Runs all three analysis modules and prints a clean, consolidated summary
of the numbers needed for the paper revision.

Usage (from project root):
    python scripts/analysis/paper_stats.py

No arguments needed. Requires:
  - results/pairwise_scores.csv
  - old/final_results/human_eval.csv
  - results/average_metrics.json (or .csv)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

# ── Section separator ──────────────────────────────────────────────────────
def section(title: str):
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def main():
    print("\n" + "#" * 70)
    print("#  TriSwitch-Hinglish — Paper Statistics Summary")
    print("#" * 70)

    # ── 1. Model Metric Averages (Table 2) ────────────────────────────────
    section("1. MODEL METRIC AVERAGES  (Table 2 in paper)")
    avg_json = ROOT / "results" / "average_metrics.json"
    if avg_json.exists():
        with open(avg_json, encoding="utf-8") as f:
            avgs = json.load(f)
        header = f"  {'Model':<10} {'N':>4}  {'BLEU':>8}  {'chrF':>8}  {'BERTScore F1':>14}"
        print(header)
        print("  " + "-" * 52)
        for row in avgs:
            print(
                f"  {row['model']:<10} {row['num_evaluations']:>4}  "
                f"{row['avg_BLEU']:.4f}    "
                f"{row['avg_chrF']:.2f}     "
                f"{row['avg_BERTScore_F1']:.4f}"
            )
    else:
        print(f"  [MISSING] {avg_json}")

    # ── 2. PASS Rate + Confidence Interval (Section 3.2 / 4.1) ───────────
    section("2. PASS RATE + 95% CI  (Section 3.2 / 4.1 in paper)")
    try:
        import confidence_intervals as ci_mod
        import csv
        import math

        rows = []
        with open(ROOT / "old" / "final_results" / "human_eval.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                sp = int(r["SP (Semantic Preservation)"])
                cv = int(r["CSV (Code-Switch Validity)"])
                rows.append(1 if (sp == 1 and cv == 1) else 0)

        n = len(rows)
        n_pass = sum(rows)
        p_hat = n_pass / n
        w_lo, w_hi = ci_mod.wilson_ci(n_pass, n)
        print(f"  n = {n},  PASS = {n_pass}  ({p_hat*100:.1f}%)")
        print(f"  95% Wilson CI: [{w_lo:.3f}, {w_hi:.3f}]")
        print(f"  → Paper phrasing: PASS rate = {p_hat:.2f}  (95% CI: {w_lo:.2f}–{w_hi:.2f})")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ── 3. Correlation Analysis (Section 4.2) ─────────────────────────────
    section("3. PASS/FAIL ↔ METRIC CORRELATION  (Section 4.2)")
    corr_csv = ROOT / "results" / "correlation_analysis.csv"
    if corr_csv.exists():
        import csv as _csv
        header = f"  {'Metric':<45} {'r_pb':>7} {'p-val':>8} {'Sig.':>6}"
        print(header)
        print("  " + "-" * 68)
        with open(corr_csv, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                print(
                    f"  {row['metric']:<45} {float(row['r_pb']):>7.4f} "
                    f"{float(row['p_value']):>8.4f} {row['significant']:>6}"
                )
    else:
        print(f"  [NOT YET GENERATED – run correlation_analysis.py first]")
        print(f"  Command: python scripts/analysis/correlation_analysis.py")

    # ── 4. Variant Metric Summary (Section 3 / new §4.0) ─────────────────
    section("4. BASE ↔ VARIANT METRIC SUMMARY  (New subsection / Figure)")
    variant_csv = ROOT / "results" / "variant_metric_summary.csv"
    if variant_csv.exists():
        import csv as _csv
        with open(variant_csv, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            all_rows = list(reader)

        # Print only OVERALL rows for brevity
        overall = [r for r in all_rows if r["domain"] == "OVERALL"]
        print(f"  {'Variant':<22} {'n':>4}  {'BLEU mean':>10} {'±std':>6}  {'BERTScore':>10} {'±std':>6}")
        print("  " + "-" * 65)
        for r in overall:
            print(
                f"  {r['variant']:<22} {r['n']:>4}  "
                f"{float(r['bleu_mean']):.4f}     "
                f"{float(r['bleu_std']):.4f}  "
                f"{float(r['bertscore_mean']):.4f}     "
                f"{float(r['bertscore_std']):.4f}"
            )
        print(f"\n  Full per-domain table: {variant_csv}")
    else:
        print(f"  [NOT YET GENERATED – run variant_metric_summary.py first]")
        print(f"  Command: python scripts/analysis/variant_metric_summary.py")

    # ── Done ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Run individual scripts to regenerate outputs:")
    print("    python scripts/analysis/correlation_analysis.py")
    print("    python scripts/analysis/variant_metric_summary.py")
    print("    python scripts/analysis/confidence_intervals.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
