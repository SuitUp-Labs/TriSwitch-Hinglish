"""
Variant Metric Summary: Base ↔ Variant BLEU/BERTScore Across All 500 Pairs
===========================================================================
Addresses reviewer feedback: "Compute BLEU/BERTScore between base and
variants directly — since these are semantically equivalent, low BLEU
scores directly prove the thesis without model complexity."

Reads results/pairwise_scores.csv (all 500 entries) and produces:
  • Per-domain summary statistics
  • Overall aggregate across all 500 pairs

Outputs:
  results/variant_metric_summary.csv  — machine-readable
  results/variant_metric_summary.txt  — formatted for paper
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

# ── Paths (run from project root) ──────────────────────────────────────────
ROOT        = Path(__file__).parent.parent.parent
PAIRWISE    = ROOT / "results" / "pairwise_scores.csv"
OUT_CSV     = ROOT / "results" / "variant_metric_summary.csv"
OUT_TXT     = ROOT / "results" / "variant_metric_summary.txt"


def fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if v is not None else "N/A"


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None, "n": 0}
    return {
        "mean":   statistics.mean(values),
        "median": statistics.median(values),
        "min":    min(values),
        "max":    max(values),
        "std":    statistics.stdev(values) if len(values) > 1 else 0.0,
        "n":      len(values),
    }


def main():
    print(f"Loading pairwise scores from: {PAIRWISE}")

    domain_data = defaultdict(lambda: {
        "bleu_topic":  [],
        "bleu_emph":   [],
        "bert_topic":  [],
        "bert_emph":   [],
        "comet_topic": [],
        "comet_emph":  [],
    })

    all_data = {k: [] for k in ["bleu_topic", "bleu_emph", "bert_topic", "bert_emph", "comet_topic", "comet_emph"]}

    with open(PAIRWISE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain = row.get("domain", "unknown").strip() or "unknown"
            for key, col in [
                ("bleu_topic",  "bleu_base_vs_topic"),
                ("bleu_emph",   "bleu_base_vs_emph"),
                ("bert_topic",  "bertscore_base_vs_topic"),
                ("bert_emph",   "bertscore_base_vs_emph"),
                ("comet_topic", "comet_base_vs_topic"),
                ("comet_emph",  "comet_base_vs_emph"),
            ]:
                val = row.get(col, "").strip()
                if val:
                    try:
                        fval = float(val)
                        domain_data[domain][key].append(fval)
                        all_data[key].append(fval)
                    except ValueError:
                        pass

    # Build rows: per domain + overall
    domains_sorted = sorted(domain_data.keys())
    rows = []

    for domain in domains_sorted:
        d = domain_data[domain]
        for variant, keys in [("topic-fronting", ("bleu_topic", "bert_topic", "comet_topic")),
                                ("emphasis-shift", ("bleu_emph",  "bert_emph",  "comet_emph"))]:
            bleu_s  = summarize(d[keys[0]])
            bert_s  = summarize(d[keys[1]])
            comet_s = summarize(d[keys[2]])
            rows.append({
                "domain":          domain,
                "variant":         variant,
                "n":               bleu_s["n"],
                "bleu_mean":       bleu_s["mean"],
                "bleu_std":        bleu_s["std"],
                "bleu_min":        bleu_s["min"],
                "bleu_max":        bleu_s["max"],
                "bertscore_mean":  bert_s["mean"],
                "bertscore_std":   bert_s["std"],
                "bertscore_min":   bert_s["min"],
                "bertscore_max":   bert_s["max"],
                "comet_mean":      comet_s["mean"],
                "comet_std":       comet_s["std"],
            })

    # Overall row
    for variant, keys in [("topic-fronting", ("bleu_topic", "bert_topic", "comet_topic")),
                            ("emphasis-shift", ("bleu_emph",  "bert_emph",  "comet_emph"))]:
        bleu_s  = summarize(all_data[keys[0]])
        bert_s  = summarize(all_data[keys[1]])
        comet_s = summarize(all_data[keys[2]])
        rows.append({
            "domain":          "OVERALL",
            "variant":         variant,
            "n":               bleu_s["n"],
            "bleu_mean":       bleu_s["mean"],
            "bleu_std":        bleu_s["std"],
            "bleu_min":        bleu_s["min"],
            "bleu_max":        bleu_s["max"],
            "bertscore_mean":  bert_s["mean"],
            "bertscore_std":   bert_s["std"],
            "bertscore_min":   bert_s["min"],
            "bertscore_max":   bert_s["max"],
            "comet_mean":      comet_s["mean"],
            "comet_std":       comet_s["std"],
        })

    # Print summary
    print("\n" + "=" * 90)
    print("VARIANT METRIC SUMMARY: Base ↔ Variant BLEU / BERTScore / COMET (all 500 pairs)")
    print("=" * 90)
    header = f"{'Domain':<14} {'Variant':<18} {'n':>4}  {'BLEU mean':>10} {'±std':>6}  {'BERTScore':>10} {'±std':>6}  {'COMET':>8}"
    print(header)
    print("-" * 90)
    for r in rows:
        comet_str  = fmt(r["comet_mean"], 4) if r["comet_mean"] is not None else "  N/A  "
        comet_std  = fmt(r["comet_std"],  4) if r["comet_std"]  is not None else "  N/A"
        print(
            f"{r['domain']:<14} {r['variant']:<18} {r['n']:>4}  "
            f"{fmt(r['bleu_mean'], 4):>10} {fmt(r['bleu_std'], 4):>6}  "
            f"{fmt(r['bertscore_mean'], 4):>10} {fmt(r['bertscore_std'], 4):>6}  "
            f"{comet_str:>8}"
        )
    print("=" * 90)

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["domain", "variant", "n",
                  "bleu_mean", "bleu_std", "bleu_min", "bleu_max",
                  "bertscore_mean", "bertscore_std", "bertscore_min", "bertscore_max",
                  "comet_mean", "comet_std"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ Saved: {OUT_CSV}")

    # Save TXT
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("VARIANT METRIC SUMMARY: Base ↔ Variant BLEU / BERTScore (all 500 pairs)\n")
        f.write("=" * 90 + "\n\n")
        f.write("Each pair is semantically equivalent (same utterance, different word order).\n")
        f.write("Low BLEU between base and syntactic variants demonstrates the thesis:\n")
        f.write("surface-level metrics fail to capture order-invariant semantics.\n\n")
        f.write(header + "\n")
        f.write("-" * 90 + "\n")
        for r in rows:
            comet_str = fmt(r["comet_mean"], 4) if r["comet_mean"] is not None else "  N/A  "
            f.write(
                f"{r['domain']:<14} {r['variant']:<18} {r['n']:>4}  "
                f"{fmt(r['bleu_mean'], 4):>10} {fmt(r['bleu_std'], 4):>6}  "
                f"{fmt(r['bertscore_mean'], 4):>10} {fmt(r['bertscore_std'], 4):>6}  "
                f"{comet_str:>8}\n"
            )
        f.write("\n" + "=" * 90 + "\n")
    print(f"✓ Saved: {OUT_TXT}")


if __name__ == "__main__":
    main()
