"""
Confidence Intervals for Human Evaluation PASS Rate
====================================================
Addresses reviewer feedback: "Report confidence intervals given the
small 100-sample human eval set."

Computes:
  • PASS rate (SP=1 AND CSV=1) over the 100 annotated samples
  • 95% Wilson score confidence interval (robust for proportions)
  • 95% Normal approximation CI (shown for comparison)
  • Per-failure-type breakdown

Output:
  results/confidence_intervals.txt
"""

import csv
import math
from collections import Counter
from pathlib import Path

# ── Paths (run from project root) ──────────────────────────────────────────
ROOT           = Path(__file__).parent.parent.parent
HUMAN_EVAL_CSV = ROOT / "old" / "final_results" / "human_eval.csv"
OUT_TXT        = ROOT / "results" / "confidence_intervals.txt"


def wilson_ci(n_success: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.
    More accurate than normal approximation for small n.
    """
    p_hat = n_success / n
    centre = (p_hat + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = (z / (1 + z**2 / n)) * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return max(0.0, centre - margin), min(1.0, centre + margin)


def normal_ci(n_success: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Normal approximation CI (Wald interval)."""
    p_hat = n_success / n
    se = math.sqrt(p_hat * (1 - p_hat) / n)
    return max(0.0, p_hat - z * se), min(1.0, p_hat + z * se)


def main():
    print(f"Loading human evaluation from: {HUMAN_EVAL_CSV}")

    rows = []
    fail_types = []
    with open(HUMAN_EVAL_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp         = int(row["SP (Semantic Preservation)"])
            csv_valid  = int(row["CSV (Code-Switch Validity)"])
            fail_type  = row.get("FAIL_TYPE", "-").strip()
            is_pass    = 1 if (sp == 1 and csv_valid == 1) else 0
            rows.append({
                "id":       int(row["id"]),
                "SP":       sp,
                "CSV":      csv_valid,
                "PASS":     is_pass,
                "FAIL_TYPE": fail_type,
            })
            if not is_pass:
                fail_types.append(fail_type if fail_type else "?")

    n = len(rows)
    n_pass = sum(r["PASS"] for r in rows)
    n_fail = n - n_pass
    p_hat = n_pass / n

    w_lo, w_hi = wilson_ci(n_pass, n)
    a_lo, a_hi = normal_ci(n_pass, n)

    # SP-only and CSV-only rates
    n_sp  = sum(r["SP"]  for r in rows)
    n_csv = sum(r["CSV"] for r in rows)

    # Failure type breakdown
    fail_counter = Counter(fail_types)

    # ── Print ──────────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append("HUMAN EVALUATION: PASS RATE & 95% CONFIDENCE INTERVALS")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  Total annotated samples  : {n}")
    lines.append(f"  PASS (SP=1 AND CSV=1)    : {n_pass}  ({p_hat*100:.1f}%)")
    lines.append(f"  FAIL                     : {n_fail}  ({n_fail/n*100:.1f}%)")
    lines.append("")
    lines.append(f"  SP (Semantic Preserv.)   : {n_sp}/100  ({n_sp}%)")
    lines.append(f"  CSV (Code-Switch Valid.) : {n_csv}/100  ({n_csv}%)")
    lines.append("")
    lines.append("  95% Wilson score CI:")
    lines.append(f"    PASS rate = {p_hat:.3f}  [{w_lo:.3f}, {w_hi:.3f}]")
    lines.append(f"    ≈ {p_hat:.2f} ± {(w_hi - w_lo)/2:.3f}  (half-width)")
    lines.append("")
    lines.append("  95% Normal approximation CI (Wald):")
    lines.append(f"    PASS rate = {p_hat:.3f}  [{a_lo:.3f}, {a_hi:.3f}]")
    lines.append(f"    ≈ {p_hat:.2f} ± {(a_hi - a_lo)/2:.3f}  (half-width)")
    lines.append("")
    lines.append("  Failure Type Breakdown:")
    lines.append(f"  {'Type':<8} {'Count':>6}  {'Desc'}")
    lines.append("  " + "-" * 45)
    type_desc = {
        "OT":  "Output garbled / un-Hinglish characters",
        "SD":  "Semantic drift (partial meaning change)",
        "ID":  "Incomplete / truncated output",
        "MO":  "Mode/register shift (full Hindi output)",
        "UN":  "Unintelligible / unreadable output",
        "-":   "Pass (no failure)",
        "?":   "Unknown",
    }
    for ft, cnt in sorted(fail_counter.items(), key=lambda x: -x[1]):
        desc = type_desc.get(ft, "")
        lines.append(f"  {ft:<8} {cnt:>6}  {desc}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Recommended paper phrasing (Section 3.2 / 4.1):")
    lines.append(f"  \"Of the 100 manually evaluated samples, {n_pass} outputs")
    lines.append(f"  ({p_hat*100:.0f}%) were judged as PASS (preserving semantics")
    lines.append(f"  and code-switch validity). The 95% Wilson confidence")
    lines.append(f"  interval is [{w_lo:.2f}, {w_hi:.2f}].\"")

    output = "\n".join(lines)
    print("\n" + output)

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"\n✓ Saved: {OUT_TXT}")


if __name__ == "__main__":
    main()
