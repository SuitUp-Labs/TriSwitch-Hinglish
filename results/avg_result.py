import json
import csv

# Input file from previous script
INPUT_JSON = "scripts\\evaluation\\metric_results.json"
OUTPUT_SUMMARY_JSON = "results\\average_metrics.json"
OUTPUT_SUMMARY_CSV = "results\\average_metrics.csv"
OUTPUT_SUMMARY_TXT = "results\\average_metrics.txt"

print(f"Loading results from {INPUT_JSON}...")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    results = json.load(f)

print(f"Loaded {len(results)} evaluation results\n")

# Group results by model
model_scores = {}

for result in results:
    model = result["model"]
    
    if model not in model_scores:
        model_scores[model] = {
            "BLEU": [],
            "chrF": [],
            "BERTScore_F1": []
        }
    
    model_scores[model]["BLEU"].append(result["BLEU"])
    model_scores[model]["chrF"].append(result["chrF"])
    model_scores[model]["BERTScore_F1"].append(result["BERTScore_F1"])

# Calculate averages
summary = []

for model, scores in model_scores.items():
    avg_bleu = sum(scores["BLEU"]) / len(scores["BLEU"])
    avg_chrf = sum(scores["chrF"]) / len(scores["chrF"])
    avg_bertscore = sum(scores["BERTScore_F1"]) / len(scores["BERTScore_F1"])
    
    summary.append({
        "model": model,
        "num_evaluations": len(scores["BLEU"]),
        "avg_BLEU": avg_bleu,
        "avg_chrF": avg_chrf,
        "avg_BERTScore_F1": avg_bertscore
    })
    
    print(f"Model: {model}")
    print(f"  Number of evaluations: {len(scores['BLEU'])}")
    print(f"  Average BLEU: {avg_bleu:.4f}")
    print(f"  Average chrF: {avg_chrf:.4f}")
    print(f"  Average BERTScore F1: {avg_bertscore:.4f}")
    print()

# Save summary to JSON
print(f"Saving summary to {OUTPUT_SUMMARY_JSON}...")
with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as json_out:
    json.dump(summary, json_out, indent=2, ensure_ascii=False)

# Save summary to CSV
print(f"Saving summary to {OUTPUT_SUMMARY_CSV}...")
with open(OUTPUT_SUMMARY_CSV, "w", newline="", encoding="utf-8") as csv_out:
    writer = csv.DictWriter(
        csv_out,
        fieldnames=["model", "num_evaluations", "avg_BLEU", "avg_chrF", "avg_BERTScore_F1"]
    )
    writer.writeheader()
    writer.writerows(summary)

# Save summary to TXT
print(f"Saving summary to {OUTPUT_SUMMARY_TXT}...")
with open(OUTPUT_SUMMARY_TXT, "w", encoding="utf-8") as txt_out:
    txt_out.write("=" * 60 + "\n")
    txt_out.write("TRANSLATION METRICS SUMMARY\n")
    txt_out.write("=" * 60 + "\n\n")
    
    for entry in summary:
        txt_out.write(f"Model: {entry['model'].upper()}\n")
        txt_out.write(f"  Number of evaluations: {entry['num_evaluations']}\n")
        txt_out.write(f"  Average BLEU:          {entry['avg_BLEU']:.4f}\n")
        txt_out.write(f"  Average chrF:          {entry['avg_chrF']:.4f}\n")
        txt_out.write(f"  Average BERTScore F1:  {entry['avg_BERTScore_F1']:.4f}\n")
        txt_out.write("\n" + "-" * 60 + "\n\n")

print(f"\nComplete! Saved average metrics to:")
print(f"  - {OUTPUT_SUMMARY_JSON}")
print(f"  - {OUTPUT_SUMMARY_CSV}")
print(f"  - {OUTPUT_SUMMARY_TXT}")