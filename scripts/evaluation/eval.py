import json
import evaluate
import csv

# Load metrics
bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")
bertscore_metric = evaluate.load("bertscore")

def compute_core_metrics(hypotheses, references):
    """
    hypotheses: list of strings
    references: list of strings
    """
    # Compute scores
    bleu_score = bleu_metric.compute(predictions=hypotheses, references=[[r] for r in references])
    chrf_score = chrf_metric.compute(predictions=hypotheses, references=[[r] for r in references])
    bertscore_dict = bertscore_metric.compute(predictions=hypotheses, references=references, lang="en")
    
    return {
        "BLEU": bleu_score["bleu"],
        "chrF": chrf_score["score"],
        "BERTScore_F1": sum(bertscore_dict["f1"]) / len(bertscore_dict["f1"])
    }

# File paths
INPUT_JSON = "D:\\Documents\\RESEARCH\\TriSwitch_EACL-26\\TriSwitch-Hinglish\\dataset\\model-outputs\\db_with_all_translations.json"  # Change to your actual filename
OUTPUT_JSON = "scripts\\evaluation\\metric_results.json"
OUTPUT_CSV = "scripts\\evaluation\\metric_results.csv"

# Load JSON data
print(f"Loading data from {INPUT_JSON}...")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} entries")

results = []

# Loop over each entry
for idx, item in enumerate(data, 1):
    print(f"Processing entry {idx}/{len(data)} (ID: {item.get('id', 'N/A')})")
    
    ref = item.get("reference_en", "")
    
    if not ref:
        print(f"  Warning: No reference found for entry {item.get('id', idx)}")
        continue
    
    # Define model keys
    models = {
        "llama": {
            "base": item.get("base_llama_translation", ""),
            "topic": item.get("topic_fronting_llama_translation", ""),
            "emphasis": item.get("emphasis_shift_llama_translation", "")
        },
        "gemma": {
            "base": item.get("base_gemma_translation", ""),
            "topic": item.get("topic_fronting_gemma_translation", ""),
            "emphasis": item.get("emphasis_shift_gemma_translation", "")
        }
    }
    
    for model_name, outputs in models.items():
        # Check if any translation is missing
        if not all([outputs["base"], outputs["topic"], outputs["emphasis"]]):
            print(f"  Warning: Missing translations for {model_name} in entry {item.get('id', idx)}")
            continue
        
        hyps = [
            outputs["base"],
            outputs["topic"],
            outputs["emphasis"]
        ]
        refs = [ref, ref, ref]  # same reference for all three variants
        
        try:
            scores = compute_core_metrics(hyps, refs)
            
            results.append({
                "id": item["id"],
                "model": model_name,
                "hyp_base": outputs["base"],
                "hyp_topic": outputs["topic"],
                "hyp_emphasis": outputs["emphasis"],
                "reference": ref,
                "BLEU": scores["BLEU"],
                "chrF": scores["chrF"],
                "BERTScore_F1": scores["BERTScore_F1"]
            })
        except Exception as e:
            print(f"  Error computing metrics for {model_name} in entry {item.get('id', idx)}: {e}")

# Save JSON
print(f"\nSaving results to {OUTPUT_JSON}...")
with open(OUTPUT_JSON, "w", encoding="utf-8") as json_out:
    json.dump(results, json_out, indent=2, ensure_ascii=False)

# Save CSV
print(f"Saving results to {OUTPUT_CSV}...")
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_out:
    writer = csv.DictWriter(
        csv_out,
        fieldnames=["id", "model", "hyp_base", "hyp_topic", "hyp_emphasis",
                    "reference", "BLEU", "chrF", "BERTScore_F1"]
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\n✓ Complete! Processed {len(results)} model evaluations.")
print(f"✓ Saved metrics to {OUTPUT_JSON} and {OUTPUT_CSV}")