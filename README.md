# TriSwitch-Hinglish Pipeline

Reproducible evaluation pipeline for Hinglish meaning-preserving variants.

## What this pipeline produces

From raw JSON records (`base`, `variant_topic_fronting`, `variant_emphasis_shift`, metadata), it creates:

- Validated datasets and annotation template
- Pairwise experimental table
- Structural feature table
- Metric score table (BLEU + BERTScore in v1)
- Summary statistics table
- Sanity logs

## Project layout

```
TriSwitch-Hinglish/
├── data/
│   ├── raw/
│   │   └── triswitch_hinglish_500.json
│   ├── interim/
│   │   ├── cleaned_dataset.csv
│   │   ├── validated_dataset.csv
│   │   ├── pairwise_dataset.csv
│   │   └── pairwise_dataset_with_features.csv
│   └── processed/
│       ├── human_validation.csv
│       ├── metric_scores.csv
│       └── metric_summary.csv
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── logs/
│       └── dataset_sanity.txt
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── load_dataset.py
│   ├── validate_dataset.py
│   ├── build_pairwise_dataset.py
│   ├── feature_engineering.py
│   ├── run_metrics.py
│   ├── run_pipeline.py
│   └── metrics/
│       ├── bleu_metric.py
│       └── bertscore_metric.py
├── notebooks/
│   ├── 01_dataset_sanity_check.ipynb
│   ├── 02_metric_distributions.ipynb
│   └── 03_figures_for_paper.ipynb
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Run end-to-end (v1)

```bash
python src/run_pipeline.py
```

This runs:
1. `load_dataset.py` (cleaning + sanity checks)
2. `validate_dataset.py` logic (annotation template + validated dataset)
3. `build_pairwise_dataset.py`
4. `feature_engineering.py`
5. `run_metrics.py` logic (BLEU + BERTScore)

## Run individual scripts

```bash
python src/load_dataset.py
python src/validate_dataset.py
python src/build_pairwise_dataset.py
python src/feature_engineering.py
python src/run_metrics.py --metrics bleu,bertscore
```

## Design decisions

- `pattern` is treated as primary gold annotation.
- `switch_points` are enforced as list-of-int semantics.
- Text normalization uses whitespace trimming only.
- Outputs are saved as CSV for annotation/debugging/reproducibility.

## Next additions (planned)

- `src/analyze_metrics.py` and `src/plotting.py`
- BLEURT, COMET, and LLM-as-a-judge modules
- statistical tests and paper-ready figures
