# TriSwitch-Hinglish

A comprehensive dataset and evaluation framework for Hindi-English code-switched sentences, featuring 500 carefully curated samples with multiple syntactic variants and domain classifications.

## Overview

**TriSwitch-Hinglish** is a research project focused on Hindi-English code-switching, a linguistic phenomenon where speakers alternate between Hindi and English within sentences. This repository provides:

- **500 code-switched sentences** across multiple domains (travel, food, etc.)
- **Multiple syntactic variants** for each sentence (base, topic-fronting, emphasis-shift)
- **Linguistic metadata** including token counts, switch points, and code-switching patterns
- **Evaluation framework** using MarianMT model for translation/generation
- **Automatic metrics** (BLEU and BERTScore) for quality assessment

## Key Features

### Dataset Characteristics

Each entry in the dataset contains:
- **Base sentence**: Original Hindi-English code-switched sentence
- **Syntactic variants**: Topic-fronting and emphasis-shift variations
- **Token analysis**: Counts of Hindi and English tokens
- **Switch points**: Positions where language switches occur
- **Pattern notation**: Language sequence (e.g., "EN-HN-HN-EN")
- **Domain classification**: Categorization by topic (travel, food, etc.)

### Example Entry
```json
{
  "id": 1,
  "base": "train kaha tak jati hai",
  "variant_topic_fronting": "kaha tak jati hai train",
  "variant_emphasis_shift": "jati hai kaha tak train",
  "tokens_hindi": 4,
  "tokens_english": 1,
  "switch_points": [0],
  "domain": "travel",
  "pattern": "EN-HN-HN-HN-HN"
}
```

## Project Structure

```
TriSwitch-Hinglish/
│
├── dataset/
│   └── db.json                          # 500 code-switched sentences with metadata
│
├── metric/
│   ├── BERT_score.py                    # BERTScore evaluation script
│   └── BLEU_score.py                    # BLEU score evaluation script
│
├── final_results/
│   ├── marian_triswitch_outputs.csv     # Model generation outputs
│   ├── bert_score.txt                   # BERTScore results
│   ├── bleu_score.txt                   # BLEU score results
│   └── human_eval.csv                   # Human evaluation data
│
├── run_marian.py                        # Main script for MarianMT inference
├── requirements.txt                     # Python dependencies
├── LICENSE                              # MIT License
└── README.md                            # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SuitUp-Research/TriSwitch-Hinglish.git
   cd TriSwitch-Hinglish
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### 1. Run MarianMT Model Inference

Generate translations/outputs for all dataset sentences:

```bash
python run_marian.py
```

This script:
- Loads the pre-trained MarianMT model (`ar5entum/marianMT_hin_eng_cs`)
- Processes all 500 sentences from the dataset
- Generates outputs with controlled generation parameters
- Saves results to `marian_triswitch_outputs.csv`
- Shows progress every 25 samples

**Output Format:**
```csv
id,input_base,model_output,domain,pattern
1,train kaha tak jati hai,hindi-english code-switched sentence: train हा तक या है,travel,EN-HN-HN-HN-HN
```

#### 2. Calculate BLEU Score

Evaluate translation quality using BLEU metric:

```bash
cd metric
python BLEU_score.py
```

This computes corpus-level BLEU score by comparing model outputs against reference sentences.

#### 3. Calculate BERTScore

Evaluate semantic similarity using BERTScore:

```bash
cd metric
python BERT_score.py
```

This calculates Precision, Recall, and F1 scores using contextual embeddings.

## Dataset Statistics

- **Total Sentences**: 500
- **Domains**: Multiple (travel, food, and more)
- **Languages**: Hindi (Devanagari & Romanized) + English
- **Variants per Sentence**: 3 (base + 2 syntactic variations)
- **Metadata Fields**: 9 per entry

### Domain Distribution

The dataset covers various real-world scenarios:
- **Travel**: Transportation, navigation, tickets
- **Food**: Orders, restaurants, cuisine
- Additional domains as categorized in the dataset

### Code-Switching Patterns

Each sentence is annotated with its code-switching pattern (e.g., "EN-HN-HN-EN-HN"), making it easy to:
- Analyze switching frequency
- Study language alternation patterns
- Filter by specific code-switching structures

## Model Information

### MarianMT Model

- **Model**: `ar5entum/marianMT_hin_eng_cs`
- **Task**: Hindi-English code-switched text generation/translation
- **Framework**: Hugging Face Transformers
- **Generation Parameters**:
  - `max_new_tokens`: 40
  - `do_sample`: False (deterministic)
  - `early_stopping`: True

## Evaluation Metrics

### BLEU Score
Measures n-gram overlap between generated and reference sentences. Higher scores indicate better lexical similarity.

### BERTScore
Leverages contextual embeddings to evaluate semantic similarity. Provides:
- **Precision**: How much of the generated text is relevant
- **Recall**: How much of the reference is captured
- **F1**: Harmonic mean of precision and recall

## 📄 License

This project is licensed under the [MIT License](LICENSE).
