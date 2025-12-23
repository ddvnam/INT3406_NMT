# Machine Translation Project: Transformer & Medical LLM Adaptation

**Course:** Natural Language Processing (NLP)
**Topic:** English-Vietnamese Machine Translation: Building Transformer from Scratch & Fine-tuning Medical LLM.

## Overview

This project addresses two core challenges in English-Vietnamese machine translation:

1. **Task 1:** Training a Transformer model from scratch, integrating modern components such as RMSNorm, Grouped-Query Attention (GQA), SwiGLU, and Rotary Positional Embeddings (RoPE).
2. **Task 2:** Fine-tuning the Qwen Large Language Model (LLM) using LoRA combined with Glossary-driven Prompting to optimize performance for the Medical domain (VLSP Medical dataset).

## Key Features

### Task 1: Transformer from Scratch
* Improved Standard Encoder-Decoder Architecture.
* Rotary Positional Embeddings (RoPE) replacing Sinusoidal encoding.
* SwiGLU activation function and RMSNorm pre-normalization.
* Grouped-Query Attention (GQA) for memory optimization.
* Tokenizer: SentencePiece (Joint-BPE).

### Task 2: Medical LLM Fine-tuning
* Base Model: Qwen (Selected for superior Vietnamese language support).
* PEFT Technique: LoRA (Low-Rank Adaptation) applied to all linear layers (All-Linear).
* Glossary-driven Prompting: A mechanism to inject medical terminology context during inference to fix literal translation errors and ensure domain accuracy.
* Specialized data processing pipeline for the VLSP Medical dataset.

## Experimental Results

| Task | Model / Method | Dataset | BLEU Score |
| :--- | :--- | :--- | :---: |
| Task 1 | Transformer (RoPE + SwiGLU) | IWSLT'15 | 34.26 |
| Task 2 | Llama Base (Baseline) | VLSP Medical | 19.58 |
| Task 2 | Qwen Base (Baseline) | VLSP Medical | 24.96 |
| Task 2 | Qwen-LoRA + Glossary | VLSP Medical | 43.21 |

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ddvnam/INT3406_NMT.git
   cd INT3406_NMT
2. Create and activate virtual environment:

- Windows:

```bash
python -m venv nlp_env
nlp_env\Scripts\activate
```
- MacOS / Linux:

```bash
python3 -m venv nlp_env
source nlp_env/bin/activate
```
3. Install dependencies:

```bash
pip install -r requirements.txt
```

# Usage
## Task 1: Training Transformer from Scratch
To process data and train the Transformer model, navigate to the folder and run the scripts in the following order:

```bash

cd task1_transformer

# Step 1: Data Preprocessing (Cleaning and formatting)
python -m scripts.preprocess

# Step 2: Train Tokenizer (SentencePiece BPE)
python -m scripts.train_tokenizer

# Step 3: Train Model
python -m scripts.train_model

# Step 4: Inference (Translate sentences)
python -m scripts.inference

```
