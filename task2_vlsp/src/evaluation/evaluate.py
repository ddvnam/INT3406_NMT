import json
import sacrebleu
import numpy as np
import os


base_dir = os.path.dirname(os.path.abspath(__file__))
vi_ref_file = os.path.join(base_dir, "..", "..", "data", "filtered_raw_data", "test.vi")
en_ref_file = os.path.join(base_dir, "..", "..", "data", "filtered_raw_data", "test.en")

# Tải file tham chiếu
with open(vi_ref_file, 'r', encoding='utf-8') as f:
    envi = [line.strip() for line in f if line.strip()]

with open(en_ref_file, 'r', encoding='utf-8') as f:
    vien = [line.strip() for line in f if line.strip()]

all_refs = envi + vien

total_sentences = len(all_refs)
BATCH_SIZE = 3
references = []
for i in range(0, total_sentences, BATCH_SIZE):
    references.append(all_refs[i])

jsonl_file = os.path.join(base_dir, "..", "..", "tests", "qwen - 69M - 1_epoch - finetune_translation.jsonl")
predictions = []

with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        predictions.append(data.get('text', "").lower().strip())

if len(predictions) != len(references):
    raise ValueError(f"Lệch độ dài: Preds: {len(predictions)} - Refs: {len(references)}")

sacrebleu_score = sacrebleu.corpus_bleu(predictions, [references])

print(sacrebleu_score.score)