import json
import sacrebleu
import numpy as np

# Tải file tham chiếu đã dịch
vi_ref_file = "../task2_vlsp/data/filtered_raw_data/test.vi"
en_ref_file = "../task2_vlsp/data/filtered_raw_data/test.en"

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

# Tải file model dịch
jsonl_file = "../task2_vlsp/tests/translation.jsonl"
predictions = []

with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        predictions.append(data.get('text', "").lower().strip())

if len(predictions) != len(references):
    raise ValueError(f"Lệch độ dài! Preds: {len(predictions)} - Refs: {len(references)}")

# Tính BLEU score
sacrebleu_score = sacrebleu.corpus_bleu(predictions, [references])

print(sacrebleu_score.score)