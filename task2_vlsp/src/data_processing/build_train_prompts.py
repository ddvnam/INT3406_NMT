import json
import torch
from typing import Dict, List, Tuple
import sentence_transformers
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


TOP_K = 5
SIM_THRESHOLD = 0.6
BATCH_SIZE = 64

# Load built dictionary 
def load_dict(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        k.strip(): sorted(set(v if isinstance(v, list) else [v.strip()]))
        for k, v in raw.items()
        if k.strip()
    }

# Load model to find similar terms
def load_sbert_model(lang: str) -> SentenceTransformer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "keepitreal/vietnamese-sbert" if lang == "vi" else "all-MiniLM-L6-v2"
    return SentenceTransformer(path).to(device)

# Embed the terms
def build_embeddings(terms: List[str], model: SentenceTransformer) -> Tuple[List[str], torch.Tensor]:
    with torch.no_grad():
        embeddings = model.encode(terms, convert_to_tensor=True, normalize_embeddings=True)
    return terms, embeddings

# Match the noun phrase detected with top k similar terms in dictionary having similarity value higher than a threshold
def get_similar_terms(sentence: str, model: SentenceTransformer, terms: List[str], embeddings: torch.Tensor) -> List[Tuple[str, float]]:
    if not terms:
        return []
    emb = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
    cos_scores = util.cos_sim(emb, embeddings)[0]
    top_k = min(TOP_K, len(terms))
    top_results = torch.topk(cos_scores, k=top_k)
    return [(terms[i], float(s)) for i, s in zip(top_results.indices, top_results.values) if s >= SIM_THRESHOLD]

# Construct reference block for prompt
def format_ref_block(pairs: List[Tuple[str, List[str]]], src_label: str) -> str:
    if not pairs:
        return ""
    lines = []
    for src_term, tgt_terms in pairs:
        quoted_tgts = '; '.join([f'"{t}"' for t in tgt_terms])
        # lines.append(f'- "{src_term}" → <target>{quoted_tgts}</target>')
        lines.append(f'   + "**{src_term}**" → **{quoted_tgts}**')

    return "**Refer to these medical terms for consistency**:\n" + "\n".join(lines) + "\n"

# Construct prompts
def build_chat_messages(src: str, tgt: str, lang: str, matched: List[Tuple[str, float]],
                        en2vi: Dict[str, List[str]], vi2en: Dict[str, List[str]]) -> List[Dict[str, str]]:
    if lang == "en":
        dedup = [(en, en2vi.get(en, [])) for en, _ in matched]
        ref_block = format_ref_block(dedup, "en")
        header = "Translate the following English text into natural and accurate Vietnamese."
    else:
        dedup = [(vi, vi2en.get(vi, [])) for vi, _ in matched]
        ref_block = format_ref_block(dedup, "vi")
        header = "Translate the following Vietnamese text into natural and accurate English."

    user_msg = f"""{header}: {src.strip()}{ref_block}""".strip()

    return [
        {"role": "system", "content": "You are a professional translator. Translate all texts carefully. Do not change or approximate any numbers, dates, laboratory values, or medication dosages. Keep all measurement units unchanged."},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": tgt.strip()}
    ]

# Construct prompts for all sentences in a test file
def process_dataset(src_path: str, tgt_path: str, output_path: str, lang: str,
                    model: SentenceTransformer, terms: List[str], embeddings: torch.Tensor,
                    en2vi: Dict[str, List[str]], vi2en: Dict[str, List[str]]):
   
    total_processed = 0
    
    with open(output_path, "w", encoding="utf-8") as f_init:
        pass 

    # Read file
    with open(src_path, "r", encoding="utf-8") as f_src, open(tgt_path, "r", encoding="utf-8") as f_tgt:
        src_lines = f_src.readlines()
        tgt_lines = f_tgt.readlines()
    
    # Make sure translated file having the same sentences as the raw one
    min_len = min(len(src_lines), len(tgt_lines))
    
    # Process by batch
    for i in tqdm(range(0, min_len, BATCH_SIZE), desc=f'Processing Batches ({lang})'):
            
        batch_src = src_lines[i : i + BATCH_SIZE]
        batch_tgt = tgt_lines[i : i + BATCH_SIZE]
        
        clean_batch_src = []
        clean_batch_tgt = []
        
        for s, t in zip(batch_src, batch_tgt):
            if s.strip() and t.strip():
                clean_batch_src.append(s.strip())
                clean_batch_tgt.append(t.strip())
        
        if not clean_batch_src:
            continue

        # Encode & Search
        with torch.no_grad():
            batch_emb = model.encode(clean_batch_src, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
            cos_scores = util.cos_sim(batch_emb, embeddings)
            top_vals, top_inds = torch.topk(cos_scores, k=min(TOP_K, len(terms)))

        batch_results = []
        for idx, src_text in enumerate(clean_batch_src):
            
            tgt_text = clean_batch_tgt[idx]
            
            cur_indices = top_inds[idx]
            cur_scores = top_vals[idx]
            
            matched = []
            for k_idx, score in zip(cur_indices, cur_scores):
                if score >= SIM_THRESHOLD:
                    matched.append((terms[k_idx], float(score)))
            
            messages = build_chat_messages(src_text, tgt_text, lang, matched, en2vi, vi2en)
            batch_results.append({"messages": messages})

        if batch_results:
            with open(output_path, "a", encoding="utf-8") as f_out:
                for r in batch_results:
                    json.dump(r, f_out, ensure_ascii=False)
                    f_out.write("\n")
            total_processed += len(batch_results)

    print(f"Saved total {total_processed} chat-formatted samples to {output_path}")
            
def main(en_ind_traindata, vi_ind_traindata):
    en2vi = load_dict(r"..\task2_vlsp\data\medical_terms\final_dic_en.json")
    vi2en = load_dict(r"..\task2_vlsp\data\medical_terms\final_dic_vi.json")
    
    model_en = load_sbert_model("en")
    en_terms, en_emb = build_embeddings(sorted(en2vi.keys()), model_en)
    process_dataset(en_ind_traindata, vi_ind_traindata, r"..\task2_vlsp\data\processed\improved_prompts_ind_train_en2vi.jsonl", "en", model_en, en_terms, en_emb, en2vi, vi2en)
   
    model_vi = load_sbert_model("vi")
    vi_terms, vi_emb = build_embeddings(sorted(vi2en.keys()), model_vi)
    process_dataset(vi_ind_traindata, en_ind_traindata, r"..\task2_vlsp\data\processed\improved_prompts_ind_train_vi2en.jsonl", "vi", model_vi, vi_terms, vi_emb, en2vi, vi2en)

main(r"..\task2_vlsp\data\raw\train.en", r"..\task2_vlsp\data\raw\train.vi")
