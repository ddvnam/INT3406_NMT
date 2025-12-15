from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy

# Đổi từ điển tiếng anh sang tiếng việt
def convert_dic(json):
    input_dict = {k.strip(): sorted(set(v if isinstance(v, list) else [v.strip()])) for k, v in json.items() if k.strip()}

    inverted = {}
    for k, values in input_dict.items():
        for v in values:
            inverted.setdefault(v.strip(), set()).add(k.strip())
    vi2en_dict = {k: sorted(v) for k, v in inverted.items()}
    return vi2en_dict

# Lưu dict vào file json
def write_to_json_file(existing_data, new_data, file_path):
    existing_data.update(new_data)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
        print(f"Đã hợp nhất và ghi dữ liệu mới vào file {file_path}")

# Xác định các cụm danh từ có thể liên quan đến chuyên ngành rồi dịch để tạo từ điển tham chiếu
class Dictionary_Augmentation(torch.nn.Module):
    def __init__(self, model_name: str, in_lang: str):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.language = in_lang

    def forward(self, en_ind_corpus: list, batch_size: int = 32) -> dict:
        
        unique_phrases = set()
        
        # Xác định cụm danh từ bằng en_core_web_sm
        for doc in NLP.pipe(en_ind_corpus, batch_size=2000, disable=["ner", "lemmatizer"]):
            for chunk in doc.noun_chunks:
                # Lọc bỏ các cụm từ chứa số
                if not any(token.pos_ == 'NUM' for token in chunk):
                    unique_phrases.add(chunk.text.strip())


        # Dịch cụm danh từ vừa xác định qua model dịch VietAI/envit5-translation
        final_mapping = {}
        sorted_phrases = sorted(list(unique_phrases))
        
        
        for i in range(0, len(sorted_phrases), batch_size):
            batch_phrases = sorted_phrases[i : i + batch_size]
            
           
            input_texts = [f"{self.language}: {phrase}" for phrase in batch_phrases]
            
            inputs = self.tokenizer(
                input_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                encoded = self.model.generate(**inputs, max_length=512)
            
            outputs = self.tokenizer.batch_decode(encoded, skip_special_tokens=True)
            
            for en_phrase, vi_output in zip(batch_phrases, outputs):
                clean_vi = vi_output[4:] if len(vi_output) > 4 else vi_output
                
                if en_phrase not in final_mapping:
                    final_mapping[en_phrase] = set()
                final_mapping[en_phrase].add(clean_vi.strip())

            if (i // batch_size + 1) % 100 == 0:
                self.save_dictionary(final_mapping)

        self.save_dictionary(final_mapping)
        
        return {k: list(v) for k, v in final_mapping.items()}

    def save_dictionary(self, current_mapping):
        to_write = {k: list(v) for k, v in current_mapping.items()}
        file_path = f"../task2_vlsp/data/medical_terms/final_dic_{self.language}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(to_write, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    NLP = spacy.load("en_core_web_sm")
    with open(r"..\task2_vlsp\data\raw\train.en", 'r', encoding='utf-8') as en_ind_corpus:
        en_list = []
        for i, line in enumerate(en_ind_corpus):
                en_list.append(line.strip())
    aug_dic_method = Dictionary_Augmentation("VietAI/envit5-translation", in_lang="en")
    final_mapping = aug_dic_method(en_list)
    write_to_json_file(final_mapping, {}, r"..\task2_vlsp\data\medical_terms\final_dic_en.json")
    write_to_json_file(convert_dic(final_mapping), {}, r"..\task2_vlsp\data\medical_terms\final_dic_vi.json")