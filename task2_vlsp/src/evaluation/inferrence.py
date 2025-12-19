from unsloth import FastLanguageModel
from datasets import load_dataset
import time
from tqdm import tqdm
import re
import pandas as pd
import json
import os

# Xử lý output của model để lấy đoạn dịch
def extract_output(text: str) -> str:
    pattern = r"<\|im_start\|>assistant\s*(.*?)\s*<\|im_end\|>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        cleaned_text = match.group(1).strip()
        cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL).strip()
    else:
        cleaned_text = text.strip().split("\n")[-1]
    cleaned_text = re.sub(r'<\|.*?\|>', '', cleaned_text, flags=re.DOTALL).strip()
    return cleaned_text

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.join(base_dir, "..", "..", "models", "qwen_lora", "qwen3_1.7B_new")
    max_seq_length = 1024
    dtype = None

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    # Load tập test
    test_envi = load_dataset("json", data_files=os.path.join(base_dir, "..", "..", "data", "processed", "improved_prompts_ind_test_en2vi.jsonl"), split="train")
    prompts_envi = [example["messages"] for example in test_envi]
    test_vien = load_dataset("json", data_files=os.path.join(base_dir, "..", "..", "data", "processed", "improved_prompts_ind_test_vi2en.jsonl"), split="train")
    prompts_vien = [example["messages"] for example in test_vien]

    all_prompts = prompts_envi + prompts_vien

    start_time = time.time()
    results = []
    total_sentences = len(all_prompts)

    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Infer model với số lượng 1/3 tập test
    BATCH_SIZE = 3
    for i in tqdm(range(0, total_sentences, BATCH_SIZE), desc="Inference"):
            inputs = tokenizer.apply_chat_template(
                all_prompts[i],
                return_tensors="pt", 
                padding=True,  
                truncation=True, 
                max_length=max_seq_length
            ).to("cuda")
            
            outputs = model.generate(
                inputs,
                max_new_tokens=256,
                use_cache=True,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            response = extract_output(tokenizer.decode(outputs[0], skip_special_tokens=True))
            results.append(response)
        
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Thời gian infer: {runtime}")

    # Lưu file dịch của model
    output_jsonl_filename = os.path.join(base_dir, "..", "..", "tests", "translation.jsonl")

    try:
        with open(output_jsonl_filename, 'w', encoding='utf-8') as f:
            for response_text in results:
                data_entry = {
                    "text": response_text
                }
                
                json_line = json.dumps(data_entry, ensure_ascii=False)
                f.write(json_line + '\n')

        print(f"Lưu thành công {len(results)} dòng.")

    except Exception as e:
        print(f"Lỗi khi lưu file JSONL: {e}")

if __name__ == '__main__':
    main()