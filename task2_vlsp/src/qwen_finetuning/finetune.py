from unsloth import FastLanguageModel
import torch
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset, concatenate_datasets


model_name = "unsloth/Qwen3-1.7B"
new_model_name = "../task2_vlsp/models/qwen_lora/qwen3_1.7B_new"

# Load model base của Qwen
model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = model_name,
  max_seq_length = 1024,
  load_in_4bit = True, 
  load_in_8bit = False,
  full_finetuning = False,
)

# Cấu hình các ma trận LORA vào model
model = FastLanguageModel.get_peft_model(
  model,
  r = 64,
  target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
],
  lora_alpha = 128,
  lora_dropout = 0, 
  bias = "none",

  use_gradient_checkpointing = "unsloth",
  random_state = 3407,
  use_rslora = False,
  loftq_config = None,
)

# Cấu hình quá trình training cho model
sft_config = SFTConfig(
    output_dir=new_model_name,

    dataset_num_proc=8,
    packing=True,

    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=8,

    warmup_steps=10,
    num_train_epochs=1,
    learning_rate=2e-4,

    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),

    logging_steps=25,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",

    save_strategy="epoch",
    save_steps=100,
    save_total_limit=1,
    seed=3407,

    dataloader_pin_memory=False,
    report_to="none",
)

def apply_chat_template(example):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=True,              
        add_generation_prompt=False  
    )
    return {"text": text.strip()}

train_ind_envi = load_dataset("json", data_files="../task2_vlsp/data/processed/improved_prompts_ind_train_en2vi.jsonl", split="train")
train_ind_vien = load_dataset("json", data_files="../task2_vlsp/data/processed/improved_prompts_ind_train_vi2en.jsonl", split="train")

train_ind_envi_dataset = train_ind_envi.map(apply_chat_template, remove_columns=["messages"])
train_ind_vien_dataset = train_ind_vien.map(apply_chat_template, remove_columns=["messages"])

full_train_dataset = concatenate_datasets([train_ind_envi_dataset, train_ind_vien_dataset])

# Chia train/test
dataset_split = full_train_dataset.train_test_split(test_size=0.1, seed=3407)
train_dataset = dataset_split["train"]
test_dataset = dataset_split["test"]

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer_stats = trainer.train()

trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)
