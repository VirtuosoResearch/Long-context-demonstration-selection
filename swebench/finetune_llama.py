import os
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

BASE_MODEL = "princeton-nlp/SWE-Llama-7b"
OUTPUT_DIR = "./outputs/swe-llama-7b-lora"
MERGED_DIR = "./outputs/swe-llama-7b-lora-merged"
SEED = 42
SAMPLE_SIZE = 200  
MAX_LEN = 4096   
LR = 2e-4
NUM_EPOCHS = 1
BATCH_SIZE_PER_DEVICE = 1
GRAD_ACCUM = 16
USE_BF16 = True

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    bnb_4bit_use_double_quant=True,
)

INSTRUCTION = """You are a software engineer. Read the issue and failing tests, then output a git patch that fixes the bug.

<issue>
{problem_statement}
</issue>

<failing_tests>
{fail_to_pass}
</failing_tests>

<repo>
{repo} @ {base_commit}
</repo>

Output format:
BEGIN_PATCH
<unified diff content>
END_PATCH
"""

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_example(ex: Dict) -> Dict:
    fail = "\n".join(ex.get("FAIL_TO_PASS", []) or [])
    prompt = INSTRUCTION.format(
        problem_statement=ex.get("problem_statement", ""),
        fail_to_pass=fail,
        repo=ex.get("repo", ""),
        base_commit=ex.get("base_commit", "")
    )
    patch = ex.get("patch", "").strip()
    text = prompt.strip() + "\n\n***BEGIN_PATCH***\n" + patch + "\n***END_PATCH***"
    return {"text": text}


def load_and_prepare_dataset(sample_size: int, seed: int):
    ds_all = load_dataset("SWE-bench/SWE-bench_Lite")
    split_name = "train" if "train" in ds_all else ("dev" if "dev" in ds_all else "test")
    ds = ds_all[split_name]
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = idx[:sample_size]
    ds = ds.select(idx)
    ds = ds.map(format_example, remove_columns=ds.column_names)
    return ds


def build_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES
    )
    model = get_peft_model(model, lora_cfg)
    return model, tokenizer


def tokenize_fn(examples: Dict, tokenizer: AutoTokenizer):
    out = tokenizer(
        examples["text"],
        max_length=MAX_LEN,
        truncation=True,
        padding=False
    )
    out["labels"] = out["input_ids"].copy()
    return out


def train_and_save():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ds = load_and_prepare_dataset(SAMPLE_SIZE, SEED)
    model, tokenizer = build_model_and_tokenizer()

    tokenized = ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=USE_BF16,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized,
        args=args,
        packing=False,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    os.makedirs(MERGED_DIR, exist_ok=True)
    print("Merging LoRA weights into base model...")
    merged = trainer.model.merge_and_unload()
    # in case pad_token
    if merged.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        merged.config.pad_token_id = tokenizer.pad_token_id
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"Adapter saved to: {OUTPUT_DIR}")
    print(f"Merged full model saved to: {MERGED_DIR}")


if __name__ == "__main__":
    train_and_save()
