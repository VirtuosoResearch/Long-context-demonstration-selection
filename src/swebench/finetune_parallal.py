import os
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    Trainer, default_data_collator
)
from peft import LoraConfig, get_peft_model, TaskType

from dataclasses import dataclass
from typing import Optional, Dict, List
import torch
import argparse
import json

SEED = 42
SAMPLE_SIZE = 300
MAX_LEN = 12000
LR = 6e-4
NUM_EPOCHS = 10
BATCH_SIZE_PER_DEVICE = 1
GRAD_ACCUM = 8
USE_BF16 = True

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

from dataclasses import dataclass
from typing import Optional, Dict, List, Union
from transformers import PreTrainedTokenizerBase

@dataclass
class DataCollatorForCausalLMWithIgnore:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        batch_input_ids = [f["input_ids"].tolist() if isinstance(f["input_ids"], torch.Tensor) else f["input_ids"] for f in features]
        batch_attention_mask = [f["attention_mask"].tolist() if isinstance(f["attention_mask"], torch.Tensor) else f["attention_mask"] for f in features]
        batch_labels = [f["labels"].tolist() if isinstance(f["labels"], torch.Tensor) else f["labels"] for f in features]

        max_len = max(len(ids) for ids in batch_input_ids)
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            m = int(self.pad_to_multiple_of)
            max_len = ((max_len + m - 1) // m) * m

        def pad_seq(seq: List[int], value: int, length: int) -> List[int]:
            n = len(seq)
            if n < length:
                return seq + [value] * (length - n)
            return seq[:length]

        input_ids = [pad_seq(ids, pad_id, max_len) for ids in batch_input_ids]
        attention_mask = [pad_seq(mask, 0, max_len) for mask in batch_attention_mask]
        labels = [pad_seq(lab, -100, max_len) for lab in batch_labels]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def format_example(ex):
    return {"prompt": ex["text"], "patch": ex["patch"]}

def load_and_prepare_dataset(dataset, split, category, sample_size):
    ds_all = load_dataset(dataset)
    ds = ds_all[split]
    with open(f"./classified/{dataset.split("/")[1]}_{split}.json", "r", encoding="utf-8") as f:
        category_data = json.load(f)
    cat_ids = set(category_data.get(category, []))
    ds = ds.filter(lambda ex: ex["instance_id"] in cat_ids)
    print(f"dataset size: {len(ds)}")
    if len(ds)>sample_size:
        ds = ds.select(range(sample_size))
    ds = ds.map(format_example, remove_columns=ds.column_names)
    return ds

def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_memory = {0: "45GiB", 1: "45GiB", "cpu": "20GiB"} 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa", 
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES
    )
    model = get_peft_model(model, lora_cfg)
    return model, tokenizer

def _truncate_pair_with_mask(enc_prompt, enc_patch, max_len):
    p_ids = enc_prompt["input_ids"]
    t_ids = enc_patch["input_ids"]
    p_mask = enc_prompt["attention_mask"]
    t_mask = enc_patch["attention_mask"]

    if len(p_ids) >= max_len:
        p_ids = p_ids[:max_len]
        p_mask = p_mask[:max_len]
        t_ids = []
        t_mask = []
    else:
        remain = max_len - len(p_ids)
        t_ids = t_ids[:remain]
        t_mask = t_mask[:remain]

    ids = p_ids + t_ids
    mask = p_mask + t_mask
    labels = [-100] * len(p_ids) + t_ids
    return ids, mask, labels

def tokenize_fn(examples, tokenizer):
    prompts = examples["prompt"]
    patches = examples["patch"]
    input_ids, attention_mask, labels = [], [], []

    for pmt, ptc in zip(prompts, patches):
        enc_p = tokenizer(pmt, add_special_tokens=False, truncation=True, max_length=MAX_LEN)
        enc_t = tokenizer(ptc, add_special_tokens=False, truncation=True, max_length=MAX_LEN)
        ids, mask, lab = _truncate_pair_with_mask(enc_p, enc_t, MAX_LEN)
        # print(tokenizer.decode(ids, attention_mask=mask))
        # print(lab)
        # exit(0)
        input_ids.append(ids)
        attention_mask.append(mask)
        labels.append(lab)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main(args):
    set_seed(SEED)
    OUTPUT_DIR = os.path.join(args.output, args.model.split("/")[1]+f"{args.category}_lora")
    MERGED_DIR = os.path.join(args.output, args.model.split("/")[1]+f"{args.category}_lora_merged")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MERGED_DIR, exist_ok=True)

    ds = load_and_prepare_dataset(args.dataset, args.split, args.category, SAMPLE_SIZE)
    model, tokenizer = build_model_and_tokenizer(args.model)

    tokenized = ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True, remove_columns=["prompt","patch"])

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=USE_BF16,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        gradient_checkpointing=True,
        group_by_length=True,
        dataloader_pin_memory=False,
    )


    collator = DataCollatorForCausalLMWithIgnore(tokenizer, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    trainer.train()

    merged = trainer.model.merge_and_unload()
    if merged.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        merged.config.pad_token_id = tokenizer.pad_token_id
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"Adapter saved to: {OUTPUT_DIR}")
    print(f"Merged full model saved to: {MERGED_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_bm25_13K")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--model", type=str, default="princeton-nlp/SWE-Llama-7b")
    parser.add_argument("--output", type=str, default="./models")
    parser.add_argument("--category", type=str, default="bugfix")
    args = parser.parse_args()
    main(args)
