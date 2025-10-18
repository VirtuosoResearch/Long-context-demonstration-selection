#!/usr/bin/env python
"""
Minimal SFT LoRA/QLoRA trainer for **HumanevalPack (70/30 split)**
Restricted to: **Qwen-Coder**, **CodeLlama**, **CodeGemma** families.

- Single batch size flag: `--batch_size` (used for train & eval)
- QLoRA on by default (bitsandbytes); graceful fallback to fp16/bf16 LoRA
- Saves **ONLY** LoRA adapter `state_dict` to `<output_dir>/adapter_state_dict.pt`
- Transformers v5-ready (`dtype`, `eval_strategy`, `processing_class`)

Usage
-----
python sft_min_qlora.py \
  --model_name Qwen/Qwen2.5-Coder-1.5B \
  --humaneval_language python \
  --output_dir ./out/qwen-coder-qlora \
  --batch_size 8 --grad_accum 2 --epochs 3 --lr 2e-4
"""
from __future__ import annotations
import argparse
import os
import warnings
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

ALLOWED_FAMILIES = ("Qwen2.5-Coder", "Qwen-Coder", "CodeLlama", "CodeGemma")


def b(x: str) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y", "t"}

SYSTEM_PROMPT = "You are a helpful coding assistant."

def check_family(model_name: str):
    if not any(f in model_name for f in ALLOWED_FAMILIES):
        warnings.warn(
            f"Model '{model_name}' does not look like one of {ALLOWED_FAMILIES}. Proceeding anyway.")


def extract_pair(ex: Dict[str, str]) -> Dict[str, str]:
    prompt = ex.get("prompt", "")
    sol = (
        ex.get("canonical_solution")
        or ex.get("solution")
        or ex.get("completion")
        or ex.get("reference_solution")
        or ""
    )
    return {"prompt": prompt, "response": sol}


def render_sample(ex: Dict[str, str], tokenizer: AutoTokenizer) -> str:
    user = ex.get("prompt", "")
    resp = ex.get("response", "")
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": resp},
        ], tokenize=False)
    return (
        f"<|system|>{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>{user}<|end|>\n"
        f"<|assistant|>{resp}<|end|>"
    )


def make_tokenize_fn(tokenizer: AutoTokenizer, max_len: int):
    def _fn(ex: Dict[str, str]):
        text = render_sample(ex, tokenizer)
        return tokenizer(text, truncation=True, max_length=max_len, padding=False, return_attention_mask=True)
    return _fn


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    check_family(args.model_name)

    # QLoRA config
    bnb_cfg = None
    if args.use_4bit:
        try:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception as e:
            warnings.warn(f"bitsandbytes not available ({e}); running without 4-bit.")
            bnb_cfg = None

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_cfg,
        dtype=dtype,
        device_map="auto",
    )

    # LoRA
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                          bias="none", task_type="CAUSAL_LM", target_modules=target_modules)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Gradient checkpointing safe setup
    model.config.use_cache = False
    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    # Data: HumanevalPack 70/30
    raw = load_dataset("bigcode/humanevalpack", args.humaneval_language)
    ds = raw["test"].shuffle(seed=args.shuffle_seed)
    n = len(ds); n_train = int(0.7 * n)
    ds_tr = ds.select(range(0, n_train)).map(extract_pair)
    ds_ev = ds.select(range(n_train, n)).map(extract_pair)

    tok_fn = make_tokenize_fn(tokenizer, args.max_seq_len)
    ds_tr = ds_tr.map(tok_fn, remove_columns=None)
    ds_ev = ds_ev.map(tok_fn, remove_columns=None)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    # Training
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        bf16=args.bf16,
        gradient_checkpointing=True,
        report_to=["none"],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tr,
        eval_dataset=ds_ev,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save ONLY adapter state_dict
    try:
        sd = model.get_peft_model_state_dict()
    except AttributeError:
        sd = model.state_dict()
    out = os.path.join(args.output_dir, "adapter_state_dict.pt")
    torch.save(sd, out)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter state_dict to: {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal LoRA/QLoRA SFT on HumanevalPack")
    parser.add_argument("--model_name", type=str, required=True,
                   help="Qwen/Qwen2.5-Coder-*, meta-llama/CodeLlama-*, google/codegemma-* or compatible")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--humaneval_language", type=str, default="python")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_seq_len", type=int, default=2048)

    parser.add_argument("--bf16", type=b, default=True)
    parser.add_argument("--use_4bit", type=b, default=True, help="Enable QLoRA (bitsandbytes 4-bit)")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()
    main(args)