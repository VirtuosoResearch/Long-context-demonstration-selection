from __future__ import annotations
import argparse
import os
import warnings
from typing import Dict, Optional, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from huggingface_hub import list_repo_files

SYSTEM_PROMPT = "You are a helpful coding assistant."


def b(x: str) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y", "t"}


def cpft_to_pair_row(ex: Dict[str, str]) -> Dict[str, str]:
    instr = (ex.get("subject") or ex.get("message") or "").strip()
    old = (ex.get("old_contents") or "").strip()
    new = (ex.get("new_contents") or "").strip()

    prompt = ""
    response = ""
    if instr and old and new:
        prompt = (
            f"[INSTRUCTION]\n{instr}\n\n"
            f"[BEFORE]\n{old}\n\n"
            f"[REQ]\nApply the instruction to produce the updated file."
        )
        response = new
    return {"prompt": prompt, "response": response}


def sample_has_pair(ex: Dict[str, str]) -> bool:
    return bool(ex.get("prompt")) and bool(ex.get("response"))


def render_sample(ex: Dict[str, str], tok: AutoTokenizer) -> str:
    user = ex["prompt"]
    resp = ex["response"]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": resp},
            ],
            tokenize=False,
        )
    return (
        f"<|system|>{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>{user}<|end|>\n"
        f"<|assistant|>{resp}<|end|>"
    )


def make_tokenize_fn(tok: AutoTokenizer, max_len: int):
    def _fn(ex: Dict[str, str]):
        text = render_sample(ex, tok)
        return tok(text, truncation=True, max_length=max_len, padding=False, return_attention_mask=True)
    return _fn


class SimplePacker:
    def __init__(self, tokenizer: AutoTokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len
        self.eos = tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, List[int]]]):
        ids: List[int] = []
        for f in features:
            ii = f["input_ids"]
            ids.extend(ii)
            if not ids or ids[-1] != self.eos:
                ids.append(self.eos)

        blocks = []
        for i in range(0, len(ids), self.max_len):
            chunk = ids[i : i + self.max_len]
            if len(chunk) < self.max_len:
                break
            blocks.append(chunk)

        if not blocks:
            maxl = max(len(f["input_ids"]) for f in features)
            input_ids = torch.full((len(features), maxl), self.tok.pad_token_id, dtype=torch.long)
            attn = torch.zeros_like(input_ids)
            labels = torch.full_like(input_ids, -100)
            for i, f in enumerate(features):
                x = torch.tensor(f["input_ids"], dtype=torch.long)
                n = len(x)
                input_ids[i, :n] = x
                attn[i, :n] = 1
                labels[i, :n] = x 
            return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

        input_ids = torch.tensor(blocks, dtype=torch.long)
        attn = torch.ones_like(input_ids)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

# Datasets v3 parquet
def _list_parquet_files(dataset_id: str, subset: str, revision: str) -> List[str]:
    files = list_repo_files(repo_id=dataset_id, repo_type="dataset", revision=revision)
    prefix = f"{subset}/"
    hits = [f for f in files if f.startswith(prefix) and f.endswith(".parquet")]
    return [f"hf://datasets/{dataset_id}@{revision}/{p}" for p in hits]


def load_commitpackft_subset_v3(subset: str):
    dataset_id = "bigcode/commitpackft"

    files = _list_parquet_files(dataset_id, subset, revision="refs/convert/parquet")
    if files:
        return load_dataset("parquet", data_files={"train": files}, split="train")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    quant_cfg = None
    if args.use_4bit:
        try:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception as e:
            warnings.warn(f"bitsandbytes not available ({e}); running without 4-bit.")
            quant_cfg = None

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_cfg,
        device_map="auto",
    )
    base.config.pad_token_id = tok.pad_token_id
    base.config.eos_token_id = tok.eos_token_id
    if getattr(tok, "bos_token_id", None) is not None:
        base.config.bos_token_id = tok.bos_token_id
    if getattr(base, "generation_config", None) is not None:
        base.generation_config.pad_token_id = tok.pad_token_id
        base.generation_config.eos_token_id = tok.eos_token_id

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    model.config.use_cache = False
    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    print(args.cp_subset)
    ds_all = load_commitpackft_subset_v3(args.cp_subset)

    ds_all = ds_all.map(cpft_to_pair_row, remove_columns=None)
    ds_all = ds_all.filter(sample_has_pair)

    if args.max_samples > 0 and len(ds_all) > args.max_samples:
        ds_all = ds_all.shuffle(seed=args.shuffle_seed).select(range(args.max_samples))

    # 70/30 split
    ds_all = ds_all.shuffle(seed=args.shuffle_seed)
    n = len(ds_all)
    print(f"total data points: {n}")
    n_train = int(0.7 * n)
    ds_tr_raw = ds_all.select(range(0, n_train))
    ds_ev_raw = ds_all.select(range(n_train, n))

    tok_fn = make_tokenize_fn(tok, args.max_seq_len)
    ds_tr = ds_tr_raw.map(tok_fn, remove_columns=None)
    ds_ev = ds_ev_raw.map(tok_fn, remove_columns=None)

    ds_tr = ds_tr.filter(lambda x: 8 <= len(x["input_ids"]) <= args.max_seq_len)
    ds_ev = ds_ev.filter(lambda x: 8 <= len(x["input_ids"]) <= args.max_seq_len)

    if args.packing:
        data_collator = SimplePacker(tok, args.max_seq_len)
    else:
        def data_collator(features):
            maxl = max(len(f["input_ids"]) for f in features)
            input_ids = torch.full((len(features), maxl), tok.pad_token_id, dtype=torch.long)
            attn = torch.zeros_like(input_ids)
            labels = torch.full_like(input_ids, -100)
            for i, f in enumerate(features):
                x = torch.tensor(f["input_ids"], dtype=torch.long)
                n = len(x)
                input_ids[i, :n] = x
                attn[i, :n] = 1
                labels[i, :n] = x
            return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    if args.paper_recipe:
        if args.paper_model == "octocoder":
            if args.override_hparams is False:
                args.lr = 5e-4
                args.batch_size = 32
                args.grad_accum = max(1, args.grad_accum)
                args.max_steps = 100
        elif args.paper_model == "octogeex":
            if args.override_hparams is False:
                args.lr = 5e-5
                args.batch_size = 48
                args.grad_accum = max(1, args.grad_accum)
                args.max_steps = 35
        args.lr_scheduler_type = "cosine"
        args.warmup_ratio = 0.02
        args.max_seq_len = 2048
        args.packing = True

    use_steps = args.max_steps > 0
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=(args.epochs if not use_steps else 1.0),
        max_steps=(args.max_steps if use_steps else -1),
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=args.bf16,
        gradient_checkpointing=True,
        report_to=["none"],
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tr,
        eval_dataset=ds_ev,
        data_collator=data_collator,
        tokenizer=tok,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if args.early_stop else None,
    )

    trainer.train()

    pt_path = os.path.join(args.output_dir, "adapter_state_dict.pt")
    torch.save(get_peft_model_state_dict(model), pt_path)
    print(f"Saved LoRA state_dict to: {pt_path}")

    # model.save_pretrained(args.output_dir)
    # tok.save_pretrained(args.output_dir)
    # print(f"Also saved adapter folder (and tokenizer) to: {args.output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--cp_subset", type=str, default="c++")

    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle_seed", type=int, default=42)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--logging_steps", type=int, default=10)

    ap.add_argument("--bf16", type=b, default=True)
    ap.add_argument("--use_4bit", type=b, default=True)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)

    ap.add_argument("--paper_recipe", type=bool, default=True)
    ap.add_argument("--packing", type=b, default=True)
    ap.add_argument("--paper_model", type=str, default="octocoder", choices=["octocoder", "octogeex"])
    ap.add_argument("--override_hparams", type=b, default=False)

    ap.add_argument("--early_stop", type=b, default=True)

    args = ap.parse_args()
    main(args)
