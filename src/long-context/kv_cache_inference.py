import argparse
import copy
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
from transformers.cache_utils import DynamicCache

from utils.data import load_data


def _setup_tokenizer(model_name: str):
    if model_name.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.padding_side == "left":
        tokenizer.padding_side = "right"
    if tokenizer.eos_token_id is None and tokenizer.sep_token is not None:
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token
    return tokenizer


def _normalize_text(dp: Dict, is_first: bool, add_newlines: bool) -> Tuple[str, str]:
    inp = dp["input"]
    out = dp["output"]
    if add_newlines:
        if not is_first:
            inp = "\n" + inp
        out = "\n" + out
    return inp, out


def _normalize_option(opt: str, add_newlines: bool) -> str:
    return ("\n" + opt) if add_newlines else opt


def _choose_fixed_demos(retrieval_data: List[Dict], k: int, strategy: str, seed: int) -> List[Dict]:
    if k <= 0:
        return []
    if k > len(retrieval_data):
        raise ValueError(f"k={k} is larger than retrieval set size={len(retrieval_data)}")
    if strategy == "first":
        return retrieval_data[:k]
    if strategy == "random":
        rng = random.Random(seed)
        idx = rng.sample(range(len(retrieval_data)), k)
        return [retrieval_data[i] for i in idx]
    raise ValueError(f"Unknown demo strategy: {strategy}")


def _accuracy(preds: List[str], gts: List[str]) -> float:
    correct = 0
    for p, g in zip(preds, gts):
        correct += int(p.strip() == g.strip())
    return correct / max(1, len(gts))


def _forward_with_optional_flops(model, model_kwargs: Dict, measure_flops: bool):
    if not measure_flops:
        return model(**model_kwargs), 0

    activities = [torch.profiler.ProfilerActivity.CPU]
    use_cuda_prof = torch.cuda.is_available() and any(
        p.is_cuda for p in model.parameters() if p.requires_grad
    )
    if use_cuda_prof:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=activities, with_flops=True) as prof:
        outputs = model(**model_kwargs)
        if use_cuda_prof:
            torch.cuda.synchronize()

    flops = 0
    for evt in prof.key_averages():
        v = getattr(evt, "flops", 0)
        if v is not None:
            flops += int(v)
    return outputs, flops


@torch.no_grad()
def _score_option_full(model, tokenizer, demo_text: str, question_text: str, option_text: str, device: str) -> float:
    prefix_ids = tokenizer(demo_text + question_text, add_special_tokens=False)["input_ids"]
    option_ids = tokenizer(option_text, add_special_tokens=False)["input_ids"]
    if len(option_ids) == 0:
        return 0.0

    all_ids = prefix_ids + option_ids
    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :].contiguous()  # predict next token
    labels = input_ids[:, 1:].contiguous()

    start = len(prefix_ids) - 1
    end = start + len(option_ids)
    token_logits = logits[:, start:end, :].reshape(-1, logits.size(-1))
    token_labels = labels[:, start:end].reshape(-1)
    nll = F.cross_entropy(token_logits, token_labels, reduction="sum")
    return float(nll.item())


@torch.no_grad()
def _score_option_full_with_flops(model, tokenizer, demo_text: str, question_text: str, option_text: str, device: str):
    prefix_ids = tokenizer(demo_text + question_text, add_special_tokens=False)["input_ids"]
    option_ids = tokenizer(option_text, add_special_tokens=False)["input_ids"]
    if len(option_ids) == 0:
        return 0.0, 0

    all_ids = prefix_ids + option_ids
    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
    outputs, flops = _forward_with_optional_flops(model, {"input_ids": input_ids}, measure_flops=True)
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    start = len(prefix_ids) - 1
    end = start + len(option_ids)
    token_logits = logits[:, start:end, :].reshape(-1, logits.size(-1))
    token_labels = labels[:, start:end].reshape(-1)
    nll = F.cross_entropy(token_logits, token_labels, reduction="sum")
    return float(nll.item()), flops


def _attention_proxy_full(seq_len: int) -> int:
    # Rough attention compute proxy per full forward: sum_{t=1..L} t = L*(L+1)/2
    return seq_len * (seq_len + 1) // 2


def _attention_proxy_incremental(start_ctx: int, new_tokens: int) -> int:
    # With KV cache, adding tokens one-by-one from context C:
    # proxy = sum_{i=1..new_tokens} (C + i)
    return new_tokens * start_ctx + new_tokens * (new_tokens + 1) // 2


class KVCachedICLScorer:
    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.demo_past = None
        self.demo_len = 0

    @torch.no_grad()
    def build_demo_cache(self, demo_text: str, measure_flops: bool = False):
        demo_ids = self.tokenizer(demo_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        if demo_ids.numel() == 0:
            raise ValueError("Demonstration text is empty.")
        out, flops = _forward_with_optional_flops(
            self.model, {"input_ids": demo_ids, "use_cache": True}, measure_flops=measure_flops
        )
        self.demo_past = out.past_key_values
        self.demo_len = demo_ids.shape[1]
        return flops

    @torch.no_grad()
    def prefill_question(self, question_text: str, measure_flops: bool = False):
        q_ids = self.tokenizer(question_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        if q_ids.numel() == 0:
            raise ValueError("Question text is empty.")
        attn = torch.ones((1, self.demo_len + q_ids.shape[1]), dtype=torch.long, device=self.device)
        out, flops = _forward_with_optional_flops(
            self.model,
            {
                "input_ids": q_ids,
                "past_key_values": self.demo_past,
                "attention_mask": attn,
                "use_cache": True,
            },
            measure_flops=measure_flops,
        )
        return out.logits[:, -1, :], out.past_key_values, q_ids.shape[1], flops

    @torch.no_grad()
    def score_options_nll(
        self, first_logits, past_after_question, question_len: int, options: List[str], measure_flops: bool = False
    ):
        tokenized = [self.tokenizer(opt, add_special_tokens=False)["input_ids"] for opt in options]
        lengths = [len(x) for x in tokenized]
        if any(l == 0 for l in lengths):
            raise ValueError("Found empty option text after tokenization.")

        bsz = len(options)
        max_len = max(lengths)
        input_ids = torch.full(
            (bsz, max_len),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        token_mask = torch.zeros((bsz, max_len), dtype=torch.float32, device=self.device)
        for i, ids in enumerate(tokenized):
            t = torch.tensor(ids, dtype=torch.long, device=self.device)
            input_ids[i, : len(ids)] = t
            token_mask[i, : len(ids)] = 1.0

        # Repeat question cache for all options using Cache API.
        if isinstance(past_after_question, tuple):
            batched_past = DynamicCache.from_legacy_cache(past_after_question)
        else:
            batched_past = copy.deepcopy(past_after_question)
        batched_past.batch_repeat_interleave(bsz)

        base_len = self.demo_len + question_len
        attn = torch.ones((bsz, base_len + max_len), dtype=torch.long, device=self.device)
        for i, l in enumerate(lengths):
            if l < max_len:
                attn[i, base_len + l :] = 0

        out, flops = _forward_with_optional_flops(
            self.model,
            {
                "input_ids": input_ids,
                "past_key_values": batched_past,
                "attention_mask": attn,
                "use_cache": False,
            },
            measure_flops=measure_flops,
        )

        first = first_logits.expand(bsz, -1).unsqueeze(1)
        if max_len == 1:
            pred_logits = first
        else:
            pred_logits = torch.cat([first, out.logits[:, :-1, :]], dim=1)

        losses = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            input_ids.reshape(-1),
            reduction="none",
        ).view(bsz, max_len)
        nll = (losses * token_mask).sum(dim=1)
        scores = {opt: float(nll[i].item()) for i, opt in enumerate(options)}
        if measure_flops:
            return scores, flops
        return scores


def run_experiment(args):
    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    tokenizer = _setup_tokenizer(args.gpt2)
    model = AutoModelForCausalLM.from_pretrained(args.gpt2).to(device)
    model.eval()

    add_newlines = not args.gpt2.startswith("gpt2")

    retrieval_data = load_data(
        task=None,
        split=args.retrieval_split,
        k=args.k,
        seed=args.seed,
        datasets=args.dataset.split(","),
        is_null=False,
    )
    eval_data = load_data(
        task=None,
        split=args.eval_split,
        k=args.k,
        seed=args.seed,
        datasets=args.dataset.split(","),
        is_null=False,
    )
    if len(eval_data) == 0 or len(retrieval_data) == 0:
        raise ValueError("Loaded empty retrieval/eval data.")

    options = eval_data[0]["options"]
    fixed_demos = _choose_fixed_demos(retrieval_data, args.k, args.demo_strategy, args.seed)

    demo_parts = []
    for i, dp in enumerate(fixed_demos):
        di, do = _normalize_text(dp, is_first=(i == 0), add_newlines=add_newlines)
        demo_parts.extend([di, do])
    demo_text = "".join(demo_parts)
    demo_len = len(tokenizer(demo_text, add_special_tokens=False)["input_ids"])

    print("\n===== Preliminary KV-cache Experiment =====")
    print(f"dataset={args.dataset}, retrieval_split={args.retrieval_split}, eval_split={args.eval_split}")
    print(f"k={args.k}, demo_strategy={args.demo_strategy}, eval_size={len(eval_data)}")

    if args.run_mode in ["time", "both"]:
        base_preds = []
        base_proxy = 0
        t0 = time.perf_counter()
        for dp in tqdm(eval_data, total=len(eval_data), desc="baseline"):
            q_text, _ = _normalize_text(dp, is_first=False, add_newlines=add_newlines)
            scores = {}
            for opt in options:
                opt_text = _normalize_option(opt, add_newlines=add_newlines)
                scores[opt] = _score_option_full(model, tokenizer, demo_text, q_text, opt_text, device)

                q_len = len(tokenizer(q_text, add_special_tokens=False)["input_ids"])
                o_len = len(tokenizer(opt_text, add_special_tokens=False)["input_ids"])
                base_proxy += _attention_proxy_full(demo_len + q_len + o_len)
            base_preds.append(min(scores, key=scores.get))
        baseline_time = time.perf_counter() - t0
        baseline_acc = _accuracy(base_preds, [dp["output"] for dp in eval_data])

        kv = KVCachedICLScorer(model=model, tokenizer=tokenizer, device=device)
        kv_proxy = 0
        t1 = time.perf_counter()
        kv.build_demo_cache(demo_text)
        kv_proxy += _attention_proxy_full(demo_len)

        kv_preds = []
        for dp in tqdm(eval_data, total=len(eval_data), desc="kv-cache"):
            q_text, _ = _normalize_text(dp, is_first=False, add_newlines=add_newlines)
            q_len = len(tokenizer(q_text, add_special_tokens=False)["input_ids"])
            first_logits, q_past, q_len_runtime, _ = kv.prefill_question(q_text)
            assert q_len_runtime == q_len
            kv_proxy += _attention_proxy_incremental(demo_len, q_len)

            opt_texts = [_normalize_option(opt, add_newlines=add_newlines) for opt in options]
            scores_text = kv.score_options_nll(first_logits, q_past, q_len, opt_texts)
            scores = {opt: scores_text[opt_text] for opt, opt_text in zip(options, opt_texts)}
            for opt_text in opt_texts:
                o_len = len(tokenizer(opt_text, add_special_tokens=False)["input_ids"])
                kv_proxy += _attention_proxy_incremental(demo_len + q_len, o_len)
            kv_preds.append(min(scores, key=scores.get))
        kv_time = time.perf_counter() - t1
        kv_acc = _accuracy(kv_preds, [dp["output"] for dp in eval_data])

        print(f"baseline_acc={baseline_acc:.6f}, kv_acc={kv_acc:.6f}")
        print(f"baseline_time={baseline_time:.3f}s, kv_time={kv_time:.3f}s, speedup={baseline_time / max(kv_time, 1e-9):.3f}x")
        print(f"baseline_attention_proxy={base_proxy}, kv_attention_proxy={kv_proxy}, reduction={base_proxy / max(kv_proxy, 1):.3f}x")
        print("NOTE: attention_proxy is a FLOPs-related proxy, not exact hardware FLOPs.")

    if args.run_mode in ["flops", "both"]:
        # Real FLOPs are measured on the exact same eval set/protocol as timing.
        flops_eval_data = eval_data
        print(f"real_flops_eval_size={len(flops_eval_data)}")

        baseline_flops = 0
        for dp in tqdm(flops_eval_data, total=len(flops_eval_data), desc="baseline-flops"):
            q_text, _ = _normalize_text(dp, is_first=False, add_newlines=add_newlines)
            for opt in options:
                opt_text = _normalize_option(opt, add_newlines=add_newlines)
                _, f = _score_option_full_with_flops(model, tokenizer, demo_text, q_text, opt_text, device)
                baseline_flops += f

        kv = KVCachedICLScorer(model=model, tokenizer=tokenizer, device=device)
        kv_flops = kv.build_demo_cache(demo_text, measure_flops=True)
        for dp in tqdm(flops_eval_data, total=len(flops_eval_data), desc="kv-cache-flops"):
            q_text, _ = _normalize_text(dp, is_first=False, add_newlines=add_newlines)
            first_logits, q_past, q_len, f_q = kv.prefill_question(q_text, measure_flops=True)
            kv_flops += f_q
            opt_texts = [_normalize_option(opt, add_newlines=add_newlines) for opt in options]
            _, f_opt = kv.score_options_nll(first_logits, q_past, q_len, opt_texts, measure_flops=True)
            kv_flops += f_opt

        print(
            f"baseline_real_flops={baseline_flops}, "
            f"kv_cache_real_flops={kv_flops}, "
            f"reduction={baseline_flops / max(kv_flops, 1):.3f}x"
        )
        print("NOTE: real_flops are from torch.profiler(with_flops=True); use same GPU/setup for fair comparison.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="e.g. sst2, poem_sentiment")
    parser.add_argument("--gpt2", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrieval_split", type=str, default="test")
    parser.add_argument("--eval_split", type=str, default="dev")
    parser.add_argument("--demo_strategy", type=str, default="first", choices=["first", "random"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_mode", type=str, default="time", choices=["time", "flops", "both"])
    args = parser.parse_args()

    run_experiment(args)
