import argparse
import random
import time
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from kv_cache_inference import (
    KVCachedICLScorer,
    _accuracy,
    _attention_proxy_full,
    _attention_proxy_incremental,
    _normalize_option,
    _normalize_text,
    _score_option_full,
    _score_option_full_with_flops,
    _setup_tokenizer,
)
from utils.data import load_data


def _choose_prefix_fixed_random_demos(
    retrieval_data: List[Dict], k: int, fixed_prefix_pct: float, seed: int
) -> List[Dict]:
    if k <= 0:
        return []
    if k > len(retrieval_data):
        raise ValueError(f"k={k} is larger than retrieval set size={len(retrieval_data)}")
    if not (0.0 <= fixed_prefix_pct <= 100.0):
        raise ValueError(f"fixed_prefix_pct should be in [0, 100], got {fixed_prefix_pct}")

    fixed_k = int(k * fixed_prefix_pct / 100.0)
    fixed_k = max(0, min(k, fixed_k))
    random_k = k - fixed_k

    fixed_part = retrieval_data[:fixed_k]
    if random_k == 0:
        return fixed_part

    pool = retrieval_data[fixed_k:]
    if random_k > len(pool):
        raise ValueError(
            "Not enough samples to draw random demonstrations from the non-fixed pool: "
            f"need {random_k}, but only {len(pool)} available."
        )
    rng = random.Random(seed)
    random_idx = rng.sample(range(len(pool)), random_k)
    random_part = [pool[i] for i in random_idx]
    return fixed_part + random_part


def run_experiment(args):
    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    tokenizer = _setup_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    add_newlines = not args.model_name.startswith("gpt2")

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
    fixed_demos = _choose_prefix_fixed_random_demos(
        retrieval_data=retrieval_data,
        k=args.k,
        fixed_prefix_pct=args.fixed_prefix_pct,
        seed=args.seed,
    )

    demo_parts = []
    for i, dp in enumerate(fixed_demos):
        di, do = _normalize_text(dp, is_first=(i == 0), add_newlines=add_newlines)
        demo_parts.extend([di, do])
    demo_text = "".join(demo_parts)
    demo_len = len(tokenizer(demo_text, add_special_tokens=False)["input_ids"])

    print("\n===== KV-cache Experiment (prefix fixed + random) =====")
    print(f"dataset={args.dataset}, retrieval_split={args.retrieval_split}, eval_split={args.eval_split}")
    print(f"k={args.k}, fixed_prefix_pct={args.fixed_prefix_pct}, eval_size={len(eval_data)}")

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
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrieval_split", type=str, default="test")
    parser.add_argument("--eval_split", type=str, default="dev")
    parser.add_argument(
        "--fixed_prefix_pct",
        type=float,
        default=50.0,
        help="The first num%% demos are fixed, and the rest are random.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_mode", type=str, default="time", choices=["time", "flops", "both"])
    args = parser.parse_args()

    run_experiment(args)
