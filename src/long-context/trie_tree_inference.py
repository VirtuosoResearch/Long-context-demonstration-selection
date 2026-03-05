import argparse
import math
import random
import time
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from kv_cache_inference import (
    KVCachedICLScorer,
    _accuracy,
    _attention_proxy_full,
    _attention_proxy_incremental,
    _forward_with_optional_flops,
    _normalize_option,
    _normalize_text,
    _score_option_full,
    _score_option_full_with_flops,
    _setup_tokenizer,
)
from utils.data import load_data


def _sample_subsets_lexicographic(
    n_candidates: int, k: int, num_subsets: int, seed: int
) -> List[Tuple[int, ...]]:
    if k <= 0:
        raise ValueError("k must be > 0")
    if k > n_candidates:
        raise ValueError(f"k={k} is larger than candidate pool size={n_candidates}")
    max_unique = math.comb(n_candidates, k)
    if num_subsets > max_unique:
        raise ValueError(
            f"num_subsets={num_subsets} exceeds number of unique subsets C({n_candidates},{k})={max_unique}"
        )

    rng = random.Random(seed)
    subsets = set()
    while len(subsets) < num_subsets:
        subset = tuple(sorted(rng.sample(range(n_candidates), k)))
        subsets.add(subset)
    return sorted(subsets)


def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i


@torch.no_grad()
def _append_segment_to_cache(
    model,
    tokenizer,
    device: str,
    segment_text: str,
    prev_past,
    prev_len: int,
    measure_flops: bool = False,
):
    seg_ids = tokenizer(segment_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    seg_len = seg_ids.shape[1]
    if seg_len == 0:
        return prev_past, prev_len, 0

    model_kwargs = {"input_ids": seg_ids, "use_cache": True}
    if prev_past is not None:
        attn = torch.ones((1, prev_len + seg_len), dtype=torch.long, device=device)
        model_kwargs["past_key_values"] = prev_past
        model_kwargs["attention_mask"] = attn

    out, flops = _forward_with_optional_flops(model, model_kwargs, measure_flops=measure_flops)
    return out.past_key_values, prev_len + seg_len, flops


def _build_segment_text(dp: Dict, pos: int, add_newlines: bool) -> str:
    inp, out = _normalize_text(dp, is_first=(pos == 0), add_newlines=add_newlines)
    return inp + out


def _build_demo_text_from_subset(
    retrieval_data: List[Dict], subset_ids: Sequence[int], add_newlines: bool
) -> str:
    parts: List[str] = []
    for pos, rid in enumerate(subset_ids):
        parts.append(_build_segment_text(retrieval_data[rid], pos=pos, add_newlines=add_newlines))
    return "".join(parts)


def _run_one_subset_baseline_eval(
    model,
    tokenizer,
    device: str,
    demo_text: str,
    options: List[str],
    option_texts: List[str],
    eval_data: List[Dict],
    add_newlines: bool,
    measure_flops: bool = False,
    subset_idx: int = 0,
    num_subsets: int = 0,
):
    preds = []
    total_proxy = 0
    total_flops = 0
    demo_len = len(tokenizer(demo_text, add_special_tokens=False)["input_ids"])

    child_desc = f"baseline-eval ({subset_idx + 1}/{num_subsets})" if num_subsets > 0 else "baseline-eval"
    for dp in tqdm(
        eval_data,
        total=len(eval_data),
        desc=child_desc,
        leave=False,
        position=1,
    ):
        q_text, _ = _normalize_text(dp, is_first=False, add_newlines=add_newlines)
        q_len = len(tokenizer(q_text, add_special_tokens=False)["input_ids"])

        scores = {}
        for opt, opt_text in zip(options, option_texts):
            if measure_flops:
                nll, f = _score_option_full_with_flops(model, tokenizer, demo_text, q_text, opt_text, device)
                total_flops += f
            else:
                nll = _score_option_full(model, tokenizer, demo_text, q_text, opt_text, device)
            o_len = len(tokenizer(opt_text, add_special_tokens=False)["input_ids"])
            total_proxy += _attention_proxy_full(demo_len + q_len + o_len)
            scores[opt] = nll
        preds.append(min(scores, key=scores.get))

    return preds, total_proxy, total_flops


def _run_one_subset_eval(
    kv: KVCachedICLScorer,
    tokenizer,
    eval_data: List[Dict],
    options: List[str],
    option_texts: List[str],
    option_lens: List[int],
    add_newlines: bool,
    measure_flops: bool = False,
    subset_idx: int = 0,
    num_subsets: int = 0,
):
    preds = []
    total_proxy = 0
    total_flops = 0

    child_desc = f"eval_split ({subset_idx + 1}/{num_subsets})" if num_subsets > 0 else "eval_split"
    for dp in tqdm(
        eval_data,
        total=len(eval_data),
        desc=child_desc,
        leave=False,
        position=1,
    ):
        q_text, _ = _normalize_text(dp, is_first=False, add_newlines=add_newlines)
        q_len = len(tokenizer(q_text, add_special_tokens=False)["input_ids"])

        first_logits, q_past, q_len_runtime, f_q = kv.prefill_question(q_text, measure_flops=measure_flops)
        if q_len_runtime != q_len:
            raise RuntimeError("Question token length mismatch during runtime.")
        total_proxy += _attention_proxy_incremental(kv.demo_len, q_len)
        total_flops += f_q

        if measure_flops:
            scores_text, f_opt = kv.score_options_nll(
                first_logits, q_past, q_len, option_texts, measure_flops=True
            )
            total_flops += f_opt
        else:
            scores_text = kv.score_options_nll(first_logits, q_past, q_len, option_texts)

        for o_len in option_lens:
            total_proxy += _attention_proxy_incremental(kv.demo_len + q_len, o_len)

        scores = {opt: scores_text[opt_text] for opt, opt_text in zip(options, option_texts)}
        preds.append(min(scores, key=scores.get))

    return preds, total_proxy, total_flops


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
    if len(retrieval_data) == 0 or len(eval_data) == 0:
        raise ValueError("Loaded empty retrieval/eval data.")

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        eval_data = eval_data[: args.max_eval_samples]

    options = eval_data[0]["options"]
    option_texts = [_normalize_option(opt, add_newlines=add_newlines) for opt in options]
    option_lens = [len(tokenizer(opt_text, add_special_tokens=False)["input_ids"]) for opt_text in option_texts]
    gts = [dp["output"] for dp in eval_data]
    option_index = {opt: i for i, opt in enumerate(options)}

    subsets = _sample_subsets_lexicographic(
        n_candidates=len(retrieval_data),
        k=args.k,
        num_subsets=args.num_subsets,
        seed=args.seed,
    )

    print("\n===== Chain KV-cache Experiment =====")
    print(f"dataset={args.dataset}, retrieval_split={args.retrieval_split}, eval_split={args.eval_split}")
    print(f"k={args.k}, num_subsets={len(subsets)}, eval_size={len(eval_data)}")
    print("subset order: intra-subset ascending IDs; inter-subset lexicographic ascending")

    baseline_subset_accs = []
    baseline_subset_preds: List[List[str]] = []
    baseline_total_proxy = 0
    baseline_total_flops = 0
    baseline_elapsed = 0.0

    if args.run_mode in ["time", "both", "flops"]:
        t_base = time.perf_counter()
        for subset_idx, subset_ids in enumerate(
            tqdm(subsets, total=len(subsets), desc="baseline-subset-eval", position=0)
        ):
            demo_text = _build_demo_text_from_subset(
                retrieval_data=retrieval_data,
                subset_ids=subset_ids,
                add_newlines=add_newlines,
            )
            preds, proxy_eval, flops_eval = _run_one_subset_baseline_eval(
                model=model,
                tokenizer=tokenizer,
                device=device,
                demo_text=demo_text,
                options=options,
                option_texts=option_texts,
                eval_data=eval_data,
                add_newlines=add_newlines,
                measure_flops=args.run_mode in ["flops", "both"],
                subset_idx=subset_idx,
                num_subsets=len(subsets),
            )
            baseline_total_proxy += proxy_eval
            baseline_total_flops += flops_eval
            baseline_subset_preds.append(preds)
            baseline_subset_accs.append(_accuracy(preds, gts))
        baseline_elapsed = time.perf_counter() - t_base

    kv = KVCachedICLScorer(model=model, tokenizer=tokenizer, device=device)
    subset_accs = []
    subset_preds: List[List[str]] = []

    # A single chain is enough because subsets are globally lexicographically sorted.
    chain_pasts = [None]  # index d stores cache after d demonstrations
    chain_lens = [0]
    prev_subset: Tuple[int, ...] = ()

    total_chain_proxy = 0
    total_chain_flops = 0

    t0 = time.perf_counter()
    for subset_idx, subset_ids in enumerate(
        tqdm(subsets, total=len(subsets), desc="subset-eval", position=0)
    ):
        lcp = _common_prefix_len(prev_subset, subset_ids)
        chain_pasts = chain_pasts[: lcp + 1]
        chain_lens = chain_lens[: lcp + 1]

        for pos in range(lcp, len(subset_ids)):
            dp = retrieval_data[subset_ids[pos]]
            seg_text = _build_segment_text(dp, pos=pos, add_newlines=add_newlines)
            new_past, new_len, f = _append_segment_to_cache(
                model=model,
                tokenizer=tokenizer,
                device=device,
                segment_text=seg_text,
                prev_past=chain_pasts[-1],
                prev_len=chain_lens[-1],
                measure_flops=args.run_mode in ["flops", "both"],
            )
            seg_len = new_len - chain_lens[-1]
            total_chain_proxy += _attention_proxy_incremental(chain_lens[-1], seg_len)
            total_chain_flops += f
            chain_pasts.append(new_past)
            chain_lens.append(new_len)

        kv.demo_past = chain_pasts[-1]
        kv.demo_len = chain_lens[-1]
        preds, proxy_eval, flops_eval = _run_one_subset_eval(
            kv=kv,
            tokenizer=tokenizer,
            eval_data=eval_data,
            options=options,
            option_texts=option_texts,
            option_lens=option_lens,
            add_newlines=add_newlines,
            measure_flops=args.run_mode in ["flops", "both"],
            subset_idx=subset_idx,
            num_subsets=len(subsets),
        )
        total_chain_proxy += proxy_eval
        total_chain_flops += flops_eval
        subset_preds.append(preds)
        subset_accs.append(_accuracy(preds, gts))
        prev_subset = subset_ids

    elapsed = time.perf_counter() - t0

    ensemble_preds = []
    for i in range(len(eval_data)):
        cnt = Counter(preds[i] for preds in subset_preds)
        best_opt = max(options, key=lambda opt: (cnt.get(opt, 0), -option_index[opt]))
        ensemble_preds.append(best_opt)
    ensemble_acc = _accuracy(ensemble_preds, gts)

    baseline_ensemble_acc = None
    if baseline_subset_preds:
        baseline_ensemble_preds = []
        for i in range(len(eval_data)):
            cnt = Counter(preds[i] for preds in baseline_subset_preds)
            best_opt = max(options, key=lambda opt: (cnt.get(opt, 0), -option_index[opt]))
            baseline_ensemble_preds.append(best_opt)
        baseline_ensemble_acc = _accuracy(baseline_ensemble_preds, gts)

    if baseline_subset_accs:
        print("\n----- Baseline (direct full inference) -----")
        print(f"baseline_mean_subset_acc={sum(baseline_subset_accs) / max(1, len(baseline_subset_accs)):.6f}")
        print(f"baseline_best_subset_acc={max(baseline_subset_accs):.6f}")
        print(f"baseline_worst_subset_acc={min(baseline_subset_accs):.6f}")
        if baseline_ensemble_acc is not None:
            print(f"baseline_ensemble_acc={baseline_ensemble_acc:.6f}")

    print("\n----- Trie/Chain KV cache -----")

    print(f"mean_subset_acc={sum(subset_accs) / max(1, len(subset_accs)):.6f}")
    print(f"best_subset_acc={max(subset_accs):.6f}")
    print(f"worst_subset_acc={min(subset_accs):.6f}")
    print(f"ensemble_acc={ensemble_acc:.6f}")

    if args.run_mode in ["time", "both"]:
        if baseline_subset_accs:
            print(f"baseline_total_time={baseline_elapsed:.3f}s")
        print(f"chain_total_time={elapsed:.3f}s")
        if baseline_subset_accs:
            print(f"time_speedup_vs_baseline={baseline_elapsed / max(elapsed, 1e-9):.3f}x")
            print(f"baseline_attention_proxy={baseline_total_proxy}")
        print(f"chain_attention_proxy={total_chain_proxy}")
        if baseline_subset_accs:
            print(f"attention_proxy_reduction={baseline_total_proxy / max(total_chain_proxy, 1):.3f}x")
        print("NOTE: attention_proxy is a FLOPs-related proxy, not exact hardware FLOPs.")
    if args.run_mode in ["flops", "both"]:
        if baseline_subset_accs:
            print(f"baseline_real_flops={baseline_total_flops}")
        print(f"chain_real_flops={total_chain_flops}")
        if baseline_subset_accs:
            print(f"flops_reduction_vs_baseline={baseline_total_flops / max(total_chain_flops, 1):.3f}x")
        print("NOTE: real_flops are from torch.profiler(with_flops=True); use same GPU/setup for fair comparison.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="e.g. sst2, poem_sentiment")
    parser.add_argument("--gpt2", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--num_subsets", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrieval_split", type=str, default="test")
    parser.add_argument("--eval_split", type=str, default="dev")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_mode", type=str, default="time", choices=["time", "flops", "both"])
    parser.add_argument("--max_eval_samples", type=int, default=None)
    args = parser.parse_args()

    run_experiment(args)
