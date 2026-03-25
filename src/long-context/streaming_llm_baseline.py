#!/usr/bin/env python3
"""
StreamingLLM baseline (Xiao et al., ICLR 2024) for comparison with LCC.

Keeps the KV cache of the first `sink_tokens` tokens (attention sinks)
plus the last `recent_tokens` tokens from the demonstration prefix.
Total cache size = sink_tokens + recent_tokens.

Usage:
  python streaming_llm_baseline.py \
      --dataset sst2 \
      --model_name Qwen/Qwen2.5-7B-Instruct \
      --k 50 --sink_tokens 4 --recent_tokens 128 \
      --eval_split dev --retrieval_split test

  # Compare multiple cache budgets:
  python streaming_llm_baseline.py \
      --dataset sst2 \
      --model_name Qwen/Qwen2.5-7B-Instruct \
      --k 50 --sink_tokens 4 \
      --recent_tokens_list 16,32,64,128,256 \
      --eval_split dev --retrieval_split test
"""

import argparse, copy, gc, math, random, time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
from transformers.cache_utils import DynamicCache

from utils.data import load_data


# =====================================================================
# Tokenizer / text utilities (reused from main codebase)
# =====================================================================

def setup_tokenizer(name):
    tok = (GPT2Tokenizer.from_pretrained(name) if name.startswith("gpt2")
           else AutoTokenizer.from_pretrained(name))
    if tok.padding_side == "left":
        tok.padding_side = "right"
    if tok.eos_token_id is None and tok.sep_token is not None:
        tok.eos_token = tok.sep_token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.bos_token_id is None:
        tok.bos_token = tok.eos_token
    return tok


def normalize_text(dp, is_first, add_newlines):
    q = ("\n" + dp["input"]) if (add_newlines and not is_first) else dp["input"]
    a = ("\n" + dp["output"]) if add_newlines else dp["output"]
    return q, a


def normalize_option(opt, add_nl):
    return ("\n" + opt) if add_nl else opt


def build_demo_text(demos, add_newlines):
    parts = []
    for i, dp in enumerate(demos):
        q, a = normalize_text(dp, is_first=(i == 0), add_newlines=add_newlines)
        parts.append(q + a)
    return "".join(parts)


def choose_fixed_demos(data, k, strategy, seed):
    if strategy == "first":
        return data[:k]
    return [data[i] for i in random.Random(seed).sample(range(len(data)), k)]


def accuracy(preds, gts):
    return sum(int(p.strip() == g.strip()) for p, g in zip(preds, gts)) / max(1, len(gts))


def strip_leading_format_tokens(ans_ids, tokenizer):
    strip_tokens = set()
    for ch in ["\n", " ", "\t"]:
        toks = tokenizer(ch, add_special_tokens=False)["input_ids"]
        strip_tokens.update(toks)
    idx = 0
    while idx < ans_ids.shape[1] and ans_ids[0, idx].item() in strip_tokens:
        idx += 1
    if idx >= ans_ids.shape[1]:
        return ans_ids, 0
    return ans_ids[:, idx:], idx


# =====================================================================
# Cache utilities
# =====================================================================

def _ensure_cache_obj(past):
    if past is None:
        return None
    if isinstance(past, DynamicCache):
        return past
    if isinstance(past, tuple):
        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(past)
        return DynamicCache(past)
    return past


def _cache_to_legacy(past):
    if past is None:
        return None
    if isinstance(past, tuple):
        return past
    if hasattr(past, "to_legacy_cache"):
        return past.to_legacy_cache()
    return tuple(past)


def _pos_ids(start, length, device, bsz=1):
    p = torch.arange(start, start + length, dtype=torch.long, device=device).unsqueeze(0)
    return p.expand(bsz, -1) if bsz > 1 else p


# =====================================================================
# StreamingLLM: build truncated KV cache
# =====================================================================

def _get_rope_params(model):
    """Extract RoPE parameters (inv_freq / base / dim) from the model."""
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads

    # Try to get rope_theta from config (most modern models)
    rope_base = getattr(config, "rope_theta", 10000.0)

    # RoPE inv_freq: 1 / (base^(2i/d)) for i = 0, 1, ..., d/2-1
    inv_freq = 1.0 / (rope_base ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    ))
    return inv_freq, head_dim


def _apply_rope_delta(key, old_positions, new_positions, inv_freq, device):
    """
    Re-encode keys from old_positions to new_positions.

    RoPE is multiplicative in angle space, so:
      RoPE(key, new_pos) = RoPE(RoPE_inv(key, old_pos), new_pos)
                         = apply_rotary(key, new_pos - old_pos)

    key: [bsz, num_heads, seq_len, head_dim]
    old_positions: [seq_len] original position indices
    new_positions: [seq_len] target position indices
    """
    seq_len = key.shape[2]
    head_dim = key.shape[3]

    delta_pos = (new_positions - old_positions).float().to(device)  # [seq_len]
    inv_freq_dev = inv_freq.to(device)

    # freqs: [seq_len, head_dim/2]
    freqs = torch.outer(delta_pos, inv_freq_dev)
    cos_f = torch.cos(freqs)  # [seq_len, head_dim/2]
    sin_f = torch.sin(freqs)  # [seq_len, head_dim/2]

    # Reshape for broadcasting: [1, 1, seq_len, head_dim/2]
    cos_f = cos_f.unsqueeze(0).unsqueeze(0)
    sin_f = sin_f.unsqueeze(0).unsqueeze(0)

    # Split key into even and odd
    k_even = key[..., 0::2]   # [bsz, heads, seq_len, head_dim/2]
    k_odd  = key[..., 1::2]

    # Apply rotation: (k_even + j*k_odd) * (cos + j*sin)
    # Real part: k_even * cos - k_odd * sin
    # Imag part: k_even * sin + k_odd * cos
    new_even = k_even * cos_f - k_odd * sin_f
    new_odd  = k_even * sin_f + k_odd * cos_f

    # Interleave back
    new_key = torch.stack([new_even, new_odd], dim=-1)
    new_key = new_key.reshape(key.shape)

    return new_key.to(key.dtype)


@torch.no_grad()
def build_streaming_cache(model, demo_ids, sink_tokens, recent_tokens, device):
    """
    Run full forward pass on demo_ids, then keep only:
      - first `sink_tokens` KV states  (attention sinks)
      - last  `recent_tokens` KV states (rolling window)

    For RoPE models, the retained keys are re-rotated to contiguous
    positions (0, 1, ..., cache_len-1) following StreamingLLM Section 3.2:
    "For encoding like RoPE, we cache the Keys of tokens prior to
    introducing the rotary transformation. Then, we apply position
    transformation to the keys in the rolling cache at each decoding phase."

    Since HuggingFace stores post-RoPE keys, we undo the original RoPE
    and re-apply with new contiguous positions.

    Returns:
      cache: DynamicCache with truncated + re-rotated KV
      cache_len: total number of tokens in cache
      true_demo_len: original demo length
    """
    T = demo_ids.shape[1]
    true_demo_len = T

    # Check if model uses RoPE
    model_type = getattr(model.config, "model_type", "")
    uses_rope = any(t in model_type for t in ["llama", "qwen", "mistral", "falcon", "phi", "gemma"])

    if uses_rope:
        inv_freq, head_dim = _get_rope_params(model)
    else:
        inv_freq, head_dim = None, None

    # Full forward to get complete KV cache
    out = model(input_ids=demo_ids, use_cache=True)
    full_past = out.past_key_values

    # Convert to legacy format for slicing
    if hasattr(full_past, "to_legacy_cache"):
        legacy = full_past.to_legacy_cache()
    elif hasattr(full_past, "key_cache"):
        legacy = [(full_past.key_cache[i], full_past.value_cache[i])
                  for i in range(len(full_past.key_cache))]
    else:
        legacy = list(full_past)

    # Determine which positions to keep
    s = min(sink_tokens, T)
    r = min(recent_tokens, T - s)
    cache_len = s + r

    # Original positions of kept tokens
    if r > 0:
        old_positions = torch.cat([
            torch.arange(0, s),
            torch.arange(T - r, T),
        ])
    else:
        old_positions = torch.arange(0, s)

    # New contiguous positions
    new_positions = torch.arange(0, cache_len)

    # Build truncated cache with RoPE correction
    truncated = DynamicCache()
    for layer_idx, (k, v) in enumerate(legacy):
        # k, v shape: [bsz, num_heads, seq_len, head_dim]
        if r > 0:
            k_kept = torch.cat([k[:, :, :s, :], k[:, :, -r:, :]], dim=2)
            v_kept = torch.cat([v[:, :, :s, :], v[:, :, -r:, :]], dim=2)
        else:
            k_kept = k[:, :, :s, :]
            v_kept = v[:, :, :s, :]

        # Re-rotate keys for RoPE models
        if uses_rope and inv_freq is not None:
            k_kept = _apply_rope_delta(
                k_kept, old_positions, new_positions, inv_freq, device
            )

        truncated.update(k_kept, v_kept, layer_idx)

    del out, full_past, legacy
    return truncated, cache_len, true_demo_len


# =====================================================================
# StreamingLLM Scorer
# =====================================================================

class StreamingLLMScorer:
    """
    Scores options using StreamingLLM-style truncated KV cache.
    Mirrors the interface of SSMHybridICLScorer for fair comparison.
    """

    def __init__(self, model, tokenizer, device, sink_tokens=4, recent_tokens=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.demo_cache = None
        self.cache_len = 0
        self.true_demo_len = 0

    @torch.no_grad()
    def build_demo_cache(self, demo_ids):
        self.demo_cache, self.cache_len, self.true_demo_len = \
            build_streaming_cache(
                self.model, demo_ids,
                self.sink_tokens, self.recent_tokens, self.device
            )

    @torch.no_grad()
    def prefill_question(self, question_text):
        q_ids = self.tokenizer(question_text, return_tensors="pt",
                               add_special_tokens=False)["input_ids"].to(self.device)
        if q_ids.numel() == 0:
            raise ValueError("Empty question.")

        # Deep copy cache so we don't mutate the demo cache
        cache = copy.deepcopy(self.demo_cache)

        attn = torch.ones(1, self.cache_len + q_ids.shape[1],
                          dtype=torch.long, device=self.device)

        # StreamingLLM: positions are contiguous within cache
        pos = _pos_ids(self.cache_len, q_ids.shape[1], self.device)

        out = self.model(
            input_ids=q_ids, past_key_values=cache,
            attention_mask=attn, position_ids=pos, use_cache=True
        )
        return out.logits[:, -1, :], out.past_key_values, q_ids.shape[1]

    @torch.no_grad()
    def score_options_nll(self, first_logits, past_after_q, q_len, options):
        tokenized = [self.tokenizer(o, add_special_tokens=False)["input_ids"]
                     for o in options]
        lengths = [len(t) for t in tokenized]
        if any(l == 0 for l in lengths):
            raise ValueError("Empty option.")
        bsz = len(options)
        mx = max(lengths)

        ids = torch.full((bsz, mx), self.tokenizer.pad_token_id,
                         dtype=torch.long, device=self.device)
        mask = torch.zeros(bsz, mx, device=self.device)
        for i, t in enumerate(tokenized):
            ids[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)
            mask[i, :len(t)] = 1.0

        bp = copy.deepcopy(_ensure_cache_obj(past_after_q))
        bp.batch_repeat_interleave(bsz)

        base = self.cache_len + q_len
        attn = torch.ones(bsz, base + mx, dtype=torch.long, device=self.device)
        pos = _pos_ids(base, mx, self.device, bsz)
        for i, l in enumerate(lengths):
            if l < mx:
                attn[i, base + l:] = 0

        out = self.model(input_ids=ids, past_key_values=bp,
                         attention_mask=attn, position_ids=pos, use_cache=False)
        first = first_logits.expand(bsz, -1).unsqueeze(1)
        pred = torch.cat([first, out.logits[:, :-1]], dim=1) if mx > 1 else first
        losses = F.cross_entropy(
            pred.reshape(-1, pred.size(-1)), ids.reshape(-1),
            reduction="none"
        ).view(bsz, mx)
        nll = (losses * mask).sum(1)
        return {o: float(nll[i]) for i, o in enumerate(options)}


# =====================================================================
# Evaluation
# =====================================================================

def run_eval(args, recent_tokens):
    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    tokenizer = setup_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    add_nl = not args.model_name.startswith("gpt2")

    retrieval_data = load_data(task=None, split=args.retrieval_split, k=args.k,
                               seed=args.seed, datasets=args.dataset.split(","), is_null=False)
    eval_data = load_data(task=None, split=args.eval_split, k=args.k,
                          seed=args.seed, datasets=args.dataset.split(","), is_null=False)
    if not retrieval_data or not eval_data:
        raise ValueError("Empty data.")

    per_query_random = (args.num_eval_demo_sets > 1)
    rng_eval = random.Random(args.seed + 7777)

    full_kv_preds, stream_preds = [], []
    full_losses, stream_losses = [], []
    ds_results = {}

    for dp in tqdm(eval_data, desc=f"eval sink={args.sink_tokens} recent={recent_tokens}"):
        q_text, ans_text = normalize_text(dp, is_first=False, add_newlines=add_nl)
        q_ids = tokenizer(q_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
        a_ids = tokenizer(ans_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)

        # Pick demos
        if per_query_random:
            indices = rng_eval.sample(range(len(retrieval_data)),
                                      min(args.k, len(retrieval_data)))
            demos = [retrieval_data[i] for i in indices]
        else:
            if not hasattr(run_eval, '_fixed_demos'):
                run_eval._fixed_demos = choose_fixed_demos(
                    retrieval_data, args.k, args.demo_strategy, args.seed)
            demos = run_eval._fixed_demos

        demo_text = build_demo_text(demos, add_newlines=add_nl)
        demo_ids = tokenizer(demo_text, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(device)
        opts = dp["options"]

        # ─── Full-KV baseline ───
        opt_scores = {}
        for opt in opts:
            opt_text = normalize_option(opt, add_nl)
            opt_ids = tokenizer(opt_text, return_tensors="pt",
                                add_special_tokens=False)["input_ids"].to(device)
            if opt_ids.numel() == 0:
                opt_scores[opt] = float("inf")
                continue
            with torch.no_grad():
                all_ids = torch.cat([demo_ids, q_ids, opt_ids], dim=1)
                out = model(input_ids=all_ids, use_cache=False)
                start = demo_ids.shape[1] + q_ids.shape[1] - 1
                pred_logits = out.logits[:, start:start + opt_ids.shape[1], :]
                nll = F.cross_entropy(
                    pred_logits.reshape(-1, pred_logits.size(-1)),
                    opt_ids.reshape(-1), reduction="sum"
                ).item()
            opt_scores[opt] = nll
        full_kv_preds.append(min(opt_scores, key=opt_scores.get))

        # Full-KV CE on answer
        if q_ids.numel() > 0 and a_ids.numel() > 0:
            with torch.no_grad():
                all_ids = torch.cat([demo_ids, q_ids, a_ids], dim=1)
                out = model(input_ids=all_ids, use_cache=False)
                dl = demo_ids.shape[1]
                ql = q_ids.shape[1]
                t_logits = out.logits[:, dl + ql - 1: dl + ql - 1 + a_ids.shape[1], :]
                a_stripped, n_stripped = strip_leading_format_tokens(a_ids, tokenizer)
                if a_stripped.numel() > 0:
                    t_logits_s = t_logits[:, n_stripped:, :]
                    t_ce = F.cross_entropy(
                        t_logits_s.reshape(-1, t_logits_s.size(-1)),
                        a_stripped.reshape(-1), reduction="mean"
                    ).item()
                    full_losses.append(t_ce)

        # ─── StreamingLLM ───
        scorer = StreamingLLMScorer(
            model, tokenizer, device,
            sink_tokens=args.sink_tokens,
            recent_tokens=recent_tokens
        )
        scorer.build_demo_cache(demo_ids)

        first, q_past, q_len = scorer.prefill_question(q_text)
        opt_texts = [normalize_option(o, add_nl) for o in opts]
        scores_t = scorer.score_options_nll(first, q_past, q_len, opt_texts)
        scores = {o: scores_t[ot] for o, ot in zip(opts, opt_texts)}
        stream_preds.append(min(scores, key=scores.get))

        # StreamingLLM CE on answer
        if q_ids.numel() > 0 and a_ids.numel() > 0:
            with torch.no_grad():
                # Rebuild cache for CE computation
                cache, cache_len, _ = build_streaming_cache(
                    model, demo_ids, args.sink_tokens, recent_tokens, device)
                pos_q = _pos_ids(cache_len, q_ids.shape[1], device)
                attn_q = torch.ones(1, cache_len + q_ids.shape[1],
                                    dtype=torch.long, device=device)
                out_q = model(input_ids=q_ids, past_key_values=cache,
                              attention_mask=attn_q, position_ids=pos_q, use_cache=True)
                first_logit = out_q.logits[:, -1:]

                base = cache_len + q_ids.shape[1]
                pos_a = _pos_ids(base, a_ids.shape[1], device)
                attn_a = torch.ones(1, base + a_ids.shape[1],
                                    dtype=torch.long, device=device)
                out_a = model(input_ids=a_ids, past_key_values=out_q.past_key_values,
                              attention_mask=attn_a, position_ids=pos_a, use_cache=False)

                if a_ids.shape[1] == 1:
                    s_logits = first_logit
                else:
                    s_logits = torch.cat([first_logit, out_a.logits[:, :-1]], dim=1)

                a_stripped, n_stripped = strip_leading_format_tokens(a_ids, tokenizer)
                if a_stripped.numel() > 0:
                    s_logits_s = s_logits[:, n_stripped:, :]
                    s_ce = F.cross_entropy(
                        s_logits_s.reshape(-1, s_logits_s.size(-1)),
                        a_stripped.reshape(-1), reduction="mean"
                    ).item()
                    stream_losses.append(s_ce)

        # Per-dataset tracking
        ds = dp.get("task", dp.get("dataset", "unknown"))
        if ds not in ds_results:
            ds_results[ds] = {"full_p": [], "stream_p": [], "gt": []}
        ds_results[ds]["full_p"].append(full_kv_preds[-1])
        ds_results[ds]["stream_p"].append(stream_preds[-1])
        ds_results[ds]["gt"].append(dp["output"])

    # ─── Aggregate ───
    gts = [dp["output"] for dp in eval_data]
    full_acc = accuracy(full_kv_preds, gts)
    stream_acc = accuracy(stream_preds, gts)
    mean_full_ce = sum(full_losses) / max(1, len(full_losses))
    mean_stream_ce = sum(stream_losses) / max(1, len(stream_losses))

    # Cache size for FLOPs proxy
    cache_size = args.sink_tokens + recent_tokens

    result = {
        "sink_tokens": args.sink_tokens,
        "recent_tokens": recent_tokens,
        "cache_size": cache_size,
        "full_kv_acc": full_acc,
        "stream_acc": stream_acc,
        "acc_gap": full_acc - stream_acc,
        "full_ce": mean_full_ce,
        "stream_ce": mean_stream_ce,
        "ce_gap": mean_stream_ce - mean_full_ce,
        "ds_results": ds_results,
    }

    print(f"\n{'='*60}")
    print(f"  StreamingLLM: sink={args.sink_tokens}, recent={recent_tokens}, "
          f"cache={cache_size}")
    print(f"{'='*60}")
    print(f"Full-KV Accuracy     = {full_acc:.4f}")
    print(f"StreamingLLM Accuracy = {stream_acc:.4f}")
    print(f"Accuracy gap          = {full_acc - stream_acc:+.4f}")
    print(f"Full-KV CE            = {mean_full_ce:.4f}")
    print(f"StreamingLLM CE       = {mean_stream_ce:.4f}")
    print(f"CE gap                = {mean_stream_ce - mean_full_ce:+.4f}")

    print(f"\n{'Dataset':<20} {'N':>5} {'FullKV':>10} {'Stream':>10} {'Gap':>10}")
    print(f"{'-'*55}")
    for ds in sorted(ds_results.keys()):
        r = ds_results[ds]
        n = len(r["gt"])
        fa = accuracy(r["full_p"], r["gt"])
        sa = accuracy(r["stream_p"], r["gt"])
        print(f"{ds:<20} {n:>5} {fa:>10.4f} {sa:>10.4f} {fa-sa:>+10.4f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# =====================================================================
# Main
# =====================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retrieval_split", type=str, default="test")
    p.add_argument("--eval_split", type=str, default="dev")
    p.add_argument("--demo_strategy", type=str, default="first")
    p.add_argument("--num_eval_demo_sets", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--sink_tokens", type=int, default=4)
    p.add_argument("--recent_tokens", type=int, default=128,
                   help="Number of recent tokens to keep (single run)")
    p.add_argument("--recent_tokens_list", type=str, default="",
                   help="Comma-separated list for sweep, e.g. '16,32,64,128,256'")

    args = p.parse_args()

    if args.recent_tokens_list:
        recent_list = [int(x.strip()) for x in args.recent_tokens_list.split(",")]
    else:
        recent_list = [args.recent_tokens]

    all_results = []
    for rt in recent_list:
        res = run_eval(args, rt)
        all_results.append(res)

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  Summary: StreamingLLM sweep (sink={args.sink_tokens})")
        print(f"{'='*70}")
        print(f"{'Recent':>8} {'Cache':>8} {'FullKV Acc':>12} {'Stream Acc':>12} "
              f"{'Gap':>8} {'FullKV CE':>12} {'Stream CE':>12} {'CE Gap':>10}")
        print(f"{'-'*82}")
        for r in all_results:
            print(f"{r['recent_tokens']:>8} {r['cache_size']:>8} "
                  f"{r['full_kv_acc']:>12.4f} {r['stream_acc']:>12.4f} "
                  f"{r['acc_gap']:>+8.4f} {r['full_ce']:>12.4f} "
                  f"{r['stream_ce']:>12.4f} {r['ce_gap']:>+10.4f}")


if __name__ == "__main__":
    main()