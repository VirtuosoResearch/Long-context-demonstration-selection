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

import argparse, copy, gc, math, random, re, time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
from transformers.cache_utils import DynamicCache

from utils.data import load_data


# =====================================================================
# GSM8K data loading
# =====================================================================

DEFAULT_GSM8K_DOWNSAMPLE = {
    "train": 1000,
    "test": 199,
}

MMLU_COLLEGE_SUBJECTS = [
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
]


def _extract_answer_number(text: str) -> str:
    match = re.search(r"####\s*(.+?)$", text.strip(), re.MULTILINE)
    if match:
        return match.group(1).strip().replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else ""


def _resolve_gsm8k_split(split: str) -> str:
    return split if split in {"train", "test"} else "test"


def load_gsm8k(split="train", max_samples=0):
    from datasets import load_dataset
    split = _resolve_gsm8k_split(split)
    ds = load_dataset("openai/gsm8k", "main", split=split)
    data = []
    for ex in ds:
        answer = ex["answer"]
        data.append({
            "input": ex["question"],
            "output": answer,
            "answer_number": _extract_answer_number(answer),
            # Keep field compatibility with MCQ-style eval loop.
            "options": [answer],
            "task": "gsm8k",
            "dataset": "gsm8k",
        })
    if max_samples <= 0:
        max_samples = DEFAULT_GSM8K_DOWNSAMPLE.get(split, 0)
    if max_samples > 0:
        data = data[:max_samples]
    return data


def _mmlu_answer_to_index(answer) -> int:
    if isinstance(answer, int):
        return answer
    if isinstance(answer, str):
        s = answer.strip().upper()
        if s.isdigit():
            return int(s)
        if len(s) == 1 and "A" <= s <= "Z":
            return ord(s) - ord("A")
    raise ValueError(f"Unsupported MMLU answer format: {answer!r}")


def _build_mmlu_college_data():
    from datasets import load_dataset

    data = []
    for subject in MMLU_COLLEGE_SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test")
        for ex in ds:
            choices = list(ex["choices"])
            ans_idx = _mmlu_answer_to_index(ex["answer"])
            if ans_idx < 0 or ans_idx >= len(choices):
                raise ValueError(
                    f"Invalid answer index {ans_idx} for subject={subject} "
                    f"with {len(choices)} choices."
                )
            data.append({
                "input": ex["question"],
                "output": choices[ans_idx],
                "options": choices,
                "task": subject,
                "dataset": "mmlu",
            })
    return data


def load_mmlu_college_split(args, split):
    cache_key = (
        args.seed,
        args.retrieval_split,
        args.eval_split,
    )
    if not hasattr(load_mmlu_college_split, "_cache"):
        load_mmlu_college_split._cache = {}
    cache = load_mmlu_college_split._cache
    if cache_key not in cache:
        all_data = _build_mmlu_college_data()
        rng = random.Random(args.seed)
        rng.shuffle(all_data)

        retrieval_size = int(len(all_data) * 5 / 6)
        retrieval_size = min(max(1, retrieval_size), max(1, len(all_data) - 1))
        retrieval_data = all_data[:retrieval_size]
        eval_data = all_data[retrieval_size:]

        cache[cache_key] = {
            "all": all_data,
            args.retrieval_split: retrieval_data,
            args.eval_split: eval_data,
        }

        print(
            f"[MMLU] merged college subsets: total={len(all_data)}, "
            f"{args.retrieval_split}={len(retrieval_data)}, "
            f"{args.eval_split}={len(eval_data)} (5:1 split)"
        )

    split_data = cache[cache_key].get(split, cache[cache_key]["all"])
    return list(split_data)


def _load_dataset(args, split):
    if args.dataset.strip().lower() == "gsm8k":
        return load_gsm8k(split=split, max_samples=0)
    if args.dataset.strip().lower() == "mmlu":
        return load_mmlu_college_split(args, split)
    return load_data(
        task=None, split=split, k=args.k,
        seed=args.seed, datasets=args.dataset.split(","), is_null=False
    )


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
# Analytical FLOPs
# =====================================================================

def _get_model_flop_params(model):
    """
    Extract architectural parameters needed for analytical FLOPs calculation.
    """
    cfg = model.config
    hidden = getattr(cfg, "hidden_size", 0)
    num_layers = getattr(cfg, "num_hidden_layers", 0)
    num_q_heads = getattr(cfg, "num_attention_heads", 0)
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim = hidden // max(1, num_q_heads)
    intermediate = getattr(cfg, "intermediate_size", 4 * hidden)
    vocab_size = getattr(cfg, "vocab_size", 0)

    model_type = getattr(cfg, "model_type", "").lower()
    gated_mlp_types = {
        "llama", "qwen", "qwen2", "mistral", "gemma", "phi3", "deepseek",
        "yi", "internlm", "internlm2", "baichuan", "cohere", "starcoder2",
    }
    is_gated_mlp = any(t in model_type for t in gated_mlp_types)

    return {
        "hidden": hidden,
        "num_layers": num_layers,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate": intermediate,
        "vocab_size": vocab_size,
        "is_gated_mlp": is_gated_mlp,
    }


def _analytical_flops(fp, new_tokens, ctx_tokens=0):
    """
    Compute analytical FLOPs for a transformer forward pass.

    Args:
        fp: dict from _get_model_flop_params
        new_tokens: number of new input tokens being processed
        ctx_tokens: number of tokens already in KV cache (0 for full forward)

    Per-layer FLOPs breakdown (factor of 2 per matmul for multiply-accumulate):
      QKV projection:  2 * new * hidden * (num_q_heads + 2*num_kv_heads) * head_dim
      Output proj:     2 * new * hidden * hidden
      Attention QK^T:  2 * num_q_heads * head_dim * attn_pairs
      Attention AV:    2 * num_q_heads * head_dim * attn_pairs
      MLP (gated):     3 * 2 * new * hidden * intermediate
      MLP (standard):  2 * 2 * new * hidden * intermediate
    Plus LM head:      2 * new * hidden * vocab_size
    """
    h = fp["hidden"]
    L = fp["num_layers"]
    nq = fp["num_q_heads"]
    nkv = fp["num_kv_heads"]
    hd = fp["head_dim"]
    inter = fp["intermediate"]
    vocab = fp["vocab_size"]
    n = new_tokens

    if n == 0:
        return 0.0

    # Attention pairs (causal)
    if ctx_tokens == 0:
        attn_pairs = n * (n + 1) / 2.0
    else:
        attn_pairs = n * ctx_tokens + n * (n + 1) / 2.0

    # Per-layer
    qkv_flops = 2.0 * n * h * (nq * hd + 2 * nkv * hd)
    o_flops = 2.0 * n * h * h
    attn_flops = 2.0 * 2.0 * nq * hd * attn_pairs
    if fp["is_gated_mlp"]:
        mlp_flops = 3.0 * 2.0 * n * h * inter
    else:
        mlp_flops = 2.0 * 2.0 * n * h * inter

    per_layer = qkv_flops + o_flops + attn_flops + mlp_flops
    total = L * per_layer

    # LM head
    total += 2.0 * n * h * vocab
    return total


def _analytical_flops_full(fp, seq_len):
    """FLOPs for a single full forward pass (no cache) on seq_len tokens."""
    return _analytical_flops(fp, new_tokens=seq_len, ctx_tokens=0)


def _analytical_flops_inc(fp, new_tokens, ctx_tokens):
    """FLOPs for incremental forward: new_tokens with ctx_tokens in KV cache."""
    return _analytical_flops(fp, new_tokens=new_tokens, ctx_tokens=ctx_tokens)


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


def _pos_ids(start, length, device, bsz=1):
    p = torch.arange(start, start + length, dtype=torch.long, device=device).unsqueeze(0)
    return p.expand(bsz, -1) if bsz > 1 else p


# =====================================================================
# RoPE utilities
# =====================================================================

def _detect_rope_layout(model):
    """
    Detect whether the model uses interleaved or half-split RoPE layout.
    - Interleaved: [r0, i0, r1, i1, ...] (e.g., GPT-NeoX, Pythia, Falcon)
    - Half-split:  [r0, r1, ..., i0, i1, ...] (e.g., Llama, Qwen, Mistral)
    """
    model_type = getattr(model.config, "model_type", "").lower()

    half_split_models = {
        "llama", "qwen", "qwen2", "mistral", "gemma", "phi", "phi3",
        "cohere", "deepseek", "yi", "internlm", "internlm2", "baichuan",
        "chatglm", "starcoder2",
    }
    interleaved_models = {
        "gpt_neox", "pythia", "falcon",
    }

    for name in half_split_models:
        if name in model_type:
            return "half"
    for name in interleaved_models:
        if name in model_type:
            return "interleaved"

    print(f"WARN: Unknown model_type '{model_type}' for RoPE layout, defaulting to half-split.")
    return "half"


def _get_rope_params(model):
    """Extract RoPE parameters (inv_freq / head_dim) from the model."""
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    rope_base = getattr(config, "rope_theta", 10000.0)
    inv_freq = 1.0 / (rope_base ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    ))
    return inv_freq, head_dim


def _apply_rope_delta_half(key, old_positions, new_positions, inv_freq, device):
    """Re-encode keys using HALF-SPLIT layout: [r0..r_{d/2-1}, i0..i_{d/2-1}]."""
    half_dim = key.shape[3] // 2
    delta_pos = (new_positions - old_positions).float().to(device)
    freqs = torch.outer(delta_pos, inv_freq.to(device))
    cos_f = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
    sin_f = torch.sin(freqs).unsqueeze(0).unsqueeze(0)

    k_real = key[..., :half_dim]
    k_imag = key[..., half_dim:]
    new_real = k_real * cos_f - k_imag * sin_f
    new_imag = k_real * sin_f + k_imag * cos_f
    return torch.cat([new_real, new_imag], dim=-1).to(key.dtype)


def _apply_rope_delta_interleaved(key, old_positions, new_positions, inv_freq, device):
    """Re-encode keys using INTERLEAVED layout: [r0, i0, r1, i1, ...]."""
    delta_pos = (new_positions - old_positions).float().to(device)
    freqs = torch.outer(delta_pos, inv_freq.to(device))
    cos_f = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
    sin_f = torch.sin(freqs).unsqueeze(0).unsqueeze(0)

    k_even = key[..., 0::2]
    k_odd  = key[..., 1::2]
    new_even = k_even * cos_f - k_odd * sin_f
    new_odd  = k_even * sin_f + k_odd * cos_f

    new_key = torch.stack([new_even, new_odd], dim=-1)
    return new_key.reshape(key.shape).to(key.dtype)


def _apply_rope_delta(key, old_positions, new_positions, inv_freq, device, layout="half"):
    if layout == "half":
        return _apply_rope_delta_half(key, old_positions, new_positions, inv_freq, device)
    else:
        return _apply_rope_delta_interleaved(key, old_positions, new_positions, inv_freq, device)


# =====================================================================
# StreamingLLM: build truncated KV cache
# =====================================================================

_ROPE_MODEL_TYPES = [
    "llama", "qwen", "qwen2", "mistral", "falcon", "phi", "gemma",
    "gpt_neox", "pythia", "deepseek", "yi", "internlm", "baichuan",
    "cohere", "starcoder2",
]


@torch.no_grad()
def build_streaming_cache(model, demo_ids, sink_tokens, recent_tokens, device):
    """
    Run full forward pass on demo_ids, then keep only:
      - first `sink_tokens` KV states  (attention sinks)
      - last  `recent_tokens` KV states (rolling window)

    For RoPE models, retained keys are re-rotated to contiguous positions
    (0, 1, ..., cache_len-1) following StreamingLLM Section 3.2.

    Returns: (cache, cache_len, true_demo_len)
    """
    T = demo_ids.shape[1]
    true_demo_len = T

    model_type = getattr(model.config, "model_type", "")
    uses_rope = any(t in model_type for t in _ROPE_MODEL_TYPES)

    if uses_rope:
        inv_freq, head_dim = _get_rope_params(model)
        rope_layout = _detect_rope_layout(model)
    else:
        inv_freq, head_dim, rope_layout = None, None, None

    out = model(input_ids=demo_ids, use_cache=True)
    full_past = out.past_key_values

    if hasattr(full_past, "to_legacy_cache"):
        legacy = full_past.to_legacy_cache()
    elif hasattr(full_past, "key_cache"):
        legacy = [(full_past.key_cache[i], full_past.value_cache[i])
                  for i in range(len(full_past.key_cache))]
    else:
        legacy = list(full_past)

    s = min(sink_tokens, T)
    r = min(recent_tokens, T - s)
    cache_len = s + r

    if r > 0:
        old_positions = torch.cat([torch.arange(0, s), torch.arange(T - r, T)])
    else:
        old_positions = torch.arange(0, s)
    new_positions = torch.arange(0, cache_len)

    truncated = DynamicCache()
    for layer_idx, (k, v) in enumerate(legacy):
        if r > 0:
            k_kept = torch.cat([k[:, :, :s, :], k[:, :, -r:, :]], dim=2)
            v_kept = torch.cat([v[:, :, :s, :], v[:, :, -r:, :]], dim=2)
        else:
            k_kept = k[:, :, :s, :]
            v_kept = v[:, :, :s, :]

        if uses_rope and inv_freq is not None:
            k_kept = _apply_rope_delta(
                k_kept, old_positions, new_positions, inv_freq, device,
                layout=rope_layout,
            )
        truncated.update(k_kept, v_kept, layer_idx)

    del out, full_past, legacy
    return truncated, cache_len, true_demo_len


# =====================================================================
# StreamingLLM Scorer
# =====================================================================

class StreamingLLMScorer:
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

        cache = copy.deepcopy(self.demo_cache)
        attn = torch.ones(1, self.cache_len + q_ids.shape[1],
                          dtype=torch.long, device=self.device)
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
    track_cuda_peak_mem = device.startswith("cuda")
    tokenizer = setup_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    # Model weight memory (constant overhead)
    model_mem_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    model_mem_gb = model_mem_bytes / (1024 ** 3)

    add_nl = not args.model_name.startswith("gpt2")

    retrieval_data = _load_dataset(args, args.retrieval_split)
    eval_data = _load_dataset(args, args.eval_split)
    if not retrieval_data or not eval_data:
        raise ValueError("Empty data.")

    per_query_random = (args.num_eval_demo_sets > 1)
    rng_eval = random.Random(args.seed + 7777)

    full_kv_preds, stream_preds = [], []
    full_losses, stream_losses = [], []
    query_logit_mses = []
    ds_results = {}
    n_queries = max(1, len(eval_data))
    flop_params = _get_model_flop_params(model)

    # Analytical FLOPs accumulators
    full_kv_total_flops = 0.0
    stream_query_total_flops = 0.0
    stream_demo_flops = 0.0

    # Peak memory tracking
    full_kv_peak_mem_bytes = 0
    stream_demo_peak_mem_bytes = 0
    stream_query_peak_mem_bytes = 0

    for qi, dp in enumerate(tqdm(eval_data, desc=f"eval sink={args.sink_tokens} recent={recent_tokens}")):
        q_text, ans_text = normalize_text(dp, is_first=False, add_newlines=add_nl)
        q_ids = tokenizer(q_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
        a_ids = tokenizer(ans_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
        q_full_kv_flops = 0.0
        q_stream_flops = 0.0

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
        demo_len = demo_ids.shape[1]
        q_len = q_ids.shape[1]
        opts = dp["options"]

        # ─── Full-KV baseline ───
        if track_cuda_peak_mem:
            torch.cuda.reset_peak_memory_stats()
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
            total_len = demo_len + q_len + opt_ids.shape[1]
            q_full_kv_flops += _analytical_flops_full(flop_params, total_len)
        full_kv_preds.append(min(opt_scores, key=opt_scores.get))
        if track_cuda_peak_mem:
            full_kv_peak_mem_bytes = max(full_kv_peak_mem_bytes, torch.cuda.max_memory_allocated())

        # Full-KV CE on answer
        t_logits_s = None
        first_token_id = None
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
                    first_token_id = a_stripped[:, 0].unsqueeze(1)
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

        # Demo cache build: track peak mem only on first query
        if track_cuda_peak_mem and qi == 0:
            torch.cuda.reset_peak_memory_stats()
        scorer.build_demo_cache(demo_ids)
        if track_cuda_peak_mem and qi == 0:
            stream_demo_peak_mem_bytes = torch.cuda.max_memory_allocated()

        # Demo FLOPs: computed once, amortized
        if qi == 0:
            stream_demo_flops = _analytical_flops_full(flop_params, demo_len)

        # Per-query inference: reset peak tracker
        if track_cuda_peak_mem:
            torch.cuda.reset_peak_memory_stats()

        first, q_past, q_len_runtime = scorer.prefill_question(q_text)
        q_stream_flops += _analytical_flops_inc(flop_params, q_len_runtime, scorer.cache_len)

        opt_texts = [normalize_option(o, add_nl) for o in opts]
        scores_t = scorer.score_options_nll(first, q_past, q_len_runtime, opt_texts)
        scores = {o: scores_t[ot] for o, ot in zip(opts, opt_texts)}
        for opt_text in opt_texts:
            opt_len = len(tokenizer(opt_text, add_special_tokens=False)["input_ids"])
            q_stream_flops += _analytical_flops_inc(
                flop_params, opt_len, scorer.cache_len + q_len_runtime)
        stream_preds.append(min(scores, key=scores.get))

        if track_cuda_peak_mem:
            stream_query_peak_mem_bytes = max(stream_query_peak_mem_bytes, torch.cuda.max_memory_allocated())

        # StreamingLLM CE on answer
        s_logits_s = None
        if q_ids.numel() > 0 and a_ids.numel() > 0:
            with torch.no_grad():
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

        # Per-query logit MSE
        if t_logits_s is not None and s_logits_s is not None and first_token_id is not None:
            t_first = t_logits_s[:, 0, :].gather(1, first_token_id).squeeze(1)
            s_first = s_logits_s[:, 0, :].gather(1, first_token_id).squeeze(1)
            t_first_val = float(t_first.item())
            s_first_val = float(s_first.item())
            denom = max(abs(s_first_val), abs(t_first_val))
            rel_sq = ((s_first_val - t_first_val) / denom) ** 2 if denom != 0 else 0.0
            query_logit_mses.append(rel_sq)

        # Per-dataset tracking
        ds = dp.get("task", dp.get("dataset", "unknown"))
        if ds not in ds_results:
            ds_results[ds] = {"full_p": [], "stream_p": [], "gt": []}
        ds_results[ds]["full_p"].append(full_kv_preds[-1])
        ds_results[ds]["stream_p"].append(stream_preds[-1])
        ds_results[ds]["gt"].append(dp["output"])
        full_kv_total_flops += q_full_kv_flops
        stream_query_total_flops += q_stream_flops

    # ─── Aggregate ───
    gts = [dp["output"] for dp in eval_data]
    full_acc = accuracy(full_kv_preds, gts)
    stream_acc = accuracy(stream_preds, gts)
    mean_full_ce = sum(full_losses) / max(1, len(full_losses))
    mean_stream_ce = sum(stream_losses) / max(1, len(stream_losses))
    per_sample_mse = sum(query_logit_mses) / max(1, len(query_logit_mses))

    # Analytical FLOPs: total to process all queries
    total_full_flops = full_kv_total_flops
    total_stream_flops = stream_demo_flops + stream_query_total_flops
    flops_reduction = (total_full_flops / total_stream_flops) if total_stream_flops > 0 else 0.0

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
        "per_sample_mean_MSE": per_sample_mse,
        "avg_full_flops": total_full_flops / n_queries,
        "avg_stream_flops": total_stream_flops / n_queries,
        "total_full_flops": total_full_flops,
        "total_stream_flops": total_stream_flops,
        "stream_demo_flops": stream_demo_flops,
        "flops_reduction": flops_reduction,
        "full_kv_peak_mem_bytes": full_kv_peak_mem_bytes,
        "full_kv_peak_mem_gb": full_kv_peak_mem_bytes / (1024 ** 3),
        "stream_demo_peak_mem_bytes": stream_demo_peak_mem_bytes,
        "stream_demo_peak_mem_gb": stream_demo_peak_mem_bytes / (1024 ** 3),
        "stream_query_peak_mem_bytes": stream_query_peak_mem_bytes,
        "stream_query_peak_mem_gb": stream_query_peak_mem_bytes / (1024 ** 3),
        "model_mem_bytes": model_mem_bytes,
        "model_mem_gb": model_mem_gb,
        "ds_results": ds_results,
    }

    print(f"\n{'='*60}")
    print(f"  StreamingLLM: sink={args.sink_tokens}, recent={recent_tokens}, "
          f"cache={cache_size}")
    print(f"{'='*60}")
    print(f"Full-KV Accuracy      = {full_acc:.4f}")
    print(f"StreamingLLM Accuracy = {stream_acc:.4f}")
    print(f"Accuracy gap          = {full_acc - stream_acc:+.4f}")
    print(f"Full-KV CE            = {mean_full_ce:.4f}")
    print(f"StreamingLLM CE       = {mean_stream_ce:.4f}")
    print(f"CE gap                = {mean_stream_ce - mean_full_ce:+.4f}")
    print(f"per_sample_mean_MSE   = {per_sample_mse:.6f}")
    if track_cuda_peak_mem:
        print(f"model_weights_mem     = {model_mem_gb:.3f} GB")
        print(f"full_kv_peak_mem      = {result['full_kv_peak_mem_gb']:.3f} GB "
              f"(+{result['full_kv_peak_mem_gb'] - model_mem_gb:.3f} GB over model)")
        print(f"stream_demo_peak_mem  = {result['stream_demo_peak_mem_gb']:.3f} GB "
              f"(+{result['stream_demo_peak_mem_gb'] - model_mem_gb:.3f} GB over model, one-time)")
        print(f"stream_query_peak_mem = {result['stream_query_peak_mem_gb']:.3f} GB "
              f"(+{result['stream_query_peak_mem_gb'] - model_mem_gb:.3f} GB over model, per-query)")
    print(f"total_full_flops      = {total_full_flops:.3e} ({n_queries} queries)")
    print(f"total_stream_flops    = {total_stream_flops:.3e} ({n_queries} queries)")
    if total_stream_flops > 0:
        print(f"flops_reduction       = {flops_reduction:.4f}x")

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
              f"{'Gap':>8} {'FullKV CE':>12} {'Stream CE':>12} {'CE Gap':>10} "
              f"{'MSE':>12} {'Full FLOPs':>12} {'Strm FLOPs':>12} {'FLOPs Red.':>12} "
              f"{'FullKV Mem':>11} {'S.Query Mem':>12}")
        print(f"{'-'*170}")
        for r in all_results:
            print(f"{r['recent_tokens']:>8} {r['cache_size']:>8} "
                  f"{r['full_kv_acc']:>12.4f} {r['stream_acc']:>12.4f} "
                  f"{r['acc_gap']:>+8.4f} {r['full_ce']:>12.4f} "
                  f"{r['stream_ce']:>12.4f} {r['ce_gap']:>+10.4f} "
                  f"{r['per_sample_mean_MSE']:>12.6f} "
                  f"{r['total_full_flops']:>12.3e} {r['total_stream_flops']:>12.3e} "
                  f"{r['flops_reduction']:>12.4f}x "
                  f"{r['full_kv_peak_mem_gb']:>10.3f}G "
                  f"{r['stream_query_peak_mem_gb']:>11.3f}G")


if __name__ == "__main__":
    main()