#!/usr/bin/env python3
"""
Dynamic Block-Sparse Attention (DBSA) baseline (Xiao et al., ACL 2025).

Two stages:
  1. Pre-encode demo pool with block-sparse streaming attention:
     each block attends to anchor block + j previous blocks + self.
     KV states are stored along with the positions they were encoded at.
  2. Per query: BM25-retrieve top blocks, concatenate KV, re-apply RoPE
     to contiguous new positions, then score options.

Usage:
  python dbsa_baseline.py \
      --dataset sst2 \
      --model_name Qwen/Qwen2.5-1.5B-Instruct \
      --k 50 --block_size 10 --n_prev_blocks 2 --retrieval_ratio 0.3 \
      --eval_split dev --retrieval_split test
"""

import argparse, copy, gc, math, random, re
from collections import Counter
from typing import Dict, List, Tuple

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
# Text / tokenizer utilities
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
# Minimal BM25
# =====================================================================

class SimpleBM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.doc_freqs = {}
        self.doc_lens = []
        self.doc_tfs = []
        self.n_docs = 0
        self.avg_dl = 0.0

    def index(self, documents: List[str]):
        self.doc_tfs = []
        self.doc_lens = []
        self.doc_freqs = Counter()
        for doc in documents:
            tokens = doc.lower().split()
            tf = Counter(tokens)
            self.doc_tfs.append(tf)
            self.doc_lens.append(len(tokens))
            for t in set(tokens):
                self.doc_freqs[t] += 1
        self.n_docs = len(documents)
        self.avg_dl = sum(self.doc_lens) / max(1, self.n_docs)

    def score(self, query: str) -> List[float]:
        q_tokens = query.lower().split()
        scores = []
        for i in range(self.n_docs):
            s = 0.0
            dl = self.doc_lens[i]
            tf_map = self.doc_tfs[i]
            for t in q_tokens:
                if t not in tf_map:
                    continue
                tf = tf_map[t]
                df = self.doc_freqs.get(t, 0)
                idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
                numer = tf * (self.k1 + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                s += idf * numer / denom
            scores.append(s)
        return scores


# =====================================================================
# Analytical FLOPs
# =====================================================================

def _get_model_flop_params(model):
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
        "hidden": hidden, "num_layers": num_layers,
        "num_q_heads": num_q_heads, "num_kv_heads": num_kv_heads,
        "head_dim": head_dim, "intermediate": intermediate,
        "vocab_size": vocab_size, "is_gated_mlp": is_gated_mlp,
    }


def _analytical_flops(fp, new_tokens, ctx_tokens=0):
    h, L = fp["hidden"], fp["num_layers"]
    nq, nkv, hd = fp["num_q_heads"], fp["num_kv_heads"], fp["head_dim"]
    inter, vocab = fp["intermediate"], fp["vocab_size"]
    n = new_tokens
    if n == 0:
        return 0.0
    attn_pairs = (n * (n + 1) / 2.0) if ctx_tokens == 0 else (n * ctx_tokens + n * (n + 1) / 2.0)
    qkv = 2.0 * n * h * (nq * hd + 2 * nkv * hd)
    o = 2.0 * n * h * h
    att = 4.0 * nq * hd * attn_pairs
    mlp = (3.0 if fp["is_gated_mlp"] else 2.0) * 2.0 * n * h * inter
    return L * (qkv + o + att + mlp) + 2.0 * n * h * vocab


def _analytical_flops_full(fp, seq_len):
    return _analytical_flops(fp, new_tokens=seq_len, ctx_tokens=0)


def _analytical_flops_inc(fp, new_tokens, ctx_tokens):
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


def _cache_to_legacy(past):
    if past is None:
        return None
    if isinstance(past, DynamicCache):
        if hasattr(past, "to_legacy_cache"):
            return past.to_legacy_cache()
        if hasattr(past, "key_cache"):
            return [(past.key_cache[i], past.value_cache[i])
                    for i in range(len(past.key_cache))]
    if isinstance(past, tuple):
        return past
    return list(past)


def _slice_cache_legacy(legacy, start, end):
    return [(k[:, :, start:end, :].clone(), v[:, :, start:end, :].clone())
            for k, v in legacy]


def _concat_cache_legacies(legacies):
    if not legacies:
        return None
    n_layers = len(legacies[0])
    return [(torch.cat([leg[l][0] for leg in legacies], dim=2),
             torch.cat([leg[l][1] for leg in legacies], dim=2))
            for l in range(n_layers)]


# =====================================================================
# RoPE utilities
# =====================================================================

_ROPE_MODEL_TYPES = [
    "llama", "qwen", "qwen2", "mistral", "falcon", "phi", "gemma",
    "gpt_neox", "pythia", "deepseek", "yi", "internlm", "baichuan",
    "cohere", "starcoder2",
]


def _detect_rope_layout(model):
    model_type = getattr(model.config, "model_type", "").lower()
    half = {"llama", "qwen", "qwen2", "mistral", "gemma", "phi", "phi3",
            "cohere", "deepseek", "yi", "internlm", "internlm2", "baichuan",
            "chatglm", "starcoder2"}
    interleaved = {"gpt_neox", "pythia", "falcon"}
    for n in half:
        if n in model_type:
            return "half"
    for n in interleaved:
        if n in model_type:
            return "interleaved"
    return "half"


def _get_rope_params(model):
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    rope_base = getattr(config, "rope_theta", 10000.0)
    inv_freq = 1.0 / (rope_base ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    return inv_freq, head_dim


def _apply_rope_delta_half(key, old_pos, new_pos, inv_freq, device):
    half = key.shape[3] // 2
    delta = (new_pos - old_pos).float().to(device)
    freqs = torch.outer(delta, inv_freq.to(device))
    c = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
    s = torch.sin(freqs).unsqueeze(0).unsqueeze(0)
    r, im = key[..., :half], key[..., half:]
    return torch.cat([r * c - im * s, r * s + im * c], dim=-1).to(key.dtype)


def _apply_rope_delta_interleaved(key, old_pos, new_pos, inv_freq, device):
    delta = (new_pos - old_pos).float().to(device)
    freqs = torch.outer(delta, inv_freq.to(device))
    c = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
    s = torch.sin(freqs).unsqueeze(0).unsqueeze(0)
    e, o = key[..., 0::2], key[..., 1::2]
    new_key = torch.stack([e * c - o * s, e * s + o * c], dim=-1)
    return new_key.reshape(key.shape).to(key.dtype)


def _apply_rope_delta(key, old_pos, new_pos, inv_freq, device, layout="half"):
    if layout == "half":
        return _apply_rope_delta_half(key, old_pos, new_pos, inv_freq, device)
    return _apply_rope_delta_interleaved(key, old_pos, new_pos, inv_freq, device)


def _reposition_legacy_cache(legacy, old_positions, new_positions,
                             inv_freq, device, layout):
    """Re-rotate keys from old_positions to new_positions."""
    result = []
    for k, v in legacy:
        k_new = _apply_rope_delta(k, old_positions, new_positions,
                                  inv_freq, device, layout)
        result.append((k_new, v))
    return result


# =====================================================================
# DBSA Stage 1: block-sparse encoding
# =====================================================================

@torch.no_grad()
def encode_blocks_sparse(model, tokenizer, demos, block_size, n_prev_blocks,
                         add_newlines, device, flop_params=None):
    """
    Encode demos with block-sparse streaming attention.

    Each block bi is encoded seeing:
      - anchor block (b0) KV
      - previous n_prev_blocks blocks' KV
      - its own tokens (causal)

    The context KV is re-positioned to contiguous [0, ctx_len) before each
    block forward, so the block's own tokens get positions [ctx_len, ctx_len+block_len).

    We store per-block:
      - legacy KV (post-RoPE keys at encoding positions)
      - encoding_positions: the actual position IDs used during encoding
      - block_text: for BM25

    Returns:
      block_infos: list of dicts per block
      total_flops: analytical FLOPs for encoding
    """
    # Partition demos into blocks
    n_blocks = max(1, (len(demos) + block_size - 1) // block_size)
    blocks_text = []
    for bi in range(n_blocks):
        s = bi * block_size
        e = min(s + block_size, len(demos))
        parts = []
        for j, dp in enumerate(demos[s:e]):
            q, a = normalize_text(dp, is_first=(s + j == 0), add_newlines=add_newlines)
            parts.append(q + a)
        blocks_text.append("".join(parts))

    model_type = getattr(model.config, "model_type", "")
    uses_rope = any(t in model_type for t in _ROPE_MODEL_TYPES)
    if uses_rope:
        inv_freq = _get_rope_params(model)[0]
        rope_layout = _detect_rope_layout(model)
    else:
        inv_freq, rope_layout = None, None

    block_infos = []  # list of {"kv": legacy, "positions": tensor, "n_tokens": int, "text": str}
    total_flops = 0.0

    for bi, block_text in enumerate(blocks_text):
        block_ids = tokenizer(block_text, return_tensors="pt",
                              add_special_tokens=False)["input_ids"].to(device)
        block_len = block_ids.shape[1]
        if block_len == 0:
            block_infos.append({"kv": [], "positions": torch.tensor([]),
                                "n_tokens": 0, "text": block_text})
            continue

        # --- Build context from anchor + prev blocks ---
        # Collect context block indices
        ctx_indices = []
        if bi > 0:
            ctx_indices.append(0)  # anchor
            prev_start = max(1, bi - n_prev_blocks)
            for pi in range(prev_start, bi):
                if pi not in ctx_indices:
                    ctx_indices.append(pi)

        # Gather context KV and their encoding positions
        ctx_kvs = []
        ctx_positions_list = []
        for ci in ctx_indices:
            info = block_infos[ci]
            if info["n_tokens"] > 0:
                ctx_kvs.append(info["kv"])
                ctx_positions_list.append(info["positions"])

        ctx_len = sum(info["n_tokens"] for ci in ctx_indices
                      if (info := block_infos[ci])["n_tokens"] > 0)

        # Re-position context to contiguous [0, ctx_len)
        if ctx_kvs:
            merged_ctx = _concat_cache_legacies(ctx_kvs)
            old_ctx_pos = torch.cat(ctx_positions_list)  # actual encoding positions
            new_ctx_pos = torch.arange(0, ctx_len, dtype=torch.long)

            if uses_rope and inv_freq is not None:
                merged_ctx = _reposition_legacy_cache(
                    merged_ctx, old_ctx_pos, new_ctx_pos,
                    inv_freq, device, rope_layout)

            cache = DynamicCache()
            for li, (k, v) in enumerate(merged_ctx):
                cache.update(k, v, li)
        else:
            cache = None
            ctx_len = 0

        # Forward this block
        # Block tokens get positions [ctx_len, ctx_len + block_len)
        block_encoding_positions = torch.arange(ctx_len, ctx_len + block_len, dtype=torch.long)
        pos = _pos_ids(ctx_len, block_len, device)
        attn_mask = torch.ones(1, ctx_len + block_len, dtype=torch.long, device=device)

        if flop_params is not None:
            total_flops += _analytical_flops_inc(flop_params, block_len, ctx_len)

        if cache is not None:
            out = model(input_ids=block_ids, past_key_values=cache,
                        attention_mask=attn_mask, position_ids=pos, use_cache=True)
        else:
            out = model(input_ids=block_ids, position_ids=pos, use_cache=True)

        # Extract only this block's KV
        full_legacy = _cache_to_legacy(out.past_key_values)
        block_own_kv = _slice_cache_legacy(full_legacy, ctx_len, ctx_len + block_len)

        block_infos.append({
            "kv": block_own_kv,
            "positions": block_encoding_positions,  # positions at which keys were encoded
            "n_tokens": block_len,
            "text": block_text,
        })

        del out, full_legacy
        if cache is not None:
            del cache

    return block_infos, total_flops


# =====================================================================
# DBSA Scorer
# =====================================================================

class DBSAScorer:
    def __init__(self, model, tokenizer, device, retrieval_ratio=0.3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.retrieval_ratio = retrieval_ratio
        self.block_infos = None
        self.bm25 = None
        self.n_blocks = 0

        model_type = getattr(model.config, "model_type", "")
        self.uses_rope = any(t in model_type for t in _ROPE_MODEL_TYPES)
        if self.uses_rope:
            self.inv_freq = _get_rope_params(model)[0]
            self.rope_layout = _detect_rope_layout(model)
        else:
            self.inv_freq, self.rope_layout = None, None

    def set_blocks(self, block_infos):
        self.block_infos = block_infos
        self.n_blocks = len(block_infos)
        self.bm25 = SimpleBM25()
        self.bm25.index([bi["text"] for bi in block_infos])

    @torch.no_grad()
    def _build_query_cache(self, query_text):
        """
        Select blocks via BM25, concatenate their KV, and re-apply RoPE
        to new contiguous positions [0, total_len).

        Each block's keys were encoded at specific positions (stored in
        block_infos[i]["positions"]). We undo the old RoPE and apply new
        contiguous positions via _apply_rope_delta.
        """
        scores = self.bm25.score(query_text)
        n_retrieve = max(1, int(self.n_blocks * self.retrieval_ratio))

        # Block 0 always included (attention sink)
        ranked = sorted(range(1, self.n_blocks), key=lambda i: scores[i], reverse=True)
        selected_indices = [0] + ranked[:n_retrieve - 1]
        selected_indices.sort()  # maintain original encoding order

        # Gather selected blocks' KV and their encoding positions
        selected_kvs = []
        selected_old_positions = []
        total_len = 0
        for bi in selected_indices:
            info = self.block_infos[bi]
            if info["n_tokens"] > 0:
                selected_kvs.append(info["kv"])
                selected_old_positions.append(info["positions"])
                total_len += info["n_tokens"]

        if not selected_kvs or total_len == 0:
            return None, 0

        # Concatenate KV
        merged = _concat_cache_legacies(selected_kvs)

        # Re-position: old positions (from encoding) → new contiguous [0, total_len)
        old_pos = torch.cat(selected_old_positions)  # actual encoding positions
        new_pos = torch.arange(0, total_len, dtype=torch.long)  # new contiguous

        if self.uses_rope and self.inv_freq is not None:
            merged = _reposition_legacy_cache(
                merged, old_pos, new_pos,
                self.inv_freq, self.device, self.rope_layout)

        cache = DynamicCache()
        for li, (k, v) in enumerate(merged):
            cache.update(k, v, li)

        return cache, total_len

    @torch.no_grad()
    def prefill_question(self, question_text, cache, cache_len):
        q_ids = self.tokenizer(question_text, return_tensors="pt",
                               add_special_tokens=False)["input_ids"].to(self.device)
        if q_ids.numel() == 0:
            raise ValueError("Empty question.")

        cache_copy = copy.deepcopy(cache)
        attn = torch.ones(1, cache_len + q_ids.shape[1],
                          dtype=torch.long, device=self.device)
        pos = _pos_ids(cache_len, q_ids.shape[1], self.device)

        out = self.model(
            input_ids=q_ids, past_key_values=cache_copy,
            attention_mask=attn, position_ids=pos, use_cache=True
        )
        return out.logits[:, -1, :], out.past_key_values, q_ids.shape[1]

    @torch.no_grad()
    def score_options_nll(self, first_logits, past_after_q, cache_len, q_len, options):
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

        base = cache_len + q_len
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

def run_eval(args):
    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    track_cuda_peak_mem = device.startswith("cuda")
    tokenizer = setup_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    model_mem_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    model_mem_gb = model_mem_bytes / (1024 ** 3)

    add_nl = not args.model_name.startswith("gpt2")

    retrieval_data = _load_dataset(args, args.retrieval_split)
    eval_data = _load_dataset(args, args.eval_split)
    if not retrieval_data or not eval_data:
        raise ValueError("Empty data.")

    per_query_random = (args.num_eval_demo_sets > 1)
    rng_eval = random.Random(args.seed + 7777)
    n_queries = max(1, len(eval_data))
    flop_params = _get_model_flop_params(model)

    if not per_query_random:
        demos = choose_fixed_demos(retrieval_data, args.k, args.demo_strategy, args.seed)
    else:
        demos = None

    full_kv_preds, dbsa_preds = [], []
    full_losses, dbsa_losses = [], []
    query_logit_mses = []
    ds_results = {}

    full_kv_total_flops = 0.0
    dbsa_demo_flops = 0.0
    dbsa_query_total_flops = 0.0

    full_kv_peak_mem_bytes = 0
    dbsa_demo_peak_mem_bytes = 0
    dbsa_query_peak_mem_bytes = 0

    # ─── Stage 1: Encode demo blocks (fixed demos) ───
    block_infos = None
    if demos is not None:
        if track_cuda_peak_mem:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        block_infos, dbsa_demo_flops = encode_blocks_sparse(
            model, tokenizer, demos, args.block_size, args.n_prev_blocks,
            add_nl, device, flop_params=flop_params)

        if track_cuda_peak_mem:
            dbsa_demo_peak_mem_bytes = torch.cuda.max_memory_allocated()

    scorer = DBSAScorer(model, tokenizer, device,
                        retrieval_ratio=args.retrieval_ratio)

    # ─── Stage 2: Per-query eval ───
    for qi, dp in enumerate(tqdm(eval_data, desc="eval DBSA")):
        q_text, ans_text = normalize_text(dp, is_first=False, add_newlines=add_nl)
        q_ids = tokenizer(q_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
        a_ids = tokenizer(ans_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
        q_full_kv_flops = 0.0
        q_dbsa_flops = 0.0

        # Per-query random demos
        if per_query_random:
            indices = rng_eval.sample(range(len(retrieval_data)),
                                      min(args.k, len(retrieval_data)))
            demos = [retrieval_data[i] for i in indices]
            block_infos, per_query_demo_flops = encode_blocks_sparse(
                model, tokenizer, demos, args.block_size, args.n_prev_blocks,
                add_nl, device, flop_params=flop_params)
            q_dbsa_flops += per_query_demo_flops

        scorer.set_blocks(block_infos)

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
            q_full_kv_flops += _analytical_flops_full(flop_params,
                                                       demo_len + q_len + opt_ids.shape[1])
        full_kv_preds.append(min(opt_scores, key=opt_scores.get))
        if track_cuda_peak_mem:
            full_kv_peak_mem_bytes = max(full_kv_peak_mem_bytes, torch.cuda.max_memory_allocated())

        # Full-KV CE
        t_logits_s = None
        first_token_id = None
        if q_ids.numel() > 0 and a_ids.numel() > 0:
            with torch.no_grad():
                all_ids = torch.cat([demo_ids, q_ids, a_ids], dim=1)
                out = model(input_ids=all_ids, use_cache=False)
                dl, ql = demo_ids.shape[1], q_ids.shape[1]
                t_logits = out.logits[:, dl + ql - 1:dl + ql - 1 + a_ids.shape[1], :]
                a_stripped, n_stripped = strip_leading_format_tokens(a_ids, tokenizer)
                if a_stripped.numel() > 0:
                    t_logits_s = t_logits[:, n_stripped:, :]
                    first_token_id = a_stripped[:, 0].unsqueeze(1)
                    full_losses.append(F.cross_entropy(
                        t_logits_s.reshape(-1, t_logits_s.size(-1)),
                        a_stripped.reshape(-1), reduction="mean").item())

        # ─── DBSA inference ───
        if track_cuda_peak_mem:
            torch.cuda.reset_peak_memory_stats()

        cache, cache_len = scorer._build_query_cache(q_text)
        if cache is None:
            dbsa_preds.append("")
            full_kv_total_flops += q_full_kv_flops
            dbsa_query_total_flops += q_dbsa_flops
            ds = dp.get("task", dp.get("dataset", "unknown"))
            if ds not in ds_results:
                ds_results[ds] = {"full_p": [], "dbsa_p": [], "gt": []}
            ds_results[ds]["full_p"].append(full_kv_preds[-1])
            ds_results[ds]["dbsa_p"].append("")
            ds_results[ds]["gt"].append(dp["output"])
            continue

        first, q_past, q_len_runtime = scorer.prefill_question(q_text, cache, cache_len)
        q_dbsa_flops += _analytical_flops_inc(flop_params, q_len_runtime, cache_len)

        opt_texts = [normalize_option(o, add_nl) for o in opts]
        scores_t = scorer.score_options_nll(first, q_past, cache_len, q_len_runtime, opt_texts)
        scores = {o: scores_t[ot] for o, ot in zip(opts, opt_texts)}
        for opt_text in opt_texts:
            opt_len = len(tokenizer(opt_text, add_special_tokens=False)["input_ids"])
            q_dbsa_flops += _analytical_flops_inc(flop_params, opt_len, cache_len + q_len_runtime)
        dbsa_preds.append(min(scores, key=scores.get))

        if track_cuda_peak_mem:
            dbsa_query_peak_mem_bytes = max(dbsa_query_peak_mem_bytes, torch.cuda.max_memory_allocated())

        # DBSA CE on answer
        s_logits_s = None
        if q_ids.numel() > 0 and a_ids.numel() > 0:
            with torch.no_grad():
                ce_cache, ce_cl = scorer._build_query_cache(q_text)
                if ce_cache is not None:
                    pos_q = _pos_ids(ce_cl, q_ids.shape[1], device)
                    attn_q = torch.ones(1, ce_cl + q_ids.shape[1], dtype=torch.long, device=device)
                    out_q = model(input_ids=q_ids, past_key_values=ce_cache,
                                  attention_mask=attn_q, position_ids=pos_q, use_cache=True)
                    first_logit = out_q.logits[:, -1:]
                    base = ce_cl + q_ids.shape[1]
                    pos_a = _pos_ids(base, a_ids.shape[1], device)
                    attn_a = torch.ones(1, base + a_ids.shape[1], dtype=torch.long, device=device)
                    out_a = model(input_ids=a_ids, past_key_values=out_q.past_key_values,
                                  attention_mask=attn_a, position_ids=pos_a, use_cache=False)
                    s_logits = (first_logit if a_ids.shape[1] == 1
                                else torch.cat([first_logit, out_a.logits[:, :-1]], dim=1))
                    a_stripped, n_stripped = strip_leading_format_tokens(a_ids, tokenizer)
                    if a_stripped.numel() > 0:
                        s_logits_s = s_logits[:, n_stripped:, :]
                        dbsa_losses.append(F.cross_entropy(
                            s_logits_s.reshape(-1, s_logits_s.size(-1)),
                            a_stripped.reshape(-1), reduction="mean").item())

        # Per-query logit MSE
        if t_logits_s is not None and s_logits_s is not None and first_token_id is not None:
            t_val = float(t_logits_s[:, 0, :].gather(1, first_token_id).item())
            s_val = float(s_logits_s[:, 0, :].gather(1, first_token_id).item())
            denom = max(abs(s_val), abs(t_val))
            query_logit_mses.append(((s_val - t_val) / denom) ** 2 if denom != 0 else 0.0)

        ds = dp.get("task", dp.get("dataset", "unknown"))
        if ds not in ds_results:
            ds_results[ds] = {"full_p": [], "dbsa_p": [], "gt": []}
        ds_results[ds]["full_p"].append(full_kv_preds[-1])
        ds_results[ds]["dbsa_p"].append(dbsa_preds[-1])
        ds_results[ds]["gt"].append(dp["output"])
        full_kv_total_flops += q_full_kv_flops
        dbsa_query_total_flops += q_dbsa_flops

    # ─── Aggregate ───
    gts = [dp["output"] for dp in eval_data]
    full_acc = accuracy(full_kv_preds, gts)
    dbsa_acc = accuracy(dbsa_preds, gts)
    mean_full_ce = sum(full_losses) / max(1, len(full_losses))
    mean_dbsa_ce = sum(dbsa_losses) / max(1, len(dbsa_losses))
    per_sample_mse = sum(query_logit_mses) / max(1, len(query_logit_mses))

    total_full_flops = full_kv_total_flops
    total_dbsa_flops = dbsa_demo_flops + dbsa_query_total_flops
    flops_reduction = (total_full_flops / total_dbsa_flops) if total_dbsa_flops > 0 else 0.0

    n_blocks = len(block_infos) if block_infos else 0
    n_retrieved = max(1, int(n_blocks * args.retrieval_ratio))

    result = {
        "block_size": args.block_size,
        "n_prev_blocks": args.n_prev_blocks,
        "retrieval_ratio": args.retrieval_ratio,
        "n_blocks": n_blocks,
        "n_retrieved_per_query": n_retrieved,
        "full_kv_acc": full_acc,
        "dbsa_acc": dbsa_acc,
        "acc_gap": full_acc - dbsa_acc,
        "full_ce": mean_full_ce,
        "dbsa_ce": mean_dbsa_ce,
        "ce_gap": mean_dbsa_ce - mean_full_ce,
        "per_sample_mean_MSE": per_sample_mse,
        "total_full_flops": total_full_flops,
        "total_dbsa_flops": total_dbsa_flops,
        "dbsa_demo_flops": dbsa_demo_flops,
        "flops_reduction": flops_reduction,
        "full_kv_peak_mem_bytes": full_kv_peak_mem_bytes,
        "full_kv_peak_mem_gb": full_kv_peak_mem_bytes / (1024 ** 3),
        "dbsa_demo_peak_mem_bytes": dbsa_demo_peak_mem_bytes,
        "dbsa_demo_peak_mem_gb": dbsa_demo_peak_mem_bytes / (1024 ** 3),
        "dbsa_query_peak_mem_bytes": dbsa_query_peak_mem_bytes,
        "dbsa_query_peak_mem_gb": dbsa_query_peak_mem_bytes / (1024 ** 3),
        "model_mem_bytes": model_mem_bytes,
        "model_mem_gb": model_mem_gb,
        "ds_results": ds_results,
    }

    print(f"\n{'='*60}")
    print(f"  DBSA: block_size={args.block_size}, n_prev={args.n_prev_blocks}, "
          f"retrieval={args.retrieval_ratio:.0%}, blocks={n_blocks}, "
          f"retrieved={n_retrieved}")
    print(f"{'='*60}")
    print(f"Full-KV Accuracy      = {full_acc:.4f}")
    print(f"DBSA Accuracy         = {dbsa_acc:.4f}")
    print(f"Accuracy gap          = {full_acc - dbsa_acc:+.4f}")
    print(f"Full-KV CE            = {mean_full_ce:.4f}")
    print(f"DBSA CE               = {mean_dbsa_ce:.4f}")
    print(f"CE gap                = {mean_dbsa_ce - mean_full_ce:+.4f}")
    print(f"per_sample_mean_MSE   = {per_sample_mse:.6f}")
    if track_cuda_peak_mem:
        print(f"model_weights_mem     = {model_mem_gb:.3f} GB")
        print(f"full_kv_peak_mem      = {result['full_kv_peak_mem_gb']:.3f} GB "
              f"(+{result['full_kv_peak_mem_gb'] - model_mem_gb:.3f} GB over model)")
        print(f"dbsa_demo_peak_mem    = {result['dbsa_demo_peak_mem_gb']:.3f} GB "
              f"(+{result['dbsa_demo_peak_mem_gb'] - model_mem_gb:.3f} GB over model, one-time)")
        print(f"dbsa_query_peak_mem   = {result['dbsa_query_peak_mem_gb']:.3f} GB "
              f"(+{result['dbsa_query_peak_mem_gb'] - model_mem_gb:.3f} GB over model, per-query)")
    print(f"total_full_flops      = {total_full_flops:.3e} ({n_queries} queries)")
    print(f"total_dbsa_flops      = {total_dbsa_flops:.3e} ({n_queries} queries)")
    if total_dbsa_flops > 0:
        print(f"flops_reduction       = {flops_reduction:.4f}x")

    print(f"\n{'Dataset':<20} {'N':>5} {'FullKV':>10} {'DBSA':>10} {'Gap':>10}")
    print(f"{'-'*55}")
    for ds in sorted(ds_results.keys()):
        r = ds_results[ds]
        n = len(r["gt"])
        fa = accuracy(r["full_p"], r["gt"])
        da = accuracy(r["dbsa_p"], r["gt"])
        print(f"{ds:<20} {n:>5} {fa:>10.4f} {da:>10.4f} {fa-da:>+10.4f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


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
    p.add_argument("--block_size", type=int, default=10)
    p.add_argument("--n_prev_blocks", type=int, default=2)
    p.add_argument("--retrieval_ratio", type=float, default=0.3)
    args = p.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()