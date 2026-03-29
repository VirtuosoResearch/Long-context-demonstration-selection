#!/usr/bin/env python3
"""
DuoAttention baseline (Xiao et al., 2024) — corrected implementation.

Key fixes vs. the original:
  1. Proper optimization-based gate learning with per-KV-head gated attention
     blending (α * full_attn + (1-α) * streaming_attn) trained end-to-end.
  2. Synthetic passkey retrieval data used for gate training (as in paper).
  3. Streaming heads use a proper Λ-shaped attention mask instead of zeroing
     K/V in full-length cache (zeroed K/V still affects softmax normalisation).
  4. RoPE-aware cache truncation for streaming heads during deployment.
  5. Attention-head reordering so retrieval / streaming heads are contiguous.

Remaining TODOs (efficiency, not correctness):
  - Chunked pre-filling with O(LK) time for streaming heads.
  - Fused CUDA kernels for split retrieval/streaming attention.

Usage:
  python duo_attention_baseline.py \
      --dataset sst2 \
      --model_name Qwen/Qwen2.5-1.5B-Instruct \
      --k 50 --sink_tokens 4 --recent_tokens 64 \
      --run_mode identify_eval
"""

import argparse, copy, gc, math, os, random, time, warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
from transformers.cache_utils import DynamicCache

from utils.data import load_data


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


def _analytical_flops_duo_inc(fp, new_tokens, ret_ctx_tokens, str_ctx_tokens, ret_q_heads_per_layer):
    """
    Analytical FLOPs for DuoAttention incremental forward:
    - QKV/O/MLP/LM-head are unchanged from dense attention.
    - Attention FLOPs are split into retrieval-head and streaming-head parts.
    """
    n = new_tokens
    if n <= 0:
        return 0.0

    h = fp["hidden"]
    nq = fp["num_q_heads"]
    nkv = fp["num_kv_heads"]
    hd = fp["head_dim"]
    inter = fp["intermediate"]
    vocab = fp["vocab_size"]
    num_layers = fp["num_layers"]
    is_gated_mlp = fp["is_gated_mlp"]

    # Linear layers per layer (independent of cache split)
    qkv_flops = 2.0 * n * h * (nq * hd + 2 * nkv * hd)
    o_flops = 2.0 * n * h * h
    if is_gated_mlp:
        mlp_flops = 3.0 * 2.0 * n * h * inter
    else:
        mlp_flops = 2.0 * 2.0 * n * h * inter

    base_per_layer = qkv_flops + o_flops + mlp_flops
    total = num_layers * base_per_layer

    attn_pairs_ret = n * ret_ctx_tokens + n * (n + 1) / 2.0
    attn_pairs_str = n * str_ctx_tokens + n * (n + 1) / 2.0
    if not ret_q_heads_per_layer:
        ret_q_heads_per_layer = [0] * num_layers

    # Per-layer split attention FLOPs: QK^T + AV
    for ret_q in ret_q_heads_per_layer:
        ret_q = int(max(0, min(nq, ret_q)))
        str_q = nq - ret_q
        attn_flops = 2.0 * 2.0 * hd * (
            ret_q * attn_pairs_ret + str_q * attn_pairs_str
        )
        total += attn_flops

    # LM head once
    total += 2.0 * n * h * vocab
    return total


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
# Model-structure helpers (Llama / Qwen2 / Mistral family)
# =====================================================================

def _get_layers(model):
    """Return the list of transformer decoder layers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers            # Llama, Qwen2, Mistral, Gemma
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h            # GPT-2 / GPT-Neo
    raise NotImplementedError(f"Unsupported architecture: {type(model)}")


def _get_embed_norm_head(model):
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens, model.model.norm, model.lm_head
    raise NotImplementedError(f"Unsupported architecture: {type(model)}")


def _model_uses_rope(config):
    return any(t in getattr(config, "model_type", "")
               for t in ["llama", "qwen", "mistral", "falcon", "phi", "gemma"])


# =====================================================================
# RoPE helpers
# =====================================================================

def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply rotary position embeddings (compatible with HF ≥4.38)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)


def _get_rope_inv_freq(config):
    head_dim = config.hidden_size // config.num_attention_heads
    base = getattr(config, "rope_theta", 10000.0)
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))


def _rope_correction(key_states, old_pos, new_pos, inv_freq, device):
    """
    Re-index RoPE on cached key states.
    key_states: [bsz, heads, seq_len, head_dim]
    old_pos / new_pos: 1-D tensors of length seq_len
    """
    delta = (new_pos.float() - old_pos.float()).to(device)
    inv_freq_d = inv_freq.to(device)
    freqs = torch.outer(delta, inv_freq_d)            # [seq_len, head_dim//2]
    cos_f = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
    sin_f = torch.sin(freqs).unsqueeze(0).unsqueeze(0)
    k_even, k_odd = key_states[..., 0::2], key_states[..., 1::2]
    new_even = k_even * cos_f - k_odd * sin_f
    new_odd  = k_even * sin_f + k_odd * cos_f
    return torch.stack([new_even, new_odd], dim=-1).reshape(key_states.shape).to(key_states.dtype)


# =====================================================================
# Cache / position-id utilities
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


def _extract_legacy(past):
    """Convert any HF cache object to list of (K, V) tuples."""
    if hasattr(past, "to_legacy_cache"):
        return past.to_legacy_cache()
    if hasattr(past, "key_cache"):
        return [(past.key_cache[i], past.value_cache[i])
                for i in range(len(past.key_cache))]
    return list(past)


# =====================================================================
# Streaming (Λ-shaped) mask builder
# =====================================================================

def build_streaming_mask(seq_len, sink, recent, device, dtype=torch.float32):
    """
    Build a proper Λ-shaped streaming mask (additive format).
    Returns [seq_len, seq_len] where 0 = attend, -inf = don't attend.
    Includes causal constraint.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    for i in range(seq_len):
        # Sink tokens
        end_sink = min(sink, i + 1)
        mask[i, :end_sink] = 0.0
        # Recent tokens (including self)
        start_recent = max(sink, i - recent + 1)
        mask[i, start_recent : i + 1] = 0.0
    return mask


def build_causal_mask(seq_len, device, dtype=torch.float32):
    return torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )


def _chunked_attention(Q, K, V, attn_bias, chunk_size):
    """
    Memory-efficient attention by chunking query length.
    Q, K, V: [bsz, n_heads, T, head_dim]
    attn_bias: [T, T] additive mask (0 or -inf), can be None.
    """
    bsz, n_heads, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    K_t = K.transpose(-2, -1)
    outputs = []

    if chunk_size is None or chunk_size <= 0:
        chunk_size = seq_len
    chunk_size = min(chunk_size, seq_len)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        q_chunk = Q[:, :, start:end, :]                            # [B, H, C, D]
        scores = torch.matmul(q_chunk, K_t) * scale                # [B, H, C, T]
        if attn_bias is not None:
            bias = attn_bias[start:end, :].to(dtype=scores.dtype)  # [C, T]
            scores = scores + bias.unsqueeze(0).unsqueeze(0)
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        out_chunk = torch.matmul(weights, V)                       # [B, H, C, D]
        outputs.append(out_chunk)

    return torch.cat(outputs, dim=2)


# =====================================================================
# Phase 1  —  Synthetic passkey data
# =====================================================================

_NATO = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
    "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
    "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
    "victor", "whiskey", "xray", "yankee", "zulu",
]


def generate_passkey_sample(tokenizer, filler_text, num_passkeys=10,
                            passkey_len=32, max_len=2048):
    """
    Generate one passkey-retrieval training sample (cf. paper §2.2, Fig. 3).
    Returns (input_ids [1, T], loss_start_idx) where distillation loss is
    computed only on tokens from loss_start_idx onward.
    """
    rng = random.Random()
    passkeys = [" ".join(rng.choice(_NATO) for _ in range(passkey_len))
                for _ in range(num_passkeys)]

    pk_tmpl = "Remember this sequence of words, it's passkey number {}: {}\n"
    q_tmpl  = "Passkey {}: {}\n"

    filler_ids = tokenizer(filler_text, add_special_tokens=False)["input_ids"]

    # Estimate overhead tokens
    overhead = ""
    for i, pk in enumerate(passkeys):
        overhead += pk_tmpl.format(i + 1, pk)
    overhead += "\nBased on the content above, recall all passkeys.\n"
    for i, pk in enumerate(passkeys):
        overhead += q_tmpl.format(i + 1, pk)
    overhead_len = len(tokenizer(overhead, add_special_tokens=False)["input_ids"])

    filler_budget = max(100, max_len - overhead_len - 20)
    filler_ids = filler_ids[:filler_budget]

    # Split filler into chunks; interleave passkey insertions
    chunk = len(filler_ids) // (num_passkeys + 1)
    parts = []
    for i in range(num_passkeys):
        s, e = i * chunk, (i + 1) * chunk
        parts.append(tokenizer.decode(filler_ids[s:e]))
        parts.append("\n" + pk_tmpl.format(i + 1, passkeys[i]))
    parts.append(tokenizer.decode(filler_ids[num_passkeys * chunk :]))

    # Query / answer section — loss is computed here only
    query = "\nBased on the content above, recall all passkeys.\n"
    for i, pk in enumerate(passkeys):
        query += q_tmpl.format(i + 1, pk)

    full_text = "".join(parts) + query
    full_ids = tokenizer(full_text, return_tensors="pt",
                         add_special_tokens=False)["input_ids"]
    query_ids = tokenizer(query, add_special_tokens=False)["input_ids"]
    loss_start = full_ids.shape[1] - len(query_ids)
    return full_ids, loss_start


# =====================================================================
# Phase 1  —  Gated attention (per-head blending for gate optimisation)
# =====================================================================

def _gated_attn_compute(attn_module, hidden_states, position_ids,
                        gate, sink, recent, config, chunk_size=256):
    """
    Compute α·full_attn + (1-α)·streaming_attn **per KV-head group**
    inside one attention layer.  Gate gradients flow through α.

    Works for Llama / Qwen2 / Mistral attention modules that expose
    q_proj, k_proj, v_proj, o_proj, rotary_emb.
    """
    bsz, seq_len, _ = hidden_states.shape
    n_heads    = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    head_dim   = config.hidden_size // n_heads
    groups     = n_heads // n_kv_heads

    # Q, K, V projections
    Q = attn_module.q_proj(hidden_states).view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
    K = attn_module.k_proj(hidden_states).view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    V = attn_module.v_proj(hidden_states).view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)

    # RoPE
    if hasattr(attn_module, "rotary_emb"):
        try:
            cos, sin = attn_module.rotary_emb(V, position_ids)
        except TypeError:
            cos, sin = attn_module.rotary_emb(V, seq_len=seq_len)
        Q, K = _apply_rotary_pos_emb(Q, K, cos, sin)

    # Expand KV for GQA
    if groups > 1:
        K_e = K.repeat_interleave(groups, dim=1)
        V_e = V.repeat_interleave(groups, dim=1)
    else:
        K_e, V_e = K, V

    # ---- full (causal) attention ----
    causal = build_causal_mask(seq_len, Q.device, Q.dtype)
    full_out = _chunked_attention(Q, K_e, V_e, causal, chunk_size)

    # ---- streaming (Λ) attention ----
    stream = build_streaming_mask(seq_len, sink, recent, Q.device, Q.dtype)
    stream_out = _chunked_attention(Q, K_e, V_e, stream, chunk_size)

    # ---- per-KV-head gate blending ----
    alpha = torch.sigmoid(gate).to(dtype=full_out.dtype)    # [n_kv_heads]
    alpha_e = alpha.repeat_interleave(groups).view(1, -1, 1, 1)
    blended = alpha_e * full_out + (1 - alpha_e) * stream_out

    blended = blended.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
    return attn_module.o_proj(blended)


def _gated_layer_forward(layer, hidden, position_ids, gate, sink, recent, config,
                         chunk_size=256):
    """Run a single decoder layer with gated attention (Llama-family)."""
    residual = hidden
    hidden = layer.input_layernorm(hidden)
    hidden = _gated_attn_compute(layer.self_attn, hidden, position_ids,
                                 gate, sink, recent, config, chunk_size=chunk_size)
    hidden = residual + hidden
    residual = hidden
    hidden = layer.post_attention_layernorm(hidden)
    hidden = layer.mlp(hidden)
    hidden = residual + hidden
    return hidden


def _gated_model_forward(model, input_ids, gates, sink, recent, chunk_size=256):
    """
    Full forward pass with per-head gated attention at every layer.
    Returns (last_hidden [bsz, T, D], logits [bsz, T, V]).
    Only *gates* carry gradients; all model parameters are frozen.
    """
    embed, norm, lm_head = _get_embed_norm_head(model)
    layers = _get_layers(model)
    config = model.config
    device = input_ids.device
    pos = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

    hidden = embed(input_ids)
    for i, layer in enumerate(layers):
        hidden = _gated_layer_forward(layer, hidden, pos,
                                      gates[i], sink, recent, config,
                                      chunk_size=chunk_size)
    hidden = norm(hidden)
    return hidden, lm_head(hidden)


# =====================================================================
# Phase 1  —  Gate optimisation loop
# =====================================================================

def identify_retrieval_heads(model, tokenizer, config, args, device):
    """
    Optimisation-based retrieval-head identification (paper §2.2).

    Trains per-KV-head gate values α_{l,h} by minimising:
        L = L_distill + λ·L_reg
    where L_distill = MSE(H_full, H_mixed) on passkey tokens,
          L_reg     = Σ|α_{l,h}|  (L1 / Lasso).
    """
    n_layers   = config.num_hidden_layers
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)

    print(f"\n[DuoAttention] Optimisation-based retrieval head identification")
    print(f"  layers={n_layers}, kv_heads={n_kv_heads}")

    # ---- trainable gates, initialised to 1 (all heads start as retrieval) ----
    gates = nn.Parameter(torch.ones(n_layers, n_kv_heads, device=device))
    optimizer = torch.optim.AdamW([gates], lr=args.gate_lr)

    # ---- learning rate schedule (warm-up + cool-down) ----
    warmup = args.gate_warmup_steps
    total  = args.gate_train_steps
    cooldown_start = total - args.gate_cooldown_steps

    def lr_lambda(step):
        if step < warmup:
            return args.gate_lr_min / args.gate_lr + \
                   (1 - args.gate_lr_min / args.gate_lr) * step / warmup
        if step >= cooldown_start:
            progress = (step - cooldown_start) / max(1, total - cooldown_start)
            return args.gate_lr_min / args.gate_lr + \
                   (1 - args.gate_lr_min / args.gate_lr) * (1 - progress)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- filler text for synthetic passkey samples ----
    retrieval_data = load_data(task=None, split=args.retrieval_split, k=args.k,
                               seed=args.seed, datasets=args.dataset.split(","),
                               is_null=False)
    add_nl = not args.model_name.startswith("gpt2")
    filler_text = build_demo_text(retrieval_data[:args.k], add_newlines=add_nl)

    # ---- precompute full-model hidden states for the first sample ----
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"  Training gates for {total} steps  (λ={args.gate_lambda}, "
          f"lr={args.gate_lr}, sink={args.sink_tokens}, recent={args.recent_tokens})")

    for step in tqdm(range(total), desc="gate optimisation"):
        # Generate a fresh passkey sample each step
        sample_ids, loss_start = generate_passkey_sample(
            tokenizer, filler_text,
            num_passkeys=args.gate_num_passkeys,
            passkey_len=args.gate_passkey_len,
            max_len=args.gate_max_seq_len,
        )
        sample_ids = sample_ids[:, :args.gate_max_seq_len].to(device)
        loss_start = max(0, min(loss_start, sample_ids.shape[1] - 1))

        # Full-attention hidden states (no grad)
        with torch.no_grad():
            full_out = model(input_ids=sample_ids, use_cache=False,
                             output_hidden_states=True)
            H_full = full_out.hidden_states[-1]  # [1, T, D]

        # Mixed-attention hidden states (grad flows through gates)
        H_mixed, _ = _gated_model_forward(
            model,
            sample_ids,
            gates,
            args.sink_tokens,
            args.recent_tokens,
            chunk_size=args.gate_attn_chunk_size,
        )

        # ---- distillation loss on passkey tokens only ----
        L_distill = F.mse_loss(
            H_mixed[:, loss_start:, :].float(),
            H_full[:, loss_start:, :].detach().float(),
        )

        # ---- L1 regularisation to push gates toward 0 (streaming) ----
        L_reg = torch.sigmoid(gates).sum()

        loss = L_distill + args.gate_lambda * L_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 200 == 0 or step == 0:
            g = torch.sigmoid(gates).detach()
            print(f"  step {step+1:>5d}  L_distill={L_distill.item():.6f}  "
                  f"L_reg={L_reg.item():.2f}  "
                  f"gate mean={g.mean():.3f}  min={g.min():.3f}  max={g.max():.3f}")

    gate_values = torch.sigmoid(gates).detach()
    print(f"\n  Final gate stats: min={gate_values.min():.4f}, "
          f"max={gate_values.max():.4f}, mean={gate_values.mean():.4f}")
    return gate_values


# =====================================================================
# Phase 2  —  Head reordering
# =====================================================================

def _reorder_heads(model, is_retrieval):
    """
    Reorder Q/K/V/O projection weights so that retrieval KV-heads come
    first within each layer.  This enables efficient contiguous slicing
    at inference time (paper §2.3).

    is_retrieval: [n_layers, n_kv_heads] bool tensor.
    Returns: reordered is_retrieval (retrieval heads first in each layer).
    """
    config = model.config
    n_heads    = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    head_dim   = config.hidden_size // n_heads
    groups     = n_heads // n_kv_heads

    layers = _get_layers(model)
    new_is_retrieval = is_retrieval.clone()

    for l_idx, layer in enumerate(layers):
        attn = layer.self_attn
        mask = is_retrieval[l_idx]                          # [n_kv_heads]

        # Sort: retrieval heads first (True > False when sorted descending)
        order = torch.argsort(mask.float(), descending=True)
        new_is_retrieval[l_idx] = mask[order]

        # Reorder K, V projections: [hidden, n_kv_heads * head_dim]
        for proj_name in ("k_proj", "v_proj"):
            proj = getattr(attn, proj_name)
            W = proj.weight.data.view(n_kv_heads, head_dim, -1)
            proj.weight.data = W[order].reshape(-1, W.shape[-1])
            if proj.bias is not None:
                b = proj.bias.data.view(n_kv_heads, head_dim)
                proj.bias.data = b[order].reshape(-1)

        # Reorder Q and O projections by query-head groups
        q_order = torch.cat([torch.arange(h * groups, (h + 1) * groups)
                             for h in order])
        for proj_name, dim in [("q_proj", 0), ("o_proj", 1)]:
            proj = getattr(attn, proj_name)
            if dim == 0:
                W = proj.weight.data.view(n_heads, head_dim, -1)
                proj.weight.data = W[q_order].reshape(-1, W.shape[-1])
                if proj.bias is not None:
                    b = proj.bias.data.view(n_heads, head_dim)
                    proj.bias.data = b[q_order].reshape(-1)
            else:  # o_proj: reorder input (columns)
                W = proj.weight.data.view(-1, n_heads, head_dim)
                proj.weight.data = W[:, q_order, :].reshape(W.shape[0], -1)

    return new_is_retrieval


# =====================================================================
# Phase 2  —  Build DuoAttention split cache
# =====================================================================

@torch.no_grad()
def build_duo_cache(model, demo_ids, gate_values, retrieval_ratio,
                    sink_tokens, recent_tokens, device):
    """
    Build two KV caches per layer:
      - retrieval_cache: full (all T tokens)
      - streaming_cache: sink + recent tokens only, with RoPE correction

    Returns (retrieval_cache, streaming_cache, T, is_retrieval, stream_cache_len).
    Both caches are lists of (K, V) tuples indexed by layer.
    """
    T = demo_ids.shape[1]
    config = model.config
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)

    # Binarise gates
    flat = gate_values.flatten()
    threshold = torch.quantile(flat, 1.0 - retrieval_ratio).item()
    is_retrieval = gate_values > threshold          # [n_layers, n_kv_heads]

    n_ret = is_retrieval.sum().item()
    total = gate_values.numel()
    print(f"  Retrieval heads: {n_ret}/{total} ({n_ret/total*100:.1f}%), "
          f"τ={threshold:.4f}")

    # Full forward to obtain KV cache
    out = model(input_ids=demo_ids, use_cache=True)
    legacy = _extract_legacy(out.past_key_values)

    s = min(sink_tokens, T)
    r = min(recent_tokens, T - s)
    stream_len = s + r

    # RoPE correction setup
    uses_rope = _model_uses_rope(config)
    inv_freq = _get_rope_inv_freq(config) if uses_rope else None

    # Position indices for streaming cache
    if r > 0:
        old_pos = torch.cat([torch.arange(0, s), torch.arange(T - r, T)])
        new_pos = torch.arange(0, stream_len)
    else:
        old_pos = torch.arange(0, s)
        new_pos = torch.arange(0, s)

    retrieval_cache = []   # per-layer (K, V) with full T tokens
    streaming_cache = []   # per-layer (K, V) with stream_len tokens
    ret_indices  = []      # per-layer list of KV-head indices that are retrieval
    str_indices  = []      # per-layer list of KV-head indices that are streaming

    for l, (K_full, V_full) in enumerate(legacy):
        # K_full, V_full: [bsz, n_kv_heads, T, head_dim]
        ret_idx = is_retrieval[l].nonzero(as_tuple=True)[0]
        str_idx = (~is_retrieval[l]).nonzero(as_tuple=True)[0]
        ret_indices.append(ret_idx)
        str_indices.append(str_idx)

        # --- retrieval heads: full cache ---
        K_ret = K_full[:, ret_idx, :, :] if ret_idx.numel() > 0 \
                else K_full[:, :0, :, :]
        V_ret = V_full[:, ret_idx, :, :] if ret_idx.numel() > 0 \
                else V_full[:, :0, :, :]
        retrieval_cache.append((K_ret.contiguous(), V_ret.contiguous()))

        # --- streaming heads: sink + recent ---
        if str_idx.numel() == 0:
            streaming_cache.append((K_full[:, :0, :0, :],
                                    V_full[:, :0, :0, :]))
            continue

        K_str_full = K_full[:, str_idx, :, :]
        V_str_full = V_full[:, str_idx, :, :]

        if r > 0:
            K_str = torch.cat([K_str_full[:, :, :s, :],
                               K_str_full[:, :, -r:, :]], dim=2)
            V_str = torch.cat([V_str_full[:, :, :s, :],
                               V_str_full[:, :, -r:, :]], dim=2)
        else:
            K_str = K_str_full[:, :, :s, :]
            V_str = V_str_full[:, :, :s, :]

        # RoPE correction: shift recent-token keys to contiguous positions
        if uses_rope and r > 0 and T > s + r:
            K_str = _rope_correction(K_str, old_pos, new_pos,
                                     inv_freq, device)

        streaming_cache.append((K_str.contiguous(), V_str.contiguous()))

    del out
    return (retrieval_cache, streaming_cache, T, is_retrieval,
            stream_len, ret_indices, str_indices)


# =====================================================================
# Phase 2  —  4-D per-head attention mask builder
# =====================================================================

def _build_duo_4d_mask(n_q_heads, q_len, kv_len_ret, kv_len_str,
                       n_ret_heads, n_str_heads, groups,
                       sink, recent, device, dtype=torch.float32):
    """
    Build two 4-D additive attention masks:
      retrieval_mask: [1, n_ret_q_heads, q_len, kv_len_ret]   (causal)
      streaming_mask: [1, n_str_q_heads, q_len, kv_len_str]   (Λ-shaped)

    These are used with split Q tensors during inference.
    """
    # Retrieval mask — standard causal
    ret_q = n_ret_heads * groups
    if ret_q > 0 and kv_len_ret > 0:
        offset_r = kv_len_ret - q_len
        ret_mask = torch.zeros(1, 1, q_len, kv_len_ret, dtype=dtype, device=device)
        for i in range(q_len):
            ret_mask[0, 0, i, offset_r + i + 1:] = float("-inf")
        ret_mask = ret_mask.expand(1, ret_q, -1, -1)
    else:
        ret_mask = None

    # Streaming mask — Λ-shaped (sink + recent window)
    str_q = n_str_heads * groups
    if str_q > 0 and kv_len_str > 0:
        # After RoPE correction, streaming cache positions are [0 .. stream_len-1]
        offset_s = kv_len_str - q_len
        str_mask = torch.full((1, 1, q_len, kv_len_str), float("-inf"),
                              dtype=dtype, device=device)
        for i in range(q_len):
            abs_pos = offset_s + i
            # Sink tokens
            end_sink = min(sink, abs_pos + 1)
            str_mask[0, 0, i, :end_sink] = 0.0
            # Recent tokens (contiguous after sink)
            start_rec = max(sink, abs_pos - recent + 1)
            str_mask[0, 0, i, start_rec : abs_pos + 1] = 0.0
        str_mask = str_mask.expand(1, str_q, -1, -1)
    else:
        str_mask = None

    return ret_mask, str_mask


# =====================================================================
# Phase 2  —  DuoAttention scorer (split attention per head-type)
# =====================================================================

class DuoAttentionScorer:
    """
    Scores answer options using proper DuoAttention inference:
    retrieval heads attend to the full KV cache, streaming heads attend
    to a truncated (sink + recent) cache with RoPE-corrected positions.
    """

    def __init__(self, model, tokenizer, device, gate_values,
                 retrieval_ratio, sink_tokens, recent_tokens):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device
        cfg = model.config
        n_heads = cfg.num_attention_heads
        n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
        self.groups = n_heads // n_kv_heads
        self.gate_values      = gate_values
        self.retrieval_ratio  = retrieval_ratio
        self.sink     = sink_tokens
        self.recent   = recent_tokens
        # Populated by build_demo_cache
        self.ret_cache  = None
        self.str_cache  = None
        self.T          = 0
        self.is_retrieval = None
        self.stream_len   = 0
        self.ret_indices  = None   # per-layer KV-head indices
        self.str_indices  = None

    @torch.no_grad()
    def build_demo_cache(self, demo_ids):
        (self.ret_cache, self.str_cache,
         self.T, self.is_retrieval, self.stream_len,
         self.ret_indices, self.str_indices) = build_duo_cache(
            self.model, demo_ids, self.gate_values, self.retrieval_ratio,
            self.sink, self.recent, self.device,
        )

    # ---- internal: run model with split caches ----

    def _forward_with_split_cache(self, input_ids, ret_cache, str_cache,
                                   ret_kv_len, str_kv_len):
        """
        Run model layer-by-layer, splitting Q into retrieval and streaming
        groups so each group attends to its own cache.

        Falls back to a 4-D mask based approach using the full model when
        manual layer processing is not feasible.
        """
        config = self.model.config
        n_heads    = config.num_attention_heads
        n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
        head_dim   = config.hidden_size // n_heads
        groups     = n_heads // n_kv_heads

        embed, norm, lm_head = _get_embed_norm_head(self.model)
        layers = _get_layers(self.model)
        bsz, q_len, *_ = input_ids.shape if input_ids.dim() == 3 else (input_ids.shape[0], input_ids.shape[1])

        hidden = embed(input_ids)
        # Position ids: retrieval sees full T + q_len, streaming sees stream_len + q_len
        pos_ret = _pos_ids(ret_kv_len, q_len, self.device)
        pos_str = _pos_ids(str_kv_len, q_len, self.device)

        new_ret_cache = []
        new_str_cache = []

        for l_idx, layer in enumerate(layers):
            attn = layer.self_attn
            ret_idx = self.ret_indices[l_idx]   # KV-head indices
            str_idx = self.str_indices[l_idx]
            n_ret_l = ret_idx.numel()
            n_str_l = str_idx.numel()

            residual = hidden
            hidden_normed = layer.input_layernorm(hidden)

            # --- Q, K, V projections ---
            Q_all = attn.q_proj(hidden_normed).view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
            K_new = attn.k_proj(hidden_normed).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
            V_new = attn.v_proj(hidden_normed).view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

            # Split K_new, V_new by index (not contiguous slice)
            K_new_ret = K_new[:, ret_idx] if n_ret_l > 0 else K_new[:, :0]
            V_new_ret = V_new[:, ret_idx] if n_ret_l > 0 else V_new[:, :0]
            K_new_str = K_new[:, str_idx] if n_str_l > 0 else K_new[:, :0]
            V_new_str = V_new[:, str_idx] if n_str_l > 0 else V_new[:, :0]

            # Split Q by the query-head groups that correspond to each KV head
            ret_q_idx = torch.cat([torch.arange(h * groups, (h + 1) * groups,
                                                 device=self.device) for h in ret_idx]) \
                        if n_ret_l > 0 else torch.tensor([], dtype=torch.long, device=self.device)
            str_q_idx = torch.cat([torch.arange(h * groups, (h + 1) * groups,
                                                 device=self.device) for h in str_idx]) \
                        if n_str_l > 0 else torch.tensor([], dtype=torch.long, device=self.device)
            Q_ret = Q_all[:, ret_q_idx] if ret_q_idx.numel() > 0 else Q_all[:, :0]
            Q_str = Q_all[:, str_q_idx] if str_q_idx.numel() > 0 else Q_all[:, :0]

            # --- RoPE ---
            if hasattr(attn, "rotary_emb"):
                try:
                    cos_r, sin_r = attn.rotary_emb(V_new_ret, pos_ret)
                    cos_s, sin_s = attn.rotary_emb(V_new_str, pos_str)
                except TypeError:
                    cos_r, sin_r = attn.rotary_emb(V_new_ret, seq_len=ret_kv_len + q_len)
                    cos_s, sin_s = attn.rotary_emb(V_new_str, seq_len=str_kv_len + q_len)
                Q_ret, K_new_ret = _apply_rotary_pos_emb(Q_ret, K_new_ret, cos_r, sin_r)
                Q_str, K_new_str = _apply_rotary_pos_emb(Q_str, K_new_str, cos_s, sin_s)

            attn_outputs = []

            # ---- retrieval heads ----
            if n_ret_l > 0:
                K_r_old, V_r_old = ret_cache[l_idx]
                K_r = torch.cat([K_r_old, K_new_ret], dim=2)
                V_r = torch.cat([V_r_old, V_new_ret], dim=2)
                new_ret_cache.append((K_r, V_r))
                # expand for GQA
                K_r_e = K_r.repeat_interleave(groups, dim=1) if groups > 1 else K_r
                V_r_e = V_r.repeat_interleave(groups, dim=1) if groups > 1 else V_r
                scores_r = torch.matmul(Q_ret, K_r_e.transpose(-2, -1)) / math.sqrt(head_dim)
                # causal mask
                cm = build_causal_mask(scores_r.shape[-1], self.device, scores_r.dtype)
                cm = cm[-q_len:]  # only last q_len rows
                scores_r = scores_r + cm.unsqueeze(0).unsqueeze(0)
                attn_out_r = torch.matmul(F.softmax(scores_r, dim=-1), V_r_e)
                attn_outputs.append(attn_out_r)
            else:
                new_ret_cache.append(ret_cache[l_idx])

            # ---- streaming heads ----
            if n_str_l > 0:
                K_s_old, V_s_old = str_cache[l_idx]
                K_s = torch.cat([K_s_old, K_new_str], dim=2)
                V_s = torch.cat([V_s_old, V_new_str], dim=2)
                # Streaming mask on the concatenated cache
                total_s = K_s.shape[2]
                sm = build_streaming_mask(total_s, self.sink, self.recent,
                                          self.device, K_s.dtype)
                sm = sm[-q_len:]  # last q_len rows
                K_s_e = K_s.repeat_interleave(groups, dim=1) if groups > 1 else K_s
                V_s_e = V_s.repeat_interleave(groups, dim=1) if groups > 1 else V_s
                scores_s = torch.matmul(Q_str, K_s_e.transpose(-2, -1)) / math.sqrt(head_dim)
                scores_s = scores_s + sm.unsqueeze(0).unsqueeze(0)
                w_s = F.softmax(scores_s, dim=-1)
                w_s = w_s.masked_fill(w_s.isnan(), 0.0)
                attn_out_s = torch.matmul(w_s, V_s_e)
                attn_outputs.append(attn_out_s)
                # Evict middle tokens to maintain constant size
                if total_s > self.sink + self.recent:
                    K_s = torch.cat([K_s[:, :, :self.sink, :],
                                     K_s[:, :, -(self.recent):, :]], dim=2)
                    V_s = torch.cat([V_s[:, :, :self.sink, :],
                                     V_s[:, :, -(self.recent):, :]], dim=2)
                new_str_cache.append((K_s, V_s))
            else:
                new_str_cache.append(str_cache[l_idx])

            # ---- reassemble attention output in original head order ----
            attn_out = torch.zeros(bsz, n_heads, q_len, head_dim,
                                   device=self.device, dtype=hidden.dtype)
            if len(attn_outputs) >= 1 and ret_q_idx.numel() > 0:
                attn_out[:, ret_q_idx] = attn_outputs[0]
            if len(attn_outputs) >= 2 and str_q_idx.numel() > 0:
                attn_out[:, str_q_idx] = attn_outputs[-1]
            elif len(attn_outputs) == 1 and ret_q_idx.numel() == 0 and str_q_idx.numel() > 0:
                attn_out[:, str_q_idx] = attn_outputs[0]

            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_out = attn.o_proj(attn_out)

            hidden = residual + attn_out
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden

        hidden = norm(hidden)
        logits = lm_head(hidden)
        return logits, new_ret_cache, new_str_cache

    # ---- public API ----

    @torch.no_grad()
    def prefill_question(self, question_text):
        q_ids = self.tokenizer(question_text, return_tensors="pt",
                               add_special_tokens=False)["input_ids"].to(self.device)
        logits, new_r, new_s = self._forward_with_split_cache(
            q_ids, self.ret_cache, self.str_cache, self.T, self.stream_len,
        )
        return (logits[:, -1, :], new_r, new_s,
                q_ids.shape[1], self.T + q_ids.shape[1],
                self.stream_len + q_ids.shape[1])

    @torch.no_grad()
    def score_options_nll(self, first_logits, ret_c, str_c,
                          ret_kv_len, str_kv_len, options):
        tokenized = [self.tokenizer(o, add_special_tokens=False)["input_ids"]
                     for o in options]
        lengths = [len(t) for t in tokenized]
        if any(l == 0 for l in lengths):
            raise ValueError("Empty option.")

        results = {}
        for opt_text, opt_toks in zip(options, tokenized):
            opt_ids = torch.tensor([opt_toks], dtype=torch.long, device=self.device)
            # Use first_logits for the first token
            nll = F.cross_entropy(first_logits, opt_ids[:, :1].reshape(-1)).item()
            if opt_ids.shape[1] > 1:
                r_c = [(k.clone(), v.clone()) for k, v in ret_c]
                s_c = [(k.clone(), v.clone()) for k, v in str_c]
                logits, _, _ = self._forward_with_split_cache(
                    opt_ids[:, :-1], r_c, s_c, ret_kv_len, str_kv_len,
                )
                rest_nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    opt_ids[:, 1:].reshape(-1),
                    reduction="sum",
                ).item()
                nll += rest_nll
            results[opt_text] = nll
        return results


# =====================================================================
# Evaluation
# =====================================================================

def run_eval(args, model, tokenizer, gate_values, retrieval_ratio, device):
    add_nl = not args.model_name.startswith("gpt2")
    track_cuda_peak_mem = str(device).startswith("cuda")

    retrieval_data = load_data(task=None, split=args.retrieval_split, k=args.k,
                               seed=args.seed, datasets=args.dataset.split(","),
                               is_null=False)
    eval_data = load_data(task=None, split=args.eval_split, k=args.k,
                          seed=args.seed, datasets=args.dataset.split(","),
                          is_null=False)

    # Model weight memory (constant overhead)
    model_mem_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    model_mem_gb = model_mem_bytes / (1024 ** 3)

    per_query_random = (args.num_eval_demo_sets > 1)
    rng_eval = random.Random(args.seed + 7777)

    full_kv_preds, duo_preds = [], []
    full_losses, duo_losses = [], []
    query_logit_mses = []
    ds_results = {}
    n_queries = max(1, len(eval_data))
    flop_params = _get_model_flop_params(model)

    # Analytical FLOPs accumulators
    full_kv_total_flops = 0.0
    duo_query_total_flops = 0.0
    duo_demo_flops = 0.0

    # Peak memory tracking
    full_kv_peak_mem_bytes = 0
    duo_demo_peak_mem_bytes = 0
    duo_query_peak_mem_bytes = 0

    for qi, dp in enumerate(tqdm(eval_data, desc=f"eval ratio={retrieval_ratio}")):
        q_text, ans_text = normalize_text(dp, is_first=False, add_newlines=add_nl)
        q_ids = tokenizer(q_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
        a_ids = tokenizer(ans_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
        q_full_kv_flops = 0.0
        q_duo_flops = 0.0

        if per_query_random:
            indices = rng_eval.sample(range(len(retrieval_data)),
                                      min(args.k, len(retrieval_data)))
            demos = [retrieval_data[i] for i in indices]
        else:
            if not hasattr(run_eval, "_fixed_demos"):
                run_eval._fixed_demos = choose_fixed_demos(
                    retrieval_data, args.k, args.demo_strategy, args.seed)
            demos = run_eval._fixed_demos

        demo_text = build_demo_text(demos, add_newlines=add_nl)
        demo_ids = tokenizer(demo_text, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(device)
        demo_len = demo_ids.shape[1]
        q_len = q_ids.shape[1]
        opts = dp["options"]

        # ---- Full-KV baseline ----
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
                        a_stripped.reshape(-1), reduction="mean"
                    ).item())

        # ---- DuoAttention ----
        scorer = DuoAttentionScorer(
            model, tokenizer, device, gate_values,
            retrieval_ratio, args.sink_tokens, args.recent_tokens,
        )
        if track_cuda_peak_mem and (per_query_random or qi == 0):
            torch.cuda.reset_peak_memory_stats()
        scorer.build_demo_cache(demo_ids)
        if track_cuda_peak_mem and (per_query_random or qi == 0):
            duo_demo_peak_mem_bytes = max(duo_demo_peak_mem_bytes, torch.cuda.max_memory_allocated())
        if qi == 0:
            duo_demo_flops = _analytical_flops_full(flop_params, demo_len)

        if track_cuda_peak_mem:
            torch.cuda.reset_peak_memory_stats()
        first, ret_c, str_c, q_len_runtime, r_kv, s_kv = scorer.prefill_question(q_text)
        ret_q_heads_per_layer = [int(idx.numel() * scorer.groups) for idx in scorer.ret_indices]
        ret_ctx = max(0, r_kv - q_len_runtime)
        str_ctx = max(0, s_kv - q_len_runtime)
        q_duo_flops += _analytical_flops_duo_inc(
            flop_params, q_len_runtime, ret_ctx, str_ctx, ret_q_heads_per_layer
        )
        opt_texts = [normalize_option(o, add_nl) for o in opts]
        scores_t = scorer.score_options_nll(first, ret_c, str_c, r_kv, s_kv,
                                            opt_texts)
        scores = {o: scores_t[ot] for o, ot in zip(opts, opt_texts)}
        for opt_text in opt_texts:
            opt_len = len(tokenizer(opt_text, add_special_tokens=False)["input_ids"])
            # Sequential scoring: first token uses first_logits (no forward),
            # then one incremental forward on the remaining tokens.
            run_len = max(0, opt_len - 1)
            q_duo_flops += _analytical_flops_duo_inc(
                flop_params, run_len, r_kv, s_kv, ret_q_heads_per_layer
            )
        duo_preds.append(min(scores, key=scores.get))
        if track_cuda_peak_mem:
            duo_query_peak_mem_bytes = max(duo_query_peak_mem_bytes, torch.cuda.max_memory_allocated())

        # DuoAttention CE
        d_logits_s = None
        if q_ids.numel() > 0 and a_ids.numel() > 0:
            with torch.no_grad():
                a_stripped, n_stripped = strip_leading_format_tokens(a_ids, tokenizer)
                if a_stripped.numel() > 0:
                    rc_ce = [(k.clone(), v.clone()) for k, v in ret_c]
                    sc_ce = [(k.clone(), v.clone()) for k, v in str_c]
                    logits_a, _, _ = scorer._forward_with_split_cache(
                        a_ids, rc_ce, sc_ce, r_kv, s_kv)
                    combined = torch.cat([first.unsqueeze(1), logits_a[:, :-1]], dim=1)
                    d_logits_s = combined[:, n_stripped:, :]
                    duo_losses.append(F.cross_entropy(
                        d_logits_s.reshape(-1, d_logits_s.size(-1)),
                        a_stripped.reshape(-1), reduction="mean"
                    ).item())

        # Per-query logit MSE
        if t_logits_s is not None and d_logits_s is not None and first_token_id is not None:
            t_first = t_logits_s[:, 0, :].gather(1, first_token_id).squeeze(1)
            d_first = d_logits_s[:, 0, :].gather(1, first_token_id).squeeze(1)
            t_first_val = float(t_first.item())
            d_first_val = float(d_first.item())
            denom = max(abs(d_first_val), abs(t_first_val))
            rel_sq = ((d_first_val - t_first_val) / denom) ** 2 if denom != 0 else 0.0
            query_logit_mses.append(rel_sq)

        ds = dp.get("task", dp.get("dataset", "unknown"))
        if ds not in ds_results:
            ds_results[ds] = {"full_p": [], "duo_p": [], "gt": []}
        ds_results[ds]["full_p"].append(full_kv_preds[-1])
        ds_results[ds]["duo_p"].append(duo_preds[-1])
        ds_results[ds]["gt"].append(dp["output"])
        full_kv_total_flops += q_full_kv_flops
        duo_query_total_flops += q_duo_flops

    # ---- aggregate ----
    gts = [dp["output"] for dp in eval_data]
    full_acc = accuracy(full_kv_preds, gts)
    duo_acc  = accuracy(duo_preds, gts)
    mean_full_ce = sum(full_losses) / max(1, len(full_losses))
    mean_duo_ce  = sum(duo_losses) / max(1, len(duo_losses))
    per_sample_mse = sum(query_logit_mses) / max(1, len(query_logit_mses))

    # Analytical FLOPs: total to process all queries
    total_full_flops = full_kv_total_flops
    total_duo_flops = duo_demo_flops + duo_query_total_flops
    flops_reduction = (total_full_flops / total_duo_flops) if total_duo_flops > 0 else 0.0
    cache_size = args.sink_tokens + args.recent_tokens

    result = {
        "retrieval_ratio": retrieval_ratio,
        "sink_tokens": args.sink_tokens,
        "recent_tokens": args.recent_tokens,
        "cache_size": cache_size,
        "full_kv_acc": full_acc, "duo_acc": duo_acc, "stream_acc": duo_acc,
        "acc_gap": full_acc - duo_acc,
        "full_ce": mean_full_ce, "duo_ce": mean_duo_ce, "stream_ce": mean_duo_ce,
        "ce_gap": mean_duo_ce - mean_full_ce,
        "per_sample_mean_MSE": per_sample_mse,
        "avg_full_flops": total_full_flops / n_queries,
        "avg_duo_flops": total_duo_flops / n_queries,
        "avg_stream_flops": total_duo_flops / n_queries,
        "total_full_flops": total_full_flops,
        "total_duo_flops": total_duo_flops,
        "total_stream_flops": total_duo_flops,
        "duo_demo_flops": duo_demo_flops,
        "stream_demo_flops": duo_demo_flops,
        "flops_reduction": flops_reduction,
        "full_kv_peak_mem_bytes": full_kv_peak_mem_bytes,
        "full_kv_peak_mem_gb": full_kv_peak_mem_bytes / (1024 ** 3),
        "duo_demo_peak_mem_bytes": duo_demo_peak_mem_bytes,
        "duo_demo_peak_mem_gb": duo_demo_peak_mem_bytes / (1024 ** 3),
        "duo_query_peak_mem_bytes": duo_query_peak_mem_bytes,
        "duo_query_peak_mem_gb": duo_query_peak_mem_bytes / (1024 ** 3),
        "stream_demo_peak_mem_bytes": duo_demo_peak_mem_bytes,
        "stream_demo_peak_mem_gb": duo_demo_peak_mem_bytes / (1024 ** 3),
        "stream_query_peak_mem_bytes": duo_query_peak_mem_bytes,
        "stream_query_peak_mem_gb": duo_query_peak_mem_bytes / (1024 ** 3),
        "model_mem_bytes": model_mem_bytes,
        "model_mem_gb": model_mem_gb,
        "ds_results": ds_results,
    }

    print(f"\n{'='*60}")
    print(f"  DuoAttention: ratio={retrieval_ratio}, "
          f"sink={args.sink_tokens}, recent={args.recent_tokens}, cache={cache_size}")
    print(f"{'='*60}")
    print(f"Full-KV Accuracy     = {full_acc:.4f}")
    print(f"DuoAttn Accuracy     = {duo_acc:.4f}")
    print(f"Accuracy gap         = {full_acc - duo_acc:+.4f}")
    print(f"Full-KV CE           = {mean_full_ce:.4f}")
    print(f"DuoAttn CE           = {mean_duo_ce:.4f}")
    print(f"CE gap               = {mean_duo_ce - mean_full_ce:+.4f}")
    print(f"per_sample_mean_MSE  = {per_sample_mse:.6f}")
    if track_cuda_peak_mem:
        print(f"model_weights_mem    = {model_mem_gb:.3f} GB")
        print(f"full_kv_peak_mem     = {result['full_kv_peak_mem_gb']:.3f} GB "
              f"(+{result['full_kv_peak_mem_gb'] - model_mem_gb:.3f} GB over model)")
        print(f"duo_demo_peak_mem    = {result['duo_demo_peak_mem_gb']:.3f} GB "
              f"(+{result['duo_demo_peak_mem_gb'] - model_mem_gb:.3f} GB over model, one-time)")
        print(f"duo_query_peak_mem   = {result['duo_query_peak_mem_gb']:.3f} GB "
              f"(+{result['duo_query_peak_mem_gb'] - model_mem_gb:.3f} GB over model, per-query)")
    print(f"total_full_flops     = {total_full_flops:.3e} ({n_queries} queries)")
    print(f"total_duo_flops      = {total_duo_flops:.3e} ({n_queries} queries)")
    if total_duo_flops > 0:
        print(f"flops_reduction      = {flops_reduction:.4f}x")

    print(f"\n{'Dataset':<20} {'N':>5} {'FullKV':>10} {'DuoAttn':>10} {'Gap':>10}")
    print(f"{'-'*55}")
    for ds in sorted(ds_results):
        r = ds_results[ds]
        n = len(r["gt"])
        fa = accuracy(r["full_p"], r["gt"])
        da = accuracy(r["duo_p"], r["gt"])
        print(f"{ds:<20} {n:>5} {fa:>10.4f} {da:>10.4f} {fa-da:>+10.4f}")

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
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "bf16", "fp16", "fp32"])

    # Streaming-cache config
    p.add_argument("--sink_tokens", type=int, default=4)
    p.add_argument("--recent_tokens", type=int, default=64)
    p.add_argument("--retrieval_ratio", type=float, default=0.5)
    p.add_argument("--retrieval_ratios", type=str, default="")

    # Gate-training hyper-parameters
    p.add_argument("--gate_lr", type=float, default=0.02)
    p.add_argument("--gate_lr_min", type=float, default=0.002)
    p.add_argument("--gate_lambda", type=float, default=0.05,
                   help="L1 regularisation weight for gates")
    p.add_argument("--gate_train_steps", type=int, default=2000)
    p.add_argument("--gate_warmup_steps", type=int, default=400)
    p.add_argument("--gate_cooldown_steps", type=int, default=400)
    p.add_argument("--gate_max_seq_len", type=int, default=2048)
    p.add_argument("--gate_num_passkeys", type=int, default=10)
    p.add_argument("--gate_passkey_len", type=int, default=32)
    p.add_argument("--gate_attn_chunk_size", type=int, default=256,
                   help="Query chunk size for gated attention to reduce VRAM.")

    # Run mode
    p.add_argument("--run_mode", type=str, default="identify_eval",
                   choices=["identify", "eval", "identify_eval"])
    p.add_argument("--save_gates_path", type=str, default="")
    p.add_argument("--load_gates_path", type=str, default="")

    args = p.parse_args()
    device = ("cuda" if torch.cuda.is_available()
              and args.device.startswith("cuda") else "cpu")

    tokenizer = setup_tokenizer(args.model_name)
    if args.dtype == "fp32":
        load_dtype = torch.float32
    elif args.dtype == "bf16":
        load_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        load_dtype = torch.float16
    else:
        if device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            load_dtype = torch.bfloat16
        elif device == "cuda":
            load_dtype = torch.float16
        else:
            load_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=load_dtype,
    ).to(device)
    print(f"[Config] device={device}, model_dtype={load_dtype}, "
          f"gate_attn_chunk_size={args.gate_attn_chunk_size}")
    model.eval()

    # ---- Phase 1: identify retrieval heads ----
    if args.run_mode in ("identify", "identify_eval"):
        gate_values = identify_retrieval_heads(
            model, tokenizer, model.config, args, device)

            # NOTE: we do NOT reorder model weights here. Instead, the
        # split-cache inference uses mask-based indexing so that any
        # retrieval_ratio works correctly without physical reordering.
        # (Physical reordering is a deployment optimisation for contiguous
        # slicing; it does not affect correctness.)

        if args.save_gates_path:
            torch.save(gate_values, args.save_gates_path)
            print(f"Saved gate values to: {args.save_gates_path}")

    elif args.load_gates_path:
        gate_values = torch.load(args.load_gates_path, map_location=device)
        print(f"Loaded gate values from: {args.load_gates_path}")
    else:
        raise ValueError("Must either identify heads or provide --load_gates_path")

    # ---- Phase 2: evaluate ----
    if args.run_mode in ("eval", "identify_eval"):
        ratios = ([float(x.strip()) for x in args.retrieval_ratios.split(",")]
                  if args.retrieval_ratios else [args.retrieval_ratio])

        all_results = []
        for ratio in ratios:
            res = run_eval(args, model, tokenizer, gate_values, ratio, device)
            all_results.append(res)

        if len(all_results) > 1:
            print(f"\n{'='*70}")
            print(f"  Summary: DuoAttention sweep")
            print(f"{'='*70}")
            print(f"{'Ratio':>8} {'Cache':>8} {'FullKV Acc':>12} {'DuoAcc':>12} "
                  f"{'Gap':>8} {'FullKV CE':>12} {'Duo CE':>12} {'CE Gap':>10} "
                  f"{'MSE':>10} {'FLOPs Red.':>12} {'Full FLOPs':>12} {'Duo FLOPs':>12} "
                  f"{'FullMem':>10} {'DuoQMem':>10}")
            print(f"{'-'*170}")
            for r in all_results:
                print(f"{r['retrieval_ratio']:>8.2f} {r['cache_size']:>8} "
                      f"{r['full_kv_acc']:>12.4f} {r['duo_acc']:>12.4f} "
                      f"{r['acc_gap']:>+8.4f} {r['full_ce']:>12.4f} "
                      f"{r['duo_ce']:>12.4f} {r['ce_gap']:>+10.4f} "
                      f"{r['per_sample_mean_MSE']:>10.6f} {r['flops_reduction']:>12.4f}x "
                      f"{r['total_full_flops']:>12.3e} {r['total_duo_flops']:>12.3e} "
                      f"{r['full_kv_peak_mem_gb']:>9.3f}G {r['duo_query_peak_mem_gb']:>9.3f}G")


if __name__ == "__main__":
    main()