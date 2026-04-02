import argparse
import copy
import gc
import math
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPT2Tokenizer
from transformers.cache_utils import DynamicCache


# =====================================================================
# GSM8K data loading
# =====================================================================

DEFAULT_SPLIT_DOWNSAMPLE = {
    "train": 1000,
    "test": 199,
}

def load_gsm8k(split="train", max_samples=0):
    """
    Load GSM8K from HuggingFace datasets.
    Returns list of dicts: input, output, answer_number, task, dataset
    """
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    data = []
    for ex in ds:
        q = ex["question"]
        a = ex["answer"]
        num = extract_answer_number(a)
        data.append({
            "input": q,
            "output": a,
            "answer_number": num,
            "task": "gsm8k",
            "dataset": "gsm8k",
        })
    if max_samples <= 0:
        max_samples = DEFAULT_SPLIT_DOWNSAMPLE.get(split, 0)
    if max_samples > 0:
        data = data[:max_samples]
    return data


def extract_answer_number(text: str) -> str:
    match = re.search(r"####\s*(.+?)$", text.strip(), re.MULTILINE)
    if match:
        return match.group(1).strip().replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else ""


def extract_generated_answer(text: str) -> str:
    match = re.search(r"####\s*(.+?)$", text.strip(), re.MULTILINE)
    if match:
        return match.group(1).strip().replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else ""


# =====================================================================
# HiPPO utilities
# =====================================================================

def _hippo_legs_matrix(size, device, dtype):
    n = torch.arange(size, device=device, dtype=dtype)
    p = torch.sqrt(2.0 * n + 1.0)
    a = torch.zeros(size, size, device=device, dtype=dtype)
    lower = torch.tril(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=-1)
    a[lower] = -(p[:, None] * p[None, :])[lower]
    a.diagonal().copy_(-(n + 1.0))
    return a

def _hippo_legs_input_vector(size, device, dtype):
    return torch.sqrt(2.0 * torch.arange(size, device=device, dtype=dtype) + 1.0)


# =====================================================================
# Per-group SSM compressor
# =====================================================================

class LayerGroupSSM(nn.Module):
    def __init__(self, hidden_size, ssm_dim, num_layers_in_group, num_kv_heads,
                 head_dim, num_virtual_tokens, dt=1.0, scan_chunk_size=64):
        super().__init__()
        self.ssm_dim = ssm_dim
        self.num_layers_in_group = num_layers_in_group
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_virtual_tokens = num_virtual_tokens
        self.scan_chunk_size = scan_chunk_size
        a_cont = _hippo_legs_matrix(ssm_dim, torch.device("cpu"), torch.float32)
        a_disc = torch.matrix_exp(dt * a_cont)
        self.register_buffer("A", a_disc)
        self.B = nn.Linear(hidden_size, ssm_dim)
        self._init_hippo_B(hidden_size)
        inter = ssm_dim * 4
        per_layer_kv = 2 * num_kv_heads * num_virtual_tokens * head_dim
        self.proj = nn.Sequential(
            nn.Linear(ssm_dim, inter), nn.SiLU(),
            nn.Linear(inter, inter), nn.SiLU(),
            nn.Linear(inter, num_layers_in_group * per_layer_kv),
        )

    def _init_hippo_B(self, hidden_size):
        with torch.no_grad():
            nn.init.normal_(self.B.weight, std=1.0 / math.sqrt(hidden_size))
            b = _hippo_legs_input_vector(self.ssm_dim, self.B.weight.device, self.B.weight.dtype)
            self.B.weight.mul_(b.unsqueeze(1))
            nn.init.zeros_(self.B.bias)

    def scan(self, x, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.scan_chunk_size
        bsz, T, _ = x.shape
        device = x.device
        orig_dtype = x.dtype
        D = self.ssm_dim
        u_all = self.B(x).float()
        state = torch.zeros(bsz, D, device=device, dtype=torch.float32)
        A = self.A.to(device=device, dtype=torch.float32)
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            C = end - start
            u_chunk = u_all[:, start:end, :]
            if C == 1:
                state = (state @ A.T) + u_chunk[:, 0]
                continue
            weights = torch.zeros(C, D, D, device=device, dtype=torch.float32)
            weights[C - 1] = torch.eye(D, device=device, dtype=torch.float32)
            if C >= 2:
                weights[C - 2] = A
            for t in range(C - 3, -1, -1):
                weights[t] = weights[t + 1] @ A
            total_contrib = torch.einsum('cdj,bcj->bd', weights, u_chunk)
            A_C = weights[0] @ A
            state = (state @ A_C.T) + total_contrib
        return state.to(orig_dtype)

    def forward(self, x):
        bsz = x.shape[0]
        state = self.scan(x)
        flat = self.proj(state)
        r = flat.view(bsz, self.num_layers_in_group, 2,
                       self.num_kv_heads, self.num_virtual_tokens, self.head_dim)
        return [(r[:, i, 0], r[:, i, 1]) for i in range(self.num_layers_in_group)]


class LayerGroupSSMSidecar(nn.Module):
    def __init__(self, config, ssm_dim=512, num_virtual_tokens=16, num_groups=4,
                 dt=1.0, effective_num_kv_heads=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.full_num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        if effective_num_kv_heads is None:
            self.num_kv_heads = self.full_num_kv_heads
        else:
            self.num_kv_heads = max(1, min(self.full_num_kv_heads, int(effective_num_kv_heads)))
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_virtual_tokens = num_virtual_tokens
        self.num_groups = num_groups
        self.ssm_dim = ssm_dim
        base, rem = divmod(self.num_layers, num_groups)
        self.group_ranges = []
        s = 0
        for g in range(num_groups):
            e = s + base + (1 if g < rem else 0)
            self.group_ranges.append((s, e))
            s = e
        self.groups = nn.ModuleList([
            LayerGroupSSM(config.hidden_size, ssm_dim, e - s, self.num_kv_heads,
                          self.head_dim, num_virtual_tokens, dt)
            for s, e in self.group_ranges
        ])
        info = ", ".join(f"G{g}:[{s},{e})" for g, (s, e) in enumerate(self.group_ranges))
        print(f"[Sidecar] {num_groups} groups, ssm_dim={ssm_dim}, vtokens={num_virtual_tokens}, "
              f"kv_heads={self.num_kv_heads}/{self.full_num_kv_heads} | {info}")

    def forward(self, embeddings):
        all_kv = []
        for group in self.groups:
            all_kv.extend(group(embeddings))
        return all_kv


# =====================================================================
# Cache utilities
# =====================================================================

def _ensure_cache_obj(past):
    if past is None: return None
    if isinstance(past, DynamicCache): return past
    if isinstance(past, tuple):
        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(past)
        return DynamicCache(past)
    return past

def _cache_to_legacy(past):
    if past is None: return None
    if isinstance(past, tuple): return past
    if hasattr(past, "to_legacy_cache"): return past.to_legacy_cache()
    return tuple(past)

def _build_cache_from_kv_list(kv_list, sink_kv=None, target_dtype=None, expected_num_kv_heads=None):
    cache = DynamicCache()
    sink_legacy = _cache_to_legacy(_ensure_cache_obj(sink_kv)) if sink_kv is not None else None
    for i, (vk, vv) in enumerate(kv_list):
        if target_dtype is not None:
            vk = vk.to(dtype=target_dtype)
            vv = vv.to(dtype=target_dtype)
        if expected_num_kv_heads is not None and expected_num_kv_heads > 0:
            v_heads = vk.shape[1]
            if v_heads != expected_num_kv_heads:
                if expected_num_kv_heads > v_heads and (expected_num_kv_heads % v_heads == 0):
                    rep = expected_num_kv_heads // v_heads
                    vk = vk.repeat_interleave(rep, dim=1)
                    vv = vv.repeat_interleave(rep, dim=1)
                elif v_heads > expected_num_kv_heads and (v_heads % expected_num_kv_heads == 0):
                    g = v_heads // expected_num_kv_heads
                    vk = vk.view(vk.shape[0], expected_num_kv_heads, g, vk.shape[2], vk.shape[3]).mean(dim=2)
                    vv = vv.view(vv.shape[0], expected_num_kv_heads, g, vv.shape[2], vv.shape[3]).mean(dim=2)
                else:
                    h = min(v_heads, expected_num_kv_heads)
                    vk, vv = vk[:, :h], vv[:, :h]
        if sink_legacy is not None:
            sk, sv = sink_legacy[i]
            if target_dtype is not None:
                sk = sk.to(dtype=target_dtype)
                sv = sv.to(dtype=target_dtype)
            s_heads, v_heads = sk.shape[1], vk.shape[1]
            if s_heads != v_heads:
                if s_heads > v_heads and (s_heads % v_heads == 0):
                    g = s_heads // v_heads
                    sk = sk.view(sk.shape[0], v_heads, g, sk.shape[2], sk.shape[3]).mean(dim=2)
                    sv = sv.view(sv.shape[0], v_heads, g, sv.shape[2], sv.shape[3]).mean(dim=2)
                elif v_heads > s_heads and (v_heads % s_heads == 0):
                    rep = v_heads // s_heads
                    sk = sk.repeat_interleave(rep, dim=1)
                    sv = sv.repeat_interleave(rep, dim=1)
                else:
                    h = min(s_heads, v_heads)
                    sk, sv = sk[:, :h], sv[:, :h]
                    vk, vv = vk[:, :h], vv[:, :h]
            k, v = torch.cat([sk, vk], dim=2), torch.cat([sv, vv], dim=2)
        else:
            k, v = vk, vv
        cache.update(k, v, i)
    return cache

def _fresh_cache_copy(cache):
    legacy = _cache_to_legacy(cache)
    return _ensure_cache_obj(tuple((k.clone(), v.clone()) for k, v in legacy))

def _pos_ids(start, length, device, bsz=1):
    p = torch.arange(start, start + length, dtype=torch.long, device=device).unsqueeze(0)
    return p.expand(bsz, -1) if bsz > 1 else p


# =====================================================================
# KV matching loss
# =====================================================================

def kv_matching_loss(virtual_kv, teacher_kv, num_virtual_tokens, loss_type="cosine"):
    if isinstance(teacher_kv, list):
        t_kv = teacher_kv
    else:
        t_kv = list(_cache_to_legacy(_ensure_cache_obj(teacher_kv)))
    device = virtual_kv[0][0].device
    total = torch.tensor(0.0, device=device)
    n_terms = 0
    for layer_idx, (vk, vv) in enumerate(virtual_kv):
        tk, tv = t_kv[layer_idx]
        s_heads, t_heads = vk.shape[1], tk.shape[1]
        if s_heads != t_heads:
            if t_heads > s_heads and (t_heads % s_heads == 0):
                g = t_heads // s_heads
                tk = tk.view(tk.shape[0], s_heads, g, tk.shape[2], tk.shape[3]).mean(dim=2)
                tv = tv.view(tv.shape[0], s_heads, g, tv.shape[2], tv.shape[3]).mean(dim=2)
            elif s_heads > t_heads and (s_heads % t_heads == 0):
                g = s_heads // t_heads
                vk = vk.view(vk.shape[0], t_heads, g, vk.shape[2], vk.shape[3]).mean(dim=2)
                vv = vv.view(vv.shape[0], t_heads, g, vv.shape[2], vv.shape[3]).mean(dim=2)
            else:
                h = min(s_heads, t_heads)
                vk, vv = vk[:, :h], vv[:, :h]
                tk, tv = tk[:, :h], tv[:, :h]
        T_full = tk.shape[2]
        chunk = max(1, T_full // num_virtual_tokens)
        for vi in range(num_virtual_tokens):
            s = vi * chunk
            e = min(s + chunk, T_full)
            if s >= T_full: break
            tk_p = tk[:, :, s:e].mean(dim=2)
            tv_p = tv[:, :, s:e].mean(dim=2)
            vk_i = vk[:, :, vi]
            vv_i = vv[:, :, vi]
            if loss_type == "cosine":
                lk = 1.0 - F.cosine_similarity(vk_i.flatten(1), tk_p.flatten(1), dim=-1).mean()
                lv = 1.0 - F.cosine_similarity(vv_i.flatten(1), tv_p.flatten(1), dim=-1).mean()
            else:
                lk = F.mse_loss(vk_i, tk_p)
                lv = F.mse_loss(vv_i, tv_p)
            total = total + lk + lv
            n_terms += 2
    return total / max(n_terms, 1)


# =====================================================================
# Teacher / student forward with optional hidden states
# =====================================================================

@torch.no_grad()
def _teacher_qa_forward(model, demo_ids, q_ids, ans_ids, output_hidden=False):
    all_ids = torch.cat([demo_ids, q_ids, ans_ids], dim=1)
    out = model(input_ids=all_ids, use_cache=False, output_hidden_states=output_hidden)
    start = demo_ids.shape[1] + q_ids.shape[1] - 1
    ans_len = ans_ids.shape[1]
    logits = out.logits[:, start:start + ans_len, :]
    if output_hidden:
        last_hidden = out.hidden_states[-1][:, start:start + ans_len, :]
        return logits, last_hidden
    return logits, None


def _student_qa_forward(model, virtual_kv_list, sink_kv, demo_len, true_demo_len,
                        align_true_positions, q_ids, ans_ids, output_hidden=False):
    model_kv_dtype = model.get_input_embeddings().weight.dtype
    model_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    cache = _build_cache_from_kv_list(
        virtual_kv_list, sink_kv=sink_kv, target_dtype=model_kv_dtype,
        expected_num_kv_heads=model_kv_heads)
    pos_q_start = true_demo_len if align_true_positions else demo_len
    attn_q = torch.ones(1, demo_len + q_ids.shape[1], dtype=torch.long, device=q_ids.device)
    out_q = model(input_ids=q_ids, past_key_values=cache, attention_mask=attn_q,
                  position_ids=_pos_ids(pos_q_start, q_ids.shape[1], q_ids.device),
                  use_cache=True, output_hidden_states=False)
    first_logit = out_q.logits[:, -1:]
    base = demo_len + q_ids.shape[1]
    pos_a_start = pos_q_start + q_ids.shape[1]
    attn_a = torch.ones(1, base + ans_ids.shape[1], dtype=torch.long, device=q_ids.device)
    out_a = model(input_ids=ans_ids, past_key_values=out_q.past_key_values,
                  attention_mask=attn_a,
                  position_ids=_pos_ids(pos_a_start, ans_ids.shape[1], q_ids.device),
                  use_cache=False, output_hidden_states=output_hidden)
    if ans_ids.shape[1] == 1:
        logits = first_logit
    else:
        logits = torch.cat([first_logit, out_a.logits[:, :-1]], dim=1)
    if output_hidden:
        last_hidden = out_a.hidden_states[-1]
        return logits, last_hidden
    return logits, None


@torch.no_grad()
def _teacher_qa_logits_nocache(model, demo_ids, q_ids, ans_ids):
    logits, _ = _teacher_qa_forward(model, demo_ids, q_ids, ans_ids, output_hidden=False)
    return logits

def _student_qa_logits(model, virtual_kv_list, sink_kv, demo_len, true_demo_len,
                       align_true_positions, q_ids, ans_ids):
    logits, _ = _student_qa_forward(model, virtual_kv_list, sink_kv, demo_len,
                                     true_demo_len, align_true_positions, q_ids, ans_ids,
                                     output_hidden=False)
    return logits


# =====================================================================
# Hidden-state matching loss
# =====================================================================

def hidden_state_loss(student_hidden, teacher_hidden):
    min_t = min(student_hidden.size(1), teacher_hidden.size(1))
    if min_t == 0:
        return torch.tensor(0.0, device=student_hidden.device)
    sh = student_hidden[:, :min_t, :].reshape(-1, student_hidden.size(-1))
    th = teacher_hidden[:, :min_t, :].reshape(-1, teacher_hidden.size(-1))
    return (1.0 - F.cosine_similarity(sh, th, dim=-1)).mean()


# =====================================================================
# Generation helpers (for CoT eval)
# =====================================================================

@torch.no_grad()
def full_kv_generate(model, tokenizer, demo_ids, q_ids, device,
                     max_new_tokens=512, temperature=0.0, stop_strings=None):
    if stop_strings is None:
        stop_strings = ["\n\nQ:", "\n\nQuestion:", "\nQ:"]
    prefix = torch.cat([demo_ids, q_ids], dim=1)
    out = model(input_ids=prefix, use_cache=True)
    logits = out.logits[:, -1, :]
    past = out.past_key_values
    generated_ids = []
    for _ in range(max_new_tokens):
        if temperature <= 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1)
        token_id = next_id.item()
        generated_ids.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break
        partial = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if any(ss in partial for ss in stop_strings):
            break
        if "####" in partial:
            after = partial.split("####")[-1].strip()
            if after and (len(after) > 10 or "\n" in after):
                break
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


@torch.no_grad()
def ssm_generate(model, tokenizer, demo_cache, demo_len, true_demo_len,
                 align_true_positions, q_text, device,
                 max_new_tokens=512, temperature=0.0, stop_strings=None):
    if stop_strings is None:
        stop_strings = ["\n\nQ:", "\n\nQuestion:", "\nQ:"]
    q_ids = tokenizer(q_text, return_tensors="pt",
                      add_special_tokens=False)["input_ids"].to(device)
    if q_ids.numel() == 0:
        return ""
    cache = _fresh_cache_copy(demo_cache)
    ctx_len = demo_len + q_ids.shape[1]
    attn = torch.ones(1, ctx_len, dtype=torch.long, device=device)
    pos_start = true_demo_len if align_true_positions else demo_len
    out = model(input_ids=q_ids, past_key_values=cache, attention_mask=attn,
                position_ids=_pos_ids(pos_start, q_ids.shape[1], device), use_cache=True)
    logits = out.logits[:, -1, :]
    past = out.past_key_values
    cur_pos = pos_start + q_ids.shape[1]
    cur_ctx = ctx_len
    generated_ids = []
    for _ in range(max_new_tokens):
        if temperature <= 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1)
        token_id = next_id.item()
        generated_ids.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break
        partial = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if any(ss in partial for ss in stop_strings):
            break
        if "####" in partial:
            after = partial.split("####")[-1].strip()
            if after and (len(after) > 10 or "\n" in after):
                break
        cur_ctx += 1
        attn = torch.ones(1, cur_ctx, dtype=torch.long, device=device)
        out = model(input_ids=next_id, past_key_values=past, attention_mask=attn,
                    position_ids=_pos_ids(cur_pos, 1, device), use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        cur_pos += 1
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# =====================================================================
# GSM8K text formatting
# =====================================================================

def _gsm8k_format_qa(question, answer, is_first, add_newlines):
    prefix = "" if is_first else "\n\n"
    if add_newlines:
        return f"{prefix}Q: {question}\nA: {answer}"
    else:
        return f"{prefix}Q: {question} A: {answer}"

def _gsm8k_format_question(question, add_newlines):
    if add_newlines:
        return f"\n\nQ: {question}\nA:"
    else:
        return f" Q: {question} A:"

def _gsm8k_build_demo_text(demos, add_newlines):
    parts = []
    for i, dp in enumerate(demos):
        parts.append(_gsm8k_format_qa(dp["input"], dp["output"],
                                       is_first=(i == 0), add_newlines=add_newlines))
    return "".join(parts)


# =====================================================================
# Training helpers
# =====================================================================

def _freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

def _load_causal_lm(args, device):
    if args.is_quant:
        if not torch.cuda.is_available():
            raise RuntimeError("--is_quant requires CUDA.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        return AutoModelForCausalLM.from_pretrained(
            args.model_name, quantization_config=bnb_config,
            device_map="auto", torch_dtype=torch.float16)
    return AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

def _setup_tokenizer(name):
    tok = GPT2Tokenizer.from_pretrained(name) if name.startswith("gpt2") else AutoTokenizer.from_pretrained(name)
    if tok.padding_side == "left": tok.padding_side = "right"
    if tok.eos_token_id is None and tok.sep_token is not None: tok.eos_token = tok.sep_token
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if tok.bos_token_id is None: tok.bos_token = tok.eos_token
    return tok

def _resolve_sidecar_kv_heads(args, model_config):
    full_heads = getattr(model_config, "num_key_value_heads", model_config.num_attention_heads)
    if args.is_quant and args.quant_sidecar_kv_heads > 0:
        return min(full_heads, int(args.quant_sidecar_kv_heads))
    return full_heads

def _generate_diverse_demo_texts(data, k, num_sets, seed, add_nl):
    rng = random.Random(seed)
    n = len(data)
    texts = []
    for _ in range(num_sets):
        indices = rng.sample(range(n), min(k, n))
        demos = [data[i] for i in indices]
        texts.append(_gsm8k_build_demo_text(demos, add_nl))
    return texts

def _choose_fixed_demos(data, k, strategy, seed):
    if strategy == "first": return data[:k]
    return [data[i] for i in random.Random(seed).sample(range(len(data)), k)]

def _teacher_demo_kv(model, demo_ids):
    with torch.no_grad():
        return _cache_to_legacy(model(input_ids=demo_ids, use_cache=True).past_key_values)

def _student_virtual_kv(model, sidecar, demo_ids):
    emb = model.get_input_embeddings()(demo_ids)
    return sidecar(emb.float())

def _strip_leading_format_tokens(ans_ids, tokenizer):
    strip_tokens = set()
    for ch in ["\n", " ", "\t"]:
        strip_tokens.update(tokenizer(ch, add_special_tokens=False)["input_ids"])
    idx = 0
    while idx < ans_ids.shape[1] and ans_ids[0, idx].item() in strip_tokens:
        idx += 1
    if idx >= ans_ids.shape[1]:
        return ans_ids, 0
    return ans_ids[:, idx:], idx

def _load_sidecar(sidecar, args, device, state_dict=None):
    _FROZEN = ('.A', '.A_powers', '.eigenvalues', '.V', '.V_inv')
    if state_dict is not None:
        filtered = {k: v for k, v in state_dict.items() if not any(k.endswith(s) for s in _FROZEN)}
        sidecar.load_state_dict(filtered, strict=False)
        print("Loaded sidecar from in-memory state.")
    elif args.load_sidecar_path:
        ckpt = torch.load(args.load_sidecar_path, map_location=device)
        st = ckpt["sidecar"] if isinstance(ckpt, dict) and "sidecar" in ckpt else ckpt
        filtered = {k: v for k, v in st.items() if not any(k.endswith(s) for s in _FROZEN)}
        sidecar.load_state_dict(filtered, strict=False)
        print(f"Loaded sidecar from: {args.load_sidecar_path}")
    else:
        print("WARN: No sidecar checkpoint.")


# =====================================================================
# Training loop (GSM8K)
# =====================================================================

def run_distillation_training(args):
    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    tokenizer = _setup_tokenizer(args.model_name)
    model = _load_causal_lm(args, device)
    _freeze(model)

    align = args.align_true_positions
    if align and getattr(model.config, "model_type", "") == "qwen2":
        print("WARN: align_true_positions disabled for Qwen2."); align = False

    add_nl = not args.model_name.startswith("gpt2")

    retrieval_data = load_gsm8k(split=args.retrieval_split, max_samples=args.max_retrieval_samples)
    train_data = load_gsm8k(split=args.train_split, max_samples=args.max_train_samples)
    if not retrieval_data or not train_data:
        raise ValueError("Empty data.")

    demo_texts = _generate_diverse_demo_texts(
        retrieval_data, args.k, args.num_demo_sets, args.seed, add_nl)
    print(f"Generated {len(demo_texts)} diverse demo texts (k={args.k}).")

    sidecar = LayerGroupSSMSidecar(
        model.config, ssm_dim=args.ssm_dim, num_virtual_tokens=args.num_virtual_tokens,
        num_groups=args.num_groups,
        effective_num_kv_heads=_resolve_sidecar_kv_heads(args, model.config),
    ).to(device=device, dtype=torch.float32)

    if args.load_sidecar_path:
        _load_sidecar(sidecar, args, device)
    sidecar.train()

    optimizer = torch.optim.AdamW(sidecar.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    total_steps = args.epochs * len(demo_texts)
    warmup = min(100, total_steps // 10)
    def lr_fn(step):
        if step < warmup: return step / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    rng = random.Random(args.seed)
    step = 0
    running = {"loss": 0.0, "kv": 0.0, "logit": 0.0, "hid": 0.0, "ce": 0.0}
    qa_batch_size = max(1, args.train_batch_size)

    use_kv = args.loss_w_kv > 0
    use_logit = args.loss_w_logit > 0
    use_hid = args.loss_w_hid > 0
    use_ce = args.loss_w_ce > 0
    need_qa = use_logit or use_hid or use_ce

    print(f"\n===== Training (GSM8K CoT): epochs={args.epochs}, demo_sets={len(demo_texts)}, "
          f"ssm_dim={args.ssm_dim}, groups={args.num_groups}, "
          f"vtokens={args.num_virtual_tokens}, sink={args.sink_tokens}")
    print(f"  loss weights: kv={args.loss_w_kv}, logit={args.loss_w_logit}, "
          f"hid={args.loss_w_hid}, ce={args.loss_w_ce}\n")

    for epoch in range(args.epochs):
        order = list(range(len(demo_texts)))
        rng.shuffle(order)
        for di in tqdm(order, desc=f"epoch-{epoch+1}"):
            demo_text = demo_texts[di]
            demo_ids = tokenizer(demo_text, return_tensors="pt",
                                 add_special_tokens=False)["input_ids"].to(device)
            if demo_ids.numel() == 0: continue
            true_demo_len = demo_ids.shape[1]

            virtual_kv = _student_virtual_kv(model, sidecar, demo_ids)

            sink_kv = None
            st = min(args.sink_tokens, demo_ids.shape[1])
            if st > 0:
                with torch.no_grad():
                    sink_kv = model(input_ids=demo_ids[:, :st], use_cache=True).past_key_values
            demo_len = st + sidecar.num_virtual_tokens

            # ── L_KV ──
            loss_kv = torch.tensor(0.0, device=device)
            if use_kv:
                teacher_kv = _teacher_demo_kv(model, demo_ids)
                loss_kv = kv_matching_loss(virtual_kv, teacher_kv, sidecar.num_virtual_tokens,
                                           loss_type=args.kv_loss_type)

            # ── L_Logit + L_hid + L_CE ──
            loss_logit = torch.tensor(0.0, device=device)
            loss_hid = torch.tensor(0.0, device=device)
            loss_ce = torch.tensor(0.0, device=device)
            if need_qa:
                qa_indices = rng.sample(range(len(train_data)), min(qa_batch_size, len(train_data)))
                valid = 0
                for qi in qa_indices:
                    dp = train_data[qi]
                    q_text = _gsm8k_format_question(dp["input"], add_nl)
                    ans_text = " " + dp["output"]

                    q_ids = tokenizer(q_text, return_tensors="pt",
                                      add_special_tokens=False)["input_ids"].to(device)
                    a_ids = tokenizer(ans_text, return_tensors="pt",
                                      add_special_tokens=False)["input_ids"].to(device)
                    if q_ids.numel() == 0 or a_ids.numel() == 0: continue

                    # Truncate long CoT
                    if a_ids.shape[1] > args.max_answer_tokens:
                        a_ids = a_ids[:, :args.max_answer_tokens]

                    with torch.no_grad():
                        t_logits, t_hidden = _teacher_qa_forward(
                            model, demo_ids, q_ids, a_ids, output_hidden=use_hid)

                    s_logits, s_hidden = _student_qa_forward(
                        model, virtual_kv, sink_kv, demo_len, true_demo_len,
                        align, q_ids, a_ids, output_hidden=use_hid)

                    if use_logit:
                        temp = args.kd_temperature
                        kl = F.kl_div(
                            F.log_softmax(s_logits / temp, dim=-1),
                            F.softmax(t_logits.detach() / temp, dim=-1),
                            reduction="batchmean") * (temp ** 2)
                        loss_logit = loss_logit + kl

                    if use_hid and s_hidden is not None and t_hidden is not None:
                        loss_hid = loss_hid + hidden_state_loss(s_hidden, t_hidden.detach())

                    if use_ce:
                        a_stripped, n_stripped = _strip_leading_format_tokens(a_ids, tokenizer)
                        if a_stripped.numel() > 0:
                            s_logits_stripped = s_logits[:, n_stripped:, :]
                            min_t = min(s_logits_stripped.size(1), a_stripped.size(1))
                            if min_t > 0:
                                ce = F.cross_entropy(
                                    s_logits_stripped[:, :min_t].reshape(-1, s_logits_stripped.size(-1)),
                                    a_stripped[:, :min_t].reshape(-1), reduction="mean")
                                loss_ce = loss_ce + ce

                    valid += 1
                if valid > 0:
                    loss_logit = loss_logit / valid
                    loss_hid = loss_hid / valid
                    loss_ce = loss_ce / valid

            loss = (args.loss_w_kv * loss_kv + args.loss_w_logit * loss_logit
                    + args.loss_w_hid * loss_hid + args.loss_w_ce * loss_ce)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sidecar.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            running["loss"] += loss.item()
            running["kv"] += (loss_kv.item() if isinstance(loss_kv, torch.Tensor) else loss_kv)
            running["logit"] += (loss_logit.item() if isinstance(loss_logit, torch.Tensor) else loss_logit)
            running["hid"] += (loss_hid.item() if isinstance(loss_hid, torch.Tensor) else loss_hid)
            running["ce"] += (loss_ce.item() if isinstance(loss_ce, torch.Tensor) else loss_ce)

            if step % args.log_every == 0:
                d = float(args.log_every)
                print(f"step={step} loss={running['loss']/d:.4f} "
                      f"kv={running['kv']/d:.4f} logit={running['logit']/d:.4f} "
                      f"hid={running['hid']/d:.4f} ce={running['ce']/d:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.6f}")
                running = {"loss": 0.0, "kv": 0.0, "logit": 0.0, "hid": 0.0, "ce": 0.0}

            del virtual_kv, sink_kv, demo_ids, loss, loss_kv, loss_logit, loss_hid, loss_ce
            if device.startswith("cuda") and args.empty_cache_every > 0 and (step % args.empty_cache_every == 0):
                gc.collect(); torch.cuda.empty_cache()
            if 0 < args.max_steps <= step: break
        if 0 < args.max_steps <= step: break

    if args.save_sidecar_path:
        torch.save({"sidecar": sidecar.state_dict(), "args": vars(args),
                     "model_name": args.model_name}, args.save_sidecar_path)
        print(f"Saved: {args.save_sidecar_path}")
    return {k: v.detach().cpu() for k, v in sidecar.state_dict().items()}


# =====================================================================
# Eval: CoT generation + teacher-forced fidelity metrics
# =====================================================================

def run_experiment(args, sidecar_state_dict=None):
    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    tokenizer = _setup_tokenizer(args.model_name)
    model = _load_causal_lm(args, device)
    model.eval()

    align = args.align_true_positions
    if align and getattr(model.config, "model_type", "") == "qwen2":
        print("WARN: align_true_positions disabled for Qwen2."); align = False

    add_nl = not args.model_name.startswith("gpt2")

    retrieval_data = load_gsm8k(split=args.retrieval_split, max_samples=args.max_retrieval_samples)
    eval_data = load_gsm8k(split=args.eval_split, max_samples=args.max_eval_samples)
    if not retrieval_data or not eval_data:
        raise ValueError("Empty data.")

    per_query_random = (args.num_eval_demo_sets > 1)

    sidecar = LayerGroupSSMSidecar(
        model.config, ssm_dim=args.ssm_dim, num_virtual_tokens=args.num_virtual_tokens,
        num_groups=args.num_groups,
        effective_num_kv_heads=_resolve_sidecar_kv_heads(args, model.config),
    ).to(device=device, dtype=torch.float32)
    _load_sidecar(sidecar, args, device, sidecar_state_dict)
    sidecar.eval()

    rng_eval = random.Random(args.seed + 7777)

    # ── Accumulators ──
    full_kv_correct, ssm_correct, total = 0, 0, 0
    full_losses, ssm_losses = [], []
    logit_kls, logit_cossims, top1_agrees, top5_overlaps = [], [], [], []
    query_logit_mses = []
    examples_log = []

    if per_query_random:
        print(f"\nPer-query random demos (k={args.k}) from {len(retrieval_data)} candidates.")
        fixed_demos = None
    else:
        print(f"\nFixed demos (strategy={args.demo_strategy}).")
        fixed_demos = _choose_fixed_demos(retrieval_data, args.k, args.demo_strategy, args.seed)

    for dp_idx, dp in enumerate(tqdm(eval_data, desc="eval-gsm8k")):
        q_raw = dp["input"]
        gt_number = dp["answer_number"]

        # Pick demos
        if per_query_random:
            indices = rng_eval.sample(range(len(retrieval_data)),
                                      min(args.k, len(retrieval_data)))
            demos = [retrieval_data[i] for i in indices]
        else:
            demos = fixed_demos

        demo_text = _gsm8k_build_demo_text(demos, add_newlines=add_nl)
        demo_ids = tokenizer(demo_text, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(device)
        original_demo_len = demo_ids.shape[1]

        q_text = _gsm8k_format_question(q_raw, add_nl)
        q_ids = tokenizer(q_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)

        # Full CoT answer for teacher-forced metrics
        ans_text = " " + dp["output"]
        a_ids = tokenizer(ans_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
        # Truncate for memory
        max_a = args.max_answer_tokens
        if a_ids.shape[1] > max_a:
            a_ids = a_ids[:, :max_a]

        # ─── Full-KV generation ───
        with torch.no_grad():
            full_gen = full_kv_generate(
                model, tokenizer, demo_ids, q_ids, device,
                max_new_tokens=args.max_gen_tokens)
        full_pred = extract_generated_answer(full_gen)
        full_match = (full_pred == gt_number)
        full_kv_correct += int(full_match)

        # ─── Full-KV teacher-forced CE ───
        t_logits_s = None
        first_token_id = None
        a_stripped, n_stripped = _strip_leading_format_tokens(a_ids, tokenizer)
        if q_ids.numel() > 0 and a_stripped.numel() > 0:
            with torch.no_grad():
                t_logits = _teacher_qa_logits_nocache(model, demo_ids, q_ids, a_ids)
                t_logits_s = t_logits[:, n_stripped:, :]
                first_token_id = a_stripped[:, 0].unsqueeze(1)
                t_ce = F.cross_entropy(
                    t_logits_s.reshape(-1, t_logits_s.size(-1)),
                    a_stripped.reshape(-1), reduction="mean").item()
                full_losses.append(t_ce)

        # ─── SSM: compress demos ───
        with torch.no_grad():
            emb = model.get_input_embeddings()(demo_ids).to(dtype=next(sidecar.parameters()).dtype)
            virtual_kv = sidecar(emb)
            sink_kv = None
            st = min(args.sink_tokens, demo_ids.shape[1])
            if st > 0:
                sink_kv = model(input_ids=demo_ids[:, :st], use_cache=True).past_key_values
            ssm_demo_len = st + sidecar.num_virtual_tokens

        # ─── SSM generation ───
        model_kv_dtype = model.get_input_embeddings().weight.dtype
        model_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        ssm_cache = _build_cache_from_kv_list(
            virtual_kv, sink_kv=sink_kv, target_dtype=model_kv_dtype,
            expected_num_kv_heads=model_kv_heads)

        with torch.no_grad():
            ssm_gen = ssm_generate(
                model, tokenizer, ssm_cache, ssm_demo_len, original_demo_len,
                align, q_text, device, max_new_tokens=args.max_gen_tokens)
        ssm_pred = extract_generated_answer(ssm_gen)
        ssm_match = (ssm_pred == gt_number)
        ssm_correct += int(ssm_match)

        total += 1

        # ─── SSM teacher-forced CE + fidelity metrics ───
        s_logits_s = None
        if q_ids.numel() > 0 and a_stripped.numel() > 0:
            with torch.no_grad():
                s_logits = _student_qa_logits(
                    model, virtual_kv, sink_kv,
                    ssm_demo_len, original_demo_len, align, q_ids, a_ids)
                s_logits_s = s_logits[:, n_stripped:, :]
                s_ce = F.cross_entropy(
                    s_logits_s.reshape(-1, s_logits_s.size(-1)),
                    a_stripped.reshape(-1), reduction="mean").item()
                ssm_losses.append(s_ce)

        # ─── Logit fidelity (teacher-forced, SSM vs Full-KV) ───
        if t_logits_s is not None and s_logits_s is not None:
            min_t = min(t_logits_s.size(1), s_logits_s.size(1))
            if min_t > 0:
                tl = t_logits_s[:, :min_t, :]
                sl = s_logits_s[:, :min_t, :]

                # KL divergence
                logit_kls.append(F.kl_div(
                    F.log_softmax(sl, dim=-1), F.softmax(tl, dim=-1),
                    reduction="batchmean").item())

                # Cosine similarity
                logit_cossims.append(F.cosine_similarity(
                    sl.reshape(-1, sl.size(-1)),
                    tl.reshape(-1, tl.size(-1)), dim=-1).mean().item())

                # Top-1 agreement
                top1_agrees.append(
                    (sl.argmax(-1) == tl.argmax(-1)).float().mean().item())

                # Top-5 overlap
                t5_t = tl.topk(5, dim=-1).indices
                t5_s = sl.topk(5, dim=-1).indices
                ovlp = sum(
                    len(set(t5_t[0, t].tolist()) & set(t5_s[0, t].tolist())) / 5.0
                    for t in range(min_t))
                top5_overlaps.append(ovlp / min_t)

                # Per-query relative squared error on first content token logit
                if first_token_id is not None:
                    t_first = tl[:, 0, :].gather(1, first_token_id).squeeze(1)
                    s_first = sl[:, 0, :].gather(1, first_token_id).squeeze(1)
                    t_val = float(t_first.item())
                    s_val = float(s_first.item())
                    denom = max(s_val, t_val)
                    rel_sq = ((s_val - t_val) / denom) ** 2 if denom != 0 else 0.0
                    query_logit_mses.append(rel_sq)

        # Log a few examples
        if dp_idx < 5:
            examples_log.append({
                "question": q_raw[:120],
                "gt": gt_number,
                "full_pred": full_pred,
                "ssm_pred": ssm_pred,
                "full_gen": full_gen[:200],
                "ssm_gen": ssm_gen[:200],
            })

        # Cleanup
        del virtual_kv, sink_kv, ssm_cache
        if device.startswith("cuda") and dp_idx % 50 == 0:
            gc.collect(); torch.cuda.empty_cache()

    # ─── Aggregate ───
    full_acc = full_kv_correct / max(1, total)
    ssm_acc = ssm_correct / max(1, total)

    valid_ces = [(f, s) for f, s in zip(full_losses, ssm_losses)
                 if not (math.isnan(f) or math.isnan(s))]
    valid_full = [f for f, _ in valid_ces]
    valid_ssm = [s for _, s in valid_ces]
    mean_full_ce = sum(valid_full) / max(1, len(valid_full))
    mean_ssm_ce = sum(valid_ssm) / max(1, len(valid_ssm))

    mean_kl = sum(logit_kls) / max(1, len(logit_kls)) if logit_kls else float("nan")
    mean_cos = sum(logit_cossims) / max(1, len(logit_cossims)) if logit_cossims else float("nan")
    mean_top1 = sum(top1_agrees) / max(1, len(top1_agrees)) if top1_agrees else float("nan")
    mean_top5 = sum(top5_overlaps) / max(1, len(top5_overlaps)) if top5_overlaps else float("nan")
    n_compare = len(query_logit_mses)
    per_sample_mse = sum(query_logit_mses) / max(1, n_compare)

    d_means = max(mean_ssm_ce, mean_full_ce)
    mean_sq_loss = ((mean_ssm_ce - mean_full_ce) / d_means) ** 2 if d_means > 0 else 0.0

    mode_str = "per-query random" if per_query_random else "fixed"
    print(f"\n{'='*65}")
    print(f"  GSM8K CoT Eval  (demos: {mode_str}, k={args.k})")
    print(f"{'='*65}")
    print(f"Total samples       = {total}")
    print(f"Full-KV Accuracy    = {full_acc:.4f}  ({full_kv_correct}/{total})")
    print(f"SSM Accuracy        = {ssm_acc:.4f}  ({ssm_correct}/{total})")
    print(f"Accuracy gap        = {full_acc - ssm_acc:+.4f}")

    print(f"\n  Logit Fidelity (SSM vs Full-KV, teacher-forced)")
    print(f"KL(SSM||FullKV)     = {mean_kl:.6f}")
    print(f"Cosine similarity   = {mean_cos:.6f}")
    print(f"Top-1 agreement     = {mean_top1:.6f}")
    print(f"Top-5 overlap       = {mean_top5:.6f}")

    print(f"\n  CE Comparison (content tokens only, format stripped)")
    print(f"num_samples         = {n_compare}")
    print(f"mean_CE_full_kv     = {mean_full_ce:.6f}")
    print(f"mean_CE_ssm         = {mean_ssm_ce:.6f}")
    print(f"CE_gap (ssm-full)   = {mean_ssm_ce - mean_full_ce:+.6f}")
    print(f"mean_norm_sq_loss   = {mean_sq_loss:.6f}")
    print(f"per_sample_mean_MSE = {per_sample_mse:.6f}")

    if examples_log:
        print(f"\n{'='*65}")
        print(f"  Example Predictions (first {len(examples_log)})")
        print(f"{'='*65}")
        for i, ex in enumerate(examples_log):
            print(f"\n--- Example {i} ---")
            print(f"  Q:         {ex['question']}...")
            print(f"  GT:        {ex['gt']}")
            print(f"  Full pred: {ex['full_pred']}")
            print(f"  SSM pred:  {ex['ssm_pred']}")
            print(f"  Full gen:  {ex['full_gen']}...")
            print(f"  SSM gen:   {ex['ssm_gen']}...")


# =====================================================================
# Loss comparison (standalone)
# =====================================================================

def run_loss_comparison(args, sidecar_state_dict=None):
    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    tokenizer = _setup_tokenizer(args.model_name)
    model = _load_causal_lm(args, device)
    model.eval()

    align = args.align_true_positions
    if align and getattr(model.config, "model_type", "") == "qwen2":
        align = False

    add_nl = not args.model_name.startswith("gpt2")
    retrieval_data = load_gsm8k(split=args.retrieval_split, max_samples=args.max_retrieval_samples)
    eval_data = load_gsm8k(split=args.eval_split, max_samples=args.max_eval_samples)
    if not retrieval_data or not eval_data:
        raise ValueError("Empty data.")

    fixed_demos = _choose_fixed_demos(retrieval_data, args.k, args.demo_strategy, args.seed)
    demo_text = _gsm8k_build_demo_text(fixed_demos, add_newlines=add_nl)
    demo_ids = tokenizer(demo_text, return_tensors="pt",
                         add_special_tokens=False)["input_ids"].to(device)
    full_len = demo_ids.shape[1]

    sidecar = LayerGroupSSMSidecar(
        model.config, ssm_dim=args.ssm_dim, num_virtual_tokens=args.num_virtual_tokens,
        num_groups=args.num_groups,
        effective_num_kv_heads=_resolve_sidecar_kv_heads(args, model.config),
    ).to(device=device, dtype=torch.float32)
    _load_sidecar(sidecar, args, device, sidecar_state_dict)
    sidecar.eval()

    with torch.no_grad():
        emb = model.get_input_embeddings()(demo_ids).to(dtype=next(sidecar.parameters()).dtype)
        eval_virtual_kv = sidecar(emb)
        eval_sink_kv = None
        st = min(args.sink_tokens, demo_ids.shape[1])
        if st > 0:
            eval_sink_kv = model(input_ids=demo_ids[:, :st], use_cache=True).past_key_values
        eval_ssm_demo_len = st + sidecar.num_virtual_tokens

    full_losses, ssm_losses = [], []
    logit_kls, logit_cossims, top1_agrees, top5_overlaps = [], [], [], []
    query_logit_mses = []

    for dp in tqdm(eval_data, desc="compare-loss-gsm8k"):
        q_text = _gsm8k_format_question(dp["input"], add_nl)
        ans_text = " " + dp["output"]
        q_ids = tokenizer(q_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
        a_ids = tokenizer(ans_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
        if q_ids.numel() == 0 or a_ids.numel() == 0: continue

        if a_ids.shape[1] > args.max_answer_tokens:
            a_ids = a_ids[:, :args.max_answer_tokens]

        a_stripped, n_stripped = _strip_leading_format_tokens(a_ids, tokenizer)
        if a_stripped.numel() == 0: continue

        with torch.no_grad():
            t_logits = _teacher_qa_logits_nocache(model, demo_ids, q_ids, a_ids)
            t_logits_s = t_logits[:, n_stripped:, :]
            t_ce = F.cross_entropy(
                t_logits_s.reshape(-1, t_logits_s.size(-1)),
                a_stripped.reshape(-1), reduction="mean").item()

            s_logits = _student_qa_logits(
                model, eval_virtual_kv, eval_sink_kv,
                eval_ssm_demo_len, full_len, align, q_ids, a_ids)
            s_logits_s = s_logits[:, n_stripped:, :]
            s_ce = F.cross_entropy(
                s_logits_s.reshape(-1, s_logits_s.size(-1)),
                a_stripped.reshape(-1), reduction="mean").item()

        full_losses.append(t_ce)
        ssm_losses.append(s_ce)

        # Fidelity metrics
        min_t = min(t_logits_s.size(1), s_logits_s.size(1))
        if min_t > 0:
            tl = t_logits_s[:, :min_t, :]
            sl = s_logits_s[:, :min_t, :]
            logit_kls.append(F.kl_div(
                F.log_softmax(sl, dim=-1), F.softmax(tl, dim=-1),
                reduction="batchmean").item())
            logit_cossims.append(F.cosine_similarity(
                sl.reshape(-1, sl.size(-1)),
                tl.reshape(-1, tl.size(-1)), dim=-1).mean().item())
            top1_agrees.append(
                (sl.argmax(-1) == tl.argmax(-1)).float().mean().item())
            t5_t = tl.topk(5, dim=-1).indices
            t5_s = sl.topk(5, dim=-1).indices
            ovlp = sum(
                len(set(t5_t[0, t].tolist()) & set(t5_s[0, t].tolist())) / 5.0
                for t in range(min_t))
            top5_overlaps.append(ovlp / min_t)

            first_token_id = a_stripped[:, 0].unsqueeze(1)
            t_first = tl[:, 0, :].gather(1, first_token_id).squeeze(1)
            s_first = sl[:, 0, :].gather(1, first_token_id).squeeze(1)
            t_val, s_val = float(t_first.item()), float(s_first.item())
            denom = max(s_val, t_val)
            rel_sq = ((s_val - t_val) / denom) ** 2 if denom != 0 else 0.0
            query_logit_mses.append(rel_sq)

    if not full_losses:
        raise ValueError("No valid samples.")

    mf = sum(full_losses) / len(full_losses)
    ms = sum(ssm_losses) / len(ssm_losses)
    d = max(ms, mf)
    mse = ((ms - mf) / d) ** 2 if d > 0 else 0.0

    mean_kl = sum(logit_kls) / max(1, len(logit_kls)) if logit_kls else float("nan")
    mean_cos = sum(logit_cossims) / max(1, len(logit_cossims)) if logit_cossims else float("nan")
    mean_top1 = sum(top1_agrees) / max(1, len(top1_agrees)) if top1_agrees else float("nan")
    mean_top5 = sum(top5_overlaps) / max(1, len(top5_overlaps)) if top5_overlaps else float("nan")
    per_sample_mse = sum(query_logit_mses) / max(1, len(query_logit_mses))

    print(f"\n===== GSM8K Loss & Fidelity Comparison =====")
    print(f"samples             = {len(full_losses)}")
    print(f"mean_CE_full        = {mf:.6f}")
    print(f"mean_CE_ssm         = {ms:.6f}")
    print(f"CE_gap              = {ms - mf:+.6f}")
    print(f"norm_sq_err         = {mse:.6f}")
    print(f"\nKL(SSM||FullKV)     = {mean_kl:.6f}")
    print(f"Cosine similarity   = {mean_cos:.6f}")
    print(f"Top-1 agreement     = {mean_top1:.6f}")
    print(f"Top-5 overlap       = {mean_top5:.6f}")
    print(f"per_sample_mean_MSE = {per_sample_mse:.6f}")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retrieval_split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="test")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--demo_strategy", type=str, default="first")
    p.add_argument("--num_eval_demo_sets", type=int, default=1)
    p.add_argument("--max_retrieval_samples", type=int, default=0)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_eval_samples", type=int, default=0,
                   help="0=all. Use e.g. 200 for quick runs.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--is_quant", default=False, action="store_true")
    p.add_argument("--quant_sidecar_kv_heads", type=int, default=4)
    p.add_argument("--run_mode", type=str, default="eval",
                   choices=["eval", "train", "train_eval", "compare_loss", "train_compare_loss"])

    # Sidecar arch
    p.add_argument("--num_virtual_tokens", type=int, default=32)
    p.add_argument("--sink_tokens", type=int, default=4)
    p.add_argument("--ssm_dim", type=int, default=512)
    p.add_argument("--num_groups", type=int, default=4)
    p.add_argument("--align_true_positions", action="store_true")

    # Training
    p.add_argument("--num_demo_sets", type=int, default=500)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--kd_temperature", type=float, default=2.0)
    p.add_argument("--loss_w_kv", type=float, default=1.0)
    p.add_argument("--loss_w_logit", type=float, default=0.0)
    p.add_argument("--loss_w_hid", type=float, default=0.0)
    p.add_argument("--loss_w_ce", type=float, default=0.0)
    p.add_argument("--kv_loss_type", type=str, default="cosine", choices=["cosine", "mse"])
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--max_answer_tokens", type=int, default=256)

    # Generation
    p.add_argument("--max_gen_tokens", type=int, default=512)

    # Checkpoints
    p.add_argument("--save_sidecar_path", type=str, default="")
    p.add_argument("--load_sidecar_path", type=str, default="")
    p.add_argument("--empty_cache_every", type=int, default=10)

    args = p.parse_args()

    if args.run_mode == "train":
        run_distillation_training(args)
    elif args.run_mode == "train_eval":
        sd = run_distillation_training(args)
        run_experiment(args, sidecar_state_dict=sd)
    elif args.run_mode == "train_compare_loss":
        sd = run_distillation_training(args)
        run_loss_comparison(args, sidecar_state_dict=sd)
    elif args.run_mode == "compare_loss":
        run_loss_comparison(args)
    else:
        run_experiment(args)