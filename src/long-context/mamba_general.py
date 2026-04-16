import argparse
import copy
import gc
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPT2Tokenizer
from transformers.cache_utils import DynamicCache


# =====================================================================
# RULER data loading
# =====================================================================

def _load_ruler_data(args):
    """
    Load rbiswasfc/ruler dataset.
      - Training: all 500 from qa_2_8k + first 391 from qa_2_4k
      - Test:     last 109 from qa_2_4k
    """
    from datasets import load_dataset

    ds_8k = load_dataset("rbiswasfc/ruler", "qa_2_8k", split="validation")
    ds_4k = load_dataset("rbiswasfc/ruler", "qa_2_4k", split="validation")
    ds_4k_list = list(ds_4k)

    train_data = []
    for ex in ds_8k:
        train_data.append({
            "input": ex["input"],
            "output": ex["outputs"][0] if ex["outputs"] else "",
            "outputs": ex["outputs"],
        })
    for ex in ds_4k_list[:391]:
        train_data.append({
            "input": ex["input"],
            "output": ex["outputs"][0] if ex["outputs"] else "",
            "outputs": ex["outputs"],
        })

    test_data = []
    for ex in ds_4k_list[391:]:
        test_data.append({
            "input": ex["input"],
            "output": ex["outputs"][0] if ex["outputs"] else "",
            "outputs": ex["outputs"],
        })

    print(f"[RULER] train={len(train_data)} (8k:{len(list(ds_8k))}+4k:{391}), "
          f"test={len(test_data)}")
    return train_data, test_data


def _split_ruler_input(text):
    """
    Split ruler input into (prefix, query).
      prefix: everything before the final "\\nQuestion:" — documents + instructions
      query:  "\\nQuestion: ... \\nAnswer:" — the actual question to answer
    """
    marker = "\nQuestion:"
    idx = text.rfind(marker)
    if idx == -1:
        # Fallback: no newline variant
        marker = "Question:"
        idx = text.rfind(marker)
    if idx == -1:
        return text, ""
    return text[:idx], text[idx:]


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
        self._init_hippo_B(hidden_size, dt)
        inter = ssm_dim * 4
        per_layer_kv = 2 * num_kv_heads * num_virtual_tokens * head_dim
        self.proj = nn.Sequential(
            nn.Linear(ssm_dim, inter), nn.SiLU(),
            nn.Linear(inter, inter), nn.SiLU(),
            nn.Linear(inter, num_layers_in_group * per_layer_kv),
        )

    def _init_hippo_B(self, hidden_size, dt):
        """ZOH-discretized HiPPO-LegS B initialization: B_d = A_c^{-1}(A_d - I) · B_c."""
        with torch.no_grad():
            nn.init.normal_(self.B.weight, std=1.0 / math.sqrt(hidden_size))
            b = _hippo_legs_input_vector(self.ssm_dim, self.B.weight.device, self.B.weight.dtype)
            self.B.weight.mul_(b.unsqueeze(1))
            A_c = _hippo_legs_matrix(self.ssm_dim, self.B.weight.device, torch.float64)
            A_d = torch.matrix_exp(dt * A_c)
            M = torch.linalg.solve(
                A_c,
                A_d - torch.eye(self.ssm_dim, device=A_c.device, dtype=torch.float64),
            )
            self.B.weight.copy_((M.to(self.B.weight.dtype)) @ self.B.weight)
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
    def __init__(self, config, ssm_dim=512, num_virtual_tokens=16, num_groups=4, dt=1.0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
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
              f"kv_heads={self.num_kv_heads} | {info}")

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

def _build_cache_from_kv_list(kv_list, sink_kv=None, target_dtype=None):
    cache = DynamicCache()
    sink_legacy = _cache_to_legacy(_ensure_cache_obj(sink_kv)) if sink_kv is not None else None
    for i, (vk, vv) in enumerate(kv_list):
        if target_dtype is not None:
            vk = vk.to(dtype=target_dtype)
            vv = vv.to(dtype=target_dtype)
        if sink_legacy is not None:
            sk, sv = sink_legacy[i]
            if target_dtype is not None:
                sk = sk.to(dtype=target_dtype)
                sv = sv.to(dtype=target_dtype)
            k = torch.cat([sk, vk], dim=2)
            v = torch.cat([sv, vv], dim=2)
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
# Teacher / student forward
# =====================================================================

@torch.no_grad()
def _teacher_qa_forward(model, prefix_ids, query_ids, ans_ids):
    """Teacher: full concat forward, returns logits on answer tokens."""
    all_ids = torch.cat([prefix_ids, query_ids, ans_ids], dim=1)
    out = model(input_ids=all_ids, use_cache=False)
    start = prefix_ids.shape[1] + query_ids.shape[1] - 1
    return out.logits[:, start:start + ans_ids.shape[1], :]


def _student_qa_forward(model, virtual_kv_list, sink_kv, demo_len, true_prefix_len,
                        query_ids, ans_ids):
    """Student: virtual KV + query + answer, returns logits on answer tokens."""
    model_kv_dtype = model.get_input_embeddings().weight.dtype
    cache = _build_cache_from_kv_list(virtual_kv_list, sink_kv=sink_kv,
                                      target_dtype=model_kv_dtype)
    # Query forward
    attn_q = torch.ones(1, demo_len + query_ids.shape[1], dtype=torch.long, device=query_ids.device)
    out_q = model(input_ids=query_ids, past_key_values=cache, attention_mask=attn_q,
                  position_ids=_pos_ids(true_prefix_len, query_ids.shape[1], query_ids.device),
                  use_cache=True)
    first_logit = out_q.logits[:, -1:]

    # Answer forward
    base = demo_len + query_ids.shape[1]
    attn_a = torch.ones(1, base + ans_ids.shape[1], dtype=torch.long, device=query_ids.device)
    out_a = model(input_ids=ans_ids, past_key_values=out_q.past_key_values,
                  attention_mask=attn_a,
                  position_ids=_pos_ids(true_prefix_len + query_ids.shape[1],
                                        ans_ids.shape[1], query_ids.device),
                  use_cache=False)
    if ans_ids.shape[1] == 1:
        return first_logit
    return torch.cat([first_logit, out_a.logits[:, :-1]], dim=1)


# =====================================================================
# Generation with compressed KV
# =====================================================================

@torch.no_grad()
def _generate_with_virtual_kv(model, tokenizer, virtual_kv_list, sink_kv,
                               demo_len, true_prefix_len, query_ids,
                               max_new_tokens=64):
    """Generate answer tokens autoregressively using compressed prefix."""
    device = query_ids.device
    model_kv_dtype = model.get_input_embeddings().weight.dtype
    cache = _build_cache_from_kv_list(virtual_kv_list, sink_kv=sink_kv,
                                      target_dtype=model_kv_dtype)

    # Prefill with query
    attn = torch.ones(1, demo_len + query_ids.shape[1], dtype=torch.long, device=device)
    out = model(input_ids=query_ids, past_key_values=cache, attention_mask=attn,
                position_ids=_pos_ids(true_prefix_len, query_ids.shape[1], device),
                use_cache=True)
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    past = out.past_key_values
    cur_len = demo_len + query_ids.shape[1]

    for _ in range(max_new_tokens - 1):
        cur_len += 1
        attn = torch.ones(1, cur_len, dtype=torch.long, device=device)
        pos = _pos_ids(true_prefix_len + query_ids.shape[1] + len(generated) - 1, 1, device)
        out = model(input_ids=next_token, past_key_values=past, attention_mask=attn,
                    position_ids=pos, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated.append(next_token)

    token_ids = torch.cat(generated, dim=-1)
    return tokenizer.decode(token_ids[0], skip_special_tokens=True).strip()


@torch.no_grad()
def _generate_full_kv(model, tokenizer, full_ids, max_new_tokens=64):
    """Generate answer tokens with full KV (teacher baseline)."""
    device = full_ids.device
    out = model(input_ids=full_ids, use_cache=True)
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    past = out.past_key_values
    cur_len = full_ids.shape[1]

    for _ in range(max_new_tokens - 1):
        cur_len += 1
        attn = torch.ones(1, cur_len, dtype=torch.long, device=device)
        out = model(input_ids=next_token, past_key_values=past, attention_mask=attn,
                    use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated.append(next_token)

    token_ids = torch.cat(generated, dim=-1)
    return tokenizer.decode(token_ids[0], skip_special_tokens=True).strip()


# =====================================================================
# Evaluation metric
# =====================================================================

def _ruler_score(prediction, ground_truths):
    """
    Check if prediction contains any of the ground truth answers (case-insensitive).
    This is the standard RULER evaluation: check if the answer string appears in the output.
    """
    pred_lower = prediction.lower().strip()
    for gt in ground_truths:
        if gt.lower().strip() in pred_lower:
            return 1.0
    return 0.0


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
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        return AutoModelForCausalLM.from_pretrained(
            args.model_name, quantization_config=bnb_config,
            device_map="auto", torch_dtype=torch.float16,
        )
    return AutoModelForCausalLM.from_pretrained(args.model_name).to(device)


def _setup_tokenizer(name):
    tok = (GPT2Tokenizer.from_pretrained(name) if name.startswith("gpt2")
           else AutoTokenizer.from_pretrained(name))
    if tok.padding_side == "left": tok.padding_side = "right"
    if tok.eos_token_id is None and tok.sep_token is not None:
        tok.eos_token = tok.sep_token
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if tok.bos_token_id is None: tok.bos_token = tok.eos_token
    return tok


def _load_sidecar(sidecar, args, device, state_dict=None):
    _FROZEN = ('.A', '.A_powers', '.eigenvalues', '.V', '.V_inv')
    if state_dict is not None:
        filtered = {k: v for k, v in state_dict.items()
                    if not any(k.endswith(s) for s in _FROZEN)}
        sidecar.load_state_dict(filtered, strict=False)
        print("Loaded sidecar from in-memory state.")
    elif args.load_sidecar_path:
        ckpt = torch.load(args.load_sidecar_path, map_location=device)
        st = ckpt["sidecar"] if isinstance(ckpt, dict) and "sidecar" in ckpt else ckpt
        filtered = {k: v for k, v in st.items()
                    if not any(k.endswith(s) for s in _FROZEN)}
        sidecar.load_state_dict(filtered, strict=False)
        print(f"Loaded sidecar from: {args.load_sidecar_path}")
    else:
        print("WARN: No sidecar checkpoint.")


# =====================================================================
# Training loop
# =====================================================================

def run_distillation_training(args):
    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    tokenizer = _setup_tokenizer(args.model_name)
    model = _load_causal_lm(args, device)
    _freeze(model)

    train_data, _ = _load_ruler_data(args)
    if not train_data:
        raise ValueError("Empty training data.")

    sidecar = LayerGroupSSMSidecar(
        model.config, ssm_dim=args.ssm_dim, num_virtual_tokens=args.num_virtual_tokens,
        num_groups=args.num_groups, dt=args.dt,
    ).to(device=device, dtype=torch.float32)

    if args.load_sidecar_path:
        _load_sidecar(sidecar, args, device)
    sidecar.train()

    optimizer = torch.optim.AdamW(sidecar.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_data)
    warmup = min(100, total_steps // 10)
    def lr_fn(step):
        if step < warmup: return step / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    rng = random.Random(args.seed)
    step = 0
    running = {"loss": 0.0, "kv": 0.0, "logit": 0.0, "ce": 0.0}

    use_kv = args.loss_w_kv > 0
    use_logit = args.loss_w_logit > 0
    use_ce = args.loss_w_ce > 0

    print(f"\n===== Training: epochs={args.epochs}, samples={len(train_data)}, "
          f"ssm_dim={args.ssm_dim}, groups={args.num_groups}, "
          f"vtokens={args.num_virtual_tokens}, sink={args.sink_tokens}, dt={args.dt}")
    print(f"  loss weights: kv={args.loss_w_kv}, logit={args.loss_w_logit}, ce={args.loss_w_ce}")
    print()

    for epoch in range(args.epochs):
        order = list(range(len(train_data)))
        rng.shuffle(order)
        for di in tqdm(order, desc=f"epoch-{epoch+1}"):
            dp = train_data[di]
            prefix_text, query_text = _split_ruler_input(dp["input"])
            ans_text = dp["output"]

            if not prefix_text or not query_text or not ans_text:
                continue

            prefix_ids = tokenizer(prefix_text, return_tensors="pt",
                                   add_special_tokens=False)["input_ids"].to(device)
            query_ids = tokenizer(query_text, return_tensors="pt",
                                  add_special_tokens=False)["input_ids"].to(device)
            ans_ids = tokenizer(ans_text, return_tensors="pt",
                                add_special_tokens=False)["input_ids"].to(device)

            if prefix_ids.numel() == 0 or query_ids.numel() == 0 or ans_ids.numel() == 0:
                continue

            true_prefix_len = prefix_ids.shape[1]

            # ── Student: SSM compress prefix ──
            emb = model.get_input_embeddings()(prefix_ids)
            virtual_kv = sidecar(emb.float())

            # Sink KV
            sink_kv = None
            st = min(args.sink_tokens, prefix_ids.shape[1])
            if st > 0:
                with torch.no_grad():
                    sink_kv = model(input_ids=prefix_ids[:, :st],
                                    use_cache=True).past_key_values
            demo_len = st + sidecar.num_virtual_tokens

            # ── L_KV: match virtual KV to teacher prefix KV ──
            loss_kv = torch.tensor(0.0, device=device)
            if use_kv:
                with torch.no_grad():
                    teacher_kv = _cache_to_legacy(
                        model(input_ids=prefix_ids, use_cache=True).past_key_values)
                loss_kv = kv_matching_loss(virtual_kv, teacher_kv,
                                           sidecar.num_virtual_tokens,
                                           loss_type=args.kv_loss_type)

            # ── L_logit + L_CE: distill on answer tokens ──
            loss_logit = torch.tensor(0.0, device=device)
            loss_ce = torch.tensor(0.0, device=device)
            if use_logit or use_ce:
                with torch.no_grad():
                    t_logits = _teacher_qa_forward(model, prefix_ids, query_ids, ans_ids)

                s_logits = _student_qa_forward(
                    model, virtual_kv, sink_kv, demo_len, true_prefix_len,
                    query_ids, ans_ids)

                if use_logit:
                    temp = args.kd_temperature
                    loss_logit = F.kl_div(
                        F.log_softmax(s_logits / temp, dim=-1),
                        F.softmax(t_logits.detach() / temp, dim=-1),
                        reduction="batchmean") * (temp ** 2)

                if use_ce:
                    loss_ce = F.cross_entropy(
                        s_logits.reshape(-1, s_logits.size(-1)),
                        ans_ids.reshape(-1),
                        reduction="mean")

            # ── Combined ──
            loss = (args.loss_w_kv * loss_kv
                    + args.loss_w_logit * loss_logit
                    + args.loss_w_ce * loss_ce)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sidecar.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            running["loss"] += loss.item()
            running["kv"] += (loss_kv.item() if isinstance(loss_kv, torch.Tensor) else loss_kv)
            running["logit"] += (loss_logit.item() if isinstance(loss_logit, torch.Tensor) else loss_logit)
            running["ce"] += (loss_ce.item() if isinstance(loss_ce, torch.Tensor) else loss_ce)

            if step % args.log_every == 0:
                d = float(args.log_every)
                print(f"step={step} loss={running['loss']/d:.4f} "
                      f"kv={running['kv']/d:.4f} logit={running['logit']/d:.4f} "
                      f"ce={running['ce']/d:.4f} lr={scheduler.get_last_lr()[0]:.6f}")
                running = {k: 0.0 for k in running}

            del virtual_kv, sink_kv, prefix_ids, query_ids, ans_ids
            del loss, loss_kv, loss_logit, loss_ce
            if device.startswith("cuda") and args.empty_cache_every > 0 and step % args.empty_cache_every == 0:
                gc.collect(); torch.cuda.empty_cache()
            if 0 < args.max_steps <= step: break
        if 0 < args.max_steps <= step: break

    if args.save_sidecar_path:
        torch.save({"sidecar": sidecar.state_dict(), "args": vars(args),
                     "model_name": args.model_name}, args.save_sidecar_path)
        print(f"Saved: {args.save_sidecar_path}")

    return {k: v.detach().cpu() for k, v in sidecar.state_dict().items()}


# =====================================================================
# Eval
# =====================================================================

def run_experiment(args, sidecar_state_dict=None):
    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    tokenizer = _setup_tokenizer(args.model_name)
    model = _load_causal_lm(args, device)
    model.eval()

    _, test_data = _load_ruler_data(args)
    if not test_data:
        raise ValueError("Empty test data.")

    sidecar = LayerGroupSSMSidecar(
        model.config, ssm_dim=args.ssm_dim, num_virtual_tokens=args.num_virtual_tokens,
        num_groups=args.num_groups, dt=args.dt,
    ).to(device=device, dtype=torch.float32)
    _load_sidecar(sidecar, args, device, sidecar_state_dict)
    sidecar.eval()

    full_scores, ssm_scores = [], []
    full_preds, ssm_preds, gts = [], [], []
    full_ces, ssm_ces = [], []

    print(f"\nEvaluating on {len(test_data)} test samples...\n")

    for dp in tqdm(test_data, desc="eval"):
        prefix_text, query_text = _split_ruler_input(dp["input"])
        ans_text = dp["output"]
        ground_truths = dp["outputs"]

        if not prefix_text or not query_text:
            continue

        prefix_ids = tokenizer(prefix_text, return_tensors="pt",
                               add_special_tokens=False)["input_ids"].to(device)
        query_ids = tokenizer(query_text, return_tensors="pt",
                              add_special_tokens=False)["input_ids"].to(device)
        ans_ids = tokenizer(ans_text, return_tensors="pt",
                            add_special_tokens=False)["input_ids"].to(device)
        true_prefix_len = prefix_ids.shape[1]

        # ─── Full-KV baseline ───
        full_input_ids = tokenizer(dp["input"], return_tensors="pt",
                                   add_special_tokens=False)["input_ids"].to(device)
        with torch.no_grad():
            full_pred = _generate_full_kv(model, tokenizer, full_input_ids,
                                          max_new_tokens=args.max_gen_tokens)
        full_preds.append(full_pred)
        full_scores.append(_ruler_score(full_pred, ground_truths))

        # Full-KV CE
        if ans_ids.numel() > 0:
            with torch.no_grad():
                t_logits = _teacher_qa_forward(model, prefix_ids, query_ids, ans_ids)
                full_ce = F.cross_entropy(
                    t_logits.reshape(-1, t_logits.size(-1)),
                    ans_ids.reshape(-1), reduction="mean").item()
                full_ces.append(full_ce)

        # ─── SSM ───
        with torch.no_grad():
            emb = model.get_input_embeddings()(prefix_ids).to(
                dtype=next(sidecar.parameters()).dtype)
            virtual_kv = sidecar(emb)
            sink_kv = None
            st = min(args.sink_tokens, prefix_ids.shape[1])
            if st > 0:
                sink_kv = model(input_ids=prefix_ids[:, :st],
                                use_cache=True).past_key_values
            ssm_demo_len = st + sidecar.num_virtual_tokens

            # Generate
            ssm_pred = _generate_with_virtual_kv(
                model, tokenizer, virtual_kv, sink_kv,
                ssm_demo_len, true_prefix_len, query_ids,
                max_new_tokens=args.max_gen_tokens)
        ssm_preds.append(ssm_pred)
        ssm_scores.append(_ruler_score(ssm_pred, ground_truths))

        # SSM CE
        if ans_ids.numel() > 0:
            with torch.no_grad():
                s_logits = _student_qa_forward(
                    model, virtual_kv, sink_kv, ssm_demo_len, true_prefix_len,
                    query_ids, ans_ids)
                ssm_ce = F.cross_entropy(
                    s_logits.reshape(-1, s_logits.size(-1)),
                    ans_ids.reshape(-1), reduction="mean").item()
                ssm_ces.append(ssm_ce)

        gts.append(ans_text)

    # ─── Results ───
    n = len(full_scores)
    full_acc = sum(full_scores) / max(1, n)
    ssm_acc = sum(ssm_scores) / max(1, n)
    mean_full_ce = sum(full_ces) / max(1, len(full_ces))
    mean_ssm_ce = sum(ssm_ces) / max(1, len(ssm_ces))

    print(f"\n{'='*65}")
    print(f"  RULER QA Eval  (n={n})")
    print(f"{'='*65}")
    print(f"Full-KV Accuracy    = {full_acc:.4f}")
    print(f"SSM Accuracy        = {ssm_acc:.4f}")
    print(f"Accuracy gap        = {full_acc - ssm_acc:+.4f}")
    print(f"mean_CE_full_kv     = {mean_full_ce:.4f}")
    print(f"mean_CE_ssm         = {mean_ssm_ce:.4f}")
    print(f"CE gap              = {mean_ssm_ce - mean_full_ce:+.4f}")

    # Print some examples
    print(f"\n  Sample predictions (first 5):")
    for i in range(min(5, n)):
        print(f"  [{i}] GT: {gts[i]}")
        print(f"       Full-KV: {full_preds[i]}")
        print(f"       SSM:     {ssm_preds[i]}")
        print()


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--is_quant", default=False, action="store_true")
    p.add_argument("--run_mode", type=str, default="eval",
                   choices=["eval", "train", "train_eval"])

    # SSM architecture
    p.add_argument("--num_virtual_tokens", type=int, default=16)
    p.add_argument("--sink_tokens", type=int, default=4)
    p.add_argument("--ssm_dim", type=int, default=512)
    p.add_argument("--num_groups", type=int, default=4)
    p.add_argument("--dt", type=float, default=1.0,
                   help="Discretization step for HiPPO.")

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--kd_temperature", type=float, default=2.0)
    p.add_argument("--loss_w_kv", type=float, default=1.0)
    p.add_argument("--loss_w_logit", type=float, default=0.5)
    p.add_argument("--loss_w_ce", type=float, default=0.5)
    p.add_argument("--kv_loss_type", type=str, default="cosine", choices=["cosine", "mse"])
    p.add_argument("--log_every", type=int, default=20)

    # Eval
    p.add_argument("--max_gen_tokens", type=int, default=64,
                   help="Max tokens to generate during evaluation.")

    # IO
    p.add_argument("--save_sidecar_path", type=str, default="")
    p.add_argument("--load_sidecar_path", type=str, default="")
    p.add_argument("--empty_cache_every", type=int, default=0)

    args = p.parse_args()

    if args.run_mode == "train":
        run_distillation_training(args)
    elif args.run_mode == "train_eval":
        sd = run_distillation_training(args)
        run_experiment(args, sidecar_state_dict=sd)
    else:
        run_experiment(args)