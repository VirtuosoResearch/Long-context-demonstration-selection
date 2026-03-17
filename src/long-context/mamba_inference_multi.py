"""
SSM Sidecar for long-context prefix compression (v2).

Key improvements over v1:
  1. Per-layer-group SSMs: each group handles a subset of layers, reducing
     the output dimensionality per SSM by ~num_groups×.
  2. Direct KV matching loss: compares virtual KV against chunked-mean-pooled
     teacher KV, giving the SSM a strong, direct gradient signal.
  3. MLP projection: multi-layer projection with nonlinearity replaces a
     single Linear fan-out.
  4. Diverse training demos: randomly samples many different k-shot demo
     compositions so the sidecar generalises to any demo set.
"""

import argparse
import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
from transformers.cache_utils import DynamicCache

from utils.data import load_data


# =====================================================================
# HiPPO utilities
# =====================================================================

def _hippo_legs_matrix(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    n = torch.arange(size, device=device, dtype=dtype)
    p = torch.sqrt(2.0 * n + 1.0)
    a = torch.zeros(size, size, device=device, dtype=dtype)
    lower = torch.tril(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=-1)
    a[lower] = -(p[:, None] * p[None, :])[lower]
    a.diagonal().copy_(-(n + 1.0))
    return a


def _hippo_legs_input_vector(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.sqrt(2.0 * torch.arange(size, device=device, dtype=dtype) + 1.0)


# =====================================================================
# Per-group SSM compressor
# =====================================================================

class LayerGroupSSM(nn.Module):
    """
    One SSM compressor for a GROUP of transformer layers.
    Input:  token embeddings [B, T, H]
    Output: virtual (K, V) for each layer in the group.
    """

    def __init__(
        self,
        hidden_size: int,
        ssm_dim: int,
        num_layers_in_group: int,
        num_kv_heads: int,
        head_dim: int,
        num_virtual_tokens: int,
        dt: float = 1.0,
    ):
        super().__init__()
        self.ssm_dim = ssm_dim
        self.num_layers_in_group = num_layers_in_group
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_virtual_tokens = num_virtual_tokens

        # Discretised HiPPO-LegS A matrix
        a_cont = _hippo_legs_matrix(ssm_dim, torch.device("cpu"), torch.float32)
        self.A = nn.Parameter(torch.matrix_exp(dt * a_cont))

        # Input projection with HiPPO-aware init
        self.B = nn.Linear(hidden_size, ssm_dim)
        self._init_hippo_B(hidden_size)

        # Multi-step projection: ssm_dim -> intermediate -> per-layer KV
        inter = ssm_dim * 4
        per_layer_kv = 2 * num_kv_heads * num_virtual_tokens * head_dim
        self.proj = nn.Sequential(
            nn.Linear(ssm_dim, inter),
            nn.SiLU(),
            nn.Linear(inter, inter),
            nn.SiLU(),
            nn.Linear(inter, num_layers_in_group * per_layer_kv),
        )

    def _init_hippo_B(self, hidden_size: int):
        with torch.no_grad():
            nn.init.normal_(self.B.weight, std=1.0 / math.sqrt(hidden_size))
            b = _hippo_legs_input_vector(self.ssm_dim, self.B.weight.device, self.B.weight.dtype)
            self.B.weight.mul_(b.unsqueeze(1))
            nn.init.zeros_(self.B.bias)

    def scan(self, x: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
        """Recurrent scan over x [B, T, H], returns final state [B, D]."""
        bsz, T, _ = x.shape
        state = torch.zeros(bsz, self.ssm_dim, device=x.device, dtype=x.dtype)
        a = self.A.to(device=x.device, dtype=x.dtype)
        for s in range(0, T, chunk_size):
            u = self.B(x[:, s : s + chunk_size])  # [B, chunk, D]
            for t in range(u.shape[1]):
                state = F.linear(state, a) + u[:, t]
        return state

    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        bsz = x.shape[0]
        state = self.scan(x)
        flat = self.proj(state)
        r = flat.view(bsz, self.num_layers_in_group, 2,
                       self.num_kv_heads, self.num_virtual_tokens, self.head_dim)
        return [(r[:, i, 0], r[:, i, 1]) for i in range(self.num_layers_in_group)]


# =====================================================================
# Sidecar – orchestrates all layer groups
# =====================================================================

class LayerGroupSSMSidecar(nn.Module):
    """
    Splits transformer layers into ``num_groups`` groups, each with an
    independent SSM.  All groups consume the *same* embedding sequence so
    that inference does NOT require a teacher forward pass.
    """

    def __init__(self, config, ssm_dim: int = 512, num_virtual_tokens: int = 16,
                 num_groups: int = 4, dt: float = 1.0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_virtual_tokens = num_virtual_tokens
        self.num_groups = num_groups
        self.ssm_dim = ssm_dim

        base, rem = divmod(self.num_layers, num_groups)
        self.group_ranges: List[Tuple[int, int]] = []
        s = 0
        for g in range(num_groups):
            e = s + base + (1 if g < rem else 0)
            self.group_ranges.append((s, e))
            s = e

        self.groups = nn.ModuleList([
            LayerGroupSSM(
                hidden_size=config.hidden_size,
                ssm_dim=ssm_dim,
                num_layers_in_group=e - s,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                num_virtual_tokens=num_virtual_tokens,
                dt=dt,
            )
            for s, e in self.group_ranges
        ])

        info = ", ".join(f"G{g}:[{s},{e})" for g, (s, e) in enumerate(self.group_ranges))
        print(f"[Sidecar] {num_groups} groups, ssm_dim={ssm_dim}, vtokens={num_virtual_tokens} | {info}")

    def forward(self, embeddings: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        embeddings: [B, T, H]
        Returns: list of (K, V) per transformer layer, len == num_layers.
        """
        all_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for group in self.groups:
            all_kv.extend(group(embeddings))
        return all_kv


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


def _build_cache_from_kv_list(
    kv_list: List[Tuple[torch.Tensor, torch.Tensor]],
    sink_kv=None,
) -> DynamicCache:
    """
    Build a DynamicCache from a list of (K, V) tuples.
    Optionally prepend sink-token KV for each layer.
    Preserves gradient flow for kv_list tensors.
    """
    cache = DynamicCache()
    sink_legacy = None
    if sink_kv is not None:
        sink_legacy = _cache_to_legacy(_ensure_cache_obj(sink_kv))
    for i, (vk, vv) in enumerate(kv_list):
        if sink_legacy is not None:
            sk, sv = sink_legacy[i]
            k = torch.cat([sk, vk], dim=2)
            v = torch.cat([sv, vv], dim=2)
        else:
            k, v = vk, vv
        cache.update(k, v, i)
    return cache


def _fresh_cache_copy(cache: DynamicCache) -> DynamicCache:
    """Deep-copy a DynamicCache for inference (breaks grad, safe for eval)."""
    legacy = _cache_to_legacy(cache)
    fresh = tuple((k.clone(), v.clone()) for k, v in legacy)
    return _ensure_cache_obj(fresh)


def _pos_ids(start: int, length: int, device: torch.device, bsz: int = 1):
    p = torch.arange(start, start + length, dtype=torch.long, device=device).unsqueeze(0)
    return p.expand(bsz, -1) if bsz > 1 else p


# =====================================================================
# KV matching loss
# =====================================================================

def kv_matching_loss(
    virtual_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    teacher_kv,
    num_virtual_tokens: int,
    loss_type: str = "cosine",
) -> torch.Tensor:
    """
    Compare virtual KV against chunked-mean-pooled teacher KV.
    Provides a strong, direct gradient to the sidecar without going
    through a full model forward.
    """
    t_legacy = _cache_to_legacy(_ensure_cache_obj(teacher_kv))
    device = virtual_kv[0][0].device
    total = torch.tensor(0.0, device=device)
    n_terms = 0

    for layer_idx, (vk, vv) in enumerate(virtual_kv):
        tk, tv = t_legacy[layer_idx]          # [B, heads, T_full, hdim]
        T_full = tk.shape[2]
        chunk = max(1, T_full // num_virtual_tokens)

        for vi in range(num_virtual_tokens):
            s = vi * chunk
            e = min(s + chunk, T_full)
            if s >= T_full:
                break
            tk_p = tk[:, :, s:e].mean(dim=2)  # [B, heads, hdim]
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
# Scorer – used at eval / compare-loss time
# =====================================================================

class SSMHybridICLScorer:
    def __init__(self, model, tokenizer, device, sidecar: LayerGroupSSMSidecar,
                 sink_tokens: int = 0, align_true_positions: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = model.get_input_embeddings().weight.dtype
        self.sidecar = sidecar
        self.sink_tokens = max(0, sink_tokens)
        self.align_true_positions = align_true_positions

        self.demo_cache: Optional[DynamicCache] = None
        self.demo_len = 0
        self.true_demo_len = 0

    @torch.no_grad()
    def build_demo_cache(self, demo_text: str):
        ids = self.tokenizer(demo_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        if ids.numel() == 0:
            raise ValueError("Empty demo text.")
        self.true_demo_len = ids.shape[1]

        emb = self.model.get_input_embeddings()(ids).to(dtype=self.dtype)
        virtual_kv = self.sidecar(emb)

        # Optional sink tokens (first few real tokens get full KV).
        sink_kv = None
        st = min(self.sink_tokens, ids.shape[1])
        if st > 0:
            sink_kv = self.model(input_ids=ids[:, :st], use_cache=True).past_key_values

        self.demo_cache = _build_cache_from_kv_list(virtual_kv, sink_kv=sink_kv)
        self.demo_len = st + self.sidecar.num_virtual_tokens

    @torch.no_grad()
    def prefill_question(self, question_text: str):
        q_ids = self.tokenizer(question_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        if q_ids.numel() == 0:
            raise ValueError("Empty question.")
        cache = _fresh_cache_copy(self.demo_cache)
        attn = torch.ones(1, self.demo_len + q_ids.shape[1], dtype=torch.long, device=self.device)
        pos_start = self.true_demo_len if self.align_true_positions else self.demo_len
        out = self.model(input_ids=q_ids, past_key_values=cache, attention_mask=attn,
                         position_ids=_pos_ids(pos_start, q_ids.shape[1], self.device), use_cache=True)
        return out.logits[:, -1, :], out.past_key_values, q_ids.shape[1]

    @torch.no_grad()
    def score_options_nll(self, first_logits, past_after_q, q_len: int, options: List[str]):
        tokenized = [self.tokenizer(o, add_special_tokens=False)["input_ids"] for o in options]
        lengths = [len(t) for t in tokenized]
        if any(l == 0 for l in lengths):
            raise ValueError("Empty option.")
        bsz = len(options)
        mx = max(lengths)
        ids = torch.full((bsz, mx), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        mask = torch.zeros(bsz, mx, device=self.device)
        for i, t in enumerate(tokenized):
            ids[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)
            mask[i, :len(t)] = 1.0

        bp = copy.deepcopy(_ensure_cache_obj(past_after_q))
        bp.batch_repeat_interleave(bsz)

        base = self.demo_len + q_len
        attn = torch.ones(bsz, base + mx, dtype=torch.long, device=self.device)
        opt_pos_start = (self.true_demo_len + q_len) if self.align_true_positions else base
        pos = _pos_ids(opt_pos_start, mx, self.device, bsz)
        for i, l in enumerate(lengths):
            if l < mx:
                attn[i, base + l:] = 0

        out = self.model(input_ids=ids, past_key_values=bp, attention_mask=attn,
                         position_ids=pos, use_cache=False)
        first = first_logits.expand(bsz, -1).unsqueeze(1)
        pred = torch.cat([first, out.logits[:, :-1]], dim=1) if mx > 1 else first
        losses = F.cross_entropy(pred.reshape(-1, pred.size(-1)), ids.reshape(-1), reduction="none").view(bsz, mx)
        nll = (losses * mask).sum(1)
        return {o: float(nll[i]) for i, o in enumerate(options)}


# =====================================================================
# Training helpers
# =====================================================================

def _freeze(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def _generate_diverse_demo_texts(
    retrieval_data: List[Dict], k: int, num_sets: int, seed: int, add_newlines: bool,
) -> List[str]:
    rng = random.Random(seed)
    n = len(retrieval_data)
    texts = []
    for _ in range(num_sets):
        indices = rng.sample(range(n), min(k, n))
        demos = [retrieval_data[i] for i in indices]
        texts.append(_build_demo_text(demos, add_newlines))
    return texts


def _teacher_demo_kv(model, demo_ids: torch.Tensor):
    """Full KV cache from frozen teacher."""
    with torch.no_grad():
        return model(input_ids=demo_ids, use_cache=True).past_key_values


def _student_virtual_kv(model, sidecar, demo_ids: torch.Tensor):
    """Virtual KV from sidecar (grad enabled)."""
    emb = model.get_input_embeddings()(demo_ids)
    emb = emb.to(dtype=next(sidecar.parameters()).dtype)
    return sidecar(emb)


@torch.no_grad()
def _teacher_qa_logits(model, teacher_kv, demo_len: int, q_ids, ans_ids):
    """Teacher logits on answer tokens (no grad)."""
    t_kv = _ensure_cache_obj(teacher_kv)
    attn_q = torch.ones(1, demo_len + q_ids.shape[1], dtype=torch.long, device=q_ids.device)
    out_q = model(input_ids=q_ids, past_key_values=copy.deepcopy(t_kv),
                  attention_mask=attn_q, use_cache=True)
    first = out_q.logits[:, -1:]
    base = demo_len + q_ids.shape[1]
    attn_a = torch.ones(1, base + ans_ids.shape[1], dtype=torch.long, device=q_ids.device)
    out_a = model(input_ids=ans_ids, past_key_values=out_q.past_key_values,
                  attention_mask=attn_a, use_cache=False)
    if ans_ids.shape[1] == 1:
        return first
    return torch.cat([first, out_a.logits[:, :-1]], dim=1)


def _student_qa_logits(model, virtual_kv_list, sink_kv, demo_len, true_demo_len,
                       align_true_positions, q_ids, ans_ids):
    """Student logits on answer tokens (grad flows through virtual_kv_list)."""
    cache = _build_cache_from_kv_list(virtual_kv_list, sink_kv=sink_kv)
    pos_q_start = true_demo_len if align_true_positions else demo_len
    attn_q = torch.ones(1, demo_len + q_ids.shape[1], dtype=torch.long, device=q_ids.device)
    out_q = model(input_ids=q_ids, past_key_values=cache,
                  attention_mask=attn_q,
                  position_ids=_pos_ids(pos_q_start, q_ids.shape[1], q_ids.device),
                  use_cache=True)
    first = out_q.logits[:, -1:]
    base = demo_len + q_ids.shape[1]
    pos_a_start = pos_q_start + q_ids.shape[1]
    attn_a = torch.ones(1, base + ans_ids.shape[1], dtype=torch.long, device=q_ids.device)
    out_a = model(input_ids=ans_ids, past_key_values=out_q.past_key_values,
                  attention_mask=attn_a,
                  position_ids=_pos_ids(pos_a_start, ans_ids.shape[1], q_ids.device),
                  use_cache=False)
    if ans_ids.shape[1] == 1:
        return first
    return torch.cat([first, out_a.logits[:, :-1]], dim=1)


# =====================================================================
# Main training loop
# =====================================================================

def run_distillation_training(args):
    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    tokenizer = _setup_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    _freeze(model)

    align = args.align_true_positions
    if align and getattr(model.config, "model_type", "") == "qwen2":
        print("WARN: align_true_positions disabled for Qwen2.")
        align = False

    add_nl = not args.model_name.startswith("gpt2")
    retrieval_data = load_data(task=None, split=args.retrieval_split, k=args.k,
                               seed=args.seed, datasets=args.dataset.split(","), is_null=False)
    train_data = load_data(task=None, split=args.train_split, k=args.k,
                           seed=args.seed, datasets=args.dataset.split(","), is_null=False)
    if not retrieval_data or not train_data:
        raise ValueError("Empty data.")

    # --- Diverse demo texts for training ---
    demo_texts = _generate_diverse_demo_texts(
        retrieval_data, args.k, args.num_demo_sets, args.seed, add_nl,
    )
    print(f"Generated {len(demo_texts)} diverse demo texts (k={args.k}).")

    # --- Sidecar ---
    sidecar = LayerGroupSSMSidecar(
        model.config,
        ssm_dim=args.ssm_dim,
        num_virtual_tokens=args.num_virtual_tokens,
        num_groups=args.num_groups,
    ).to(device=device, dtype=model.get_input_embeddings().weight.dtype)
    sidecar.train()

    optimizer = torch.optim.AdamW(sidecar.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    total_steps = args.epochs * len(demo_texts)
    warmup = min(100, total_steps // 10)

    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    rng = random.Random(args.seed)
    step = 0
    running = {"loss": 0.0, "kv": 0.0, "logit": 0.0}
    qa_batch_size = max(1, args.train_batch_size)

    print(f"\n===== Training: epochs={args.epochs}, demo_sets={len(demo_texts)}, "
          f"ssm_dim={args.ssm_dim}, groups={args.num_groups}, "
          f"vtokens={args.num_virtual_tokens}, sink={args.sink_tokens} =====\n")

    for epoch in range(args.epochs):
        order = list(range(len(demo_texts)))
        rng.shuffle(order)
        for di in tqdm(order, desc=f"epoch-{epoch+1}"):
            demo_text = demo_texts[di]
            demo_ids = tokenizer(demo_text, return_tensors="pt",
                                 add_special_tokens=False)["input_ids"].to(device)
            if demo_ids.numel() == 0:
                continue
            true_demo_len = demo_ids.shape[1]

            # ---- Teacher full KV (no grad) ----
            teacher_kv = _teacher_demo_kv(model, demo_ids)

            # ---- Student virtual KV (grad) ----
            virtual_kv = _student_virtual_kv(model, sidecar, demo_ids)

            # ---- Sink KV (no grad, optional) ----
            sink_kv = None
            st = min(args.sink_tokens, demo_ids.shape[1])
            if st > 0:
                with torch.no_grad():
                    sink_kv = model(input_ids=demo_ids[:, :st], use_cache=True).past_key_values
            demo_len = st + sidecar.num_virtual_tokens

            # ---- Loss 1: KV matching ----
            loss_kv = kv_matching_loss(virtual_kv, teacher_kv, sidecar.num_virtual_tokens,
                                       loss_type=args.kv_loss_type)

            # ---- Loss 2: QA logit distillation ----
            loss_logit = torch.tensor(0.0, device=device)
            if args.loss_w_logit > 0:
                qa_indices = rng.sample(range(len(train_data)), min(qa_batch_size, len(train_data)))
                valid = 0
                for qi in qa_indices:
                    dp = train_data[qi]
                    q_text, ans_text = _normalize_text(dp, is_first=False, add_newlines=add_nl)
                    q_ids = tokenizer(q_text, return_tensors="pt",
                                      add_special_tokens=False)["input_ids"].to(device)
                    a_ids = tokenizer(ans_text, return_tensors="pt",
                                      add_special_tokens=False)["input_ids"].to(device)
                    if q_ids.numel() == 0 or a_ids.numel() == 0:
                        continue

                    t_logits = _teacher_qa_logits(model, teacher_kv, true_demo_len, q_ids, a_ids)
                    s_logits = _student_qa_logits(
                        model, virtual_kv, sink_kv, demo_len, true_demo_len, align, q_ids, a_ids,
                    )
                    temp = args.kd_temperature
                    kl = F.kl_div(
                        F.log_softmax(s_logits / temp, dim=-1),
                        F.softmax(t_logits.detach() / temp, dim=-1),
                        reduction="batchmean",
                    ) * (temp ** 2)
                    ce = F.cross_entropy(
                        s_logits.reshape(-1, s_logits.size(-1)),
                        a_ids.reshape(-1),
                        reduction="mean",
                    )
                    loss_logit = loss_logit + kl + 0.5 * ce
                    valid += 1
                if valid > 0:
                    loss_logit = loss_logit / valid

            # ---- Combined loss ----
            loss = args.loss_w_kv * loss_kv + args.loss_w_logit * loss_logit

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sidecar.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            running["loss"] += loss.item()
            running["kv"] += loss_kv.item()
            running["logit"] += (loss_logit.item() if isinstance(loss_logit, torch.Tensor) else loss_logit)

            if step % args.log_every == 0:
                d = float(args.log_every)
                print(f"step={step} loss={running['loss']/d:.4f} "
                      f"kv={running['kv']/d:.4f} logit={running['logit']/d:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.6f}")
                running = {"loss": 0.0, "kv": 0.0, "logit": 0.0}

            if 0 < args.max_steps <= step:
                break
        if 0 < args.max_steps <= step:
            break

    if args.save_sidecar_path:
        torch.save({"sidecar": sidecar.state_dict(), "args": vars(args),
                     "model_name": args.model_name}, args.save_sidecar_path)
        print(f"Saved: {args.save_sidecar_path}")

    return {k: v.detach().cpu() for k, v in sidecar.state_dict().items()}


# =====================================================================
# Eval experiment
# =====================================================================

def run_experiment(args, sidecar_state_dict=None):
    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    tokenizer = _setup_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    align = args.align_true_positions
    if align and getattr(model.config, "model_type", "") == "qwen2":
        print("WARN: align_true_positions disabled for Qwen2.")
        align = False

    add_nl = not args.model_name.startswith("gpt2")
    retrieval_data = load_data(task=None, split=args.retrieval_split, k=args.k,
                               seed=args.seed, datasets=args.dataset.split(","), is_null=False)
    eval_data = load_data(task=None, split=args.eval_split, k=args.k,
                          seed=args.seed, datasets=args.dataset.split(","), is_null=False)
    fixed_demos = _choose_fixed_demos(retrieval_data, args.k, args.demo_strategy, args.seed)
    demo_text = _build_demo_text(fixed_demos, add_newlines=add_nl)
    original_demo_len = len(tokenizer(demo_text, add_special_tokens=False)["input_ids"])

    sidecar = LayerGroupSSMSidecar(
        model.config, ssm_dim=args.ssm_dim, num_virtual_tokens=args.num_virtual_tokens,
        num_groups=args.num_groups,
    ).to(device=device, dtype=model.get_input_embeddings().weight.dtype)
    _load_sidecar(sidecar, args, device, sidecar_state_dict)
    sidecar.eval()

    scorer = SSMHybridICLScorer(model, tokenizer, device, sidecar,
                                sink_tokens=args.sink_tokens, align_true_positions=align)

    t0 = time.perf_counter()
    scorer.build_demo_cache(demo_text)

    preds = []
    proxy = _attn_proxy_full(scorer.demo_len)

    for dp in tqdm(eval_data, desc="ssm-eval"):
        q_text, _ = _normalize_text(dp, is_first=False, add_newlines=add_nl)
        q_len_tok = len(tokenizer(q_text, add_special_tokens=False)["input_ids"])

        first, q_past, q_len = scorer.prefill_question(q_text)
        proxy += _attn_proxy_inc(scorer.demo_len, q_len)

        opts = dp["options"]
        opt_texts = [_normalize_option(o, add_nl) for o in opts]
        scores_t = scorer.score_options_nll(first, q_past, q_len, opt_texts)
        scores = {o: scores_t[ot] for o, ot in zip(opts, opt_texts)}
        for ot in opt_texts:
            proxy += _attn_proxy_inc(scorer.demo_len + q_len,
                                     len(tokenizer(ot, add_special_tokens=False)["input_ids"]))
        preds.append(min(scores, key=scores.get))

    elapsed = time.perf_counter() - t0
    acc = _accuracy(preds, [dp["output"] for dp in eval_data])

    print(f"\n===== SSM Hybrid Eval =====")
    print(f"Accuracy        = {acc:.6f}")
    print(f"Time            = {elapsed:.3f}s")
    print(f"Original prefix = {original_demo_len} tokens")
    print(f"Compressed      = {scorer.demo_len} tokens")
    print(f"Attn proxy      = {proxy}")


# =====================================================================
# Loss comparison (SSM vs full KV)
# =====================================================================

def run_loss_comparison(args, sidecar_state_dict=None):
    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    tokenizer = _setup_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    align = args.align_true_positions
    if align and getattr(model.config, "model_type", "") == "qwen2":
        align = False

    add_nl = not args.model_name.startswith("gpt2")
    retrieval_data = load_data(task=None, split=args.retrieval_split, k=args.k,
                               seed=args.seed, datasets=args.dataset.split(","), is_null=False)
    eval_data = load_data(task=None, split=args.eval_split, k=args.k,
                          seed=args.seed, datasets=args.dataset.split(","), is_null=False)
    if not retrieval_data or not eval_data:
        raise ValueError("Empty data.")

    fixed_demos = _choose_fixed_demos(retrieval_data, args.k, args.demo_strategy, args.seed)
    demo_text = _build_demo_text(fixed_demos, add_newlines=add_nl)
    demo_ids = tokenizer(demo_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    # Teacher baseline
    with torch.no_grad():
        teacher_kv = model(input_ids=demo_ids, use_cache=True).past_key_values
    full_len = demo_ids.shape[1]

    # SSM scorer
    sidecar = LayerGroupSSMSidecar(
        model.config, ssm_dim=args.ssm_dim, num_virtual_tokens=args.num_virtual_tokens,
        num_groups=args.num_groups,
    ).to(device=device, dtype=model.get_input_embeddings().weight.dtype)
    _load_sidecar(sidecar, args, device, sidecar_state_dict)
    sidecar.eval()

    scorer = SSMHybridICLScorer(model, tokenizer, device, sidecar,
                                sink_tokens=args.sink_tokens, align_true_positions=align)
    scorer.build_demo_cache(demo_text)

    full_losses, ssm_losses = [], []
    for dp in tqdm(eval_data, desc="compare-loss"):
        q_text, ans_text = _normalize_text(dp, is_first=False, add_newlines=add_nl)
        q_ids = tokenizer(q_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        a_ids = tokenizer(ans_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        if q_ids.numel() == 0 or a_ids.numel() == 0:
            continue

        # Teacher CE
        with torch.no_grad():
            t_logits = _teacher_qa_logits(model, teacher_kv, full_len, q_ids, a_ids)
            t_ce = F.cross_entropy(t_logits.reshape(-1, t_logits.size(-1)), a_ids.reshape(-1), reduction="mean").item()

        # SSM CE
        with torch.no_grad():
            emb = model.get_input_embeddings()(demo_ids).to(dtype=next(sidecar.parameters()).dtype)
            vkv = sidecar(emb)
            sink_kv = None
            st = min(args.sink_tokens, demo_ids.shape[1])
            if st > 0:
                sink_kv = model(input_ids=demo_ids[:, :st], use_cache=True).past_key_values
            demo_len = st + sidecar.num_virtual_tokens
            s_logits = _student_qa_logits(
                model, vkv, sink_kv, demo_len, full_len, align, q_ids, a_ids,
            )
            s_ce = F.cross_entropy(s_logits.reshape(-1, s_logits.size(-1)), a_ids.reshape(-1), reduction="mean").item()

        full_losses.append(t_ce)
        ssm_losses.append(s_ce)

    if not full_losses:
        raise ValueError("No valid samples.")

    mf = sum(full_losses) / len(full_losses)
    ms = sum(ssm_losses) / len(ssm_losses)
    mse = ((ms - mf) / max(ms, mf)) ** 2

    print(f"\n===== Loss Comparison =====")
    print(f"samples       = {len(full_losses)}")
    print(f"mean_CE_full  = {mf:.6f}")
    print(f"mean_CE_ssm   = {ms:.6f}")
    print(f"norm_sq_err   = {mse:.6f}")


# =====================================================================
# Text / data utilities
# =====================================================================

def _setup_tokenizer(name: str):
    tok = GPT2Tokenizer.from_pretrained(name) if name.startswith("gpt2") else AutoTokenizer.from_pretrained(name)
    if tok.padding_side == "left":
        tok.padding_side = "right"
    if tok.eos_token_id is None and tok.sep_token is not None:
        tok.eos_token = tok.sep_token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.bos_token_id is None:
        tok.bos_token = tok.eos_token
    return tok


def _normalize_text(dp: Dict, is_first: bool, add_newlines: bool) -> Tuple[str, str]:
    q = ("\n" + dp["input"]) if (add_newlines and not is_first) else dp["input"]
    a = ("\n" + dp["output"]) if add_newlines else dp["output"]
    return q, a


def _normalize_option(opt: str, add_newlines: bool) -> str:
    return ("\n" + opt) if add_newlines else opt


def _build_demo_text(demos: List[Dict], add_newlines: bool) -> str:
    parts = []
    for i, dp in enumerate(demos):
        q, a = _normalize_text(dp, is_first=(i == 0), add_newlines=add_newlines)
        parts.append(q + a)
    return "".join(parts)


def _choose_fixed_demos(data, k, strategy, seed):
    if strategy == "first":
        return data[:k]
    return [data[i] for i in random.Random(seed).sample(range(len(data)), k)]


def _accuracy(preds, gts):
    return sum(int(p.strip() == g.strip()) for p, g in zip(preds, gts)) / max(1, len(gts))


def _attn_proxy_full(n):
    return n * (n + 1) // 2


def _attn_proxy_inc(ctx, new):
    return new * ctx + new * (new + 1) // 2


def _load_sidecar(sidecar, args, device, state_dict=None):
    if state_dict is not None:
        sidecar.load_state_dict(state_dict, strict=True)
        print("Loaded sidecar from in-memory state.")
    elif args.load_sidecar_path:
        ckpt = torch.load(args.load_sidecar_path, map_location=device)
        st = ckpt["sidecar"] if isinstance(ckpt, dict) and "sidecar" in ckpt else ckpt
        sidecar.load_state_dict(st, strict=True)
        print(f"Loaded sidecar from: {args.load_sidecar_path}")
    else:
        print("WARN: No sidecar checkpoint; results may be random.")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retrieval_split", type=str, default="test")
    p.add_argument("--eval_split", type=str, default="dev")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--demo_strategy", type=str, default="first")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--run_mode", type=str, default="eval",
                   choices=["eval", "train", "train_eval", "compare_loss", "train_compare_loss"])

    # Sidecar architecture
    p.add_argument("--num_virtual_tokens", type=int, default=16)
    p.add_argument("--sink_tokens", type=int, default=4)
    p.add_argument("--ssm_dim", type=int, default=512)
    p.add_argument("--num_groups", type=int, default=4)
    p.add_argument("--align_true_positions", action="store_true")

    # Training
    p.add_argument("--num_demo_sets", type=int, default=500)
    p.add_argument("--train_batch_size", type=int, default=4, help="QA pairs per demo set per step")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--kd_temperature", type=float, default=2.0)
    p.add_argument("--loss_w_kv", type=float, default=1.0)
    p.add_argument("--loss_w_logit", type=float, default=0.5)
    p.add_argument("--kv_loss_type", type=str, default="cosine", choices=["cosine", "mse"])
    p.add_argument("--log_every", type=int, default=20)

    # Checkpoints
    p.add_argument("--save_sidecar_path", type=str, default="")
    p.add_argument("--load_sidecar_path", type=str, default="")

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