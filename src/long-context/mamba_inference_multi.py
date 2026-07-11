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

from utils.data import load_data


# =====================================================================
# MMLU data loading
# =====================================================================

MMLU_COLLEGE_SUBJECTS = [
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
]


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
    if args.dataset.strip().lower() == "mmlu":
        return load_mmlu_college_split(args, split)
    return load_data(
        task=None, split=split, k=args.k,
        seed=args.seed, datasets=args.dataset.split(","), is_null=False
    )


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
    def __init__(self, hidden_size, ssm_dim, num_layers_in_group, num_kv_heads,
                 head_dim, num_virtual_tokens, dt=1.0, scan_chunk_size=64):
        super().__init__()
        self.ssm_dim = ssm_dim
        self.num_layers_in_group = num_layers_in_group
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_virtual_tokens = num_virtual_tokens
        self.scan_chunk_size = scan_chunk_size

        # Discretised HiPPO-LegS A matrix — FROZEN, not trained.
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
        """
        Chunk-parallel scan using precomputed A powers (real-valued).

        For a chunk of C inputs u_0..u_{C-1} and incoming state s:
          s_out = A^C @ s + A^{C-1} @ u_0 + A^{C-2} @ u_1 + ... + I @ u_{C-1}

        The sum is a single batched einsum per chunk, reducing T kernel
        launches to ~T/C launches.  A powers are computed on-the-fly per
        chunk (C matmuls of [D,D]) to avoid storing a large buffer.
        """
        if chunk_size is None:
            chunk_size = self.scan_chunk_size
        bsz, T, _ = x.shape
        device, dtype = x.device, x.dtype
        D = self.ssm_dim

        # Project all inputs at once: [B, T, D]
        u_all = self.B(x)

        state = torch.zeros(bsz, D, device=device, dtype=dtype)
        A = self.A.to(device=device, dtype=dtype)  # [D, D]

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            C = end - start
            u_chunk = u_all[:, start:end, :]  # [B, C, D]

            if C == 1:
                state = (state @ A.T) + u_chunk[:, 0]
                continue

            # Build weight matrices on the fly: weights[t] = A^{C-1-t}
            # weights[C-1] = I, weights[C-2] = A, weights[C-3] = A^2, ...
            # Compute iteratively: only C matmuls of [D,D]
            weights = torch.zeros(C, D, D, device=device, dtype=dtype)
            weights[C - 1] = torch.eye(D, device=device, dtype=dtype)
            if C >= 2:
                weights[C - 2] = A
            for t in range(C - 3, -1, -1):
                weights[t] = weights[t + 1] @ A

            # Batched contribution: einsum over chunk dim
            # weights: [C, D, D], u_chunk: [B, C, D]
            # contrib[b, c, d] = sum_j weights[c, d, j] * u_chunk[b, c, j]
            # total: [B, D] = sum over c
            total_contrib = torch.einsum('cdj,bcj->bd', weights, u_chunk)

            # A^C = weights[0] @ A (since weights[0] = A^{C-1})
            A_C = weights[0] @ A  # [D, D]
            state = (state @ A_C.T) + total_contrib

        return state

    def forward(self, x):
        bsz = x.shape[0]
        state = self.scan(x)
        flat = self.proj(state)
        r = flat.view(bsz, self.num_layers_in_group, 2,
                       self.num_kv_heads, self.num_virtual_tokens, self.head_dim)
        return [(r[:, i, 0], r[:, i, 1]) for i in range(self.num_layers_in_group)]


# =====================================================================
# Sidecar
# =====================================================================

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
        print(f"[Sidecar] {num_groups} groups, ssm_dim={ssm_dim}, vtokens={num_virtual_tokens} | {info}")

    def forward(self, embeddings):
        all_kv = []
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


def _build_cache_from_kv_list(kv_list, sink_kv=None):
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


def _fresh_cache_copy(cache):
    legacy = _cache_to_legacy(cache)
    fresh = tuple((k.clone(), v.clone()) for k, v in legacy)
    return _ensure_cache_obj(fresh)


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
            if s >= T_full:
                break
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
# Scorer
# =====================================================================

class SSMHybridICLScorer:
    def __init__(self, model, tokenizer, device, sidecar, sink_tokens=0,
                 align_true_positions=False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = model.get_input_embeddings().weight.dtype
        self.sidecar = sidecar
        self.sink_tokens = max(0, sink_tokens)
        self.align_true_positions = align_true_positions
        self.demo_cache = None
        self.demo_len = 0
        self.true_demo_len = 0

    @torch.no_grad()
    def build_demo_cache(self, demo_text):
        ids = self.tokenizer(demo_text, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(self.device)
        if ids.numel() == 0:
            raise ValueError("Empty demo text.")
        self.true_demo_len = ids.shape[1]
        emb = self.model.get_input_embeddings()(ids).to(dtype=self.dtype)
        virtual_kv = self.sidecar(emb)
        sink_kv = None
        st = min(self.sink_tokens, ids.shape[1])
        if st > 0:
            sink_kv = self.model(input_ids=ids[:, :st], use_cache=True).past_key_values
        self.demo_cache = _build_cache_from_kv_list(virtual_kv, sink_kv=sink_kv)
        self.demo_len = st + self.sidecar.num_virtual_tokens

    @torch.no_grad()
    def prefill_question(self, question_text):
        q_ids = self.tokenizer(question_text, return_tensors="pt",
                               add_special_tokens=False)["input_ids"].to(self.device)
        if q_ids.numel() == 0:
            raise ValueError("Empty question.")
        cache = _fresh_cache_copy(self.demo_cache)
        attn = torch.ones(1, self.demo_len + q_ids.shape[1], dtype=torch.long, device=self.device)
        pos_start = self.true_demo_len if self.align_true_positions else self.demo_len
        out = self.model(input_ids=q_ids, past_key_values=cache, attention_mask=attn,
                         position_ids=_pos_ids(pos_start, q_ids.shape[1], self.device),
                         use_cache=True)
        return out.logits[:, -1, :], out.past_key_values, q_ids.shape[1]

    @torch.no_grad()
    def score_options_nll(self, first_logits, past_after_q, q_len, options):
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
        losses = F.cross_entropy(pred.reshape(-1, pred.size(-1)), ids.reshape(-1),
                                 reduction="none").view(bsz, mx)
        nll = (losses * mask).sum(1)
        return {o: float(nll[i]) for i, o in enumerate(options)}


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
            raise RuntimeError("--is_quant requires CUDA (bitsandbytes 4-bit).")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        return AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    return AutoModelForCausalLM.from_pretrained(args.model_name).to(device)


def _generate_diverse_demo_texts(retrieval_data, k, num_sets, seed, add_newlines):
    rng = random.Random(seed)
    n = len(retrieval_data)
    texts = []
    for _ in range(num_sets):
        indices = rng.sample(range(n), min(k, n))
        demos = [retrieval_data[i] for i in indices]
        texts.append(_build_demo_text(demos, add_newlines))
    return texts


def _teacher_demo_kv(model, demo_ids):
    with torch.no_grad():
        out = model(input_ids=demo_ids, use_cache=True)
        return _cache_to_legacy(out.past_key_values)


def _student_virtual_kv(model, sidecar, demo_ids):
    emb = model.get_input_embeddings()(demo_ids)
    emb = emb.to(dtype=next(sidecar.parameters()).dtype)
    return sidecar(emb)


@torch.no_grad()
def _teacher_qa_forward(model, demo_ids, q_ids, ans_ids, output_hidden=False):
    """
    Teacher forward: single concatenated pass, returns logits on answer tokens.
    If output_hidden=True, also returns last-layer hidden states on answer tokens.
    """
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
    """
    Student forward with virtual KV cache. Returns logits on answer tokens.
    If output_hidden=True, also returns last-layer hidden states on answer tokens.
    """
    cache = _build_cache_from_kv_list(virtual_kv_list, sink_kv=sink_kv)
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
        return logits, out_a.hidden_states[-1]
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


def hidden_state_loss(student_hidden, teacher_hidden):
    min_t = min(student_hidden.size(1), teacher_hidden.size(1))
    if min_t == 0:
        return torch.tensor(0.0, device=student_hidden.device)
    sh = student_hidden[:, :min_t, :].reshape(-1, student_hidden.size(-1))
    th = teacher_hidden[:, :min_t, :].reshape(-1, teacher_hidden.size(-1))
    return (1.0 - F.cosine_similarity(sh, th, dim=-1)).mean()


def _strip_leading_format_tokens(ans_ids, tokenizer):
    """
    Strip leading formatting tokens (newline, space) from answer ids
    so that CE comparison only measures actual content prediction.
    Returns (stripped_ids, num_stripped).
    """
    strip_tokens = set()
    for ch in ["\n", " ", "\t"]:
        toks = tokenizer(ch, add_special_tokens=False)["input_ids"]
        strip_tokens.update(toks)

    idx = 0
    while idx < ans_ids.shape[1] and ans_ids[0, idx].item() in strip_tokens:
        idx += 1
    if idx >= ans_ids.shape[1]:
        # Don't strip everything — keep at least 1 token.
        return ans_ids, 0
    return ans_ids[:, idx:], idx


# =====================================================================
# Training loop
# =====================================================================

def run_distillation_training(args):
    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    tokenizer = _setup_tokenizer(args.model_name)
    model = _load_causal_lm(args, device)
    _freeze(model)

    align = args.align_true_positions
    if align and getattr(model.config, "model_type", "") == "qwen2":
        print("WARN: align_true_positions disabled for Qwen2.")
        align = False

    add_nl = not args.model_name.startswith("gpt2")

    # Retrieve demos from retrieval_split (test), train QA from train_split
    retrieval_data = _load_dataset(args, args.retrieval_split)
    train_data = _load_dataset(args, args.train_split)
    if not retrieval_data or not train_data:
        raise ValueError("Empty data.")

    demo_texts = _generate_diverse_demo_texts(
        retrieval_data, args.k, args.num_demo_sets, args.seed, add_nl,
    )
    print(f"Generated {len(demo_texts)} diverse demo texts (k={args.k}).")

    sidecar = LayerGroupSSMSidecar(
        model.config, ssm_dim=args.ssm_dim, num_virtual_tokens=args.num_virtual_tokens,
        num_groups=args.num_groups,
    ).to(device=device, dtype=model.get_input_embeddings().weight.dtype)

    if args.load_sidecar_path:
        _load_sidecar(sidecar, args, device)
        print(f"Loaded sidecar from: {args.load_sidecar_path}")

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
    running = {"loss": 0.0, "kv": 0.0, "logit": 0.0, "hid": 0.0, "ce": 0.0}
    qa_batch_size = max(1, args.train_batch_size)
    use_kv = args.loss_w_kv > 0
    use_logit = args.loss_w_logit > 0
    use_hid = args.loss_w_hid > 0
    use_ce = args.loss_w_ce > 0
    need_qa = use_logit or use_hid or use_ce

    print(f"\n===== Training: epochs={args.epochs}, demo_sets={len(demo_texts)}, "
          f"ssm_dim={args.ssm_dim}, groups={args.num_groups}, "
          f"vtokens={args.num_virtual_tokens}, sink={args.sink_tokens}")
    print(f"  loss weights: kv={args.loss_w_kv}, logit={args.loss_w_logit}, "
          f"hid={args.loss_w_hid}, ce={args.loss_w_ce}")
    print()

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

            virtual_kv = _student_virtual_kv(model, sidecar, demo_ids)

            sink_kv = None
            st = min(args.sink_tokens, demo_ids.shape[1])
            if st > 0:
                with torch.no_grad():
                    sink_kv = model(input_ids=demo_ids[:, :st], use_cache=True).past_key_values
            demo_len = st + sidecar.num_virtual_tokens

            loss_kv = torch.tensor(0.0, device=device)
            if use_kv:
                teacher_kv = _teacher_demo_kv(model, demo_ids)
                loss_kv = kv_matching_loss(virtual_kv, teacher_kv, sidecar.num_virtual_tokens,
                                           loss_type=args.kv_loss_type)

            loss_logit = torch.tensor(0.0, device=device)
            loss_hid = torch.tensor(0.0, device=device)
            loss_ce = torch.tensor(0.0, device=device)
            if need_qa:
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
                            reduction="batchmean",
                        ) * (temp ** 2)
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
                                    s_logits_stripped[:, :min_t, :].reshape(-1, s_logits_stripped.size(-1)),
                                    a_stripped[:, :min_t].reshape(-1),
                                    reduction="mean",
                                )
                                loss_ce = loss_ce + ce
                    valid += 1
                if valid > 0:
                    loss_logit = loss_logit / valid
                    loss_hid = loss_hid / valid
                    loss_ce = loss_ce / valid

            loss = (args.loss_w_kv * loss_kv
                    + args.loss_w_logit * loss_logit
                    + args.loss_w_hid * loss_hid
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
            running["hid"] += (loss_hid.item() if isinstance(loss_hid, torch.Tensor) else loss_hid)
            running["ce"] += (loss_ce.item() if isinstance(loss_ce, torch.Tensor) else loss_ce)

            if step % args.log_every == 0:
                d = float(args.log_every)
                print(f"step={step} loss={running['loss']/d:.4f} "
                      f"kv={running['kv']/d:.4f} logit={running['logit']/d:.4f} "
                      f"hid={running['hid']/d:.4f} ce={running['ce']/d:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.6f}")
                running = {"loss": 0.0, "kv": 0.0, "logit": 0.0, "hid": 0.0, "ce": 0.0}

            # Drop large per-step tensor references early; no effect on numerics.
            del virtual_kv, sink_kv, demo_ids, loss, loss_kv, loss_logit, loss_hid, loss_ce
            if device.startswith("cuda") and args.empty_cache_every > 0 and (step % args.empty_cache_every == 0):
                gc.collect()
                torch.cuda.empty_cache()

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
    model = _load_causal_lm(args, device)
    model.eval()
    track_cuda_peak_mem = device.startswith("cuda")
    model_mem_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    model_mem_gb = model_mem_bytes / (1024 ** 3)

    align = args.align_true_positions
    if align and getattr(model.config, "model_type", "") == "qwen2":
        print("WARN: align_true_positions disabled for Qwen2.")
        align = False

    add_nl = not args.model_name.startswith("gpt2")
    retrieval_data = _load_dataset(args, args.retrieval_split)
    eval_data = _load_dataset(args, args.eval_split)

    per_query_random = (args.num_eval_demo_sets > 1)
    n_queries = max(1, len(eval_data))

    sidecar = LayerGroupSSMSidecar(
        model.config, ssm_dim=args.ssm_dim, num_virtual_tokens=args.num_virtual_tokens,
        num_groups=args.num_groups,
    ).to(device=device, dtype=model.get_input_embeddings().weight.dtype)
    _load_sidecar(sidecar, args, device, sidecar_state_dict)
    sidecar.eval()
    sidecar_mem_bytes = sum(p.nelement() * p.element_size() for p in sidecar.parameters())
    sidecar_mem_gb = sidecar_mem_bytes / (1024 ** 3)
    flop_params = _get_model_flop_params(model) if args.flops else None

    rng_eval = random.Random(args.seed + 7777)

    full_kv_preds, ssm_preds = [], []
    full_losses, ssm_losses = [], []
    logit_kls, logit_cossims, top1_agrees, top5_overlaps = [], [], [], []
    query_logit_mses = []
    ds_results = {}  # ds -> {"full_p":[], "ssm_p":[], "gt":[]}
    full_kv_total_flops = 0.0
    ssm_demo_flops = 0.0
    ssm_query_total_flops = 0.0
    full_kv_peak_mem_bytes = 0
    ssm_demo_peak_mem_bytes = 0
    ssm_query_peak_mem_bytes = 0

    if per_query_random:
        print(f"\nPer-query random demos (k={args.k}) from {len(retrieval_data)} candidates.")
    else:
        print(f"\nFixed demos (strategy={args.demo_strategy}).")

    for qi, dp in enumerate(tqdm(eval_data, desc="eval")):
        q_text, ans_text = _normalize_text(dp, is_first=False, add_newlines=add_nl)
        q_ids = tokenizer(q_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
        a_ids = tokenizer(ans_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)

        # Pick demos for this query
        if per_query_random:
            indices = rng_eval.sample(range(len(retrieval_data)),
                                      min(args.k, len(retrieval_data)))
            demos = [retrieval_data[i] for i in indices]
        else:
            if not hasattr(run_experiment, '_fixed_demos'):
                run_experiment._fixed_demos = _choose_fixed_demos(
                    retrieval_data, args.k, args.demo_strategy, args.seed)
            demos = run_experiment._fixed_demos

        demo_text = _build_demo_text(demos, add_newlines=add_nl)
        demo_ids = tokenizer(demo_text, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(device)
        original_demo_len = demo_ids.shape[1]
        opts = dp["options"]
        q_full_kv_flops = 0.0
        q_ssm_flops = 0.0

        # ─── Full-KV: concatenated forward ───
        if track_cuda_peak_mem:
            torch.cuda.reset_peak_memory_stats()
        opt_scores = {}
        for opt in opts:
            opt_text = _normalize_option(opt, add_nl)
            opt_ids = tokenizer(opt_text, return_tensors="pt",
                                add_special_tokens=False)["input_ids"].to(device)
            if opt_ids.numel() == 0:
                opt_scores[opt] = float("inf")
                continue
            if args.flops:
                total_len = original_demo_len + q_ids.shape[1] + opt_ids.shape[1]
                q_full_kv_flops += _analytical_flops_full(flop_params, total_len)
            with torch.no_grad():
                all_ids = torch.cat([demo_ids, q_ids, opt_ids], dim=1)
                out = model(input_ids=all_ids, use_cache=False)
                start = demo_ids.shape[1] + q_ids.shape[1] - 1
                pred_logits = out.logits[:, start:start + opt_ids.shape[1], :]
                nll = F.cross_entropy(
                    pred_logits.reshape(-1, pred_logits.size(-1)),
                    opt_ids.reshape(-1), reduction="sum",
                ).item()
            opt_scores[opt] = nll
        full_kv_preds.append(min(opt_scores, key=opt_scores.get))
        if track_cuda_peak_mem:
            full_kv_peak_mem_bytes = max(full_kv_peak_mem_bytes, torch.cuda.max_memory_allocated())

        # Full-KV CE
        t_logits_s = None
        first_token_id = None
        if q_ids.numel() > 0 and a_ids.numel() > 0:
            with torch.no_grad():
                t_logits = _teacher_qa_logits_nocache(model, demo_ids, q_ids, a_ids)
                a_stripped, n_stripped = _strip_leading_format_tokens(a_ids, tokenizer)
                if a_stripped.numel() > 0:
                    t_logits_s = t_logits[:, n_stripped:, :]
                    first_token_id = a_stripped[:, 0].unsqueeze(1)
                    t_ce = F.cross_entropy(
                        t_logits_s.reshape(-1, t_logits_s.size(-1)),
                        a_stripped.reshape(-1), reduction="mean",
                    ).item()
                    full_losses.append(t_ce)

        # ─── SSM: compress this query's demos ───
        if track_cuda_peak_mem and qi == 0:
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            emb = model.get_input_embeddings()(demo_ids).to(dtype=next(sidecar.parameters()).dtype)
            virtual_kv = sidecar(emb)
            sink_kv = None
            st = min(args.sink_tokens, demo_ids.shape[1])
            if st > 0:
                sink_kv = model(input_ids=demo_ids[:, :st], use_cache=True).past_key_values
            ssm_demo_len = st + sidecar.num_virtual_tokens
        if track_cuda_peak_mem and qi == 0:
            ssm_demo_peak_mem_bytes = torch.cuda.max_memory_allocated()
        # Demo FLOPs: one-time amortized cost, aligned with streaming baseline style.
        if args.flops and qi == 0:
            ssm_demo_flops = _sidecar_analytical_flops(sidecar, original_demo_len)
            if st > 0:
                ssm_demo_flops += _analytical_flops_full(flop_params, st)

        # SSM accuracy: build scorer on the fly for this query's demos
        scorer = SSMHybridICLScorer(model, tokenizer, device, sidecar,
                                    sink_tokens=args.sink_tokens, align_true_positions=align)
        scorer.demo_cache = _build_cache_from_kv_list(virtual_kv, sink_kv=sink_kv)
        scorer.demo_len = ssm_demo_len
        scorer.true_demo_len = original_demo_len

        if track_cuda_peak_mem:
            torch.cuda.reset_peak_memory_stats()
        first, q_past, q_len = scorer.prefill_question(q_text)
        if args.flops:
            q_ssm_flops += _analytical_flops_inc(flop_params, q_len, ssm_demo_len)
        opt_texts = [_normalize_option(o, add_nl) for o in opts]
        scores_t = scorer.score_options_nll(first, q_past, q_len, opt_texts)
        scores = {o: scores_t[ot] for o, ot in zip(opts, opt_texts)}
        if args.flops:
            for opt_text in opt_texts:
                opt_len = len(tokenizer(opt_text, add_special_tokens=False)["input_ids"])
                q_ssm_flops += _analytical_flops_inc(
                    flop_params, opt_len, ssm_demo_len + q_len
                )
        ssm_preds.append(min(scores, key=scores.get))
        if track_cuda_peak_mem:
            ssm_query_peak_mem_bytes = max(ssm_query_peak_mem_bytes, torch.cuda.max_memory_allocated())

        # SSM CE
        s_logits_s = None
        if q_ids.numel() > 0 and a_ids.numel() > 0:
            with torch.no_grad():
                s_logits = _student_qa_logits(
                    model, virtual_kv, sink_kv,
                    ssm_demo_len, original_demo_len, align, q_ids, a_ids,
                )
                a_stripped, n_stripped = _strip_leading_format_tokens(a_ids, tokenizer)
                if a_stripped.numel() > 0:
                    s_logits_s = s_logits[:, n_stripped:, :]
                    s_ce = F.cross_entropy(
                        s_logits_s.reshape(-1, s_logits_s.size(-1)),
                        a_stripped.reshape(-1), reduction="mean",
                    ).item()
                    ssm_losses.append(s_ce)

                    # Logit fidelity against full-KV teacher logits.
                    t_logits_full = _teacher_qa_logits_nocache(model, demo_ids, q_ids, a_ids)
                    t_logits_stripped = t_logits_full[:, n_stripped:, :]
                    min_t = min(t_logits_stripped.size(1), s_logits_s.size(1))
                    if min_t > 0:
                        tl = t_logits_stripped[:, :min_t, :]
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

        # Per-query relative squared error on the correct answer's first-token logit.
        if t_logits_s is not None and s_logits_s is not None and first_token_id is not None:
            t_first = t_logits_s[:, 0, :].gather(1, first_token_id).squeeze(1)
            s_first = s_logits_s[:, 0, :].gather(1, first_token_id).squeeze(1)
            t_first_val = float(t_first.item())
            s_first_val = float(s_first.item())
            denom = max(abs(s_first_val), abs(t_first_val))
            rel_sq = ((s_first_val - t_first_val) / denom) ** 2 if denom != 0 else 0.0
            query_logit_mses.append(rel_sq)

        # Track per-dataset
        ds = dp.get("task", dp.get("dataset", "unknown"))
        if ds not in ds_results:
            ds_results[ds] = {"full_p": [], "ssm_p": [], "gt": []}
        ds_results[ds]["full_p"].append(full_kv_preds[-1])
        ds_results[ds]["ssm_p"].append(ssm_preds[-1])
        ds_results[ds]["gt"].append(dp["output"])
        if args.flops:
            full_kv_total_flops += q_full_kv_flops
            ssm_query_total_flops += q_ssm_flops

    # ─── Aggregate ───
    full_acc = _accuracy(full_kv_preds, [dp["output"] for dp in eval_data])
    ssm_acc = _accuracy(ssm_preds, [dp["output"] for dp in eval_data])

    valid_ces = [(f, s) for f, s in zip(full_losses, ssm_losses) if not (math.isnan(f) or math.isnan(s))]
    valid_full = [f for f, _ in valid_ces]
    valid_ssm = [s for _, s in valid_ces]
    mean_full_ce = sum(valid_full) / max(1, len(valid_full))
    mean_ssm_ce = sum(valid_ssm) / max(1, len(valid_ssm))

    mean_kl = sum(logit_kls) / max(1, len(logit_kls)) if logit_kls else float("nan")
    mean_cos = sum(logit_cossims) / max(1, len(logit_cossims)) if logit_cossims else float("nan")
    mean_top1 = sum(top1_agrees) / max(1, len(top1_agrees)) if top1_agrees else float("nan")
    mean_top5 = sum(top5_overlaps) / max(1, len(top5_overlaps)) if top5_overlaps else float("nan")

    n_compare = len(query_logit_mses)
    per_sample_mse = sum(query_logit_mses) / max(1, len(query_logit_mses))
    d_means = max(mean_ssm_ce, mean_full_ce)
    mean_sq_loss = ((mean_ssm_ce - mean_full_ce) / d_means) ** 2 if d_means > 0 else 0.0
    total_full_flops = full_kv_total_flops
    total_ssm_flops = ssm_demo_flops + ssm_query_total_flops
    flops_reduction = (total_full_flops / total_ssm_flops) if total_ssm_flops > 0 else 0.0

    mode_str = "per-query random" if per_query_random else "fixed"
    print(f"\n{'='*60}")
    print(f"  SSM Hybrid Eval Results  (demos: {mode_str}, k={args.k})")
    print(f"{'='*60}")
    print(f"Full-KV Accuracy    = {full_acc:.6f}")
    print(f"SSM Accuracy        = {ssm_acc:.6f}")
    print(f"Accuracy gap        = {full_acc - ssm_acc:+.6f}")

    print(f"\n{'='*60}")
    print(f"  Per-Dataset Breakdown")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'N':>5} {'FullKV Acc':>10} {'SSM Acc':>10} {'Gap':>10}")
    print(f"{'-'*55}")
    for ds in sorted(ds_results.keys()):
        r = ds_results[ds]
        n = len(r["gt"])
        fa = _accuracy(r["full_p"], r["gt"])
        sa = _accuracy(r["ssm_p"], r["gt"])
        print(f"{ds:<20} {n:>5} {fa:>10.4f} {sa:>10.4f} {fa-sa:>+10.4f}")

    print(f"\n{'='*60}")
    print(f"  Loss Comparison (content tokens only, format stripped)")
    print(f"{'='*60}")
    print(f"KL(SSM||FullKV)     = {mean_kl:.6f}")
    print(f"Cosine similarity   = {mean_cos:.6f}")
    print(f"Top-1 agreement     = {mean_top1:.6f}")
    print(f"Top-5 overlap       = {mean_top5:.6f}")
    print(f"num_samples         = {n_compare}")
    print(f"mean_CE_full_kv     = {mean_full_ce:.6f}")
    print(f"mean_CE_ssm         = {mean_ssm_ce:.6f}")
    print(f"CE_gap (ssm-full)   = {mean_ssm_ce - mean_full_ce:+.6f}")
    print(f"mean_norm_sq_loss   = {mean_sq_loss:.6f}")
    print(f"per_sample_mean_MSE = {per_sample_mse:.6f}")
    if track_cuda_peak_mem:
        base_ssm_gb = model_mem_gb + sidecar_mem_gb
        full_peak_gb = full_kv_peak_mem_bytes / (1024 ** 3)
        ssm_demo_peak_gb = ssm_demo_peak_mem_bytes / (1024 ** 3)
        ssm_query_peak_gb = ssm_query_peak_mem_bytes / (1024 ** 3)
        print(f"model_weights_mem   = {model_mem_gb:.3f} GB")
        print(f"sidecar_weights_mem = {sidecar_mem_gb:.3f} GB")
        print(f"full_kv_peak_mem    = {full_peak_gb:.3f} GB "
              f"(+{full_peak_gb - model_mem_gb:.3f} GB over model)")
        print(f"ssm_demo_peak_mem   = {ssm_demo_peak_gb:.3f} GB "
              f"(+{ssm_demo_peak_gb - base_ssm_gb:.3f} GB over model+sidecar, one-time)")
        print(f"ssm_query_peak_mem  = {ssm_query_peak_gb:.3f} GB "
              f"(+{ssm_query_peak_gb - base_ssm_gb:.3f} GB over model+sidecar, per-query)")
    if args.flops:
        print(f"total_full_flops    = {total_full_flops:.3e} ({n_queries} queries)")
        print(f"total_ssm_flops     = {total_ssm_flops:.3e} ({n_queries} queries)")
        if total_ssm_flops > 0:
            print(f"flops_reduction     = {flops_reduction:.4f}x")


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
    retrieval_data = _load_dataset(args, args.retrieval_split)
    eval_data = _load_dataset(args, args.eval_split)
    if not retrieval_data or not eval_data:
        raise ValueError("Empty data.")

    fixed_demos = _choose_fixed_demos(retrieval_data, args.k, args.demo_strategy, args.seed)
    demo_text = _build_demo_text(fixed_demos, add_newlines=add_nl)
    demo_ids = tokenizer(demo_text, return_tensors="pt",
                         add_special_tokens=False)["input_ids"].to(device)
    full_len = demo_ids.shape[1]

    sidecar = LayerGroupSSMSidecar(
        model.config, ssm_dim=args.ssm_dim, num_virtual_tokens=args.num_virtual_tokens,
        num_groups=args.num_groups,
    ).to(device=device, dtype=model.get_input_embeddings().weight.dtype)
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
    for dp in tqdm(eval_data, desc="compare-loss"):
        q_text, ans_text = _normalize_text(dp, is_first=False, add_newlines=add_nl)
        q_ids = tokenizer(q_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
        a_ids = tokenizer(ans_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
        if q_ids.numel() == 0 or a_ids.numel() == 0:
            continue

        a_stripped, n_stripped = _strip_leading_format_tokens(a_ids, tokenizer)
        if a_stripped.numel() == 0:
            continue

        with torch.no_grad():
            t_logits = _teacher_qa_logits_nocache(model, demo_ids, q_ids, a_ids)
            t_logits_s = t_logits[:, n_stripped:, :]
            t_ce = F.cross_entropy(
                t_logits_s.reshape(-1, t_logits_s.size(-1)),
                a_stripped.reshape(-1), reduction="mean",
            ).item()

            s_logits = _student_qa_logits(
                model, eval_virtual_kv, eval_sink_kv,
                eval_ssm_demo_len, full_len, align, q_ids, a_ids,
            )
            s_logits_s = s_logits[:, n_stripped:, :]
            s_ce = F.cross_entropy(
                s_logits_s.reshape(-1, s_logits_s.size(-1)),
                a_stripped.reshape(-1), reduction="mean",
            ).item()

        full_losses.append(t_ce)
        ssm_losses.append(s_ce)

    if not full_losses:
        raise ValueError("No valid samples.")

    mf = sum(full_losses) / len(full_losses)
    ms = sum(ssm_losses) / len(ssm_losses)
    d = max(ms, mf)
    mse = ((ms - mf) / d) ** 2 if d > 0 else 0.0

    print(f"\n===== Loss Comparison (content tokens only) =====")
    print(f"samples       = {len(full_losses)}")
    print(f"mean_CE_full  = {mf:.6f}")
    print(f"mean_CE_ssm   = {ms:.6f}")
    print(f"CE_gap        = {ms - mf:+.6f}")
    print(f"norm_sq_err   = {mse:.6f}")


# =====================================================================
# Text / data utilities
# =====================================================================

def _setup_tokenizer(name):
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


def _normalize_text(dp, is_first, add_newlines):
    q = ("\n" + dp["input"]) if (add_newlines and not is_first) else dp["input"]
    a = ("\n" + dp["output"]) if add_newlines else dp["output"]
    return q, a


def _normalize_option(opt, add_newlines):
    return ("\n" + opt) if add_newlines else opt


def _build_demo_text(demos, add_newlines):
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
    h = fp["hidden"]
    l = fp["num_layers"]
    nq = fp["num_q_heads"]
    nkv = fp["num_kv_heads"]
    hd = fp["head_dim"]
    inter = fp["intermediate"]
    vocab = fp["vocab_size"]
    n = new_tokens

    if n == 0:
        return 0.0

    if ctx_tokens == 0:
        attn_pairs = n * (n + 1) / 2.0
    else:
        attn_pairs = n * ctx_tokens + n * (n + 1) / 2.0

    qkv_flops = 2.0 * n * h * (nq * hd + 2 * nkv * hd)
    o_flops = 2.0 * n * h * h
    attn_flops = 2.0 * 2.0 * nq * hd * attn_pairs
    if fp["is_gated_mlp"]:
        mlp_flops = 3.0 * 2.0 * n * h * inter
    else:
        mlp_flops = 2.0 * 2.0 * n * h * inter

    per_layer = qkv_flops + o_flops + attn_flops + mlp_flops
    total = l * per_layer
    total += 2.0 * n * h * vocab
    return total


def _analytical_flops_full(fp, seq_len):
    return _analytical_flops(fp, new_tokens=seq_len, ctx_tokens=0)


def _analytical_flops_inc(fp, new_tokens, ctx_tokens):
    return _analytical_flops(fp, new_tokens=new_tokens, ctx_tokens=ctx_tokens)


def _sidecar_analytical_flops(sidecar, seq_len):
    total = 0.0
    t = seq_len

    for group in sidecar.groups:
        d = group.ssm_dim
        hidden = sidecar.hidden_size
        c = group.scan_chunk_size

        total += 2.0 * t * hidden * d

        pos = 0
        while pos < t:
            chunk_len = min(c, t - pos)
            if chunk_len == 1:
                total += 2.0 * d * d + d
            else:
                total += max(0, chunk_len - 2) * 2.0 * d * d * d
                total += chunk_len * 2.0 * d * d
                total += 2.0 * d * d * d
                total += 2.0 * d * d
            pos += chunk_len

        inter = d * 4
        per_layer_kv = 2 * group.num_kv_heads * group.num_virtual_tokens * group.head_dim
        output_dim = group.num_layers_in_group * per_layer_kv
        total += 2.0 * d * inter
        total += 2.0 * inter * inter
        total += 2.0 * inter * output_dim

    return total


def _load_sidecar(sidecar, args, device, state_dict=None):
    if state_dict is not None:
        # Filter out frozen buffers (A, eigenvalues, V, V_inv) that may differ
        # between old checkpoints (A was nn.Parameter) and new code (A is buffer).
        filtered = {k: v for k, v in state_dict.items()
                    if not any(k.endswith(s) for s in ('.A', '.A_powers', '.eigenvalues', '.V', '.V_inv'))}
        sidecar.load_state_dict(filtered, strict=False)
        print("Loaded sidecar from in-memory state (trainable params only).")
    elif args.load_sidecar_path:
        ckpt = torch.load(args.load_sidecar_path, map_location=device)
        st = ckpt["sidecar"] if isinstance(ckpt, dict) and "sidecar" in ckpt else ckpt
        filtered = {k: v for k, v in st.items()
                    if not any(k.endswith(s) for s in ('.A', '.A_powers', '.eigenvalues', '.V', '.V_inv'))}
        sidecar.load_state_dict(filtered, strict=False)
        print(f"Loaded sidecar from: {args.load_sidecar_path}")
    else:
        print("WARN: No sidecar checkpoint; results may be random.")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retrieval_split", type=str, default="test")
    p.add_argument("--eval_split", type=str, default="dev")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--demo_strategy", type=str, default="first")
    p.add_argument("--num_eval_demo_sets", type=int, default=1,
                   help=">1 enables per-query random demo sampling at eval time.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--is_quant", default=False, action="store_true")
    p.add_argument("--flops", default=False, action="store_true",
                   help="Enable analytical FLOPs accounting and print FLOPs reduction.")
    p.add_argument("--run_mode", type=str, default="eval",
                   choices=["eval", "train", "train_eval", "compare_loss", "train_compare_loss"])

    p.add_argument("--num_virtual_tokens", type=int, default=16)
    p.add_argument("--sink_tokens", type=int, default=4)
    p.add_argument("--ssm_dim", type=int, default=512)
    p.add_argument("--num_groups", type=int, default=4)
    p.add_argument("--align_true_positions", action="store_true")

    p.add_argument("--num_demo_sets", type=int, default=500)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--kd_temperature", type=float, default=2.0)
    p.add_argument("--loss_w_kv", type=float, default=1.0)
    p.add_argument("--loss_w_logit", type=float, default=0.0)
    p.add_argument("--loss_w_hid", type=float, default=0.0,
                   help="Weight for hidden-state matching loss (cosine distance on last layer).")
    p.add_argument("--loss_w_ce", type=float, default=0.0,
                   help="Weight for supervised cross-entropy loss on answer tokens.")
    p.add_argument("--kv_loss_type", type=str, default="cosine", choices=["cosine", "mse"])
    p.add_argument("--log_every", type=int, default=20)

    p.add_argument("--save_sidecar_path", type=str, default="")
    p.add_argument("--load_sidecar_path", type=str, default="")
    p.add_argument("--empty_cache_every", type=int, default=0,
                   help="If >0, run gc.collect()+torch.cuda.empty_cache() every N train steps.")

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