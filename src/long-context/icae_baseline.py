"""
ICAE Baseline Implementation
=============================================================
Based on: "In-context Autoencoder for Context Compression in a
Large Language Model" (Ge et al., ICLR 2024)

Architecture:
  Encoder = LLM + LoRA(Q,V) + learnable memory-token embeddings
  Decoder = frozen LLM (same weights, LoRA disabled)

Training:
  Phase 1 – Pretrain with AE + LM objectives on raw text
  Phase 2 – Fine-tune on (context, prompt, response) triples
            where memory slots come from the SAME context that
            the prompt/response refer to  (Section 2.3 of paper)

Evaluation:
  Compatible with the existing SSM-sidecar eval framework.
=============================================================
"""

import argparse, copy, gc, math, os, random, re, time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPT2Tokenizer,
)

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


def _load_dataset(args, split, k=None):
    if args.dataset.strip().lower() == "gsm8k":
        return load_gsm8k(split=split, max_samples=0)
    if args.dataset.strip().lower() == "mmlu":
        return load_mmlu_college_split(args, split)
    use_k = args.k if k is None else k
    return load_data(
        task=None, split=split, k=use_k,
        seed=args.seed, datasets=args.dataset.split(","), is_null=False
    )


# =====================================================================
# LoRA
# =====================================================================

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank residual."""
    def __init__(self, orig: nn.Linear, rank: int = 64, alpha: float = 1.0):
        super().__init__()
        self.orig = orig
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        in_f, out_f = orig.in_features, orig.out_features
        param_device = orig.weight.device
        # Keep trainable LoRA weights in fp32 for optimizer stability.
        self.lora_A = nn.Parameter(torch.empty(in_f, rank, device=param_device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f, device=param_device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        orig.weight.requires_grad = False
        if orig.bias is not None:
            orig.bias.requires_grad = False
        self._enabled = True

    def forward(self, x):
        base = self.orig(x)
        if not self._enabled:
            return base
        # Compute LoRA update in fp32 for stability, then cast back.
        x_lora = x.to(self.lora_A.dtype)
        return base + (x_lora @ self.lora_A @ self.lora_B).to(base.dtype) * self.scale


# ── helpers to find Q / V projections across architectures ──

_QV_NAMES = {'q_proj', 'v_proj',       # LLaMA / Qwen / Mistral
             'query', 'value'}          # some older HF models

def _apply_lora(model, rank=64, alpha=1.0):
    """Inject LoRA into every Q and V linear layer. Returns list[LoRALinear]."""
    lora_layers: List[LoRALinear] = []
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        short = name.split('.')[-1]
        if short not in _QV_NAMES:
            continue
        parent = model
        parts = name.split('.')
        for p in parts[:-1]:
            parent = getattr(parent, p)
        lora = LoRALinear(mod, rank=rank, alpha=alpha)
        setattr(parent, parts[-1], lora)
        lora_layers.append(lora)
    # GPT-2 fallback (combined c_attn)
    if not lora_layers:
        for name, mod in list(model.named_modules()):
            if isinstance(mod, nn.Linear) and name.endswith('c_attn'):
                parent = model
                parts = name.split('.')
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                lora = LoRALinear(mod, rank=rank, alpha=alpha)
                setattr(parent, parts[-1], lora)
                lora_layers.append(lora)
    print(f"[LoRA] injected into {len(lora_layers)} layers, rank={rank}")
    return lora_layers


def _set_lora(layers, enabled: bool):
    for l in layers:
        l._enabled = enabled


def _lora_params(layers):
    ps = []
    for l in layers:
        ps += [l.lora_A, l.lora_B]
    return ps


# =====================================================================
# ICAE Model
# =====================================================================

class ICAE(nn.Module):
    """
    In-context Autoencoder.

    Encoder: LLM (with LoRA on Q, V) processes [context, mem_tokens]
             → last-layer hidden states at mem_token positions = memory slots
    Decoder: same LLM *without* LoRA, conditions on memory slots via
             inputs_embeds.
    """
    def __init__(self, model, num_memory_slots=128, lora_rank=64, lora_alpha=1.0):
        super().__init__()
        self.model = model
        self.H = model.config.hidden_size
        self.k = num_memory_slots
        emb_weight = model.get_input_embeddings().weight

        # Learnable memory-token embeddings  (em(m_i) in the paper)
        self.mem_emb = nn.Parameter(
            torch.randn(num_memory_slots, self.H, device=emb_weight.device, dtype=torch.float32) * 0.02
        )
        # Special [AE] token embedding
        self.ae_emb = nn.Parameter(
            torch.randn(1, self.H, device=emb_weight.device, dtype=torch.float32) * 0.02
        )

        self.lora_layers = _apply_lora(model, rank=lora_rank, alpha=lora_alpha)

    # ── encoder ──────────────────────────────────────────────
    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[B, L] → memory_slots [B, k, H]"""
        B = input_ids.shape[0]
        tok_emb = self.model.get_input_embeddings()(input_ids)          # [B,L,H]
        mem = self.mem_emb.unsqueeze(0).expand(B, -1, -1)               # [B,k,H]
        inp = torch.cat([tok_emb, mem.to(tok_emb.dtype)], dim=1)        # [B,L+k,H]

        _set_lora(self.lora_layers, True)
        out = self.model(inputs_embeds=inp, use_cache=False,
                         output_hidden_states=True)
        _set_lora(self.lora_layers, False)

        return out.hidden_states[-1][:, -self.k:, :]                    # [B,k,H]

    # ── decoder helpers ──────────────────────────────────────
    def _decode_fwd(self, inputs_embeds):
        """Run frozen decoder (LoRA off) and return logits."""
        _set_lora(self.lora_layers, False)
        return self.model(inputs_embeds=inputs_embeds, use_cache=False).logits

    # ── AE objective ─────────────────────────────────────────
    def decode_ae(self, mem_slots: torch.Tensor, target_ids: torch.Tensor):
        """
        Reconstruct original context from memory slots.
        Decoder input:  [m̃_1 … m̃_k, [AE], e(w_1) … e(w_{L-1})]
        Target:         w_1, w_2, …, w_L
        Returns logits [B, L, V].
        """
        B, L = target_ids.shape
        ae = self.ae_emb.unsqueeze(0).expand(B, -1, -1).to(mem_slots.dtype)
        tgt_emb = self.model.get_input_embeddings()(target_ids)

        dec_in = torch.cat([mem_slots, ae, tgt_emb[:, :-1]], dim=1)     # [B, k+L, H]
        logits = self._decode_fwd(dec_in)
        return logits[:, self.k: self.k + L, :]                         # [B, L, V]

    # ── LM (continuation) objective ──────────────────────────
    def decode_lm(self, mem_slots: torch.Tensor, cont_ids: torch.Tensor):
        """
        Predict continuation tokens given memory slots.
        Decoder input:  [m̃_1 … m̃_k, e(w_{L+1}) … e(w_{L+N-1})]
        Target:         w_{L+1}, …, w_{L+N}
        Returns logits [B, N, V].
        """
        B, N = cont_ids.shape
        cont_emb = self.model.get_input_embeddings()(cont_ids)
        dec_in = torch.cat([mem_slots, cont_emb[:, :-1]], dim=1)        # [B, k+N-1, H]
        logits = self._decode_fwd(dec_in)
        return logits[:, self.k - 1: self.k - 1 + N, :]                # [B, N, V]

    # ── Instruction (prompt → response) ──────────────────────
    def decode_instruction(self, mem_slots, prompt_ids, response_ids):
        """
        Decoder input:  [m̃_1…m̃_k, p_1…p_M, r_1…r_{N-1}]
        Target:         r_1, …, r_N
        Returns logits [B, N, V].
        """
        B = prompt_ids.shape[0]
        M, N = prompt_ids.shape[1], response_ids.shape[1]
        p_emb = self.model.get_input_embeddings()(prompt_ids)
        r_emb = self.model.get_input_embeddings()(response_ids)
        dec_in = torch.cat([mem_slots, p_emb, r_emb[:, :-1]], dim=1)
        logits = self._decode_fwd(dec_in)
        start = self.k + M - 1
        return logits[:, start: start + N, :]

    # ── Trainable parameters ─────────────────────────────────
    def trainable_params(self):
        return [self.mem_emb, self.ae_emb] + _lora_params(self.lora_layers)

    def num_trainable(self):
        return sum(p.numel() for p in self.trainable_params())


# =====================================================================
# ICAE Scorer (for ICL evaluation, mirrors SSMHybridICLScorer)
# =====================================================================

class ICAEScorer:
    """Score ICL options by NLL using ICAE-compressed demo context."""

    def __init__(self, model, icae: ICAE, tokenizer, device):
        self.model = model
        self.icae = icae
        self.tok = tokenizer
        self.dev = device
        self.dtype = model.get_input_embeddings().weight.dtype
        self.mem_slots = None       # [1, k, H]
        self.demo_len_orig = 0      # original demo token count

    @torch.no_grad()
    def set_demo(self, demo_text: str):
        ids = self.tok(demo_text, return_tensors="pt",
                       add_special_tokens=False)["input_ids"].to(self.dev)
        self.demo_len_orig = ids.shape[1]
        self.mem_slots = self.icae.encode(ids).detach()

    @torch.no_grad()
    def score_options(self, question_text: str, options: List[str]) -> Dict[str, float]:
        """Return {option_text: NLL} for each option."""
        q_ids = self.tok(question_text, return_tensors="pt",
                         add_special_tokens=False)["input_ids"].to(self.dev)
        results = {}
        for opt in options:
            o_ids = self.tok(opt, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(self.dev)
            if o_ids.numel() == 0:
                results[opt] = float("inf")
                continue
            # build decoder input: [mem_slots, q_emb, o_emb]
            q_emb = self.model.get_input_embeddings()(q_ids)
            o_emb = self.model.get_input_embeddings()(o_ids)
            dec_in = torch.cat([
                self.mem_slots,
                q_emb,
                o_emb[:, :-1],
            ], dim=1)

            _set_lora(self.icae.lora_layers, False)
            logits = self.model(inputs_embeds=dec_in, use_cache=False).logits

            k = self.icae.k
            M = q_ids.shape[1]
            N = o_ids.shape[1]
            start = k + M - 1
            pred = logits[:, start: start + N, :]
            nll = F.cross_entropy(pred.reshape(-1, pred.size(-1)),
                                  o_ids.reshape(-1), reduction="sum").item()
            results[opt] = nll
        return results

    @torch.no_grad()
    def qa_logits(self, q_ids, a_ids):
        """Get logits on answer tokens (for CE / fidelity metrics)."""
        q_emb = self.model.get_input_embeddings()(q_ids)
        a_emb = self.model.get_input_embeddings()(a_ids)
        dec_in = torch.cat([self.mem_slots, q_emb, a_emb[:, :-1]], dim=1)
        _set_lora(self.icae.lora_layers, False)
        logits = self.model(inputs_embeds=dec_in, use_cache=False).logits
        start = self.icae.k + q_ids.shape[1] - 1
        return logits[:, start: start + a_ids.shape[1], :]


# =====================================================================
# Text / Data utilities  (mirrors the sidecar codebase)
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


def _default_ckpt_path(dataset, stage, preferred_path=""):
    """
    Checkpoint path policy:
      1) Always save under ./checkpoint/icae/
      2) Filename must include task name
    """
    task = re.sub(r"[^A-Za-z0-9._-]+", "_", str(dataset)).strip("._-")
    if not task:
        task = "unknown_task"
    out_dir = os.path.join(".", "checkpoint", "icae")
    os.makedirs(out_dir, exist_ok=True)
    if preferred_path:
        base = os.path.basename(preferred_path)
        stem, ext = os.path.splitext(base)
        if not ext:
            ext = ".pt"
        if task not in stem:
            stem = f"{task}_{stem}" if stem else f"{task}_icae_{stage}"
        return os.path.join(out_dir, f"{stem}{ext}")
    return os.path.join(out_dir, f"{task}_icae_{stage}.pt")


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


def _strip_leading_format_tokens(a_ids, tokenizer):
    strip_toks = set()
    for ch in ["\n", " ", "\t"]:
        strip_toks.update(tokenizer(ch, add_special_tokens=False)["input_ids"])
    idx = 0
    while idx < a_ids.shape[1] and a_ids[0, idx].item() in strip_toks:
        idx += 1
    if idx >= a_ids.shape[1]:
        return a_ids, 0
    return a_ids[:, idx:], idx


def _freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def _load_causal_lm(args, device):
    if device.startswith("cuda"):
        preferred_dtype = torch.float32
        common_kwargs = {"device_map": "auto", "torch_dtype": preferred_dtype}
        if args.is_quant:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=preferred_dtype,
                                     bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
            return AutoModelForCausalLM.from_pretrained(
                args.model_name, quantization_config=bnb, **common_kwargs
            )
        return AutoModelForCausalLM.from_pretrained(args.model_name, **common_kwargs)

    if args.is_quant:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        return AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=bnb,
                                                    device_map="auto", torch_dtype=torch.float16)
    return AutoModelForCausalLM.from_pretrained(args.model_name).to(device)


def _resolve_runtime_device(device_arg: str) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if device_arg == "auto":
        return "cuda"
    if device_arg.startswith("cuda"):
        return device_arg
    return "cpu"


def _model_input_device(model):
    return model.get_input_embeddings().weight.device


@torch.no_grad()
def _teacher_qa_logits_nocache(model, demo_ids, q_ids, ans_ids):
    all_ids = torch.cat([demo_ids, q_ids, ans_ids], dim=1)
    out = model(input_ids=all_ids, use_cache=False)
    start = demo_ids.shape[1] + q_ids.shape[1] - 1
    return out.logits[:, start:start + ans_ids.shape[1], :]


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
    L = fp["num_layers"]
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
    total = L * per_layer
    total += 2.0 * n * h * vocab
    return total


def _analytical_flops_full(fp, seq_len):
    return _analytical_flops(fp, new_tokens=seq_len, ctx_tokens=0)


def _analytical_flops_inc(fp, new_tokens, ctx_tokens):
    return _analytical_flops(fp, new_tokens=new_tokens, ctx_tokens=ctx_tokens)


# =====================================================================
# Build (context, prompt, response) triples for fine-tuning
# =====================================================================

def _build_ft_triples(retrieval_data, train_data, k, num_triples, seed, add_newlines):
    """
    Build coherent (context_text, prompt_text, response_text) triples
    matching the ICAE paper Section 2.3:
      - context  = k-shot demo text  (the text to be compressed)
      - prompt   = a question from the SAME task distribution
      - response = the gold answer for that question

    Each triple pairs a *specific* demo context with a *specific* QA,
    so memory slots are always the compression of the context that the
    prompt/response depend on.
    """
    rng = random.Random(seed)
    n_ret = len(retrieval_data)
    n_train = len(train_data)
    triples = []

    for _ in range(num_triples):
        # Sample k demos as context
        demo_indices = rng.sample(range(n_ret), min(k, n_ret))
        demos = [retrieval_data[i] for i in demo_indices]
        ctx_text = _build_demo_text(demos, add_newlines=add_newlines)

        # Sample 1 QA pair as the prompt/response for this context
        qi = rng.randint(0, n_train - 1)
        dp = train_data[qi]
        q_text, a_text = _normalize_text(dp, is_first=False, add_newlines=add_newlines)

        triples.append((ctx_text, q_text, a_text))

    return triples


# =====================================================================
# Phase 1: Pretraining  (AE + LM)
# =====================================================================

def run_pretrain(args):
    device = _resolve_runtime_device(args.device)
    tokenizer = _setup_tokenizer(args.model_name)
    model = _load_causal_lm(args, device)
    io_device = _model_input_device(model)
    if hasattr(model, "hf_device_map"):
        print(f"[Model] hf_device_map={model.hf_device_map}")
    _freeze(model)

    icae = ICAE(model, num_memory_slots=args.num_memory_slots,
                lora_rank=args.lora_rank)
    print(f"[ICAE] trainable params: {icae.num_trainable():,}")

    # ── data: use raw text from retrieval data + train data for self-supervised pretrain ──
    add_nl = not args.model_name.startswith("gpt2")
    raw_data = _load_dataset(args, args.retrieval_split, k=999999)
    raw_data += _load_dataset(args, args.train_split, k=999999)
    # Build raw text snippets for AE/LM pretraining
    texts = []
    for dp in raw_data:
        q, a = _normalize_text(dp, is_first=True, add_newlines=add_nl)
        texts.append(q + a)
    print(f"[Pretrain] {len(texts)} text snippets")

    optimizer = torch.optim.AdamW(icae.trainable_params(), lr=args.pretrain_lr,
                                  weight_decay=args.weight_decay)
    max_len = args.pretrain_max_len  # default 512
    lam = args.pretrain_lambda       # AE weight (paper: 0.4-0.6)

    rng = random.Random(args.seed)
    step = 0
    running = {"loss": 0.0, "ae": 0.0, "lm": 0.0}

    for epoch in range(args.pretrain_epochs):
        order = list(range(len(texts)))
        rng.shuffle(order)
        for ti in tqdm(order, desc=f"pretrain-{epoch+1}"):
            ids = tokenizer(texts[ti], return_tensors="pt", add_special_tokens=False,
                            max_length=max_len * 2, truncation=True)["input_ids"].to(io_device)
            L = min(ids.shape[1], max_len)
            if L < 10:
                continue

            ctx_ids = ids[:, :L]
            mem_slots = icae.encode(ctx_ids)  # [1, k, H]

            # ── AE loss ──
            ae_logits = icae.decode_ae(mem_slots, ctx_ids)
            loss_ae = F.cross_entropy(ae_logits.float().reshape(-1, ae_logits.size(-1)),
                                      ctx_ids.reshape(-1), reduction="mean")

            # ── LM loss (if continuation available) ──
            loss_lm = torch.tensor(0.0, device=io_device)
            cont_len = ids.shape[1] - L
            if cont_len >= 5:
                N = min(cont_len, max_len)
                cont_ids = ids[:, L: L + N]
                lm_logits = icae.decode_lm(mem_slots, cont_ids)
                loss_lm = F.cross_entropy(lm_logits.float().reshape(-1, lm_logits.size(-1)),
                                          cont_ids.reshape(-1), reduction="mean")

            loss = lam * loss_ae + (1 - lam) * loss_lm

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(icae.trainable_params(), max_norm=args.grad_clip)
            optimizer.step()
            step += 1
            running["loss"] += loss.item()
            running["ae"] += loss_ae.item()
            running["lm"] += (loss_lm.item() if isinstance(loss_lm, torch.Tensor) else loss_lm)

            if step % args.log_every == 0:
                d = float(args.log_every)
                print(f"[pretrain] step={step}  loss={running['loss']/d:.4f}  "
                      f"ae={running['ae']/d:.4f}  lm={running['lm']/d:.4f}")
                running = {"loss": 0.0, "ae": 0.0, "lm": 0.0}

            del mem_slots, loss, loss_ae, loss_lm
            if step % max(1, args.empty_cache_every) == 0 and device.startswith("cuda"):
                gc.collect(); torch.cuda.empty_cache()
            if 0 < args.max_steps <= step:
                break
        if 0 < args.max_steps <= step:
            break

    state = {k: v.detach().cpu() for k, v in icae.state_dict().items()
             if any(k.startswith(p) for p in ("mem_emb", "ae_emb", "lora_layers"))
             or "lora_" in k}
    save_pretrain_path = _default_ckpt_path(
        args.dataset, "pretrain", preferred_path=args.save_pretrain_path
    )
    torch.save({"icae": state, "args": vars(args)}, save_pretrain_path)
    print(f"Saved pretrained ICAE: {save_pretrain_path}")
    return icae, state


# =====================================================================
# Phase 2: Instruction Fine-tuning
#
# Paper Section 2.3:  Each sample is a coherent (context, prompt, response)
# triple.  The ICAE encodes *this sample's context* into memory slots,
# then the decoder produces the response given (memory_slots, prompt).
# =====================================================================

def run_finetune(args, icae=None, model=None, tokenizer=None):
    device = _resolve_runtime_device(args.device)
    if tokenizer is None:
        tokenizer = _setup_tokenizer(args.model_name)
    if model is None:
        model = _load_causal_lm(args, device)
        _freeze(model)
    io_device = _model_input_device(model)
    if icae is None:
        icae = ICAE(model, num_memory_slots=args.num_memory_slots,
                    lora_rank=args.lora_rank)
        if args.load_pretrain_path:
            ckpt = torch.load(args.load_pretrain_path, map_location="cpu")
            icae.load_state_dict(ckpt["icae"], strict=False)
            print(f"Loaded pretrained ICAE from: {args.load_pretrain_path}")

    add_nl = not args.model_name.startswith("gpt2")

    retrieval_data = _load_dataset(args, args.retrieval_split, k=args.k)
    train_data = _load_dataset(args, args.train_split, k=args.k)

    # ── Build coherent (context, prompt, response) triples ──
    # Each triple: context = k-shot demos, prompt = question, response = answer.
    # The memory slots are the compression of *this specific context*.
    triples = _build_ft_triples(
        retrieval_data, train_data, args.k,
        num_triples=args.num_ft_triples, seed=args.seed,
        add_newlines=add_nl,
    )
    print(f"[FT] Built {len(triples)} coherent (context, prompt, response) triples")

    optimizer = torch.optim.AdamW(icae.trainable_params(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    total_steps = args.epochs * len(triples)
    warmup = min(100, total_steps // 10)

    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    rng = random.Random(args.seed)
    step = 0
    running = {"loss": 0.0}

    for epoch in range(args.epochs):
        order = list(range(len(triples)))
        rng.shuffle(order)
        for ti in tqdm(order, desc=f"ft-{epoch+1}"):
            ctx_text, q_text, a_text = triples[ti]

            ctx_ids = tokenizer(ctx_text, return_tensors="pt",
                                add_special_tokens=False)["input_ids"].to(io_device)
            q_ids = tokenizer(q_text, return_tensors="pt",
                              add_special_tokens=False)["input_ids"].to(io_device)
            a_ids = tokenizer(a_text, return_tensors="pt",
                              add_special_tokens=False)["input_ids"].to(io_device)

            if ctx_ids.numel() == 0 or q_ids.numel() == 0 or a_ids.numel() == 0:
                continue

            # Encode THIS sample's context into memory slots
            mem_slots = icae.encode(ctx_ids)  # [1, k, H]

            # Decode: predict response given (memory_slots, prompt)
            logits = icae.decode_instruction(mem_slots, q_ids, a_ids)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                   a_ids.reshape(-1), reduction="mean")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(icae.trainable_params(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            running["loss"] += loss.item()

            if step % args.log_every == 0:
                print(f"[ft] step={step}  loss={running['loss']/args.log_every:.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.6f}")
                running = {"loss": 0.0}

            del mem_slots, loss, ctx_ids, q_ids, a_ids
            if step % max(1, args.empty_cache_every) == 0 and device.startswith("cuda"):
                gc.collect(); torch.cuda.empty_cache()
            if 0 < args.max_steps <= step:
                break
        if 0 < args.max_steps <= step:
            break

    state = {k: v.detach().cpu() for k, v in icae.state_dict().items()
             if "lora_" in k or k in ("mem_emb", "ae_emb")}
    save_sidecar_path = _default_ckpt_path(
        args.dataset, "finetune", preferred_path=args.save_sidecar_path
    )
    torch.save({"icae": state, "args": vars(args)}, save_sidecar_path)
    print(f"Saved fine-tuned ICAE: {save_sidecar_path}")
    return icae, state


# =====================================================================
# Evaluation  (compatible with sidecar eval framework)
# =====================================================================

def run_experiment(args, icae=None, model=None, tokenizer=None):
    device = _resolve_runtime_device(args.device)
    track_cuda_peak_mem = device.startswith("cuda")
    if tokenizer is None:
        tokenizer = _setup_tokenizer(args.model_name)
    if model is None:
        model = _load_causal_lm(args, device)
    io_device = _model_input_device(model)
    if hasattr(model, "hf_device_map"):
        print(f"[Model] hf_device_map={model.hf_device_map}")
    model.eval()
    model_mem_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    model_mem_gb = model_mem_bytes / (1024 ** 3)
    flop_params = _get_model_flop_params(model)

    add_nl = not args.model_name.startswith("gpt2")
    retrieval_data = _load_dataset(args, args.retrieval_split, k=args.k)
    eval_data = _load_dataset(args, args.eval_split, k=args.k)

    if icae is None:
        icae = ICAE(model, num_memory_slots=args.num_memory_slots,
                    lora_rank=args.lora_rank)
        if args.load_sidecar_path:
            ckpt = torch.load(args.load_sidecar_path, map_location="cpu")
            st = ckpt["icae"] if isinstance(ckpt, dict) and "icae" in ckpt else ckpt
            icae.load_state_dict(st, strict=False)
            print(f"Loaded ICAE from: {args.load_sidecar_path}")
        else:
            print("WARN: no ICAE checkpoint; results may be random.")
    icae.eval()

    scorer = ICAEScorer(model, icae, tokenizer, io_device)
    per_query_random = (args.num_eval_demo_sets > 1)
    rng_eval = random.Random(args.seed + 7777)

    full_kv_preds, icae_preds = [], []
    full_losses, icae_losses = [], []
    query_logit_mses = []
    logit_kls, logit_cossims, top1_agrees, top5_overlaps = [], [], [], []
    ds_results = {}
    n_queries = max(1, len(eval_data))

    # Analytical FLOPs accumulators
    full_kv_total_flops = 0.0
    icae_demo_flops = 0.0
    icae_query_total_flops = 0.0

    # Peak memory
    full_kv_peak_mem_bytes = 0
    icae_demo_peak_mem_bytes = 0
    icae_query_peak_mem_bytes = 0

    for qi, dp in enumerate(tqdm(eval_data, desc="eval")):
        q_text, ans_text = _normalize_text(dp, is_first=False, add_newlines=add_nl)
        q_ids = tokenizer(q_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(io_device)
        a_ids = tokenizer(ans_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(io_device)
        q_len = q_ids.shape[1]
        opts = dp["options"]
        q_full_kv_flops = 0.0
        q_icae_flops = 0.0

        # Pick demos
        if per_query_random:
            indices = rng_eval.sample(range(len(retrieval_data)),
                                      min(args.k, len(retrieval_data)))
            demos = [retrieval_data[i] for i in indices]
        else:
            if not hasattr(run_experiment, '_fixed'):
                run_experiment._fixed = _choose_fixed_demos(
                    retrieval_data, args.k, args.demo_strategy, args.seed)
            demos = run_experiment._fixed

        demo_text = _build_demo_text(demos, add_newlines=add_nl)
        demo_ids = tokenizer(demo_text, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(io_device)
        demo_len = demo_ids.shape[1]

        # ── Full-KV baseline ──
        if track_cuda_peak_mem:
            torch.cuda.reset_peak_memory_stats()
        opt_scores = {}
        for opt in opts:
            opt_text = _normalize_option(opt, add_newlines=add_nl)
            opt_ids = tokenizer(opt_text, return_tensors="pt",
                                add_special_tokens=False)["input_ids"].to(io_device)
            if opt_ids.numel() == 0:
                opt_scores[opt] = float("inf")
                continue
            with torch.no_grad():
                all_ids = torch.cat([demo_ids, q_ids, opt_ids], dim=1)
                out = model(input_ids=all_ids, use_cache=False)
                s = demo_ids.shape[1] + q_ids.shape[1] - 1
                pred = out.logits[:, s: s + opt_ids.shape[1], :]
                nll = F.cross_entropy(pred.reshape(-1, pred.size(-1)),
                                      opt_ids.reshape(-1), reduction="sum").item()
            opt_scores[opt] = nll
            total_len = demo_len + q_len + opt_ids.shape[1]
            q_full_kv_flops += _analytical_flops_full(flop_params, total_len)
        full_kv_preds.append(min(opt_scores, key=opt_scores.get))
        if track_cuda_peak_mem:
            full_kv_peak_mem_bytes = max(full_kv_peak_mem_bytes, torch.cuda.max_memory_allocated())

        # Full-KV CE on gold answer
        t_logits_s = None
        first_token_id = None
        if q_ids.numel() > 0 and a_ids.numel() > 0:
            with torch.no_grad():
                t_logits = _teacher_qa_logits_nocache(model, demo_ids, q_ids, a_ids)
                a_stripped, n_s = _strip_leading_format_tokens(a_ids, tokenizer)
                if a_stripped.numel() > 0:
                    t_logits_s = t_logits[:, n_s:, :]
                    first_token_id = a_stripped[:, 0].unsqueeze(1)
                    full_losses.append(F.cross_entropy(
                        t_logits_s.reshape(-1, t_logits_s.size(-1)),
                        a_stripped.reshape(-1), reduction="mean").item())

        # ── ICAE ──
        if track_cuda_peak_mem and (per_query_random or qi == 0):
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            scorer.set_demo(demo_text)
        if track_cuda_peak_mem and (per_query_random or qi == 0):
            icae_demo_peak_mem_bytes = max(icae_demo_peak_mem_bytes, torch.cuda.max_memory_allocated())

        # ICAE demo compression pass: encode on [demo + mem_tokens].
        if per_query_random or qi == 0:
            icae_demo_flops += _analytical_flops_full(flop_params, demo_len + args.num_memory_slots)

        if track_cuda_peak_mem:
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            opt_texts = [_normalize_option(o, add_newlines=add_nl) for o in opts]
            scores_raw = scorer.score_options(q_text, opt_texts)
            scores = {o: scores_raw[ot] for o, ot in zip(opts, opt_texts)}
        for opt_text in opt_texts:
            opt_len = len(tokenizer(opt_text, add_special_tokens=False)["input_ids"])
            q_icae_flops += _analytical_flops_full(
                flop_params, args.num_memory_slots + q_len + opt_len
            )
        icae_preds.append(min(scores, key=scores.get))
        if track_cuda_peak_mem:
            icae_query_peak_mem_bytes = max(icae_query_peak_mem_bytes, torch.cuda.max_memory_allocated())

        # ICAE CE on gold answer
        s_logits_s = None
        if q_ids.numel() > 0 and a_ids.numel() > 0:
            with torch.no_grad():
                s_logits = scorer.qa_logits(q_ids, a_ids)
                a_stripped, n_s = _strip_leading_format_tokens(a_ids, tokenizer)
                if a_stripped.numel() > 0:
                    s_logits_s = s_logits[:, n_s:, :]
                    icae_losses.append(F.cross_entropy(
                        s_logits_s.reshape(-1, s_logits_s.size(-1)),
                        a_stripped.reshape(-1), reduction="mean").item())

                    # Logit fidelity
                    if t_logits_s is not None:
                        t_full = _teacher_qa_logits_nocache(model, demo_ids, q_ids, a_ids)
                        tl = t_full[:, n_s:, :]
                        sl = s_logits_s
                        mn = min(tl.size(1), sl.size(1))
                        if mn > 0:
                            tl, sl = tl[:, :mn], sl[:, :mn]
                            logit_kls.append(F.kl_div(
                                F.log_softmax(sl, -1), F.softmax(tl, -1),
                                reduction="batchmean").item())
                            logit_cossims.append(F.cosine_similarity(
                                sl.reshape(-1, sl.size(-1)),
                                tl.reshape(-1, tl.size(-1)), dim=-1).mean().item())
                            top1_agrees.append(
                                (sl.argmax(-1) == tl.argmax(-1)).float().mean().item())
                            t5t = tl.topk(5, -1).indices
                            t5s = sl.topk(5, -1).indices
                            ovlp = sum(len(set(t5t[0, t].tolist()) & set(t5s[0, t].tolist())) / 5.0
                                       for t in range(mn))
                            top5_overlaps.append(ovlp / mn)

        # Per-query logit MSE (aligned with streaming_llm_baseline.py)
        if t_logits_s is not None and s_logits_s is not None and first_token_id is not None:
            t_first = t_logits_s[:, 0, :].gather(1, first_token_id).squeeze(1)
            s_first = s_logits_s[:, 0, :].gather(1, first_token_id).squeeze(1)
            t_first_val = float(t_first.item())
            s_first_val = float(s_first.item())
            denom = max(abs(s_first_val), abs(t_first_val))
            rel_sq = ((s_first_val - t_first_val) / denom) ** 2 if denom != 0 else 0.0
            query_logit_mses.append(rel_sq)

        ds = dp.get("task", dp.get("dataset", "unknown"))
        if ds not in ds_results:
            ds_results[ds] = {"full_p": [], "icae_p": [], "gt": []}
        ds_results[ds]["full_p"].append(full_kv_preds[-1])
        ds_results[ds]["icae_p"].append(icae_preds[-1])
        ds_results[ds]["gt"].append(dp["output"])
        full_kv_total_flops += q_full_kv_flops
        icae_query_total_flops += q_icae_flops

    # ── Aggregate ──
    gts = [dp["output"] for dp in eval_data]
    full_acc = _accuracy(full_kv_preds, gts)
    icae_acc = _accuracy(icae_preds, gts)

    mean_f = sum(full_losses) / max(1, len(full_losses))
    mean_i = sum(icae_losses) / max(1, len(icae_losses))
    per_sample_mse = sum(query_logit_mses) / max(1, len(query_logit_mses))
    mean_kl = sum(logit_kls) / max(1, len(logit_kls)) if logit_kls else float("nan")
    mean_cos = sum(logit_cossims) / max(1, len(logit_cossims)) if logit_cossims else float("nan")
    mean_t1 = sum(top1_agrees) / max(1, len(top1_agrees)) if top1_agrees else float("nan")
    mean_t5 = sum(top5_overlaps) / max(1, len(top5_overlaps)) if top5_overlaps else float("nan")
    total_full_flops = full_kv_total_flops
    total_icae_flops = icae_demo_flops + icae_query_total_flops
    flops_reduction = (total_full_flops / total_icae_flops) if total_icae_flops > 0 else 0.0

    result = {
        # Keep schema aligned with streaming_llm_baseline.py for downstream scripts.
        "full_kv_acc": full_acc,
        "icae_acc": icae_acc,
        "stream_acc": icae_acc,
        "acc_gap": full_acc - icae_acc,
        "full_ce": mean_f,
        "icae_ce": mean_i,
        "stream_ce": mean_i,
        "ce_gap": mean_i - mean_f,
        "per_sample_mean_MSE": per_sample_mse,
        "avg_full_flops": total_full_flops / n_queries,
        "avg_icae_flops": total_icae_flops / n_queries,
        "total_full_flops": total_full_flops,
        "total_icae_flops": total_icae_flops,
        "icae_demo_flops": icae_demo_flops,
        "flops_reduction": flops_reduction,
        "full_kv_peak_mem_bytes": full_kv_peak_mem_bytes,
        "full_kv_peak_mem_gb": full_kv_peak_mem_bytes / (1024 ** 3),
        "icae_demo_peak_mem_bytes": icae_demo_peak_mem_bytes,
        "icae_demo_peak_mem_gb": icae_demo_peak_mem_bytes / (1024 ** 3),
        "icae_query_peak_mem_bytes": icae_query_peak_mem_bytes,
        "icae_query_peak_mem_gb": icae_query_peak_mem_bytes / (1024 ** 3),
        "model_mem_bytes": model_mem_bytes,
        "model_mem_gb": model_mem_gb,
        "ds_results": ds_results,
        # ICAE-specific fidelity metrics are retained as extras.
        "logit_kl_icae_vs_full": mean_kl,
        "logit_cosine_similarity": mean_cos,
        "top1_agreement": mean_t1,
        "top5_overlap": mean_t5,
    }

    print(f"\n{'='*60}")
    print(f"  ICAE Baseline Eval  (k_mem={args.num_memory_slots}, "
          f"lora_r={args.lora_rank}, k_demo={args.k})")
    print(f"{'='*60}")
    print(f"Full-KV Accuracy  = {full_acc:.6f}")
    print(f"ICAE Accuracy     = {icae_acc:.6f}")
    print(f"Accuracy gap      = {full_acc - icae_acc:+.6f}")

    print(f"\n{'Dataset':<20} {'N':>5} {'FullKV':>10} {'ICAE':>10} {'Gap':>10}")
    print(f"{'-'*55}")
    for ds in sorted(ds_results):
        r = ds_results[ds]
        n = len(r["gt"])
        fa = _accuracy(r["full_p"], r["gt"])
        ia = _accuracy(r["icae_p"], r["gt"])
        print(f"{ds:<20} {n:>5} {fa:>10.4f} {ia:>10.4f} {fa-ia:>+10.4f}")

    print(f"\n{'='*60}")
    print(f"  Loss / Fidelity Metrics")
    print(f"{'='*60}")
    print(f"mean_CE_full_kv   = {mean_f:.6f}")
    print(f"mean_CE_icae      = {mean_i:.6f}")
    print(f"CE_gap            = {mean_i - mean_f:+.6f}")
    print(f"per_sample_mean_MSE   = {per_sample_mse:.6f}")
    print(f"KL(ICAE||FullKV)  = {mean_kl:.6f}")
    print(f"Cosine sim        = {mean_cos:.6f}")
    print(f"Top-1 agreement   = {mean_t1:.6f}")
    print(f"Top-5 overlap     = {mean_t5:.6f}")
    if track_cuda_peak_mem:
        print(f"model_weights_mem = {model_mem_gb:.3f} GB")
        print(f"full_kv_peak_mem  = {full_kv_peak_mem_bytes / (1024 ** 3):.3f} GB "
              f"(+{full_kv_peak_mem_bytes / (1024 ** 3) - model_mem_gb:.3f} GB over model)")
        print(f"icae_demo_peak_mem= {icae_demo_peak_mem_bytes / (1024 ** 3):.3f} GB "
              f"(+{icae_demo_peak_mem_bytes / (1024 ** 3) - model_mem_gb:.3f} GB over model)")
        print(f"icae_query_peak_mem= {icae_query_peak_mem_bytes / (1024 ** 3):.3f} GB "
              f"(+{icae_query_peak_mem_bytes / (1024 ** 3) - model_mem_gb:.3f} GB over model)")
    print(f"total_full_flops  = {total_full_flops:.3e} ({n_queries} queries)")
    print(f"total_icae_flops  = {total_icae_flops:.3e} ({n_queries} queries)")
    if total_icae_flops > 0:
        print(f"flops_reduction   = {flops_reduction:.4f}x")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ICAE Baseline")

    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retrieval_split", type=str, default="test")
    p.add_argument("--eval_split", type=str, default="dev")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--demo_strategy", type=str, default="first")
    p.add_argument("--num_eval_demo_sets", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--is_quant", default=False, action="store_true")

    p.add_argument("--run_mode", type=str, default="eval",
                   choices=["eval", "pretrain", "finetune", "pretrain_finetune",
                            "pretrain_finetune_eval", "finetune_eval"])

    # ICAE architecture
    p.add_argument("--num_memory_slots", type=int, default=128,
                   help="k in the paper. 128 → 4x compression for 512-token context.")
    p.add_argument("--lora_rank", type=int, default=64)

    # Pretraining
    p.add_argument("--pretrain_epochs", type=int, default=1)
    p.add_argument("--pretrain_lr", type=float, default=1e-4)
    p.add_argument("--pretrain_max_len", type=int, default=512)
    p.add_argument("--pretrain_lambda", type=float, default=0.5,
                   help="AE weight in λ*L_AE + (1-λ)*L_LM. Paper recommends 0.4-0.6.")

    # Fine-tuning
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--num_ft_triples", type=int, default=2000,
                   help="Number of (context, prompt, response) triples to generate.")
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--empty_cache_every", type=int, default=50)

    # Checkpoint paths
    p.add_argument("--save_pretrain_path", type=str, default="")
    p.add_argument("--load_pretrain_path", type=str, default="")
    p.add_argument("--save_sidecar_path", type=str, default="")
    p.add_argument("--load_sidecar_path", type=str, default="")

    args = p.parse_args()

    if args.run_mode == "pretrain":
        run_pretrain(args)

    elif args.run_mode == "finetune":
        run_finetune(args)

    elif args.run_mode == "pretrain_finetune":
        icae, _ = run_pretrain(args)
        run_finetune(args, icae=icae,
                     model=icae.model,
                     tokenizer=_setup_tokenizer(args.model_name))

    elif args.run_mode == "pretrain_finetune_eval":
        icae, _ = run_pretrain(args)
        tok = _setup_tokenizer(args.model_name)
        icae, _ = run_finetune(args, icae=icae, model=icae.model, tokenizer=tok)
        run_experiment(args, icae=icae, model=icae.model, tokenizer=tok)

    elif args.run_mode == "finetune_eval":
        tok = _setup_tokenizer(args.model_name)
        model = _load_causal_lm(args, _resolve_runtime_device(args.device))
        _freeze(model)
        icae, _ = run_finetune(args, model=model, tokenizer=tok)
        run_experiment(args, icae=icae, model=model, tokenizer=tok)

    elif args.run_mode == "eval":
        run_experiment(args)