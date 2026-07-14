#!/usr/bin/env python3
"""
I2CL baseline (Implicit In-context Learning).

This implementation follows the same evaluation/data/metric style used in:
  - streaming_llm_baseline.py
  - icae_baseline.py
  - gist_token_baseline.py

Core idea:
  1) Vectorize each demonstration in activation space by extracting layer-wise
     attention/MLP outputs at the final token position.
  2) Aggregate demonstrations by element-wise mean to build a context vector.
  3) Inject context vectors at inference by linearly mixing module outputs:
       attn' = beta_a * attn + lambda_a * ctx_attn
       mlp'  = beta_m * mlp  + lambda_m * ctx_mlp
"""

import argparse
import gc
import random
import re
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPT2Tokenizer

from utils.data import load_data


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
        data.append(
            {
                "input": ex["question"],
                "output": answer,
                "answer_number": _extract_answer_number(answer),
                "options": [answer],
                "task": "gsm8k",
                "dataset": "gsm8k",
            }
        )
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
            data.append(
                {
                    "input": ex["question"],
                    "output": choices[ans_idx],
                    "options": choices,
                    "task": subject,
                    "dataset": "mmlu",
                }
            )
    return data


def load_mmlu_college_split(args, split):
    cache_key = (args.seed, args.retrieval_split, args.eval_split)
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
        task=None,
        split=split,
        k=args.k,
        seed=args.seed,
        datasets=args.dataset.split(","),
        is_null=False,
    )


def setup_tokenizer(name):
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
        strip_tokens.update(tokenizer(ch, add_special_tokens=False)["input_ids"])
    idx = 0
    while idx < ans_ids.shape[1] and ans_ids[0, idx].item() in strip_tokens:
        idx += 1
    if idx >= ans_ids.shape[1]:
        return ans_ids, 0
    return ans_ids[:, idx:], idx


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
    is_gated = any(
        t in model_type
        for t in {
            "llama",
            "qwen",
            "qwen2",
            "mistral",
            "gemma",
            "phi3",
            "deepseek",
            "yi",
            "internlm",
            "internlm2",
            "baichuan",
            "cohere",
            "starcoder2",
        }
    )
    return {
        "hidden": hidden,
        "num_layers": num_layers,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate": intermediate,
        "vocab_size": vocab_size,
        "is_gated_mlp": is_gated,
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
    attn_pairs = n * (n + 1) / 2.0 if ctx_tokens == 0 else n * ctx_tokens + n * (n + 1) / 2.0
    qkv_flops = 2.0 * n * h * (nq * hd + 2 * nkv * hd)
    o_flops = 2.0 * n * h * h
    attn_flops = 2.0 * 2.0 * nq * hd * attn_pairs
    mlp_flops = (3.0 if fp["is_gated_mlp"] else 2.0) * 2.0 * n * h * inter
    total = L * (qkv_flops + o_flops + attn_flops + mlp_flops)
    total += 2.0 * n * h * vocab
    return total


def _analytical_flops_full(fp, seq_len):
    return _analytical_flops(fp, new_tokens=seq_len, ctx_tokens=0)


def _resolve_runtime_device(device_arg: str) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if device_arg == "auto":
        return "cuda"
    if device_arg.startswith("cuda"):
        return device_arg
    return "cpu"


def _load_model(args, runtime_device: str):
    if runtime_device.startswith("cuda"):
        if args.dtype == "auto":
            load_dtype = torch.float32
        elif args.dtype == "fp32":
            load_dtype = torch.float32
        elif args.dtype == "bf16":
            load_dtype = torch.bfloat16
        elif args.dtype == "fp16":
            load_dtype = torch.float16
        else:
            raise ValueError(f"Unsupported --dtype: {args.dtype}")
        if args.is_quant:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=load_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            return AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=bnb,
                dtype=load_dtype,
                device_map="auto",
            )
        return AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=load_dtype,
            device_map="auto",
        )
    if args.is_quant:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        return AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb,
            dtype=torch.float16,
            device_map="auto",
        )
    return AutoModelForCausalLM.from_pretrained(args.model_name).to(runtime_device)


def _model_input_device(model):
    return model.get_input_embeddings().weight.device


def _truncate_demo_ids(demo_ids: torch.Tensor, max_demo_tokens: int) -> torch.Tensor:
    if max_demo_tokens is None or max_demo_tokens <= 0:
        return demo_ids
    if demo_ids.shape[1] <= max_demo_tokens:
        return demo_ids
    return demo_ids[:, -max_demo_tokens:]


def _pick_demos_for_query(retrieval_data, args, per_query_random, rng_eval):
    if per_query_random:
        idx = rng_eval.sample(range(len(retrieval_data)), min(args.k, len(retrieval_data)))
        return [retrieval_data[i] for i in idx]
    if not hasattr(_pick_demos_for_query, "_fixed"):
        _pick_demos_for_query._fixed = choose_fixed_demos(
            retrieval_data, args.k, args.demo_strategy, args.seed
        )
    return _pick_demos_for_query._fixed


def _get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError("Unsupported model architecture: cannot locate decoder layers.")


class I2CLScorer:
    def __init__(self, model, tokenizer, device, lambda_a_init=0.1, beta_a_init=1.0, lambda_m_init=0.1, beta_m_init=1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.layers = _get_decoder_layers(model)
        self.n_layers = len(self.layers)
        self.ctx_attn = None
        self.ctx_mlp = None
        self.demo_len_orig = 0
        # Paper-aligned per-layer coefficients: 4L trainable scalars.
        self.lambda_a = torch.full((self.n_layers,), float(lambda_a_init), device=device, dtype=torch.float32)
        self.beta_a = torch.full((self.n_layers,), float(beta_a_init), device=device, dtype=torch.float32)
        self.lambda_m = torch.full((self.n_layers,), float(lambda_m_init), device=device, dtype=torch.float32)
        self.beta_m = torch.full((self.n_layers,), float(beta_m_init), device=device, dtype=torch.float32)

    @staticmethod
    def _extract_tensor(module_out):
        if isinstance(module_out, tuple):
            return module_out[0]
        return module_out

    @staticmethod
    def _replace_tensor(module_out, new_tensor):
        if isinstance(module_out, tuple):
            rest = list(module_out[1:])
            return tuple([new_tensor] + rest)
        return new_tensor

    @torch.no_grad()
    def build_context_from_demo_text(self, demo_text: str):
        demo_ids = self.tokenizer(demo_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        self.demo_len_orig = int(demo_ids.shape[1])
        return self.build_context_from_demo_ids(demo_ids)

    @torch.no_grad()
    def build_context_from_demo_ids(self, demo_ids: torch.Tensor):
        attn_vecs = [None] * self.n_layers
        mlp_vecs = [None] * self.n_layers
        hooks = []

        for li, layer in enumerate(self.layers):
            if hasattr(layer, "self_attn"):
                def _make_attn_capture(idx):
                    def _hook(_m, _inp, out):
                        x = self._extract_tensor(out)
                        attn_vecs[idx] = x[:, -1, :].detach()
                    return _hook
                hooks.append(layer.self_attn.register_forward_hook(_make_attn_capture(li)))
            elif hasattr(layer, "attn"):
                def _make_attn_capture(idx):
                    def _hook(_m, _inp, out):
                        x = self._extract_tensor(out)
                        attn_vecs[idx] = x[:, -1, :].detach()
                    return _hook
                hooks.append(layer.attn.register_forward_hook(_make_attn_capture(li)))
            else:
                raise ValueError("Layer without attention module.")

            if hasattr(layer, "mlp"):
                def _make_mlp_capture(idx):
                    def _hook(_m, _inp, out):
                        x = self._extract_tensor(out)
                        mlp_vecs[idx] = x[:, -1, :].detach()
                    return _hook
                hooks.append(layer.mlp.register_forward_hook(_make_mlp_capture(li)))
            else:
                raise ValueError("Layer without MLP module.")

        self.model(input_ids=demo_ids, use_cache=False)
        for h in hooks:
            h.remove()

        for li in range(self.n_layers):
            if attn_vecs[li] is None or mlp_vecs[li] is None:
                raise RuntimeError(f"Failed to capture activations for layer={li}.")
        return attn_vecs, mlp_vecs

    @torch.no_grad()
    def set_demo(self, demos: List[dict], add_newlines: bool, max_demo_tokens: int = 0):
        per_demo_attn = []
        per_demo_mlp = []
        demo_token_lens = []
        calibration_items = []
        for i, dp in enumerate(demos):
            q, a = normalize_text(dp, is_first=(i == 0), add_newlines=add_newlines)
            text = q + a
            ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
            if max_demo_tokens and max_demo_tokens > 0 and ids.shape[1] > max_demo_tokens:
                ids = ids[:, -max_demo_tokens:]
            demo_token_lens.append(int(ids.shape[1]))
            a_vecs, m_vecs = self.build_context_from_demo_ids(ids)
            per_demo_attn.append(torch.stack(a_vecs, dim=0).squeeze(1))
            per_demo_mlp.append(torch.stack(m_vecs, dim=0).squeeze(1))
            q_single, a_single = normalize_text(dp, is_first=True, add_newlines=add_newlines)
            q_ids = self.tokenizer(q_single, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
            a_ids = self.tokenizer(a_single, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
            if q_ids.numel() > 0 and a_ids.numel() > 0:
                calibration_items.append((q_ids, a_ids))

        attn_tensor = torch.stack(per_demo_attn, dim=0)
        mlp_tensor = torch.stack(per_demo_mlp, dim=0)
        self.ctx_attn = attn_tensor.mean(dim=0)
        self.ctx_mlp = mlp_tensor.mean(dim=0)
        return demo_token_lens, calibration_items

    def _register_injection_hooks(
        self,
        lambda_a_vec: torch.Tensor,
        beta_a_vec: torch.Tensor,
        lambda_m_vec: torch.Tensor,
        beta_m_vec: torch.Tensor,
        noise_std: float = 0.0,
    ):
        if self.ctx_attn is None or self.ctx_mlp is None:
            raise ValueError("I2CL context is not set. Call set_demo() first.")

        hooks = []
        for li, layer in enumerate(self.layers):
            ctx_a = self.ctx_attn[li].to(self.device)
            ctx_m = self.ctx_mlp[li].to(self.device)
            la = lambda_a_vec[li]
            ba = beta_a_vec[li]
            lm = lambda_m_vec[li]
            bm = beta_m_vec[li]

            def _make_attn_inject(ctx_vec, lam, beta):
                def _hook(_m, _inp, out):
                    x = self._extract_tensor(out)
                    bsz, seq_len, _ = x.shape
                    ctx = ctx_vec.view(1, 1, -1).expand(bsz, seq_len, -1).to(x.dtype)
                    y = beta.to(dtype=x.dtype, device=x.device) * x + lam.to(dtype=x.dtype, device=x.device) * ctx
                    if noise_std > 0:
                        y = y + noise_std * torch.randn_like(y)
                    return self._replace_tensor(out, y)
                return _hook

            def _make_mlp_inject(ctx_vec, lam, beta):
                def _hook(_m, _inp, out):
                    x = self._extract_tensor(out)
                    bsz, seq_len, _ = x.shape
                    ctx = ctx_vec.view(1, 1, -1).expand(bsz, seq_len, -1).to(x.dtype)
                    y = beta.to(dtype=x.dtype, device=x.device) * x + lam.to(dtype=x.dtype, device=x.device) * ctx
                    if noise_std > 0:
                        y = y + noise_std * torch.randn_like(y)
                    return self._replace_tensor(out, y)
                return _hook

            if hasattr(layer, "self_attn"):
                hooks.append(layer.self_attn.register_forward_hook(_make_attn_inject(ctx_a, la, ba)))
            else:
                hooks.append(layer.attn.register_forward_hook(_make_attn_inject(ctx_a, la, ba)))
            hooks.append(layer.mlp.register_forward_hook(_make_mlp_inject(ctx_m, lm, bm)))
        return hooks

    def _forward_with_coeffs(
        self,
        input_ids: torch.Tensor,
        lambda_a_vec: torch.Tensor,
        beta_a_vec: torch.Tensor,
        lambda_m_vec: torch.Tensor,
        beta_m_vec: torch.Tensor,
        noise_std: float = 0.0,
        use_no_grad: bool = True,
    ):
        hooks = self._register_injection_hooks(
            lambda_a_vec=lambda_a_vec,
            beta_a_vec=beta_a_vec,
            lambda_m_vec=lambda_m_vec,
            beta_m_vec=beta_m_vec,
            noise_std=noise_std,
        )
        try:
            if use_no_grad:
                with torch.no_grad():
                    out = self.model(input_ids=input_ids, use_cache=False)
            else:
                out = self.model(input_ids=input_ids, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        return out

    @torch.no_grad()
    def forward_with_i2cl(self, input_ids: torch.Tensor):
        return self._forward_with_coeffs(
            input_ids=input_ids,
            lambda_a_vec=self.lambda_a,
            beta_a_vec=self.beta_a,
            lambda_m_vec=self.lambda_m,
            beta_m_vec=self.beta_m,
            noise_std=0.0,
            use_no_grad=True,
        )

    @torch.no_grad()
    def score_option_nll(self, question_text: str, option_text: str):
        q_ids = self.tokenizer(question_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        o_ids = self.tokenizer(option_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        if o_ids.numel() == 0:
            return float("inf")
        all_ids = torch.cat([q_ids, o_ids], dim=1)
        out = self.forward_with_i2cl(all_ids)
        q_len = q_ids.shape[1]
        pred = out.logits[:, q_len - 1 : q_len - 1 + o_ids.shape[1], :]
        nll = F.cross_entropy(pred.reshape(-1, pred.size(-1)), o_ids.reshape(-1), reduction="sum").item()
        return nll

    @torch.no_grad()
    def qa_logits(self, q_ids: torch.Tensor, a_ids: torch.Tensor):
        all_ids = torch.cat([q_ids, a_ids], dim=1)
        out = self.forward_with_i2cl(all_ids)
        q_len = q_ids.shape[1]
        return out.logits[:, q_len - 1 : q_len - 1 + a_ids.shape[1], :]

    def calibrate_with_noisy_self_calibration(
        self,
        calibration_items: List[Tuple[torch.Tensor, torch.Tensor]],
        steps: int,
        lr: float,
        noise_std: float,
        seed: int,
    ):
        if steps <= 0 or len(calibration_items) == 0:
            return {"steps": 0, "mean_loss": float("nan"), "seq_lens_used": []}

        lambda_a = torch.nn.Parameter(self.lambda_a.clone())
        beta_a = torch.nn.Parameter(self.beta_a.clone())
        lambda_m = torch.nn.Parameter(self.lambda_m.clone())
        beta_m = torch.nn.Parameter(self.beta_m.clone())
        opt = torch.optim.Adam([lambda_a, beta_a, lambda_m, beta_m], lr=lr)

        rng = random.Random(seed)
        used_lens = []
        losses = []
        for _ in range(steps):
            q_ids, a_ids = calibration_items[rng.randrange(len(calibration_items))]
            all_ids = torch.cat([q_ids, a_ids], dim=1)
            out = self._forward_with_coeffs(
                input_ids=all_ids,
                lambda_a_vec=lambda_a,
                beta_a_vec=beta_a,
                lambda_m_vec=lambda_m,
                beta_m_vec=beta_m,
                noise_std=noise_std,
                use_no_grad=False,
            )
            q_len = q_ids.shape[1]
            pred = out.logits[:, q_len - 1 : q_len - 1 + a_ids.shape[1], :]
            loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), a_ids.reshape(-1), reduction="mean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            used_lens.append(int(all_ids.shape[1]))
            losses.append(float(loss.item()))

        self.lambda_a = lambda_a.detach()
        self.beta_a = beta_a.detach()
        self.lambda_m = lambda_m.detach()
        self.beta_m = beta_m.detach()
        return {
            "steps": len(used_lens),
            "mean_loss": float(sum(losses) / max(1, len(losses))),
            "seq_lens_used": used_lens,
        }

    def coeff_stats(self):
        def _stats(v: torch.Tensor):
            return {
                "mean": float(v.mean().item()),
                "std": float(v.std(unbiased=False).item()),
                "min": float(v.min().item()),
                "max": float(v.max().item()),
            }
        return {
            "lambda_a": _stats(self.lambda_a),
            "beta_a": _stats(self.beta_a),
            "lambda_m": _stats(self.lambda_m),
            "beta_m": _stats(self.beta_m),
        }


def run_eval(args):
    runtime_device = _resolve_runtime_device(args.device)
    tokenizer = setup_tokenizer(args.model_name)
    model = _load_model(args, runtime_device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "hf_device_map"):
        print(f"[Model] hf_device_map={model.hf_device_map}")

    io_device = _model_input_device(model)
    track_cuda_peak_mem = io_device.type == "cuda"
    if track_cuda_peak_mem:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model_mem_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    model_mem_gb = model_mem_bytes / (1024 ** 3)

    add_nl = not args.model_name.startswith("gpt2")
    retrieval_data = _load_dataset(args, args.retrieval_split)
    eval_data = _load_dataset(args, args.eval_split)
    if not retrieval_data or not eval_data:
        raise ValueError("Empty data.")

    scorer = I2CLScorer(
        model,
        tokenizer,
        io_device,
        lambda_a_init=args.lambda_a,
        beta_a_init=args.beta_a,
        lambda_m_init=args.lambda_m,
        beta_m_init=args.beta_m,
    )

    per_query_random = args.num_eval_demo_sets > 1
    rng_eval = random.Random(args.seed + 7777)
    full_kv_preds, i2cl_preds = [], []
    full_losses, i2cl_losses = [], []
    query_logit_mses = []
    ds_results = {}
    n_queries = max(1, len(eval_data))

    fp = _get_model_flop_params(model)
    full_kv_total_flops = 0.0
    i2cl_demo_flops = 0.0
    i2cl_query_total_flops = 0.0

    full_kv_peak_mem_bytes = 0
    i2cl_demo_peak_mem_bytes = 0
    i2cl_query_peak_mem_bytes = 0

    calibration_losses = []
    coeff_stats_last = scorer.coeff_stats()

    def _prepare_demo_bundle(demos, demo_seed):
        if track_cuda_peak_mem:
            torch.cuda.reset_peak_memory_stats()
        demo_text_local = build_demo_text(demos, add_newlines=add_nl)
        demo_ids_local = tokenizer(demo_text_local, return_tensors="pt", add_special_tokens=False)["input_ids"].to(io_device)
        demo_ids_local = _truncate_demo_ids(demo_ids_local, args.max_demo_tokens)
        demo_token_lens, calibration_items = scorer.set_demo(
            demos, add_newlines=add_nl, max_demo_tokens=args.max_demo_tokens
        )

        calib_info = {"steps": 0, "mean_loss": float("nan"), "seq_lens_used": []}
        if not args.disable_noisy_self_calibration:
            calib_info = scorer.calibrate_with_noisy_self_calibration(
                calibration_items=calibration_items,
                steps=args.calibration_steps,
                lr=args.calibration_lr,
                noise_std=args.calibration_noise_std,
                seed=demo_seed + args.calibration_seed_offset,
            )

        demo_peak = torch.cuda.max_memory_allocated(io_device) if track_cuda_peak_mem else 0
        vectorize_flops = sum(_analytical_flops_full(fp, max(1, tlen)) for tlen in demo_token_lens)
        calib_forward_flops = sum(
            _analytical_flops_full(fp, max(1, tlen)) for tlen in calib_info["seq_lens_used"]
        )
        calib_total_flops = args.calibration_backward_factor * calib_forward_flops
        return {
            "demo_ids": demo_ids_local,
            "demo_len": int(demo_ids_local.shape[1]),
            "demo_phase_flops": vectorize_flops + calib_total_flops,
            "demo_peak_mem_bytes": int(demo_peak),
            "calib_info": calib_info,
        }

    fixed_bundle = None
    if not per_query_random:
        fixed_demos = _pick_demos_for_query(retrieval_data, args, per_query_random, rng_eval)
        fixed_bundle = _prepare_demo_bundle(fixed_demos, demo_seed=args.seed + 999)
        i2cl_demo_flops += fixed_bundle["demo_phase_flops"]
        i2cl_demo_peak_mem_bytes = max(i2cl_demo_peak_mem_bytes, fixed_bundle["demo_peak_mem_bytes"])
        if fixed_bundle["calib_info"]["steps"] > 0 and fixed_bundle["calib_info"]["mean_loss"] == fixed_bundle["calib_info"]["mean_loss"]:
            calibration_losses.append(float(fixed_bundle["calib_info"]["mean_loss"]))
        coeff_stats_last = scorer.coeff_stats()

    for qi, dp in enumerate(tqdm(eval_data, desc="eval i2cl")):
        q_text, ans_text = normalize_text(dp, is_first=False, add_newlines=add_nl)
        q_ids = tokenizer(q_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(io_device)
        a_ids = tokenizer(ans_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(io_device)
        opts = dp["options"]

        if per_query_random:
            demos = _pick_demos_for_query(retrieval_data, args, per_query_random, rng_eval)
            query_bundle = _prepare_demo_bundle(demos, demo_seed=args.seed + 100000 + qi)
            i2cl_demo_flops += query_bundle["demo_phase_flops"]
            i2cl_demo_peak_mem_bytes = max(i2cl_demo_peak_mem_bytes, query_bundle["demo_peak_mem_bytes"])
            if query_bundle["calib_info"]["steps"] > 0 and query_bundle["calib_info"]["mean_loss"] == query_bundle["calib_info"]["mean_loss"]:
                calibration_losses.append(float(query_bundle["calib_info"]["mean_loss"]))
            coeff_stats_last = scorer.coeff_stats()
        else:
            query_bundle = fixed_bundle

        demo_ids = query_bundle["demo_ids"]
        demo_len = query_bundle["demo_len"]
        q_len = q_ids.shape[1]
        q_full_flops = 0.0
        q_i2cl_flops = 0.0

        if track_cuda_peak_mem:
            torch.cuda.reset_peak_memory_stats()
        opt_scores = {}
        for opt in opts:
            opt_ids = tokenizer(
                normalize_option(opt, add_nl), return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(io_device)
            if opt_ids.numel() == 0:
                opt_scores[opt] = float("inf")
                continue
            with torch.no_grad():
                all_ids = torch.cat([demo_ids, q_ids, opt_ids], dim=1)
                out = model(input_ids=all_ids, use_cache=False)
                start = demo_len + q_len - 1
                pred = out.logits[:, start : start + opt_ids.shape[1], :]
                nll = F.cross_entropy(pred.reshape(-1, pred.size(-1)), opt_ids.reshape(-1), reduction="sum").item()
            opt_scores[opt] = nll
            q_full_flops += _analytical_flops_full(fp, demo_len + q_len + opt_ids.shape[1])
        full_kv_preds.append(min(opt_scores, key=opt_scores.get))
        if track_cuda_peak_mem:
            full_kv_peak_mem_bytes = max(full_kv_peak_mem_bytes, torch.cuda.max_memory_allocated(io_device))

        t_logits_s = None
        first_token_id = None
        if q_ids.numel() > 0 and a_ids.numel() > 0:
            with torch.no_grad():
                all_ids = torch.cat([demo_ids, q_ids, a_ids], dim=1)
                out = model(input_ids=all_ids, use_cache=False)
                t_logits = out.logits[:, demo_len + q_len - 1 : demo_len + q_len - 1 + a_ids.shape[1], :]
                a_stripped, n_strip = strip_leading_format_tokens(a_ids, tokenizer)
                if a_stripped.numel() > 0:
                    t_logits_s = t_logits[:, n_strip:, :]
                    first_token_id = a_stripped[:, 0].unsqueeze(1)
                    full_losses.append(
                        F.cross_entropy(
                            t_logits_s.reshape(-1, t_logits_s.size(-1)),
                            a_stripped.reshape(-1),
                            reduction="mean",
                        ).item()
                    )

        if track_cuda_peak_mem:
            torch.cuda.reset_peak_memory_stats()
        scores = {}
        for opt in opts:
            opt_text = normalize_option(opt, add_nl)
            scores[opt] = scorer.score_option_nll(q_text, opt_text)
            opt_len = len(tokenizer(opt_text, add_special_tokens=False)["input_ids"])
            q_i2cl_flops += _analytical_flops_full(fp, q_len + opt_len)
        i2cl_preds.append(min(scores, key=scores.get))
        if track_cuda_peak_mem:
            i2cl_query_peak_mem_bytes = max(i2cl_query_peak_mem_bytes, torch.cuda.max_memory_allocated(io_device))

        s_logits_s = None
        if q_ids.numel() > 0 and a_ids.numel() > 0:
            with torch.no_grad():
                s_logits = scorer.qa_logits(q_ids, a_ids)
                a_stripped, n_strip = strip_leading_format_tokens(a_ids, tokenizer)
                if a_stripped.numel() > 0:
                    s_logits_s = s_logits[:, n_strip:, :]
                    i2cl_losses.append(
                        F.cross_entropy(
                            s_logits_s.reshape(-1, s_logits_s.size(-1)),
                            a_stripped.reshape(-1),
                            reduction="mean",
                        ).item()
                    )

        if t_logits_s is not None and s_logits_s is not None and first_token_id is not None:
            t_first = t_logits_s[:, 0, :].gather(1, first_token_id).squeeze(1)
            s_first = s_logits_s[:, 0, :].gather(1, first_token_id).squeeze(1)
            t_v = float(t_first.item())
            s_v = float(s_first.item())
            denom = max(abs(s_v), abs(t_v))
            query_logit_mses.append(((s_v - t_v) / denom) ** 2 if denom != 0 else 0.0)

        ds = dp.get("task", dp.get("dataset", "unknown"))
        if ds not in ds_results:
            ds_results[ds] = {"full_p": [], "i2cl_p": [], "gt": []}
        ds_results[ds]["full_p"].append(full_kv_preds[-1])
        ds_results[ds]["i2cl_p"].append(i2cl_preds[-1])
        ds_results[ds]["gt"].append(dp["output"])

        full_kv_total_flops += q_full_flops
        i2cl_query_total_flops += q_i2cl_flops

    gts = [dp["output"] for dp in eval_data]
    full_acc = accuracy(full_kv_preds, gts)
    i2cl_acc = accuracy(i2cl_preds, gts)
    mean_full_ce = sum(full_losses) / max(1, len(full_losses))
    mean_i2cl_ce = sum(i2cl_losses) / max(1, len(i2cl_losses))
    per_sample_mse = sum(query_logit_mses) / max(1, len(query_logit_mses))

    total_full_flops = full_kv_total_flops
    total_i2cl_flops = i2cl_demo_flops + i2cl_query_total_flops
    flops_reduction = (total_full_flops / total_i2cl_flops) if total_i2cl_flops > 0 else 0.0
    mean_calibration_loss = sum(calibration_losses) / max(1, len(calibration_losses)) if calibration_losses else float("nan")

    result = {
        "full_kv_acc": full_acc,
        "i2cl_acc": i2cl_acc,
        "stream_acc": i2cl_acc,
        "acc_gap": full_acc - i2cl_acc,
        "full_ce": mean_full_ce,
        "i2cl_ce": mean_i2cl_ce,
        "stream_ce": mean_i2cl_ce,
        "ce_gap": mean_i2cl_ce - mean_full_ce,
        "per_sample_mean_MSE": per_sample_mse,
        "avg_full_flops": total_full_flops / n_queries,
        "avg_i2cl_flops": total_i2cl_flops / n_queries,
        "avg_stream_flops": total_i2cl_flops / n_queries,
        "total_full_flops": total_full_flops,
        "total_i2cl_flops": total_i2cl_flops,
        "total_stream_flops": total_i2cl_flops,
        "i2cl_demo_flops": i2cl_demo_flops,
        "stream_demo_flops": i2cl_demo_flops,
        "flops_reduction": flops_reduction,
        "full_kv_peak_mem_bytes": full_kv_peak_mem_bytes,
        "full_kv_peak_mem_gb": full_kv_peak_mem_bytes / (1024 ** 3),
        "i2cl_demo_peak_mem_bytes": i2cl_demo_peak_mem_bytes,
        "i2cl_demo_peak_mem_gb": i2cl_demo_peak_mem_bytes / (1024 ** 3),
        "i2cl_query_peak_mem_bytes": i2cl_query_peak_mem_bytes,
        "i2cl_query_peak_mem_gb": i2cl_query_peak_mem_bytes / (1024 ** 3),
        "model_mem_bytes": model_mem_bytes,
        "model_mem_gb": model_mem_gb,
        "ds_results": ds_results,
        "lambda_a": args.lambda_a,
        "beta_a": args.beta_a,
        "lambda_m": args.lambda_m,
        "beta_m": args.beta_m,
        "coeff_stats": coeff_stats_last,
        "calibration_steps": 0 if args.disable_noisy_self_calibration else args.calibration_steps,
        "calibration_noise_std": 0.0 if args.disable_noisy_self_calibration else args.calibration_noise_std,
        "calibration_mean_loss": mean_calibration_loss,
    }

    print(f"\n{'='*60}")
    print(
        f"  I2CL: la={args.lambda_a}, ba={args.beta_a}, "
        f"lm={args.lambda_m}, bm={args.beta_m}, k={args.k}"
    )
    print(f"{'='*60}")
    print(f"Full-KV Accuracy      = {full_acc:.4f}")
    print(f"I2CL Accuracy         = {i2cl_acc:.4f}")
    print(f"Accuracy gap          = {full_acc - i2cl_acc:+.4f}")
    print(f"Full-KV CE            = {mean_full_ce:.4f}")
    print(f"I2CL CE               = {mean_i2cl_ce:.4f}")
    print(f"CE gap                = {mean_i2cl_ce - mean_full_ce:+.4f}")
    if args.disable_noisy_self_calibration:
        print("noisy_self_calibration = disabled")
    else:
        print(
            f"noisy_self_calibration = enabled "
            f"(steps={args.calibration_steps}, noise_std={args.calibration_noise_std}, "
            f"mean_loss={mean_calibration_loss:.4f})"
        )
    print(f"per_sample_mean_MSE   = {per_sample_mse:.6f}")
    if track_cuda_peak_mem:
        print(f"model_weights_mem     = {model_mem_gb:.3f} GB")
        print(
            f"full_kv_peak_mem      = {result['full_kv_peak_mem_gb']:.3f} GB "
            f"(+{result['full_kv_peak_mem_gb'] - model_mem_gb:.3f} GB over model)"
        )
        print(
            f"i2cl_demo_peak_mem    = {result['i2cl_demo_peak_mem_gb']:.3f} GB "
            f"(+{result['i2cl_demo_peak_mem_gb'] - model_mem_gb:.3f} GB over model, one-time)"
        )
        print(
            f"i2cl_query_peak_mem   = {result['i2cl_query_peak_mem_gb']:.3f} GB "
            f"(+{result['i2cl_query_peak_mem_gb'] - model_mem_gb:.3f} GB over model, per-query)"
        )
    print(f"total_full_flops      = {total_full_flops:.3e} ({n_queries} queries)")
    print(f"total_i2cl_flops      = {total_i2cl_flops:.3e} ({n_queries} queries)")
    if total_i2cl_flops > 0:
        print(f"flops_reduction       = {flops_reduction:.4f}x")
    print(
        f"coeff(lambda_a/beta_a/lambda_m/beta_m)_mean = "
        f"{coeff_stats_last['lambda_a']['mean']:.4f} / {coeff_stats_last['beta_a']['mean']:.4f} / "
        f"{coeff_stats_last['lambda_m']['mean']:.4f} / {coeff_stats_last['beta_m']['mean']:.4f}"
    )
    print(f"\n{'Dataset':<20} {'N':>5} {'FullKV':>10} {'I2CL':>10} {'Gap':>10}")
    print(f"{'-'*55}")
    for ds in sorted(ds_results.keys()):
        r = ds_results[ds]
        n = len(r["gt"])
        fa = accuracy(r["full_p"], r["gt"])
        ia = accuracy(r["i2cl_p"], r["gt"])
        print(f"{ds:<20} {n:>5} {fa:>10.4f} {ia:>10.4f} {fa-ia:>+10.4f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description="I2CL baseline")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retrieval_split", type=str, default="test")
    p.add_argument("--eval_split", type=str, default="dev")
    p.add_argument("--demo_strategy", type=str, default="first")
    p.add_argument("--num_eval_demo_sets", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--is_quant", default=False, action="store_true")
    p.add_argument(
        "--max_demo_tokens",
        type=int,
        default=1536,
        help="Cap demonstration prefix length to avoid OOM (0 disables truncation).",
    )

    # Layer-wise coefficients are initialized from these values, then calibrated.
    p.add_argument("--lambda_a", type=float, default=0.1)
    p.add_argument("--beta_a", type=float, default=1.0)
    p.add_argument("--lambda_m", type=float, default=0.1)
    p.add_argument("--beta_m", type=float, default=1.0)
    p.add_argument("--disable_noisy_self_calibration", default=False, action="store_true")
    p.add_argument("--calibration_steps", type=int, default=100)
    p.add_argument("--calibration_lr", type=float, default=1e-2)
    p.add_argument("--calibration_noise_std", type=float, default=1e-2)
    p.add_argument(
        "--calibration_backward_factor",
        type=float,
        default=3.0,
        help="Approx FLOPs multiplier for calibration fwd+bwd+update.",
    )
    p.add_argument("--calibration_seed_offset", type=int, default=2026)
    args = p.parse_args()

    run_eval(args)


if __name__ == "__main__":
    main()
