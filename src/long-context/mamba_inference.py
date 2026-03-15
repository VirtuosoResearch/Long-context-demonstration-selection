import argparse
import copy
import math
import random
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
from transformers.cache_utils import DynamicCache

from utils.data import load_data


def _hippo_legs_matrix(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Build the continuous-time HiPPO-LegS transition matrix.
    This matrix is lower-triangular and stable (negative diagonal).
    """
    n = torch.arange(size, device=device, dtype=dtype)
    p = torch.sqrt(2.0 * n + 1.0)
    row = p[:, None]
    col = p[None, :]

    a = torch.zeros(size, size, device=device, dtype=dtype)
    lower = torch.tril(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=-1)
    a[lower] = -(row * col)[lower]
    a[torch.arange(size, device=device), torch.arange(size, device=device)] = -(n + 1.0)
    return a


def _hippo_legs_input_vector(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Build the canonical HiPPO-LegS input vector b_n = sqrt(2n + 1).
    """
    n = torch.arange(size, device=device, dtype=dtype)
    return torch.sqrt(2.0 * n + 1.0)


class SSMVirtualKVSidecar(nn.Module):
    """
    A compact state-space sidecar compressor.
    It compresses long demo embeddings into a fixed-size latent state,
    then projects the state into a small number of virtual KV tokens.
    """
    def __init__(
        self,
        config,
        ssm_dim: int = 256,
        num_virtual_tokens: int = 8,
        dt: float = 1.0,
        train_A: bool = False,
    ):
        super().__init__()
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        self.ssm_dim = ssm_dim
        self.num_virtual_tokens = num_virtual_tokens
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers

        # Support GQA/MQA models that expose separate KV heads.
        self.num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads

        # State update: x_t = x_{t-1} @ A + B(u_t), with HiPPO initialization for A.
        a_cont = _hippo_legs_matrix(ssm_dim, device=torch.device("cpu"), dtype=torch.float32)
        a_disc = torch.matrix_exp(dt * a_cont)
        if train_A:
            self.A = nn.Parameter(a_disc)
        else:
            self.register_buffer("A", a_disc, persistent=True)
        self.B = nn.Linear(self.hidden_size, ssm_dim)
        self._init_hippo_aware_B()

        # Project latent state into per-layer K/V virtual tokens.
        total_kv_elements = (self.num_layers * 2 * self.num_kv_heads * self.num_virtual_tokens * self.head_dim)
        self.proj = nn.Linear(ssm_dim, total_kv_elements)

    def _init_hippo_aware_B(self):
        """
        HiPPO-aware initialization for input projection B.
        Rows are scaled by sqrt(2n + 1) so lower-order coefficients receive
        balanced updates while preserving a standard random projection structure.
        """
        with torch.no_grad():
            nn.init.normal_(self.B.weight, mean=0.0, std=1.0 / math.sqrt(self.hidden_size))
            b_vec = _hippo_legs_input_vector(
                self.ssm_dim, device=self.B.weight.device, dtype=self.B.weight.dtype
            )
            self.B.weight.mul_(b_vec.unsqueeze(1))
            nn.init.zeros_(self.B.bias)

    def _scan_state(
        self,
        inputs_embeds: torch.Tensor,
        chunk_size: int = 0,
        detach_state_every_chunk: bool = False,
    ) -> torch.Tensor:
        """
        Scan sequence and return final SSM state.
        If chunk_size > 0, scan by chunks; optionally detach state between chunks
        for TBPTT-style training.
        """
        bsz, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        state = torch.zeros(bsz, self.ssm_dim, device=device, dtype=dtype)
        a = self.A.to(device=device, dtype=dtype)
        if chunk_size <= 0:
            for t in range(seq_len):
                u_t = self.B(inputs_embeds[:, t, :])
                # state is row-major [B, D], so use A^T for row-vector update.
                state = F.linear(state, a) + u_t
            return state

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk = inputs_embeds[:, start:end, :]
            for t in range(chunk.shape[1]):
                u_t = self.B(chunk[:, t, :])
                state = F.linear(state, a) + u_t
            if detach_state_every_chunk:
                state = state.detach()
        return state

    def _state_to_virtual_past(self, state: torch.Tensor):
        bsz = state.shape[0]

        # Project to flat virtual K/V tensor.
        flat_kvs = self.proj(state)  # [bsz, total_kv_elements]

        # Reshape to legacy cache format expected by HF models:
        # [batch_size, num_layers, 2(K,V), num_kv_heads, num_virtual_tokens, head_dim]
        reshaped = flat_kvs.view(
            bsz,
            self.num_layers,
            2,
            self.num_kv_heads,
            self.num_virtual_tokens,
            self.head_dim
        )

        virtual_past = ()
        for layer_idx in range(self.num_layers):
            k = reshaped[:, layer_idx, 0, :, :, :]  # [bsz, num_kv_heads, num_virtual_tokens, head_dim]
            v = reshaped[:, layer_idx, 1, :, :, :]
            virtual_past += ((k, v),)

        return virtual_past

    def forward(self, inputs_embeds):
        # Standard full-sequence scan.
        state = self._scan_state(inputs_embeds, chunk_size=0, detach_state_every_chunk=False)
        return self._state_to_virtual_past(state)

    def forward_chunked(self, inputs_embeds, chunk_size: int, detach_state_every_chunk: bool = False):
        # Chunked scan. Detach is optional and should match the training objective.
        state = self._scan_state(
            inputs_embeds,
            chunk_size=chunk_size,
            detach_state_every_chunk=detach_state_every_chunk,
        )
        return self._state_to_virtual_past(state)


class SSMHybridICLScorer:
    def __init__(self, model, tokenizer, device: str, num_virtual_tokens: int = 8, train_A: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Attach sidecar SSM compressor.
        self.sidecar = SSMVirtualKVSidecar(
            model.config,
            ssm_dim=256,
            num_virtual_tokens=num_virtual_tokens,
            train_A=train_A,
        ).to(device)

        self.demo_past = None
        self.demo_len = 0  # Fixed context length after compression.

    @torch.no_grad()
    def build_demo_cache(self, demo_text: str, measure_flops: bool = False):
        demo_ids = self.tokenizer(demo_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        if demo_ids.numel() == 0:
            raise ValueError("Demonstration text is empty.")

        # Feed long demos to the sidecar compressor instead of full attention.
        inputs_embeds = self.model.get_input_embeddings()(demo_ids)

        # Build compressed virtual-token KV cache.
        self.demo_past = self.sidecar(inputs_embeds)

        # Context length collapses to a constant virtual-token count.
        self.demo_len = self.sidecar.num_virtual_tokens

        # Rough SSM compute proxy.
        flops = inputs_embeds.shape[1] * self.sidecar.ssm_dim * self.sidecar.ssm_dim
        return flops

    @torch.no_grad()
    def prefill_question(self, question_text: str, measure_flops: bool = False):
        q_ids = self.tokenizer(question_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        if q_ids.numel() == 0:
            raise ValueError("Question text is empty.")

        # Attention mask now uses compressed prefix length.
        attn = torch.ones((1, self.demo_len + q_ids.shape[1]), dtype=torch.long, device=self.device)

        outputs = self.model(
            input_ids=q_ids,
            past_key_values=self.demo_past,
            attention_mask=attn,
            use_cache=True,
        )
        # FLOPs proxy is tracked separately via attention proxy.
        return outputs.logits[:, -1, :], outputs.past_key_values, q_ids.shape[1], 0

    @torch.no_grad()
    def score_options_nll(
        self, first_logits, past_after_question, question_len: int, options: List[str], measure_flops: bool = False
    ):
        tokenized = [self.tokenizer(opt, add_special_tokens=False)["input_ids"] for opt in options]
        lengths = [len(x) for x in tokenized]
        if any(length == 0 for length in lengths):
            raise ValueError("Found empty option text after tokenization.")
        bsz = len(options)
        max_len = max(lengths)
        
        input_ids = torch.full((bsz, max_len), fill_value=self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        token_mask = torch.zeros((bsz, max_len), dtype=torch.float32, device=self.device)
        for i, ids in enumerate(tokenized):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            token_mask[i, : len(ids)] = 1.0

        if isinstance(past_after_question, tuple):
            batched_past = DynamicCache.from_legacy_cache(past_after_question)
        else:
            batched_past = copy.deepcopy(past_after_question)
        batched_past.batch_repeat_interleave(bsz)

        # Base context length is compressed virtual prefix + question length.
        base_len = self.demo_len + question_len
        attn = torch.ones((bsz, base_len + max_len), dtype=torch.long, device=self.device)
        for i, l in enumerate(lengths):
            if l < max_len:
                attn[i, base_len + l :] = 0

        out = self.model(
            input_ids=input_ids,
            past_key_values=batched_past,
            attention_mask=attn,
            use_cache=False,
        )

        first = first_logits.expand(bsz, -1).unsqueeze(1)
        pred_logits = torch.cat([first, out.logits[:, :-1, :]], dim=1) if max_len > 1 else first

        losses = F.cross_entropy(pred_logits.reshape(-1, pred_logits.size(-1)), input_ids.reshape(-1), reduction="none").view(bsz, max_len)
        nll = (losses * token_mask).sum(dim=1)
        return {opt: float(nll[i].item()) for i, opt in enumerate(options)}


def _build_demo_text(fixed_demos: List[Dict], add_newlines: bool) -> str:
    return "".join(
        [di + do for i, dp in enumerate(fixed_demos) for di, do in [_normalize_text(dp, i == 0, add_newlines)]]
    )


def _freeze_backbone(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def _maybe_compile_scan(sidecar: SSMVirtualKVSidecar, args):
    """
    Optionally compile the scan kernel for faster recurrent updates.
    """
    if not args.compile_scan:
        return
    if not hasattr(torch, "compile"):
        print("torch.compile is not available in this PyTorch version; skip compile_scan.")
        return
    try:
        sidecar._scan_state = torch.compile(
            sidecar._scan_state,
            mode=args.compile_mode,
            fullgraph=False,
        )
        print(f"Enabled torch.compile for sidecar._scan_state (mode={args.compile_mode}).")
    except Exception as exc:
        print(f"WARNING: compile_scan failed, fallback to eager mode. Error: {exc}")


def _build_student_demo_past(
    model,
    sidecar: SSMVirtualKVSidecar,
    demo_ids: torch.Tensor,
    tbptt_chunk_size: int = 0,
    detach_state_every_chunk: bool = False,
):
    demo_embeds = model.get_input_embeddings()(demo_ids)
    if tbptt_chunk_size > 0:
        demo_past = sidecar.forward_chunked(
            demo_embeds,
            chunk_size=tbptt_chunk_size,
            detach_state_every_chunk=detach_state_every_chunk,
        )
    else:
        demo_past = sidecar(demo_embeds)
    return demo_past, sidecar.num_virtual_tokens


def _student_prefill_question_with_demo_past(
    model,
    demo_past,
    demo_len: int,
    q_ids: torch.Tensor,
):
    attn = torch.ones((1, demo_len + q_ids.shape[1]), dtype=torch.long, device=q_ids.device)
    out = model(
        input_ids=q_ids,
        past_key_values=demo_past,
        attention_mask=attn,
        use_cache=True,
    )
    return out.logits[:, -1, :], out.past_key_values, demo_len


@torch.no_grad()
def _teacher_prefill_question(model, demo_past, demo_len: int, q_ids: torch.Tensor):
    attn = torch.ones((1, demo_len + q_ids.shape[1]), dtype=torch.long, device=q_ids.device)
    out = model(
        input_ids=q_ids,
        past_key_values=demo_past,
        attention_mask=attn,
        use_cache=True,
    )
    return out.logits[:, -1, :], out.past_key_values


def _answer_pass_logits_and_hidden(model, first_logits, past_after_question, base_len: int, answer_ids: torch.Tensor):
    ans_len = answer_ids.shape[1]
    attn = torch.ones((1, base_len + ans_len), dtype=torch.long, device=answer_ids.device)
    out = model(
        input_ids=answer_ids,
        past_key_values=past_after_question,
        attention_mask=attn,
        use_cache=False,
        output_hidden_states=True,
    )
    if ans_len == 1:
        pred_logits = first_logits.unsqueeze(1)
    else:
        pred_logits = torch.cat([first_logits.unsqueeze(1), out.logits[:, :-1, :]], dim=1)
    return pred_logits, out.hidden_states[-1]


def run_distillation_training(args):
    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    tokenizer = _setup_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    _freeze_backbone(model)

    add_newlines = not args.model_name.startswith("gpt2")
    retrieval_data = load_data(
        task=None,
        split=args.retrieval_split,
        k=args.k,
        seed=args.seed,
        datasets=args.dataset.split(","),
        is_null=False,
    )
    train_data = load_data(
        task=None,
        split=args.train_split,
        k=args.k,
        seed=args.seed,
        datasets=args.dataset.split(","),
        is_null=False,
    )
    if len(train_data) == 0 or len(retrieval_data) == 0:
        raise ValueError("Loaded empty retrieval/train data.")

    fixed_demos = _choose_fixed_demos(retrieval_data, args.k, args.demo_strategy, args.seed)
    demo_text = _build_demo_text(fixed_demos, add_newlines=add_newlines)
    demo_ids = tokenizer(demo_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    if demo_ids.numel() == 0:
        raise ValueError("Demonstration text is empty after tokenization.")
    full_demo_len = demo_ids.shape[1]

    # Teacher uses full KV-cache over raw demonstrations.
    with torch.no_grad():
        teacher_demo = model(input_ids=demo_ids, use_cache=True)
        teacher_demo_past = teacher_demo.past_key_values

    student = SSMHybridICLScorer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_virtual_tokens=args.num_virtual_tokens,
        train_A=args.train_A,
    )
    sidecar = student.sidecar
    sidecar.train()
    _maybe_compile_scan(sidecar, args)

    trainable = list(sidecar.B.parameters()) + list(sidecar.proj.parameters())
    if isinstance(sidecar.A, nn.Parameter):
        trainable.append(sidecar.A)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    rng = random.Random(args.seed)
    steps = 0
    running = {"loss": 0.0, "kl": 0.0, "mse": 0.0, "ce": 0.0}
    print("\n===== Sidecar Distillation Training =====")
    print(
        f"train_size={len(train_data)}, epochs={args.epochs}, demo_len={full_demo_len}, "
        f"virtual_tokens={args.num_virtual_tokens}, tbptt_chunk_size={args.tbptt_chunk_size}, "
        f"train_batch_size={args.train_batch_size}"
    )

    train_batch_size = max(1, args.train_batch_size)
    for epoch in range(args.epochs):
        order = list(range(len(train_data)))
        rng.shuffle(order)
        num_batches = (len(order) + train_batch_size - 1) // train_batch_size
        for b in tqdm(range(num_batches), total=num_batches, desc=f"train-epoch-{epoch+1}"):
            batch_indices = order[b * train_batch_size : (b + 1) * train_batch_size]
            if len(batch_indices) == 0:
                continue

            # Build student demo cache once per optimizer step.
            student_demo_past, student_demo_len = _build_student_demo_past(
                model=model,
                sidecar=sidecar,
                demo_ids=demo_ids,
                tbptt_chunk_size=args.tbptt_chunk_size,
                detach_state_every_chunk=False,
            )

            batch_loss = None
            batch_kl, batch_mse, batch_ce = 0.0, 0.0, 0.0
            valid_items = 0

            for idx in batch_indices:
                dp = train_data[idx]
                q_text, ans_text = _normalize_text(dp, is_first=False, add_newlines=add_newlines)
                q_ids = tokenizer(q_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                ans_ids = tokenizer(ans_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                if q_ids.numel() == 0 or ans_ids.numel() == 0:
                    continue

                with torch.no_grad():
                    t_first, t_q_past = _teacher_prefill_question(model, teacher_demo_past, full_demo_len, q_ids)
                    t_logits, t_hidden = _answer_pass_logits_and_hidden(
                        model, t_first, t_q_past, full_demo_len + q_ids.shape[1], ans_ids
                    )

                s_first, s_q_past, s_demo_len = _student_prefill_question_with_demo_past(
                    model=model,
                    demo_past=student_demo_past,
                    demo_len=student_demo_len,
                    q_ids=q_ids,
                )
                s_logits, s_hidden = _answer_pass_logits_and_hidden(
                    model, s_first, s_q_past, s_demo_len + q_ids.shape[1], ans_ids
                )

                vocab_size = s_logits.size(-1)
                s_logits_2d = s_logits.reshape(-1, vocab_size)
                t_logits_2d = t_logits.reshape(-1, vocab_size)
                temp = args.kd_temperature
                kl = F.kl_div(
                    F.log_softmax(s_logits_2d / temp, dim=-1),
                    F.softmax(t_logits_2d / temp, dim=-1),
                    reduction="batchmean",
                ) * (temp * temp)
                mse = F.mse_loss(s_hidden, t_hidden)
                ce = F.cross_entropy(
                    s_logits_2d,
                    ans_ids.reshape(-1),
                    reduction="mean",
                )
                loss = args.loss_w_kl * kl + args.loss_w_mse * mse + args.loss_w_ce * ce
                batch_loss = loss if batch_loss is None else (batch_loss + loss)
                batch_kl += float(kl.item())
                batch_mse += float(mse.item())
                batch_ce += float(ce.item())
                valid_items += 1

            if valid_items == 0:
                continue

            batch_loss = batch_loss / valid_items
            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
            optimizer.step()

            steps += 1
            running["loss"] += float(batch_loss.item())
            running["kl"] += batch_kl / valid_items
            running["mse"] += batch_mse / valid_items
            running["ce"] += batch_ce / valid_items

            if steps % args.log_every == 0:
                denom = float(args.log_every)
                print(
                    f"step={steps} "
                    f"loss={running['loss']/denom:.4f} "
                    f"kl={running['kl']/denom:.4f} "
                    f"mse={running['mse']/denom:.4f} "
                    f"ce={running['ce']/denom:.4f}"
                )
                running = {"loss": 0.0, "kl": 0.0, "mse": 0.0, "ce": 0.0}

            if args.max_steps > 0 and steps >= args.max_steps:
                break
        if args.max_steps > 0 and steps >= args.max_steps:
            break

    if args.save_sidecar_path:
        ckpt = {
            "sidecar": sidecar.state_dict(),
            "args": vars(args),
            "model_name": args.model_name,
        }
        torch.save(ckpt, args.save_sidecar_path)
        print(f"Saved sidecar checkpoint to: {args.save_sidecar_path}")

# ----------------- Utility Functions -----------------
def _setup_tokenizer(model_name: str):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name) if model_name.startswith("gpt2") else AutoTokenizer.from_pretrained(model_name)
    if tokenizer.padding_side == "left":
        tokenizer.padding_side = "right"
    if tokenizer.eos_token_id is None and tokenizer.sep_token is not None:
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token
    return tokenizer

def _normalize_text(dp: Dict, is_first: bool, add_newlines: bool) -> Tuple[str, str]:
    return ("\n" + dp["input"]) if (add_newlines and not is_first) else dp["input"], ("\n" + dp["output"]) if add_newlines else dp["output"]

def _normalize_option(opt: str, add_newlines: bool) -> str:
    return ("\n" + opt) if add_newlines else opt

def _choose_fixed_demos(retrieval_data: List[Dict], k: int, strategy: str, seed: int) -> List[Dict]:
    return retrieval_data[:k] if strategy == "first" else [retrieval_data[i] for i in random.Random(seed).sample(range(len(retrieval_data)), k)]

def _accuracy(preds: List[str], gts: List[str]) -> float:
    return sum(int(p.strip() == g.strip()) for p, g in zip(preds, gts)) / max(1, len(gts))

def _attention_proxy_full(seq_len: int) -> int:
    return seq_len * (seq_len + 1) // 2

def _attention_proxy_incremental(start_ctx: int, new_tokens: int) -> int:
    return new_tokens * start_ctx + new_tokens * (new_tokens + 1) // 2


def run_experiment(args):
    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    tokenizer = _setup_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    add_newlines = not args.model_name.startswith("gpt2")
    retrieval_data = load_data(
        task=None,
        split=args.retrieval_split,
        k=args.k,
        seed=args.seed,
        datasets=args.dataset.split(","),
        is_null=False,
    )
    eval_data = load_data(
        task=None,
        split=args.eval_split,
        k=args.k,
        seed=args.seed,
        datasets=args.dataset.split(","),
        is_null=False,
    )
    options = eval_data[0]["options"]
    fixed_demos = _choose_fixed_demos(retrieval_data, args.k, args.demo_strategy, args.seed)

    demo_text = _build_demo_text(fixed_demos, add_newlines=add_newlines)
    original_demo_len = len(tokenizer(demo_text, add_special_tokens=False)["input_ids"])

    print("\n===== SSM Virtual KV-cache Experiment =====")
    print("WARNING: Without training the SSMVirtualKVSidecar, accuracy will be random!")

    # Use the SSM-hybrid scorer.
    num_virtual_tokens = args.num_virtual_tokens
    kv = SSMHybridICLScorer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_virtual_tokens=num_virtual_tokens,
        train_A=args.train_A,
    )
    if args.load_sidecar_path:
        ckpt = torch.load(args.load_sidecar_path, map_location=device)
        state = ckpt["sidecar"] if isinstance(ckpt, dict) and "sidecar" in ckpt else ckpt
        kv.sidecar.load_state_dict(state, strict=True)
        kv.sidecar.eval()
        print(f"Loaded sidecar checkpoint from: {args.load_sidecar_path}")

    t1 = time.perf_counter()
    kv.build_demo_cache(demo_text)

    kv_preds = []
    kv_proxy = 0
    # Proxy is computed on compressed context lengths.
    kv_proxy += _attention_proxy_full(kv.demo_len)

    for dp in tqdm(eval_data, total=len(eval_data), desc="ssm-hybrid-cache"):
        q_text, _ = _normalize_text(dp, is_first=False, add_newlines=add_newlines)
        q_len = len(tokenizer(q_text, add_special_tokens=False)["input_ids"])

        first_logits, q_past, q_len_runtime, _ = kv.prefill_question(q_text)
        assert q_len_runtime == q_len
        kv_proxy += _attention_proxy_incremental(kv.demo_len, q_len)

        opt_texts = [_normalize_option(opt, add_newlines=add_newlines) for opt in options]
        scores_text = kv.score_options_nll(first_logits, q_past, q_len, opt_texts)
        scores = {opt: scores_text[opt_text] for opt, opt_text in zip(options, opt_texts)}
        
        for opt_text in opt_texts:
            o_len = len(tokenizer(opt_text, add_special_tokens=False)["input_ids"])
            kv_proxy += _attention_proxy_incremental(kv.demo_len + q_len, o_len)

        kv_preds.append(min(scores, key=scores.get))
        
    kv_time = time.perf_counter() - t1
    kv_acc = _accuracy(kv_preds, [dp["output"] for dp in eval_data])

    print(f"\nSSM Hybrid Acc (Untrained) = {kv_acc:.6f}")
    print(f"Time Taken = {kv_time:.3f}s")
    print(f"Original Demo Length = {original_demo_len}")
    print(f"Compressed Demo Length = {kv.demo_len}")
    print(f"Attention Compute Proxy = {kv_proxy} (Massively reduced from baseline!)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrieval_split", type=str, default="test")
    parser.add_argument("--eval_split", type=str, default="dev")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--demo_strategy", type=str, default="first")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_mode", type=str, default="eval", choices=["eval", "train", "train_eval"])

    # Sidecar/training options
    parser.add_argument("--num_virtual_tokens", type=int, default=8)
    parser.add_argument("--train_A", action="store_true")
    parser.add_argument("--tbptt_chunk_size", type=int, default=2048)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means no hard step cap.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--kd_temperature", type=float, default=1.0)
    parser.add_argument("--loss_w_kl", type=float, default=1.0)
    parser.add_argument("--loss_w_mse", type=float, default=1.0)
    parser.add_argument("--loss_w_ce", type=float, default=0.5)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--compile_scan", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="default")
    parser.add_argument("--save_sidecar_path", type=str, default="")
    parser.add_argument("--load_sidecar_path", type=str, default="")
    args = parser.parse_args()

    if args.run_mode == "train":
        run_distillation_training(args)
    elif args.run_mode == "train_eval":
        run_distillation_training(args)
        if args.save_sidecar_path:
            args.load_sidecar_path = args.save_sidecar_path
        run_experiment(args)
    else:
        run_experiment(args)