"""
SSM-based KV-Cache Compression for In-Context Learning Transformers.
v5: Fix scale mismatch, better metric, thorough diagnostics.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from eval import get_model_from_run
from samplers import get_data_sampler
from tasks import get_task_sampler

sns.set_theme("notebook", "darkgrid")

# =====================================================================
# 1. Load pretrained ICL model
# =====================================================================
task_name = "linear_regression"
run_dir = "../models"
run_path = os.path.join(run_dir, task_name, "pretrained")

model, conf = get_model_from_run(run_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()
for p in model.parameters():
    p.requires_grad = False

n_dims = conf.model.n_dims
batch_size_conf = conf.training.batch_size

backbone_cfg = model._backbone.config
n_embd = backbone_cfg.n_embd
n_layer = backbone_cfg.n_layer
n_head = backbone_cfg.n_head
head_dim = n_embd // n_head

print(f"[Model] n_embd={n_embd}, n_layer={n_layer}, n_head={n_head}, head_dim={head_dim}")
print(f"[Task]  n_dims={n_dims}")


# =====================================================================
# 2. HiPPO utilities
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
# 3. Multi-State SSM per layer group — with KV scale normalization
# =====================================================================

class MultiStateGroupSSM(nn.Module):
    def __init__(self, input_dim, ssm_dim, num_layers_in_group, n_head,
                 head_dim, num_virtual_tokens, dt=1.0):
        super().__init__()
        self.ssm_dim = ssm_dim
        self.num_layers_in_group = num_layers_in_group
        self.n_head = n_head
        self.head_dim = head_dim
        self.num_virtual_tokens = num_virtual_tokens

        A_cont = _hippo_legs_matrix(ssm_dim, torch.device("cpu"), torch.float32)
        A_disc = torch.matrix_exp(dt * A_cont)
        self.register_buffer("A", A_disc)

        self.B_proj = nn.Linear(input_dim, ssm_dim)
        self._init_hippo_B(input_dim)

        per_layer_kv = 2 * n_head * head_dim
        total_per_token = num_layers_in_group * per_layer_kv
        self.token_readouts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ssm_dim, ssm_dim), nn.SiLU(),
                nn.Linear(ssm_dim, total_per_token),
            )
            for _ in range(num_virtual_tokens)
        ])

        # Learnable per-layer scale + bias for K and V separately.
        # Initialized to target real KV statistics after calibration.
        self.k_scale = nn.Parameter(torch.ones(num_layers_in_group))
        self.v_scale = nn.Parameter(torch.ones(num_layers_in_group))
        self.k_bias = nn.Parameter(torch.zeros(num_layers_in_group))
        self.v_bias = nn.Parameter(torch.zeros(num_layers_in_group))

        # Initialize readout last layer with small weights
        for readout in self.token_readouts:
            nn.init.normal_(readout[-1].weight, std=0.01)
            nn.init.zeros_(readout[-1].bias)

    def _init_hippo_B(self, input_dim):
        with torch.no_grad():
            nn.init.normal_(self.B_proj.weight, std=1.0 / math.sqrt(input_dim))
            b_vec = _hippo_legs_input_vector(
                self.ssm_dim, self.B_proj.weight.device, self.B_proj.weight.dtype
            )
            self.B_proj.weight.mul_(b_vec.unsqueeze(1))
            nn.init.zeros_(self.B_proj.bias)

    def forward(self, embeddings):
        bsz, T, _ = embeddings.shape
        device, dtype = embeddings.device, embeddings.dtype
        D = self.ssm_dim
        A = self.A.to(device=device, dtype=dtype)
        u_all = self.B_proj(embeddings)

        checkpoints = [
            int(round((i + 1) * T / self.num_virtual_tokens)) - 1
            for i in range(self.num_virtual_tokens)
        ]

        state = torch.zeros(bsz, D, device=device, dtype=dtype)
        saved_states = []
        cp_idx = 0
        for t in range(T):
            state = state @ A.T + u_all[:, t, :]
            if cp_idx < len(checkpoints) and t == checkpoints[cp_idx]:
                saved_states.append(state.clone())
                cp_idx += 1
        while len(saved_states) < self.num_virtual_tokens:
            saved_states.append(state.clone())

        all_K = [[] for _ in range(self.num_layers_in_group)]
        all_V = [[] for _ in range(self.num_layers_in_group)]
        for vi, s in enumerate(saved_states):
            flat = self.token_readouts[vi](s)
            r = flat.view(bsz, self.num_layers_in_group, 2, self.n_head, self.head_dim)
            for li in range(self.num_layers_in_group):
                # Apply per-layer scale and bias
                k_raw = r[:, li, 0, :, :]  # [B, n_head, head_dim]
                v_raw = r[:, li, 1, :, :]
                k_out = k_raw * self.k_scale[li] + self.k_bias[li]
                v_out = v_raw * self.v_scale[li] + self.v_bias[li]
                all_K[li].append(k_out)
                all_V[li].append(v_out)

        kv_list = []
        for li in range(self.num_layers_in_group):
            K = torch.stack(all_K[li], dim=2)
            V = torch.stack(all_V[li], dim=2)
            kv_list.append((K, V))
        return kv_list


class ICL_SSM_Sidecar(nn.Module):
    def __init__(self, n_embd, n_layer, n_head, head_dim,
                 ssm_dim=512, num_virtual_tokens=16, num_groups=4, dt=1.0):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_dim = head_dim
        self.num_virtual_tokens = num_virtual_tokens
        self.num_groups = num_groups

        base, rem = divmod(n_layer, num_groups)
        self.group_ranges = []
        s = 0
        for g in range(num_groups):
            e = s + base + (1 if g < rem else 0)
            self.group_ranges.append((s, e))
            s = e

        self.groups = nn.ModuleList([
            MultiStateGroupSSM(
                input_dim=n_embd, ssm_dim=ssm_dim,
                num_layers_in_group=e - s,
                n_head=n_head, head_dim=head_dim,
                num_virtual_tokens=num_virtual_tokens, dt=dt,
            )
            for s, e in self.group_ranges
        ])

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        info = ", ".join(f"G{g}:[{s},{e})" for g, (s, e) in enumerate(self.group_ranges))
        print(f"[Sidecar] {num_groups} groups, ssm_dim={ssm_dim}, "
              f"vtokens={num_virtual_tokens} | {info}")
        print(f"[Sidecar] trainable params: {total_params:,}")

    def forward(self, embeddings):
        all_kv = []
        for group in self.groups:
            all_kv.extend(group(embeddings))
        return all_kv

    def calibrate_scales(self, model, data_sampler, n_dims, device,
                         k_demos=30, n_pool=200, num_calibration=20):
        """
        Run a few real examples, measure the mean/std of real KV per layer,
        and set scale/bias so that the virtual KV output has matching statistics.
        """
        print("[Calibration] Measuring real KV statistics...")
        real_k_stats = {li: {"sum": 0.0, "sq_sum": 0.0, "n": 0} for li in range(self.n_layer)}
        real_v_stats = {li: {"sum": 0.0, "sq_sum": 0.0, "n": 0} for li in range(self.n_layer)}

        with torch.no_grad():
            for _ in range(num_calibration):
                ep = generate_episode(data_sampler, n_dims, n_pool, 5, 5, device=device)
                idx = np.random.choice(n_pool, size=k_demos, replace=False)
                embeds = _build_demo_embeds(model, ep["pool_xs"][idx], ep["pool_ys"][idx])
                out = model._backbone(inputs_embeds=embeds, use_cache=True)
                for li in range(self.n_layer):
                    rk, rv = out.past_key_values[li]
                    real_k_stats[li]["sum"] += rk.mean().item()
                    real_k_stats[li]["sq_sum"] += rk.std().item()
                    real_k_stats[li]["n"] += 1
                    real_v_stats[li]["sum"] += rv.mean().item()
                    real_v_stats[li]["sq_sum"] += rv.std().item()
                    real_v_stats[li]["n"] += 1

        # Set scales to match
        for g_idx, (g_start, g_end) in enumerate(self.group_ranges):
            group = self.groups[g_idx]
            for local_li in range(g_end - g_start):
                global_li = g_start + local_li
                n = real_k_stats[global_li]["n"]
                target_k_std = real_k_stats[global_li]["sq_sum"] / n
                target_v_std = real_v_stats[global_li]["sq_sum"] / n
                target_k_mean = real_k_stats[global_li]["sum"] / n
                target_v_mean = real_v_stats[global_li]["sum"] / n

                group.k_scale.data[local_li] = target_k_std
                group.v_scale.data[local_li] = target_v_std
                group.k_bias.data[local_li] = target_k_mean
                group.v_bias.data[local_li] = target_v_mean

                print(f"  Layer {global_li}: K scale={target_k_std:.3f} bias={target_k_mean:.4f}, "
                      f"V scale={target_v_std:.3f} bias={target_v_mean:.4f}")


# =====================================================================
# 4. Data
# =====================================================================

def generate_betas(b_size, n_dims, n_classes=5, alpha=1.0, device="cpu"):
    Q, _ = torch.linalg.qr(torch.randn(n_dims, n_classes))
    anchors = Q[:, :n_classes].T.to(device)
    beta = torch.randn(b_size, n_dims, 1, device=device)
    for i in range(b_size):
        cls = torch.randint(0, n_classes, (1,))
        beta[i] = (1 - alpha) * beta[i] + alpha * anchors[cls].unsqueeze(-1)
    return beta


def generate_episode(data_sampler, n_dims, n_pool, n_val, n_test,
                     n_classes=5, noise=0.3, device="cpu"):
    beta = generate_betas(1, n_dims, n_classes, device=device)
    n_total = n_pool + n_val + n_test
    xs_all = data_sampler.sample_xs(n_total, 1).to(device)
    ys_all = (xs_all @ beta).squeeze(-1)
    ys_all += noise * torch.randn_like(ys_all)
    xs_all = xs_all.squeeze(0)
    ys_all = ys_all.squeeze(0)
    return {
        "beta": beta.squeeze(0),
        "pool_xs": xs_all[:n_pool],
        "pool_ys": ys_all[:n_pool],
        "val_xs": xs_all[n_pool:n_pool + n_val],
        "val_ys": ys_all[n_pool:n_pool + n_val],
        "test_xs": xs_all[n_pool + n_val:],
        "test_ys": ys_all[n_pool + n_val:],
    }


def generate_episodes(data_sampler, n_dims, num_episodes, n_pool, n_val, n_test,
                      n_classes=5, noise=0.3, device="cpu", seed=None):
    if seed is not None:
        rng_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        np_state = np.random.get_state()
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    episodes = [
        generate_episode(data_sampler, n_dims, n_pool, n_val, n_test,
                         n_classes, noise, device)
        for _ in range(num_episodes)
    ]
    if seed is not None:
        torch.random.set_rng_state(rng_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
        np.random.set_state(np_state)
    return episodes


# =====================================================================
# 5. Inference helpers
# =====================================================================

def _expand_past(past_kv, bsz):
    return tuple(
        (k.expand(bsz, -1, -1, -1).contiguous(),
         v.expand(bsz, -1, -1, -1).contiguous())
        for k, v in past_kv
    )


def _merge_virtual_and_sink(virtual_kv_list, sink_past, n_layer):
    merged = []
    for li in range(n_layer):
        vk, vv = virtual_kv_list[li]
        if sink_past is not None:
            sk, sv = sink_past[li]
            mk = torch.cat([sk.detach(), vk], dim=2)
            mv = torch.cat([sv.detach(), vv], dim=2)
        else:
            mk, mv = vk, vv
        merged.append((mk, mv))
    return tuple(merged)


def _build_demo_embeds(model, demo_xs, demo_ys):
    xs = demo_xs.unsqueeze(0)
    ys = demo_ys.unsqueeze(0)
    zs = model._combine(xs, ys)
    return model._read_in(zs)


def _predict_batch_queries(model, past_kv, query_xs):
    """
    Predict y for each query point independently.
    Each query is [x_q, y_dummy=0] → 2 tokens, with past_kv from demos.
    Prediction is read at the x-token position (index 0).
    """
    n_q = query_xs.shape[0]
    xs_q = query_xs.unsqueeze(1)                        # [n_q, 1, n_dims]
    ys_q = torch.zeros(n_q, 1, device=query_xs.device)  # [n_q, 1]
    zs_q = model._combine(xs_q, ys_q)
    embeds_q = model._read_in(zs_q)                     # [n_q, 2, n_embd]
    expanded_past = _expand_past(past_kv, n_q)
    out = model._backbone(
        inputs_embeds=embeds_q,
        past_key_values=expanded_past,
        use_cache=False,
    )
    hidden = out.last_hidden_state                       # [n_q, 2, n_embd]
    pred = model._read_out(hidden)                       # [n_q, 2, 1]
    y_preds = pred[:, 0, 0]                              # [n_q]
    last_hs = hidden[:, -1, :]                           # [n_q, n_embd]
    return y_preds, last_hs


# =====================================================================
# 6. Metrics
# =====================================================================

def _nmse(pred, target):
    """
    Normalized MSE = MSE(pred, target) / Var(target).
    0 = perfect, 1 = predicting the mean, >1 = worse than mean.
    Robust to scale, unlike per-element RelSqE.
    """
    mse = F.mse_loss(pred, target).item()
    var = target.var().item()
    return mse / max(var, 1e-8)


def _mse(pred, target):
    return F.mse_loss(pred, target).item()


# =====================================================================
# 7. Diagnostics
# =====================================================================

def run_diagnostics(model, compressor, data_sampler, n_dims, device,
                    k_demos, n_pool, n_val, n_test, n_classes, noise):
    print(f"\n{'='*60}")
    print(f"  Running diagnostics...")
    print(f"{'='*60}\n")

    ep = generate_episode(data_sampler, n_dims, n_pool, n_val, n_test,
                          n_classes, noise, device)
    demo_idx = np.random.choice(n_pool, size=k_demos, replace=False)
    demo_xs = ep["pool_xs"][demo_idx]
    demo_ys = ep["pool_ys"][demo_idx]

    # --- Diagnostic 1: Sanity check — real KV via split vs full sequence ---
    with torch.no_grad():
        demo_embeds = _build_demo_embeds(model, demo_xs, demo_ys)
        real_out = model._backbone(inputs_embeds=demo_embeds, use_cache=True)
        real_past = real_out.past_key_values

        test_xs = ep["test_xs"]
        test_ys = ep["test_ys"]

        # Method A: our _predict_batch_queries (split, independent queries)
        preds_split, _ = _predict_batch_queries(model, real_past, test_xs)
        mse_split = _mse(preds_split, test_ys)
        nmse_split = _nmse(preds_split, test_ys)

        # Method B: standard model forward (all points together)
        n_demo = demo_xs.shape[0]
        all_xs = torch.cat([demo_xs, test_xs], dim=0).unsqueeze(0)  # [1, n_demo+n_test, n_dims]
        all_ys = torch.cat([demo_ys, test_ys], dim=0).unsqueeze(0)  # [1, n_demo+n_test]
        pred_full = model(all_xs, all_ys)                            # [1, n_demo+n_test]
        preds_full_test = pred_full[0, n_demo:]                      # [n_test]
        mse_full = _mse(preds_full_test, test_ys)
        nmse_full = _nmse(preds_full_test, test_ys)

    print(f"  [Sanity] Full-sequence forward:  MSE={mse_full:.6f}, NMSE={nmse_full:.4f}")
    print(f"  [Sanity] Split (real KV+cache):  MSE={mse_split:.6f}, NMSE={nmse_split:.4f}")
    if nmse_split > 2 * nmse_full:
        print(f"  *** WARNING: Split method is much worse than full-sequence! ***")
        print(f"  This suggests _predict_batch_queries has an issue.")
    else:
        print(f"  Split and full-sequence agree reasonably.")

    # Also check: predictions should be correlated
    corr = torch.corrcoef(torch.stack([preds_split, preds_full_test]))[0, 1].item()
    print(f"  [Sanity] Correlation(split, full) = {corr:.4f}")

    # --- Diagnostic 2: KV scale ---
    with torch.no_grad():
        virtual_kv = compressor(demo_embeds)
    print(f"\n  [Scale] Real KV vs Virtual KV (after calibration):")
    for li in [0, n_layer // 2, n_layer - 1]:
        rk, rv = real_past[li]
        vk, vv = virtual_kv[li]
        print(f"    Layer {li}:")
        print(f"      Real  K: std={rk.std():.4f}  Virtual K: std={vk.std():.4f}")
        print(f"      Real  V: std={rv.std():.4f}  Virtual V: std={vv.std():.4f}")

    # --- Diagnostic 3: gradient flow ---
    print(f"\n  [Grad] Checking gradient flow...")
    compressor.train()
    compressor.zero_grad(set_to_none=True)

    ep2 = generate_episode(data_sampler, n_dims, n_pool, n_val, n_test,
                           n_classes, noise, device)
    idx2 = np.random.choice(n_pool, size=k_demos, replace=False)
    with torch.no_grad():
        embeds2 = _build_demo_embeds(model, ep2["pool_xs"][idx2], ep2["pool_ys"][idx2])
        teacher_out2 = model._backbone(inputs_embeds=embeds2, use_cache=True)
        teacher_past2 = teacher_out2.past_key_values

    virtual_kv2 = compressor(embeds2.detach())
    student_past2 = _merge_virtual_and_sink(virtual_kv2, None, compressor.n_layer)

    q_xs = ep2["val_xs"][:3]
    with torch.no_grad():
        t_p, t_h = _predict_batch_queries(model, teacher_past2, q_xs)
    s_p, s_h = _predict_batch_queries(model, student_past2, q_xs)

    print(f"    virtual K requires_grad = {virtual_kv2[0][0].requires_grad}")
    print(f"    student pred requires_grad = {s_p.requires_grad}")

    test_loss = F.mse_loss(s_p, t_p.detach())
    test_loss.backward()

    has_grad, no_grad, total_norm = 0, 0, 0.0
    for name, p in compressor.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad += 1
            total_norm += p.grad.norm().item()
        else:
            no_grad += 1
    print(f"    params WITH grad: {has_grad}, WITHOUT: {no_grad}")
    print(f"    total grad norm: {total_norm:.2f}")

    compressor.zero_grad(set_to_none=True)

    if has_grad == 0:
        print("\n  *** CRITICAL: No gradients! ***\n")
        return False

    print(f"\n  Gradient flow OK.\n{'='*60}\n")
    return True


# =====================================================================
# 8. Training
# =====================================================================

def _kv_matching_loss(virtual_kv, teacher_past, n_layer, num_virtual_tokens):
    """
    Direct KV matching: for each layer, pool real KV into num_virtual_tokens
    chunks, match virtual KV to chunk means via cosine + MSE.
    Provides strong, direct gradient signal without going through the backbone.
    """
    loss = torch.tensor(0.0, device=virtual_kv[0][0].device)
    n_terms = 0
    for li in range(n_layer):
        tk, tv = teacher_past[li]  # [1, n_head, T, head_dim]
        vk, vv = virtual_kv[li]   # [1, n_head, vtok, head_dim]
        T_full = tk.shape[2]
        # Evenly partition all T tokens into exactly num_virtual_tokens chunks
        # E.g. T=60, vtok=16 → chunks of size [4,4,4,4, 4,4,4,4, 4,4,4,4, 3,3,3,3]
        for vi in range(num_virtual_tokens):
            s = vi * T_full // num_virtual_tokens
            e = (vi + 1) * T_full // num_virtual_tokens
            if e <= s:
                e = s + 1  # at least 1 token
            tk_pool = tk[:, :, s:e, :].mean(dim=2)  # [1, n_head, head_dim]
            tv_pool = tv[:, :, s:e, :].mean(dim=2)
            vk_i = vk[:, :, vi, :]
            vv_i = vv[:, :, vi, :]
            # Cosine + MSE hybrid for both direction and magnitude
            loss = loss + (1 - F.cosine_similarity(
                vk_i.flatten(1), tk_pool.flatten(1), dim=-1).mean())
            loss = loss + (1 - F.cosine_similarity(
                vv_i.flatten(1), tv_pool.flatten(1), dim=-1).mean())
            loss = loss + F.mse_loss(vk_i, tk_pool)
            loss = loss + F.mse_loss(vv_i, tv_pool)
            n_terms += 4
    return loss / max(n_terms, 1)


def train_compressor(
    model, compressor, data_sampler, n_dims, device,
    k_demos=30, n_pool=200, n_val=20, n_test=20,
    n_classes=5, noise=0.3,
    num_steps=3000, lr=1e-3, log_every=50,
    sink_tokens=0,
    kv_weight=1.0, logit_weight=0.5, hidden_weight=0.1,
    num_query_per_step=8,
    warmup_steps=200,
    phase2_start=1000,
    phase2_ramp=500,
):
    """
    Two-phase training:
      Phase 1 (steps 1..phase2_start): KV matching dominates.
        Direct gradient from virtual KV → compressor. Fast convergence.
      Phase 2 (steps phase2_start..num_steps): Logit+hidden loss added.
        Fine-tune with end-to-end signal through backbone.
    """
    compressor.train()
    optimizer = torch.optim.AdamW(compressor.parameters(), lr=lr, weight_decay=0.01)

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    losses_log = {"total": [], "kv": [], "logit": [], "hidden": [], "grad_norm": []}

    for step in range(1, num_steps + 1):
        ep = generate_episode(data_sampler, n_dims, n_pool, n_val, n_test,
                              n_classes, noise, device)
        pool_size = ep["pool_xs"].shape[0]
        demo_indices = np.random.choice(pool_size, size=min(k_demos, pool_size), replace=False)
        demo_xs = ep["pool_xs"][demo_indices]
        demo_ys = ep["pool_ys"][demo_indices]

        with torch.no_grad():
            demo_embeds = _build_demo_embeds(model, demo_xs, demo_ys)
            teacher_out = model._backbone(inputs_embeds=demo_embeds, use_cache=True)
            teacher_past = teacher_out.past_key_values

        virtual_kv = compressor(demo_embeds.detach())

        # ── Smooth transition: KV loss decays, logit/hidden loss ramps up ──
        loss_kv = _kv_matching_loss(
            virtual_kv, teacher_past, compressor.n_layer,
            compressor.num_virtual_tokens,
        )

        loss_logit = torch.tensor(0.0, device=device)
        loss_hidden = torch.tensor(0.0, device=device)

        if step >= phase2_start:
            # Smooth blend: KV weight decays linearly, logit/hidden ramps up
            blend = min(1.0, (step - phase2_start) / phase2_ramp)
            w_kv = 1.0 - blend      # 1.0 → 0.0
            w_logit = blend          # 0.0 → 1.0
            w_hidden = blend * 0.1   # 0.0 → 0.1

            sink_past = None
            if sink_tokens > 0 and sink_tokens <= demo_embeds.shape[1]:
                with torch.no_grad():
                    sink_out = model._backbone(
                        inputs_embeds=demo_embeds[:, :sink_tokens, :], use_cache=True
                    )
                    sink_past = sink_out.past_key_values
            student_past = _merge_virtual_and_sink(virtual_kv, sink_past, compressor.n_layer)

            n_q = min(num_query_per_step, ep["val_xs"].shape[0])
            q_indices = np.random.choice(ep["val_xs"].shape[0], size=n_q, replace=False)
            query_xs = ep["val_xs"][q_indices]

            with torch.no_grad():
                t_preds, t_hiddens = _predict_batch_queries(model, teacher_past, query_xs)
            s_preds, s_hiddens = _predict_batch_queries(model, student_past, query_xs)

            loss_logit = F.mse_loss(s_preds, t_preds.detach())
            loss_hidden = 1.0 - F.cosine_similarity(
                s_hiddens, t_hiddens.detach(), dim=-1
            ).mean()

            loss = w_kv * loss_kv + w_logit * loss_logit + w_hidden * loss_hidden
        else:
            loss = loss_kv

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Log grad norm BEFORE clipping
        raw_norm = torch.nn.utils.clip_grad_norm_(compressor.parameters(), max_norm=100.0)

        optimizer.step()
        scheduler.step()

        losses_log["total"].append(loss.item())
        losses_log["kv"].append(loss_kv.item())
        losses_log["logit"].append(loss_logit.item() if isinstance(loss_logit, torch.Tensor) else loss_logit)
        losses_log["hidden"].append(loss_hidden.item() if isinstance(loss_hidden, torch.Tensor) else loss_hidden)
        losses_log["grad_norm"].append(float(raw_norm))

        if step % log_every == 0:
            avg = {k: np.mean(v[-log_every:]) for k, v in losses_log.items()}
            if step < phase2_start:
                phase = "P1(KV)"
            else:
                cur_blend = min(1.0, (step - phase2_start) / phase2_ramp)
                phase = f"P2(b={cur_blend:.2f})"
            print(f"[step {step:>4d} {phase}] loss={avg['total']:.4f}  "
                  f"kv={avg['kv']:.4f}  logit={avg['logit']:.4f}  "
                  f"hidden={avg['hidden']:.4f}  "
                  f"gnorm={avg['grad_norm']:.1f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    compressor.eval()
    return losses_log


# =====================================================================
# 9. Evaluation
# =====================================================================

@torch.no_grad()
def evaluate_compression(model, compressor, test_episodes, k_demos, device,
                         sink_tokens=0):
    compressor.eval()
    results = {
        "mse_full_vs_gt": [], "mse_ssm_vs_gt": [], "mse_ssm_vs_full": [],
        "nmse_full_vs_gt": [], "nmse_ssm_vs_gt": [], "nmse_ssm_vs_full": [],
        "hidden_cosine": [], "pred_correlation": [],
    }

    for ep in tqdm(test_episodes, desc="eval"):
        pool_size = ep["pool_xs"].shape[0]
        demo_indices = np.random.choice(pool_size, size=min(k_demos, pool_size), replace=False)
        demo_xs = ep["pool_xs"][demo_indices]
        demo_ys = ep["pool_ys"][demo_indices]
        demo_embeds = _build_demo_embeds(model, demo_xs, demo_ys)

        teacher_out = model._backbone(inputs_embeds=demo_embeds, use_cache=True)
        teacher_past = teacher_out.past_key_values

        virtual_kv = compressor(demo_embeds)
        sink_past = None
        if sink_tokens > 0 and sink_tokens <= demo_embeds.shape[1]:
            sink_out = model._backbone(
                inputs_embeds=demo_embeds[:, :sink_tokens, :], use_cache=True
            )
            sink_past = sink_out.past_key_values
        student_past = _merge_virtual_and_sink(virtual_kv, sink_past, compressor.n_layer)

        test_xs = ep["test_xs"]
        test_ys = ep["test_ys"]
        t_preds, t_hs = _predict_batch_queries(model, teacher_past, test_xs)
        s_preds, s_hs = _predict_batch_queries(model, student_past, test_xs)

        results["mse_full_vs_gt"].append(_mse(t_preds, test_ys))
        results["mse_ssm_vs_gt"].append(_mse(s_preds, test_ys))
        results["mse_ssm_vs_full"].append(_mse(s_preds, t_preds))
        results["nmse_full_vs_gt"].append(_nmse(t_preds, test_ys))
        results["nmse_ssm_vs_gt"].append(_nmse(s_preds, test_ys))
        results["nmse_ssm_vs_full"].append(_nmse(s_preds, t_preds))
        results["hidden_cosine"].append(
            F.cosine_similarity(s_hs, t_hs, dim=-1).mean().item()
        )
        corr = torch.corrcoef(torch.stack([s_preds, t_preds]))[0, 1].item()
        results["pred_correlation"].append(corr if not math.isnan(corr) else 0.0)

    return results


@torch.no_grad()
def evaluate_full_icl_curve(model, test_episodes, device, max_demos=None):
    if max_demos is None:
        max_demos = test_episodes[0]["pool_xs"].shape[0]
    k_values = list(range(1, max_demos + 1))
    loss_curves = []
    for ep in tqdm(test_episodes, desc="ref-curve"):
        pool_xs, pool_ys = ep["pool_xs"], ep["pool_ys"]
        test_xs, test_ys = ep["test_xs"], ep["test_ys"]
        ep_losses = []
        for k in k_values:
            demo_embeds = _build_demo_embeds(model, pool_xs[:k], pool_ys[:k])
            out = model._backbone(inputs_embeds=demo_embeds, use_cache=True)
            preds, _ = _predict_batch_queries(model, out.past_key_values, test_xs)
            ep_losses.append(_nmse(preds, test_ys))
        loss_curves.append(ep_losses)
    return np.array(k_values), np.array(loss_curves)


# =====================================================================
# 10. Main
# =====================================================================

if __name__ == "__main__":
    SSM_DIM = 512
    NUM_VIRTUAL_TOKENS = 16
    NUM_GROUPS = 4
    SINK_TOKENS = 2  # number of backbone tokens to keep as real KV (not demo pairs)
    NUM_TRAIN_STEPS = 3000
    LR = 1e-3
    WARMUP = 200
    PHASE2_START = 1000
    PHASE2_RAMP = 500    # blend over 500 steps: KV weight 1→0, logit weight 0→1

    K_DEMOS = 30
    N_POOL = 200
    N_VAL = 20
    N_TEST = 20
    N_CLASSES = 5
    NOISE = 0.3

    NUM_TEST_EPISODES = 50
    NUM_QUERY_PER_STEP = 8

    data_sampler = get_data_sampler(conf.training.data, n_dims)

    # ── Fixed test episodes ──
    test_episodes = generate_episodes(
        data_sampler, n_dims, NUM_TEST_EPISODES, N_POOL, N_VAL, N_TEST,
        N_CLASSES, NOISE, device, seed=42,
    )
    print(f"[Data] {NUM_TEST_EPISODES} test episodes (seed=42)")

    # ── Build compressor ──
    compressor = ICL_SSM_Sidecar(
        n_embd=n_embd, n_layer=n_layer, n_head=n_head, head_dim=head_dim,
        ssm_dim=SSM_DIM, num_virtual_tokens=NUM_VIRTUAL_TOKENS,
        num_groups=NUM_GROUPS,
    ).to(device)

    # ── Calibrate output scales to match real KV statistics ──
    compressor.calibrate_scales(model, data_sampler, n_dims, device, k_demos=K_DEMOS)

    n_demo_tok = 2 * K_DEMOS
    n_compressed = NUM_VIRTUAL_TOKENS + SINK_TOKENS
    print(f"\n  {K_DEMOS} demos = {n_demo_tok} tokens → {n_compressed} compressed "
          f"({n_demo_tok / n_compressed:.1f}x)")
    print(f"  Sink: first {SINK_TOKENS} backbone tokens kept as real KV")

    # ── Diagnostics ──
    grad_ok = run_diagnostics(
        model, compressor, data_sampler, n_dims, device,
        K_DEMOS, N_POOL, N_VAL, N_TEST, N_CLASSES, NOISE,
    )
    if not grad_ok:
        print("Gradients blocked! Stopping.")
        exit(1)

    # ── Train ──
    train_log = train_compressor(
        model, compressor, data_sampler, n_dims, device,
        k_demos=K_DEMOS, n_pool=N_POOL, n_val=N_VAL, n_test=N_TEST,
        n_classes=N_CLASSES, noise=NOISE,
        num_steps=NUM_TRAIN_STEPS, lr=LR,
        sink_tokens=SINK_TOKENS,
        kv_weight=1.0, logit_weight=0.5, hidden_weight=0.1,
        num_query_per_step=NUM_QUERY_PER_STEP,
        warmup_steps=WARMUP,
        phase2_start=PHASE2_START,
        phase2_ramp=PHASE2_RAMP,
    )

    # ── Evaluate ──
    print(f"\n{'='*60}")
    print(f"  Evaluation on {NUM_TEST_EPISODES} test episodes")
    print(f"{'='*60}\n")

    results = evaluate_compression(
        model, compressor, test_episodes, K_DEMOS, device,
        sink_tokens=SINK_TOKENS,
    )

    # ── Reference curve ──
    print("\nComputing reference ICL curve...")
    ref_k, ref_curves = evaluate_full_icl_curve(
        model, test_episodes[:10], device,
        max_demos=min(K_DEMOS + 10, N_POOL),
    )

    # ── Print ──
    def _s(arr):
        return f"{np.mean(arr):.6f} ± {np.std(arr):.6f}"

    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    print(f"  --- MSE (absolute) ---")
    print(f"  MSE(full-KV → GT)     = {_s(results['mse_full_vs_gt'])}")
    print(f"  MSE(SSM → GT)         = {_s(results['mse_ssm_vs_gt'])}")
    print(f"  MSE(SSM → full-KV)    = {_s(results['mse_ssm_vs_full'])}")
    print(f"  --- NMSE (normalized by target variance) ---")
    print(f"  NMSE(full-KV → GT)    = {_s(results['nmse_full_vs_gt'])}")
    print(f"  NMSE(SSM → GT)        = {_s(results['nmse_ssm_vs_gt'])}")
    print(f"  NMSE(SSM → full-KV)   = {_s(results['nmse_ssm_vs_full'])}  ← compression error")
    print(f"  --- Fidelity ---")
    print(f"  Hidden cosine          = {_s(results['hidden_cosine'])}")
    print(f"  Pred correlation       = {_s(results['pred_correlation'])}")

    # ── Plot 1: training curves ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, key, title in zip(axes.flat,
                               ["total", "kv", "logit", "grad_norm"],
                               ["Total Loss", "KV Matching", "Logit MSE", "Grad Norm"]):
        vals = train_log[key]
        w = min(50, len(vals))
        sm = np.convolve(vals, np.ones(w) / w, mode="valid")
        ax.plot(sm, lw=1.5)
        if key == "grad_norm":
            ax.set_yscale("log")
        ax.set_title(title); ax.set_xlabel("Step"); ax.set_ylabel("Value")
        ax.axvline(PHASE2_START, color="red", ls="--", alpha=0.5, label="Phase 2 start")
        ax.axvline(PHASE2_START + PHASE2_RAMP, color="orange", ls="--", alpha=0.5, label="Ramp end")
        ax.legend()
    plt.tight_layout()
    plt.savefig("ssm_v5_training.png", dpi=150); plt.show()

    # ── Plot 2: ICL curve + SSM point ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ref_mean = ref_curves.mean(0); ref_std = ref_curves.std(0)
    ax.plot(ref_k, ref_mean, lw=2, color="steelblue", label="Full-KV (reference)")
    ax.fill_between(ref_k, ref_mean - ref_std, ref_mean + ref_std,
                     alpha=0.2, color="steelblue")

    ssm_v = np.mean(results["nmse_ssm_vs_gt"])
    ssm_s = np.std(results["nmse_ssm_vs_gt"])
    ax.errorbar(K_DEMOS, ssm_v, yerr=ssm_s, fmt='o', ms=10,
                color="coral", capsize=5,
                label=f"SSM ({NUM_VIRTUAL_TOKENS}vtok+{2*SINK_TOKENS}sink)")

    fk_v = np.mean(results["nmse_full_vs_gt"])
    fk_s = np.std(results["nmse_full_vs_gt"])
    ax.errorbar(K_DEMOS, fk_v, yerr=fk_s, fmt='s', ms=10,
                color="green", capsize=5, label=f"Full-KV ({K_DEMOS} demos)")

    ax.set_xlabel("# demonstrations"); ax.set_ylabel("NMSE vs GT")
    ax.set_title("Full-KV vs SSM Compressed"); ax.legend()
    plt.tight_layout()
    plt.savefig("ssm_v5_icl_curve.png", dpi=150); plt.show()

    # ── Plot 3: error bars + fidelity ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    keys = ["nmse_full_vs_gt", "nmse_ssm_vs_gt", "nmse_ssm_vs_full"]
    labs = ["Full→GT", "SSM→GT", "SSM→Full"]
    ax.bar(labs, [np.mean(results[k]) for k in keys],
           yerr=[np.std(results[k]) for k in keys],
           color=["steelblue", "coral", "mediumpurple"], alpha=0.8, capsize=5)
    ax.set_ylabel("NMSE"); ax.set_title("Error (NMSE)")

    ax = axes[1]
    ax.hist(results["hidden_cosine"], bins=20, color="teal", alpha=0.7, edgecolor="white")
    ax.axvline(np.mean(results["hidden_cosine"]), color="red", ls="--", lw=2)
    ax.set_xlabel("Cosine Sim"); ax.set_title("Hidden State Alignment")

    ax = axes[2]
    ax.hist(results["pred_correlation"], bins=20, color="orange", alpha=0.7, edgecolor="white")
    ax.axvline(np.mean(results["pred_correlation"]), color="red", ls="--", lw=2)
    ax.set_xlabel("Pearson r"); ax.set_title("Prediction Correlation")

    plt.tight_layout()
    plt.savefig("ssm_v5_error.png", dpi=150); plt.show()

    # ── Save ──
    np.savez("ssm_v5_results.npz", **{
        k: np.array(v) for k, v in results.items()
    })
    torch.save(compressor.state_dict(), "ssm_v5_compressor.pt")
    print("\nSaved: ssm_v5_results.npz, ssm_v5_compressor.pt")