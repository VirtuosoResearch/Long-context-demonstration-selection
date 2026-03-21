"""
Empirical verification for the first two checks in Proposition 3.1.

Check 1: Spectral tail energy E_N^{(g)} = sum_{j>N} sigma_j^2(K^{(g)})
  -> Compute singular value statistics per layer group, across multiple prefix lengths.

Check 2: HiPPO projection residual ||(I - P_N) u_i^{(g)}||^2
  -> Compute residual vs N on log-log scale and fit slope.

Usage:
  python verify_kv_error.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --dataset sst2 \
    --k 50 --num_groups 4 \
    --prefix_lengths 256,512,1024 \
    --max_hippo_dim 128 \
    --save_dir ./prop31_results
"""

import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer

from utils.data import load_data


# =====================================================================
# HiPPO-LegS basis construction
# =====================================================================

def build_hippo_basis(T, N, device="cpu"):
    """
    Build H_N by evaluating the first N HiPPO-LegS basis functions
    on the discrete token grid t = 0, ..., T-1.

    phi_n(t) = sqrt(2n+1) * P_n(2t/(T-1) - 1),  n = 0, ..., N-1

    Returns: Q of shape [T, N] with orthonormal columns spanning H_N.
    """
    if N > T:
        raise ValueError(f"N={N} cannot exceed T={T}.")

    x = torch.linspace(-1.0, 1.0, T, device=device, dtype=torch.float64)

    basis = torch.zeros(T, N, device=device, dtype=torch.float64)
    if N >= 1:
        basis[:, 0] = 1.0
    if N >= 2:
        basis[:, 1] = x
    for n in range(2, N):
        basis[:, n] = ((2 * n - 1) * x * basis[:, n - 1]
                       - (n - 1) * basis[:, n - 2]) / n

    scales = torch.sqrt(2.0 * torch.arange(N, device=device, dtype=torch.float64) + 1.0)
    basis = basis * scales.unsqueeze(0)

    Q, _ = torch.linalg.qr(basis)
    return Q


# =====================================================================
# Extract exact KV cache
# =====================================================================

@torch.no_grad()
def extract_kv_cache(model, input_ids):
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    kv_list = []
    if hasattr(past, "key_cache"):
        for l in range(len(past.key_cache)):
            kv_list.append((past.key_cache[l].cpu(), past.value_cache[l].cpu()))
    else:
        for k, v in past:
            kv_list.append((k.cpu(), v.cpu()))
    return kv_list


def partition_into_groups(kv_list, num_groups):
    num_layers = len(kv_list)
    if num_groups <= 0 or num_groups > num_layers:
        raise ValueError(f"num_groups={num_groups} invalid for {num_layers} layers.")
    base, rem = divmod(num_layers, num_groups)
    groups, idx = [], 0
    for g in range(num_groups):
        size = base + (1 if g < rem else 0)
        layers = []
        for _ in range(size):
            K_l = kv_list[idx][0]
            T_len = K_l.shape[2]
            K_flat = K_l[0].permute(1, 0, 2).reshape(T_len, -1)
            layers.append(K_flat)
            idx += 1
        groups.append(torch.cat(layers, dim=1).to(torch.float64))
    return groups


# =====================================================================
# Check 1: Singular value statistics
# =====================================================================

def compute_spectral_tail(groups):
    group_stats = []
    for g, K_g in enumerate(groups):
        S_sq = torch.linalg.svdvals(K_g).numpy() ** 2
        E_N = S_sq.sum() - np.cumsum(S_sq)
        group_stats.append({
            "group": g,
            "singular_values_sq": S_sq.tolist(),
            "tail_energy": E_N.tolist(),
        })
    return group_stats


# =====================================================================
# Check 2: HiPPO projection residual (fixed K singular vectors)
# =====================================================================

def compute_hippo_residual(groups, max_N, device="cpu"):
    """
    Fix the set of tracked singular vectors (top-K, K=max_N),
    then sweep subspace dimension N = 4, 8, ..., max_N.

    (a) max_{i<=K} ||(I - P_N) u_i||^2
    (b) sum_{i=1}^{K} sigma_i^2 ||(I - P_N) u_i||^2
    """
    T = groups[0].shape[0]
    K_fixed = max_N
    print(f"Building HiPPO basis: T={T}, max_N={max_N} ...")
    Q_full = build_hippo_basis(T, max_N, device=device)

    N_values = sorted(set(n for n in range(4, max_N + 1, 4)))
    if max_N not in N_values:
        N_values.append(max_N)
    N_values = sorted(N_values)

    groups_out = []

    for g, K_g in enumerate(groups):
        print(f"  Group {g}: SVD on [{K_g.shape[0]} x {K_g.shape[1]}] ...")
        U, S, _ = torch.linalg.svd(K_g, full_matrices=False)
        S_sq = (S ** 2).to(torch.float64)

        n_track = min(K_fixed, U.shape[1])
        U_track = U[:, :n_track]
        S_track = S_sq[:n_track]

        max_res_list, wt_res_list = [], []

        for N in N_values:
            Q_N = Q_full[:, :N]
            coeffs = Q_N.T @ U_track
            residual = U_track - Q_N @ coeffs
            res_sq = (residual ** 2).sum(dim=0)

            max_res_list.append(res_sq.max().item())
            wt_res_list.append((S_track * res_sq).sum().item())

        N_arr = np.array(N_values, dtype=float)
        max_arr = np.array(max_res_list)
        wt_arr = np.array(wt_res_list)

        slope_max = None
        smoothness = None
        slope_wt = None

        valid = max_arr > 0
        if valid.sum() >= 3:
            c = np.polyfit(np.log(N_arr[valid]), np.log(max_arr[valid]), 1)
            slope_max = float(c[0])
            smoothness = float(-c[0] / 2.0)
            print(f"    [max]  slope={slope_max:.3f}, s≈{smoothness:.3f}")

        valid_w = wt_arr > 0
        if valid_w.sum() >= 3:
            c = np.polyfit(np.log(N_arr[valid_w]), np.log(wt_arr[valid_w]), 1)
            slope_wt = float(c[0])
            print(f"    [weighted] slope={slope_wt:.3f}")

        groups_out.append({
            "group": g,
            "n_values": N_values,
            "max_residual": max_res_list,
            "weighted_residual": wt_res_list,
            "slope_max": slope_max,
            "smoothness_estimate": smoothness,
            "slope_weighted": slope_wt,
        })

    return {
        "max_N": max_N,
        "K_fixed": K_fixed,
        "groups": groups_out,
    }


# =====================================================================
# Text / data utilities
# =====================================================================

def parse_prefix_lengths(raw, total_len):
    if raw:
        vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        vals = [min(128, total_len), min(256, total_len),
                min(512, total_len), total_len]
    vals = sorted(set(v for v in vals if 16 <= v <= total_len))
    return vals or [total_len]


def setup_tokenizer(name):
    tok = (GPT2Tokenizer.from_pretrained(name) if name.startswith("gpt2")
           else AutoTokenizer.from_pretrained(name))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_demo_text(demos, add_newlines=True):
    parts = []
    for i, dp in enumerate(demos):
        q = dp["input"] if i == 0 else ("\n" + dp["input"])
        a = ("\n" + dp["output"]) if add_newlines else dp["output"]
        parts.append(q + a)
    return "".join(parts)


# =====================================================================
# Main
# =====================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dataset", type=str, default="sst2")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_groups", type=int, default=4)
    p.add_argument("--prefix_lengths", type=str, default="")
    p.add_argument("--max_hippo_dim", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_dir", type=str, default="./prop31_results")
    p.add_argument("--results_name", type=str, default="verify_kv_results.pt")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model_name}")
    tokenizer = setup_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    data = load_data(task=None, split=args.split, k=args.k,
                     seed=args.seed, datasets=args.dataset.split(","), is_null=False)
    if not data:
        raise ValueError("No data loaded.")

    import random
    rng = random.Random(args.seed)
    demos = [data[i] for i in rng.sample(range(len(data)), min(args.k, len(data)))]
    add_nl = not args.model_name.startswith("gpt2")
    demo_text = build_demo_text(demos, add_newlines=add_nl)

    demo_ids = tokenizer(demo_text, return_tensors="pt",
                         add_special_tokens=False)["input_ids"].to(device)
    full_T = demo_ids.shape[1]
    print(f"Demo sequence length: T_full = {full_T}")
    prefix_lengths = parse_prefix_lengths(args.prefix_lengths, full_T)
    print(f"Prefix length sweep for Check 1: {prefix_lengths}")

    groups_for_term2 = None
    term2_T = None
    term1_results = []

    # ── Check 1 ──
    print("\n=== Check 1: Spectral tail analysis ===")
    for L in prefix_lengths:
        print(f"\n-- Prefix length L={L} --")
        kv_list = extract_kv_cache(model, demo_ids[:, :L])
        print(f"   {len(kv_list)} layers, K shape: {kv_list[0][0].shape}")
        groups = partition_into_groups(kv_list, args.num_groups)
        for g, K_g in enumerate(groups):
            print(f"   Group {g}: {K_g.shape}")
        term1_results.append({
            "prefix_length": L,
            "groups": compute_spectral_tail(groups),
        })
        if L == prefix_lengths[-1]:
            groups_for_term2 = groups
            term2_T = L
        del kv_list

    del model
    torch.cuda.empty_cache()

    # ── Check 2 ──
    max_N = min(args.max_hippo_dim, term2_T)
    print(f"\n=== Check 2: HiPPO projection residual (L={term2_T}, max_N={max_N}) ===")
    term2_results = compute_hippo_residual(groups_for_term2, max_N, device="cpu")

    output = {
        "meta": {
            "model_name": args.model_name,
            "dataset": args.dataset,
            "split": args.split,
            "k": args.k,
            "seed": args.seed,
            "num_groups": args.num_groups,
            "prefix_lengths": prefix_lengths,
            "max_hippo_dim": args.max_hippo_dim,
            "term2_prefix_length": term2_T,
        },
        "term1": term1_results,
        "term2": term2_results,
    }
    out_path = os.path.join(args.save_dir, args.results_name)
    torch.save(output, out_path)
    print(f"\nAll statistics saved to: {out_path}")


if __name__ == "__main__":
    main()