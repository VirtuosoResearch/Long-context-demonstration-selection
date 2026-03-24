#!/usr/bin/env python3
"""
Compute and store KV-error verification statistics for Proposition 3.1.

This script only records data. Plotting is moved to verify_kv_plot.py.

Usage:
  python verify_kv_error.py \
      --model_name Qwen/Qwen2.5-7B-Instruct \
      --dataset sst2 --k 50 --num_groups 4 --N_max 512 \
      --output_dir prop31_results
"""

import argparse
import gc
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer

from utils.data import load_data


# =====================================================================
# Helpers
# =====================================================================

def setup_tokenizer(name):
    tok = (
        GPT2Tokenizer.from_pretrained(name)
        if name.startswith("gpt2")
        else AutoTokenizer.from_pretrained(name)
    )
    if tok.padding_side == "left":
        tok.padding_side = "right"
    if tok.eos_token_id is None and tok.sep_token is not None:
        tok.eos_token = tok.sep_token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_demo_text(demos, add_newlines):
    parts = []
    for i, dp in enumerate(demos):
        q = dp["input"] if (i == 0 or not add_newlines) else ("\n" + dp["input"])
        a = ("\n" + dp["output"]) if add_newlines else dp["output"]
        parts.append(q + a)
    return "".join(parts)


def build_discrete_legendre_basis(T: int, N_max: int) -> np.ndarray:
    if T == 1:
        return np.ones((1, min(N_max, 1)), dtype=np.float64)
    x = np.linspace(-1.0, 1.0, T)
    phi = np.zeros((T, N_max), dtype=np.float64)
    phi[:, 0] = 1.0
    if N_max > 1:
        phi[:, 1] = x
    for n in range(1, N_max - 1):
        phi[:, n + 1] = ((2 * n + 1) * x * phi[:, n] - n * phi[:, n - 1]) / (n + 1)
    q, r = np.linalg.qr(phi, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q *= signs[np.newaxis, :]
    return q


def extract_grouped_caches(model, input_ids, num_groups):
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    if hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    elif hasattr(past, "key_cache"):
        past = [
            (past.key_cache[i], past.value_cache[i])
            for i in range(len(past.key_cache))
        ]

    num_layers = len(past)
    T = past[0][0].shape[2]
    base, rem = divmod(num_layers, num_groups)
    ranges, s = [], 0
    for g in range(num_groups):
        e = s + base + (1 if g < rem else 0)
        ranges.append((s, e))
        s = e

    grouped = []
    for gs, ge in ranges:
        cols = []
        for l in range(gs, ge):
            k, v = past[l]
            k = k[0].permute(1, 0, 2).reshape(T, -1)
            v = v[0].permute(1, 0, 2).reshape(T, -1)
            cols.extend([k, v])
        grouped.append(torch.cat(cols, dim=1).to(dtype=torch.float64, device="cpu"))

    del out, past
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return grouped, ranges


def infer_model_display_name(model_name: str) -> str:
    name = model_name.lower()
    if "qwen" in name:
        return "Qwen-7B"
    if "llama" in name:
        return "Llama-8B"
    return model_name.split("/")[-1]


# =====================================================================
# Analysis
# =====================================================================

def compute_per_group(grouped_caches, Q, N_vals):
    """
    Returns:
      spectra: list of G arrays [N_max], mean |c_hat_{j,n}|^2 over columns
      tail_energy: list of G arrays [len(N_vals)], E_N = sum_{n>=N} sum_j |c_hat_{j,n}|^2
    """
    q_t = torch.from_numpy(Q)
    spectra = []
    tail_energy = []

    for g, c_g in enumerate(grouped_caches):
        print(f"    Group {g}: projecting [{c_g.shape[0]} x {c_g.shape[1]}]...")
        coeffs = q_t.T @ c_g

        coeff_sq_per_mode = (coeffs ** 2).sum(dim=1).numpy()
        spectra.append((coeffs ** 2).mean(dim=1).numpy())

        total = coeff_sq_per_mode.sum()
        cumsum = np.cumsum(coeff_sq_per_mode)
        tail = np.zeros(len(N_vals))
        for i, N in enumerate(N_vals):
            if N <= 0:
                tail[i] = total
            elif N >= len(coeff_sq_per_mode):
                tail[i] = 0.0
            else:
                tail[i] = total - cumsum[N - 1]
        tail_energy.append(tail)

    return spectra, tail_energy


def print_tail_summary(tail_energy, N_vals, eff_N_max, model_display_name):
    print(f"\n{'=' * 55}")
    print(f"  Legendre Tail Energy Summary - {model_display_name}")
    print(f"{'=' * 55}")
    for g, tail in enumerate(tail_energy):
        total = tail[0]
        for N_show in [1, 10, 20, 50, 100, 200, 512]:
            if N_show > eff_N_max:
                break
            idx = np.argmin(np.abs(N_vals - N_show))
            pct = tail[idx] / total * 100 if total > 0 else 0
            print(
                f"  Group {g}: E_{N_vals[idx]:>3d} = "
                f"{tail[idx]:.4e}  ({pct:5.1f}% of total)"
            )
        print()


def save_results(path, args, T, eff_N_max, N_vals, ranges, spectra, tail_energy, model_title):
    payload = {
        "model_name": args.model_name,
        "model_title": model_title,
        "dataset": args.dataset,
        "k": args.k,
        "seed": args.seed,
        "split": args.split,
        "num_groups": args.num_groups,
        "T": int(T),
        "N_max": int(eff_N_max),
        "N_vals": N_vals,
        "ranges": ranges,
        "spectra": spectra,
        "tail_energy": tail_energy,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    print(f"Saved results: {path}")


# =====================================================================
# Main
# =====================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--dataset", type=str, default="sst2")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--num_groups", type=int, default=4)
    p.add_argument("--N_max", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default="prop31_results")
    p.add_argument("--results_path", type=str, default="")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_title = infer_model_display_name(args.model_name)
    model_tag = model_title.lower().replace("-", "_")
    results_path = (
        args.results_path
        if args.results_path
        else os.path.join(args.output_dir, f"verify_kv_results_{model_tag}.pt")
    )

    print("Loading data...")
    retrieval_data = load_data(
        task=None,
        split=args.split,
        k=args.k,
        seed=args.seed,
        datasets=args.dataset.split(","),
        is_null=False,
    )
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(retrieval_data)), min(args.k, len(retrieval_data)))
    demos = [retrieval_data[i] for i in indices]

    add_nl = not args.model_name.startswith("gpt2")
    tok = setup_tokenizer(args.model_name)
    demo_text = build_demo_text(demos, add_newlines=add_nl)
    demo_ids = tok(demo_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    T = demo_ids.shape[1]
    print(f"Demo tokens: T={T}")

    print(f"Loading {args.model_name} (float32 for precision)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map=args.device,
    )
    model.eval()

    print("Extracting grouped KV caches...")
    grouped, ranges = extract_grouped_caches(model, demo_ids.to(model.device), args.num_groups)
    print(
        "Groups: "
        + str(
            [
                f"G{g}:[{s},{e}) D_g={grouped[g].shape[1]}"
                for g, (s, e) in enumerate(ranges)
            ]
        )
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eff_N_max = min(args.N_max, T)
    N_vals = np.unique(
        np.concatenate(
            [
                np.arange(1, 20),
                np.arange(20, 100, 5),
                np.arange(100, eff_N_max + 1, 10),
            ]
        )
    ).astype(int)

    print(f"\nBuilding Legendre basis (T={T}, N_max={eff_N_max})...")
    Q = build_discrete_legendre_basis(T, eff_N_max)

    print("Computing spectra and tail energy...")
    spectra, tail_energy = compute_per_group(grouped, Q, N_vals)

    print_tail_summary(tail_energy, N_vals, eff_N_max, model_title)
    save_results(
        results_path,
        args,
        T,
        eff_N_max,
        N_vals,
        ranges,
        spectra,
        tail_energy,
        model_title,
    )
    print("Done.")


if __name__ == "__main__":
    main()