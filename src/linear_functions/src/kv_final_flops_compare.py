from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm
import numpy as np
import math

from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names

import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')

run_dir = "../models"

from samplers import get_data_sampler
from tasks import get_task_sampler
task = "linear_regression"
#task = "sparse_linear_regression"
#task = "decision_tree"
#task = "relu_2nn_regression"

run_id = "pretrained"  # if you train more models, replace with the run_id from the table above

run_path = os.path.join(run_dir, task, run_id)
recompute_metrics = False
print(run_path)
if recompute_metrics:
    get_run_metrics(run_path)  # these are normally precomputed at the end of training

model, conf = get_model_from_run(run_path)

n_dims = conf.model.n_dims
batch_size = conf.training.batch_size

data_sampler = get_data_sampler(conf.training.data, n_dims)
task_sampler = get_task_sampler(
    conf.training.task,
    n_dims,
    batch_size,
    **conf.training.task_kwargs
)

def predict_full_label(model, xs, ys, labels):
    with torch.no_grad():
        pred = model(xs, ys)
    metric = task.get_metric()
    loss = metric(pred, labels).cpu().numpy()

    return pred, loss.mean(axis=0)


def tensor_add_sample(xs, ys, sample):
    xs = torch.cat([xs, sample.x.view(1,1,-1)], dim=1)
    ys = torch.cat([ys, sample.y.view(1,-1)], dim=1)
    return xs, ys

def tensor_add_xy(xs, ys, x, y):
    xs = torch.cat([xs, x.view(1,1,-1)], dim=1)
    ys = torch.cat([ys, y.view(1,-1)], dim=1)
    return xs, ys

def tensor_del_sample(xs, ys):
    xs = xs[:, :-1, :]
    ys = ys[:, :-1]
    return xs, ys

import torch.nn.functional as F
import copy
from datetime import datetime

device = 'cuda'
model = model.to(device)

model.eval()
use_checkpoint = True
if not use_checkpoint:
    n_total = 5
    n_labeled = 2
    n_dims = 2
    b_size = 3
else:
    n_total = 41
    n_labeled = 20
    n_dims = conf.model.n_dims
    #b_size = conf.training.batch_size
    b_size = 50

n_unlabeled = n_total - n_labeled
runs = 1
loss_full_label_list = []
loss_random_list = []
loss_random_ensemble_list = []
loss_kv_cache_re_list = []
loss_kv_final_list = []
loss_no_kv_final_list = []
flops_kv_final_list = []
flops_no_kv_final_list = []
loss_beta_list = []
loss_loss_list = []
loss_contrastive_list = []
loss_fs_inference_list = []

same_distribution = True


add_set_size = 5
profile_flops = True
# kv_cache baseline: profile full batch directly (no scaling estimate).
profile_batch_size_kv = None
# no_kv baseline: profile partial batch then scale to full batch estimate.
profile_batch_size_no_kv = 10

def distance_score(model, seq, sample_list, new_sample):
    n = len(sample_list)
    # concat sample_list to seq
    sub_seq_list = []
    for i in range(n):
        sub_seq = copy.deepcopy(seq)
        sub_seq.add_query(sample_list[i])
        sub_seq.get_input()
        sub_seq_list.append(sub_seq)

    xs, ys = sequence_to_tensor(sub_seq_list)
    new_seq = copy.deepcopy(seq)
    new_seq.add_query(new_sample)
    new_seq.get_input()
    new_xs, new_ys = sequence_to_tensor([new_seq])

    with torch.no_grad():
        embedding = model.encoder(xs, ys)
        new_embedding = model.encoder(new_xs, new_ys)
    score = torch.norm(embedding.mean(dim=0) - new_embedding[0])
    score = 1 / score
    return score

def x_distance_score(model, sample_list, new_sample):
    n = len(sample_list)
    xs = torch.stack([s.x for s in sample_list], dim=0)
    score = torch.norm(xs.mean(dim=0) - new_sample.x, dim=0)
    return score


def _expand_past_key_values(past_key_values, batch_size):
    expanded = []
    for layer_past in past_key_values:
        expanded_layer = tuple(
            tensor.expand(batch_size, *tensor.shape[1:]).contiguous()
            for tensor in layer_past
        )
        expanded.append(expanded_layer)
    return tuple(expanded)

def _trim_past_key_values(past_key_values, keep_tokens):
    trimmed = []
    for layer_past in past_key_values:
        trimmed_layer = tuple(tensor[:, :, :keep_tokens, :].contiguous() for tensor in layer_past)
        trimmed.append(trimmed_layer)
    return tuple(trimmed)

def _token_lcp_length(zs_a, zs_b):
    max_len = min(zs_a.shape[0], zs_b.shape[0])
    lcp = 0
    while lcp < max_len and torch.equal(zs_a[lcp], zs_b[lcp]):
        lcp += 1
    return lcp


def _subset_loss_with_kv_cache_from_base_past(
    model,
    metric,
    base_past_key_values,
    subset_x,
    subset_y,
    query_xs,
    query_ys,
):
    with torch.no_grad():
        if subset_x is not None and subset_x.shape[0] > 0:
            subset_x_b = subset_x.unsqueeze(0)
            subset_y_b = subset_y.unsqueeze(0)
            subset_zs = model._combine(subset_x_b, subset_y_b)
            subset_embeds = model._read_in(subset_zs)
            subset_out = model._backbone(
                inputs_embeds=subset_embeds,
                past_key_values=base_past_key_values,
                use_cache=True,
            )
            full_past_key_values = subset_out.past_key_values
        else:
            full_past_key_values = base_past_key_values

        query_x_b = query_xs.unsqueeze(1)
        query_y_b = query_ys.view(-1, 1)
        query_zs = model._combine(query_x_b, query_y_b)
        query_embeds = model._read_in(query_zs)
        expanded_past = _expand_past_key_values(full_past_key_values, query_embeds.shape[0])

        query_out = model._backbone(
            inputs_embeds=query_embeds,
            past_key_values=expanded_past,
            use_cache=False,
        ).last_hidden_state
        query_pred = model._read_out(query_out)[:, ::2, 0]
        query_loss = metric(query_pred, query_y_b).squeeze(-1)
        return float(query_loss.mean().detach().cpu().item())

def _eval_ordered_last_query_with_kv_cache(model, ordered_seq_list):
    last_preds = []
    prev_zs = None
    prev_past = None

    with torch.no_grad():
        for seq in ordered_seq_list:
            xs_1 = seq.input_x.unsqueeze(0)
            ys_1 = seq.input_y.unsqueeze(0)
            zs = model._combine(xs_1, ys_1)[0]
            embeds = model._read_in(zs.unsqueeze(0))

            last_x_token_idx = 2 * (seq.input_x.shape[0] - 1)
            if prev_zs is None:
                out = model._backbone(inputs_embeds=embeds, use_cache=True)
                hidden_suffix = out.last_hidden_state[0]
                pred_last = model._read_out(hidden_suffix[last_x_token_idx : last_x_token_idx + 1])[:, 0]
                prev_past = out.past_key_values
            else:
                lcp = _token_lcp_length(prev_zs, zs)
                # Ensure target query token is recomputed in current suffix.
                lcp = min(lcp, last_x_token_idx)
                if lcp == 0:
                    out = model._backbone(inputs_embeds=embeds, use_cache=True)
                    hidden_suffix = out.last_hidden_state[0]
                    pred_last = model._read_out(hidden_suffix[last_x_token_idx : last_x_token_idx + 1])[:, 0]
                    prev_past = out.past_key_values
                else:
                    trimmed_past = _trim_past_key_values(prev_past, lcp)
                    suffix_embeds = embeds[:, lcp:, :]
                    out = model._backbone(
                        inputs_embeds=suffix_embeds,
                        past_key_values=trimmed_past,
                        use_cache=True,
                    )
                    hidden_suffix = out.last_hidden_state[0]
                    local_idx = last_x_token_idx - lcp
                    pred_last = model._read_out(hidden_suffix[local_idx : local_idx + 1])[:, 0]
                    prev_past = out.past_key_values

            last_preds.append(pred_last.squeeze(0))
            prev_zs = zs

    return torch.stack(last_preds, dim=0)

def _candidate_avg_loss_by_kv_cache(
    model,
    xs,
    ys,
    sample_list,
    n_labeled,
    query_idx,
    set_size,
    candidate_indices,
    subset_multiplier=2,
    query_set_size=8,
):
    n_candidates = len(candidate_indices)
    m = max(1, int(np.ceil(subset_multiplier * n_candidates)))
    global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(candidate_indices)}

    query_indices = np.arange(n_labeled, xs.shape[1])
    q_size = min(query_set_size, len(query_indices))
    sampled_query_idx = np.random.choice(query_indices, size=q_size, replace=False)
    query_xs = xs[query_idx, sampled_query_idx, :]
    query_ys = ys[query_idx, sampled_query_idx]

    metric = task.get_metric()
    loss_sum = np.zeros(n_candidates, dtype=np.float64)
    loss_count = np.zeros(n_candidates, dtype=np.int64)

    prompt_x_base = xs[query_idx, :n_labeled, :].clone()
    prompt_y_base = ys[query_idx, :n_labeled].clone()
    with torch.no_grad():
        prompt_x_base_b = prompt_x_base.unsqueeze(0)
        prompt_y_base_b = prompt_y_base.unsqueeze(0)
        prompt_base_zs = model._combine(prompt_x_base_b, prompt_y_base_b)
        prompt_base_embeds = model._read_in(prompt_base_zs)
        base_out = model._backbone(inputs_embeds=prompt_base_embeds, use_cache=True)
        base_past_key_values = base_out.past_key_values

    for _ in range(m):
        subset = np.random.choice(
            n_candidates,
            size=min(set_size, n_candidates),
            replace=False,
        )
        subset_global = [candidate_indices[idx] for idx in subset]

        subset_x = torch.stack([sample_list[g_idx].x for g_idx in subset_global], dim=0)
        subset_y = torch.stack([sample_list[g_idx].y for g_idx in subset_global], dim=0)

        subset_loss = _subset_loss_with_kv_cache_from_base_past(
            model, metric, base_past_key_values, subset_x, subset_y, query_xs, query_ys
        )

        for g_idx in subset_global:
            local_idx = global_to_local[g_idx]
            loss_sum[local_idx] += subset_loss
            loss_count[local_idx] += 1

    avg_loss = np.full(n_candidates, np.inf, dtype=np.float64)
    valid_mask = loss_count > 0
    avg_loss[valid_mask] = loss_sum[valid_mask] / loss_count[valid_mask]
    return avg_loss

def _subset_loss_without_kv_cache(model, metric, prompt_x, prompt_y, query_xs, query_ys):
    with torch.no_grad():
        query_x_b = query_xs.unsqueeze(1)
        query_y_b = query_ys.view(-1, 1)
        q_size = query_x_b.shape[0]

        prompt_x_b = prompt_x.unsqueeze(0).expand(q_size, -1, -1)
        prompt_y_b = prompt_y.unsqueeze(0).expand(q_size, -1)
        full_x = torch.cat([prompt_x_b, query_x_b], dim=1)
        full_y = torch.cat([prompt_y_b, query_y_b], dim=1)

        pred = model(full_x, full_y)
        query_pred = pred[:, -1:].contiguous()
        query_loss = metric(query_pred, query_y_b).squeeze(-1)
        return float(query_loss.mean().detach().cpu().item())


def _candidate_avg_loss_without_kv_cache(
    model,
    xs,
    ys,
    sample_list,
    n_labeled,
    query_idx,
    set_size,
    candidate_indices,
    subset_multiplier=2,
    query_set_size=8,
):
    n_candidates = len(candidate_indices)
    m = max(1, int(np.ceil(subset_multiplier * n_candidates)))
    global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(candidate_indices)}

    query_indices = np.arange(n_labeled, xs.shape[1])
    q_size = min(query_set_size, len(query_indices))
    sampled_query_idx = np.random.choice(query_indices, size=q_size, replace=False)
    query_xs = xs[query_idx, sampled_query_idx, :]
    query_ys = ys[query_idx, sampled_query_idx]

    metric = task.get_metric()
    loss_sum = np.zeros(n_candidates, dtype=np.float64)
    loss_count = np.zeros(n_candidates, dtype=np.int64)

    for _ in range(m):
        subset = np.random.choice(
            n_candidates,
            size=min(set_size, n_candidates),
            replace=False,
        )
        subset_global = [candidate_indices[idx] for idx in subset]

        prompt_x = xs[query_idx, :n_labeled, :].clone()
        prompt_y = ys[query_idx, :n_labeled].clone()
        for g_idx in subset_global:
            prompt_x = torch.cat([prompt_x, sample_list[g_idx].x.unsqueeze(0)], dim=0)
            prompt_y = torch.cat([prompt_y, sample_list[g_idx].y.unsqueeze(0)], dim=0)

        subset_loss = _subset_loss_without_kv_cache(
            model, metric, prompt_x, prompt_y, query_xs, query_ys
        )

        for g_idx in subset_global:
            local_idx = global_to_local[g_idx]
            loss_sum[local_idx] += subset_loss
            loss_count[local_idx] += 1

    avg_loss = np.full(n_candidates, np.inf, dtype=np.float64)
    valid_mask = loss_count > 0
    avg_loss[valid_mask] = loss_sum[valid_mask] / loss_count[valid_mask]
    return avg_loss


def _eval_ordered_last_query_without_kv_cache(model, ordered_seq_list):
    last_preds = []
    with torch.no_grad():
        for seq in ordered_seq_list:
            xs_1 = seq.input_x.unsqueeze(0)
            ys_1 = seq.input_y.unsqueeze(0)
            pred = model(xs_1, ys_1)
            pred_last = pred[0, -1]
            last_preds.append(pred_last)
    return torch.stack(last_preds, dim=0)


def _sum_profiled_flops(prof):
    total_flops = 0
    for event in prof.key_averages():
        event_flops = getattr(event, "flops", 0)
        if event_flops is not None:
            total_flops += int(event_flops)
    return int(total_flops)


def _kv_final_select_core(
    model,
    xs,
    ys,
    beta,
    sample_list,
    n_labeled,
    set_size_list,
    candidate_avg_loss_fn,
    eval_last_query_fn,
    subset_multiplier=2,
    query_set_size=8,
    candidate_pool_size=200,
    candidate_indices=None,
    lambda_weight=0.7,
    profile_flops=False,
    profile_batch_size=None,
    profile_scale_to_full=False,
    profile_tag="kv_final_select",
):
    loss_list = []
    flops_list = []
    flops_raw_list = []

    if candidate_indices is None:
        n_all = len(sample_list)
        n_candidates = min(candidate_pool_size, n_all)
        fixed_candidate_indices = np.random.choice(n_all, size=n_candidates, replace=False).tolist()
    else:
        fixed_candidate_indices = [int(i) for i in candidate_indices]

    def _run_single_set_size(set_size, xs_local, ys_local, beta_local):
        seq_list = []
        n_candidates = len(fixed_candidate_indices)
        freq_count = np.zeros(n_candidates, dtype=np.float64)
        order_signatures = []

        for i in range(xs_local.shape[0]):
            # Put shared demos before query-specific base examples to improve LCP reuse.
            base_prompt_x = xs_local[i, :n_labeled, :].clone()
            base_prompt_y = ys_local[i, :n_labeled].clone()
            seq = Input_sequence(xs_local[i], ys_local[i], 0, beta_local[i], i)

            if set_size > 0:
                avg_loss = candidate_avg_loss_fn(
                    model,
                    xs_local,
                    ys_local,
                    sample_list,
                    n_labeled,
                    i,
                    set_size,
                    fixed_candidate_indices,
                    subset_multiplier=subset_multiplier,
                    query_set_size=query_set_size,
                )

                # loss is lower-better, so affinity uses negative loss.
                affinity = -avg_loss
                finite_mask = np.isfinite(affinity)
                if finite_mask.any():
                    min_aff = affinity[finite_mask].min()
                else:
                    min_aff = -1.0
                affinity[~finite_mask] = min_aff - 1.0

                aff_min, aff_max = affinity.min(), affinity.max()
                if aff_max > aff_min:
                    affinity_norm = (affinity - aff_min) / (aff_max - aff_min)
                else:
                    affinity_norm = np.zeros_like(affinity)

                if freq_count.sum() > 0:
                    freq_norm = freq_count / freq_count.sum()
                else:
                    freq_norm = np.zeros_like(freq_count)

                score = lambda_weight * affinity_norm + (1 - lambda_weight) * freq_norm
                selected_local = np.argsort(score)[-min(set_size, n_candidates):][::-1]

                for local_idx in selected_local:
                    freq_count[local_idx] += 1.0

                # Trie-friendly order: high-frequency demos first.
                selected_local_sorted = sorted(
                    selected_local.tolist(),
                    key=lambda idx: (-freq_count[idx], idx),
                )
                selected_global = [fixed_candidate_indices[idx] for idx in selected_local_sorted]

                for g_idx in selected_global:
                    seq.add_sample(sample_list[g_idx])
                order_signatures.append(tuple(selected_global))
            else:
                order_signatures.append(tuple())

            seq.prompt_x = torch.cat([seq.prompt_x, base_prompt_x], dim=0)
            seq.prompt_y = torch.cat([seq.prompt_y, base_prompt_y], dim=0)
            seq.get_input(query_index=-2)
            seq_list.append(seq)

        # Lexicographic query order to mimic Trie traversal.
        ordered_idx = sorted(range(len(seq_list)), key=lambda idx: order_signatures[idx])
        ordered_seq_list = [seq_list[idx] for idx in ordered_idx]
        pred_last = eval_last_query_fn(model, ordered_seq_list)
        target_last = torch.stack([seq.input_y[-1] for seq in ordered_seq_list], dim=0).to(pred_last.device)
        pred_last = pred_last.unsqueeze(1)
        target_last = target_last.unsqueeze(1)
        metric = task.get_metric()
        loss = metric(pred_last, target_last).cpu().numpy()
        loss_query = loss.mean(axis=0)[-1]
        return loss_query

    for set_size in set_size_list:
        if profile_flops:
            prof_bs = xs.shape[0] if profile_batch_size is None else min(int(profile_batch_size), xs.shape[0])
            xs_prof = xs[:prof_bs]
            ys_prof = ys[:prof_bs]
            beta_prof = beta[:prof_bs]

            activities = [ProfilerActivity.CPU]
            if xs_prof.is_cuda:
                activities.append(ProfilerActivity.CUDA)
            with profile(activities=activities, with_flops=True, record_shapes=True) as prof:
                loss_query = _run_single_set_size(set_size, xs_prof, ys_prof, beta_prof)

            raw_flops = _sum_profiled_flops(prof)
            if profile_scale_to_full and prof_bs < xs.shape[0]:
                scaled_flops = int(round(raw_flops * (xs.shape[0] / prof_bs)))
            else:
                scaled_flops = raw_flops
            flops_raw_list.append(raw_flops)
            flops_list.append(scaled_flops)
            if profile_scale_to_full and prof_bs < xs.shape[0]:
                print(
                    f"[{profile_tag}] set_size={set_size}, profiled_batch={prof_bs}/{xs.shape[0]}, "
                    f"reported_flops={scaled_flops} (scaled from raw={raw_flops})"
                )
            else:
                print(
                    f"[{profile_tag}] set_size={set_size}, profiled_batch={prof_bs}/{xs.shape[0]}, "
                    f"reported_flops={scaled_flops}"
                )
        else:
            loss_query = _run_single_set_size(set_size, xs, ys, beta)
        loss_list.append(loss_query)

    if profile_flops:
        return loss_list, flops_list, flops_raw_list
    return loss_list


def kv_final_select(
    model,
    xs,
    ys,
    beta,
    sample_list,
    n_labeled,
    set_size_list,
    subset_multiplier=2,
    query_set_size=8,
    candidate_pool_size=200,
    candidate_indices=None,
    lambda_weight=0.7,
    profile_flops=False,
    profile_batch_size=None,
):
    return _kv_final_select_core(
        model=model,
        xs=xs,
        ys=ys,
        beta=beta,
        sample_list=sample_list,
        n_labeled=n_labeled,
        set_size_list=set_size_list,
        candidate_avg_loss_fn=_candidate_avg_loss_by_kv_cache,
        eval_last_query_fn=_eval_ordered_last_query_with_kv_cache,
        subset_multiplier=subset_multiplier,
        query_set_size=query_set_size,
        candidate_pool_size=candidate_pool_size,
        candidate_indices=candidate_indices,
        lambda_weight=lambda_weight,
        profile_flops=profile_flops,
        profile_batch_size=profile_batch_size,
        profile_scale_to_full=False,
        profile_tag="kv_final_select",
    )


def kv_final_select_without_kv_cache(
    model,
    xs,
    ys,
    beta,
    sample_list,
    n_labeled,
    set_size_list,
    subset_multiplier=2,
    query_set_size=8,
    candidate_pool_size=200,
    candidate_indices=None,
    lambda_weight=0.7,
    profile_flops=False,
    profile_batch_size=None,
):
    return _kv_final_select_core(
        model=model,
        xs=xs,
        ys=ys,
        beta=beta,
        sample_list=sample_list,
        n_labeled=n_labeled,
        set_size_list=set_size_list,
        candidate_avg_loss_fn=_candidate_avg_loss_without_kv_cache,
        eval_last_query_fn=_eval_ordered_last_query_without_kv_cache,
        subset_multiplier=subset_multiplier,
        query_set_size=query_set_size,
        candidate_pool_size=candidate_pool_size,
        candidate_indices=candidate_indices,
        lambda_weight=lambda_weight,
        profile_flops=profile_flops,
        profile_batch_size=profile_batch_size,
        profile_scale_to_full=True,
        profile_tag="kv_final_select_without_kv_cache",
    )

class Sample():
    def __init__(self, x, y, c, beta):
        self.x = x
        self.y = y
        self.c = c
        self.beta = beta
    
    def __repr__(self):
        return f"x: {self.x}, y: {self.y}, c: {self.c}\n"

class Input_sequence():
    def __init__(self, xs, ys, length, beta, c):
        self.xs = xs
        self.ys = ys
        self.prompt_x = xs[:length]
        self.prompt_y = ys[:length]
        self.query_x = xs[-1]
        self.query_y = ys[-1]
        self.beta = beta
        self.c = c
    
    def add_sample(self, sample):
        self.prompt_x = torch.cat([self.prompt_x, sample.x.unsqueeze(0)], dim=0)
        self.prompt_y = torch.cat([self.prompt_y, sample.y.unsqueeze(0)], dim=0)
    
    def del_sample(self):
        self.prompt_x = self.prompt_x[:-1]
        self.prompt_y = self.prompt_y[:-1]

    def add_xy(self, x, y):
        self.prompt_x = torch.cat([self.prompt_x, x.unsqueeze(0)], dim=0)
        self.prompt_y = torch.cat([self.prompt_y, y.unsqueeze(0)], dim=0)
    
    def add_query(self, sample):
        self.query_x = sample.x
        self.query_y = sample.y
    
    def get_input(self, query_index=-1, query_range=0, query_sample=None, last_prompt=False):
        if last_prompt:
            self.input_x = self.prompt_x
            self.input_y = self.prompt_y
        elif query_sample is not None:
            self.input_x = torch.cat([self.prompt_x, query_sample.x.unsqueeze(0)], dim=0)
            self.input_y = torch.cat([self.prompt_y, query_sample.y.unsqueeze(0)], dim=0)
        elif query_range > 0:
            self.input_x = torch.cat([self.prompt_x, self.xs[-query_range-1:-1]], dim=0)
            self.input_y = torch.cat([self.prompt_y, self.ys[-query_range-1:-1]], dim=0)
        elif query_index == -1:
            self.input_x = torch.cat([self.prompt_x, self.query_x.unsqueeze(0)], dim=0)
            self.input_y = torch.cat([self.prompt_y, self.query_y.unsqueeze(0)], dim=0)
        else:
            self.input_x = torch.cat([self.prompt_x, self.xs[query_index].unsqueeze(0)], dim=0)
            self.input_y = torch.cat([self.prompt_y, self.ys[query_index].unsqueeze(0)], dim=0)
        return self.input_x, self.input_y
    
    def pad(self):
        self.prompt_x = torch.cat([self.prompt_x, self.prompt_x[-1].unsqueeze(0)], dim=0)
        self.prompt_y = torch.cat([self.prompt_y, self.prompt_y[-1].unsqueeze(0)], dim=0)
    
    def get_prompt_length(self):
        return self.prompt_x.shape[0]
    
    def __repr__(self):
        return f"prompt_x: {self.prompt_x}, prompt_y: {self.prompt_y}, query_x: {self.query_x}, query_y: {self.query_y}\n"

def sequence_to_tensor(seq_list):
    xs = torch.stack([s.input_x for s in seq_list], dim=0)
    ys = torch.stack([s.input_y for s in seq_list], dim=0)
    return xs, ys

def generate_synthetic_data(num_sequences=100, n_points=10, x_dim=8, diff_diftribution=False, alpha=0.9):
    xs_b = torch.randn(num_sequences, n_points, x_dim)
    # same mean for all points
    if diff_diftribution:
        for i in range(xs_b.shape[0]):
            mean_vectors = torch.randn(1, x_dim)
            for j in range(xs_b.shape[1]):
                xs_b[i, j, :] = ((1-alpha) * xs_b[i, j, :] + alpha * mean_vectors)
    return xs_b

def generate_orthogonal_matrix(n, m):
    if n > m:
        raise ValueError("The number of rows (n) should not be greater than the number of columns (m) for an orthogonal set.")
    
    # Generate a random matrix
    A = torch.randn(n, m)
    
    # Apply Gram-Schmidt process
    Q, _ = torch.linalg.qr(A.T)  # QR decomposition on the transpose
    return Q.T  # Transpose back to get row-wise orthogonality


runs = 1
candidate_pool_size = 100
for run in range(runs):
    task = task_sampler()
    x_single = torch.randn(n_total, n_dims)
    xs = torch.randn(b_size, n_total, n_dims)
    xs = xs.to(device)
    XS = xs
    n_classes = b_size // 10
    n_classes = 5
    anchor_points = generate_orthogonal_matrix(n_classes, n_dims).to(device)
    beta = torch.randn(b_size, n_dims, 1).to(device)
    alpha = 1
    for i in range(beta.shape[0]):
        i_class = torch.randint(0, n_classes, (1,))
        beta[i] = ((1-alpha)*beta[i] + anchor_points[i_class].T)
    ys = (xs @ beta)[:, :, 0]
    noise = 0.3
    ys += torch.randn_like(ys).to(device) * noise
    labels = ys.clone()
    with torch.no_grad():
        embedding = model.encoder(xs, ys)
    print(embedding.shape)

    # split sequences
    sample_list = []
    for i in range(xs.shape[0]):
        # -1 for remaining the last sample as query
        for j in range(xs.shape[1]-1):
            s = Sample(xs[i, j], ys[i, j], i, beta[i])
            sample_list.append(s)
    
    fixed_candidate_indices = np.random.choice(
        len(sample_list),
        size=min(candidate_pool_size, len(sample_list)),
        replace=False,
    ).tolist()

    set_size_list = list(range(1, 11))


    kv_result = kv_final_select(
        model,
        xs,
        ys,
        beta,
        sample_list,
        n_labeled,
        set_size_list,
        candidate_pool_size=candidate_pool_size,
        candidate_indices=fixed_candidate_indices,
        lambda_weight=0.7,
        profile_flops=profile_flops,
        profile_batch_size=profile_batch_size_kv,
    )
    if profile_flops:
        loss_kv_final, flops_kv_final, flops_kv_final_raw = kv_result
    else:
        loss_kv_final = kv_result
        flops_kv_final = []
        flops_kv_final_raw = []
    print("loss_kv_final: ", loss_kv_final)
    print("flops_kv_final: ", flops_kv_final)
    loss_kv_final_list.append(loss_kv_final)
    flops_kv_final_list.append(flops_kv_final)

    no_kv_result = kv_final_select_without_kv_cache(
        model,
        xs,
        ys,
        beta,
        sample_list,
        n_labeled,
        set_size_list,
        candidate_pool_size=candidate_pool_size,
        candidate_indices=fixed_candidate_indices,
        lambda_weight=0.7,
        profile_flops=profile_flops,
        profile_batch_size=profile_batch_size_no_kv,
    )
    if profile_flops:
        loss_no_kv_final, flops_no_kv_final, flops_no_kv_final_raw = no_kv_result
    else:
        loss_no_kv_final = no_kv_result
        flops_no_kv_final = []
        flops_no_kv_final_raw = []
    print("loss_no_kv_final: ", loss_no_kv_final)
    print("flops_no_kv_final: ", flops_no_kv_final)
    loss_no_kv_final_list.append(loss_no_kv_final)
    flops_no_kv_final_list.append(flops_no_kv_final)

    if profile_flops:
        flops_ratio_no_kv_over_kv = [
            (float(no_kv) / float(kv)) if float(kv) > 0 else float("inf")
            for no_kv, kv in zip(flops_no_kv_final, flops_kv_final)
        ]
        print("flops_ratio_no_kv_over_kv: ", flops_ratio_no_kv_over_kv)
        assert len(flops_kv_final) == len(set_size_list), "kv_final FLOPs missing some set_size"
        assert len(flops_no_kv_final) == len(set_size_list), "no_kv FLOPs missing some set_size"

    flops_save_path = os.path.join(
        os.getcwd(),
        "results",
        "kv_vs_no_kv_flops.npy",
    )
    flops_payload = {
        "set_size_list": np.array(set_size_list, dtype=np.int64),
        "flops_kv_final_list": np.array(flops_kv_final_list, dtype=np.float64),
        "flops_no_kv_final_list": np.array(flops_no_kv_final_list, dtype=np.float64),
    }
    np.save(flops_save_path, flops_payload, allow_pickle=True)
    print("saved_flops_npy: ", flops_save_path)
