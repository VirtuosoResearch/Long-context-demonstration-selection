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
loss_beta_list = []
loss_loss_list = []
loss_contrastive_list = []
loss_fs_inference_list = []

same_distribution = True


add_set_size = 5

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



def fs_select(
    model,
    xs,
    ys,
    beta,
    sample_list,
    b_size,
    n_labeled,
    set_size_list,
    candidate_pool_size=200,
    candidate_indices=None,
):
    n_points = xs.shape[1] - 1
    if candidate_indices is None:
        n_all = len(sample_list)
        n_candidates = min(candidate_pool_size, n_all)
        candidate_indices = np.random.choice(n_all, size=n_candidates, replace=False).tolist()
    else:
        candidate_indices = [int(i) for i in candidate_indices]
        n_candidates = len(candidate_indices)
    pos_index = torch.zeros(n_candidates, n_candidates)

    # init select index for query in each sequence
    loss_list = []
    loss_list_infer = []
    max_points = np.max(set_size_list)
    sub_sample_list = [[] for _ in range(b_size)]
    sub_sample_index = [[] for _ in range(b_size)]
    sub_sample_list_infer = [[] for _ in range(b_size)]
    sub_sample_index_infer = [[] for _ in range(b_size)]
    select_index = 0

    temp_x = []
    temp_y = []
    for i in range(b_size):
        temp_x.append(xs[i, :n_labeled, :].unsqueeze(0))
        temp_y.append(ys[i, :n_labeled].unsqueeze(0))

    task_index = range(1)
    for k in range(1, max_points+1):
        for i in task_index:

            sub_seq_x_list = []
            sub_seq_y_list = []
            available_indices = []
            n_val = 5
            i_x = temp_x[i]
            i_y = temp_y[i]
            pred_0, embeds_0 = model.forward_with_embeds(i_x, i_y)
            grad = torch.autograd.grad(pred_0[0, -1], embeds_0, retain_graph=True, create_graph=True)[0]
            grad = grad[:, 0::2, :]
            grad_0 = grad[:, -1, :]
            embeds_0 = embeds_0[:, 0::2, :]

            metric = task.get_metric()

            for j in candidate_indices:
                if j in sub_sample_index[i]:
                    continue
                sub_i_x, sub_i_y = tensor_add_sample(i_x, i_y, sample_list[j])
                sub_seq_x_list.append(sub_i_x)
                sub_seq_y_list.append(sub_i_y)
                available_indices.append(j)

            if len(available_indices) == 0:
                continue

            sub_xs = torch.cat(sub_seq_x_list, dim=0)
            sub_ys = torch.cat(sub_seq_y_list, dim=0)
            sub_label = copy.deepcopy(sub_ys)
            sub_label[:, -1] = sub_label[:, -2]
            embeds = model.embed_x(sub_xs)
            delta_embeds = embeds[:, -1, :] - embeds[:, -2, :]

            with torch.no_grad():
                pred = model(sub_xs, sub_ys)

            metric = task.get_metric()
            loss_infer = metric(pred, sub_ys)
            score = loss_infer[:, -1]
            
            #score = score_infer
            
            threshold = 1e-3
            filtered_score = score.clone()
            #score[filtered_score < threshold] = 1e9
            topk = min(10, score.shape[0])
            _, topk_indices = torch.topk(score, topk, largest=False)
            
            mean_x = 0
            mean_y = 0
            mean_beta = 0
            x_candidate = []
            y_candidate = []
            beta_candidate = []
            for idx in topk_indices:
                g_idx = available_indices[int(idx)]
                mean_x+=sample_list[g_idx].x
                mean_y+=sample_list[g_idx].y
                mean_beta+=sample_list[g_idx].beta
                beta_candidate.append(sample_list[g_idx].beta.unsqueeze(0))
                x_candidate.append(sample_list[g_idx].x.unsqueeze(0))
                y_candidate.append(sample_list[g_idx].y.unsqueeze(0))
            
            beta_candidate = torch.cat(beta_candidate, dim=0)
            x_candidate = torch.cat(x_candidate, dim=0)
            y_candidate = torch.cat(y_candidate, dim=0)
            unique_beta, counts = torch.unique(beta_candidate, return_counts=True, dim=0)
            max_idx = torch.argmax(counts)
            most_beta = unique_beta[max_idx]
            most_x = x_candidate[max_idx]
            most_y = y_candidate[max_idx]
            mean_x /= topk
            mean_y /= topk
            mean_beta /= topk
            new_x = sample_list[available_indices[int(topk_indices[0])]].x
            #new_y = new_x @ mean_beta
            new_y = (new_x @ most_beta)
            #new_sample = Sample(mean_x, mean_y, 0, 0)
            #new_sample = Sample(new_x, new_y, 0, 0)
            new_sample = Sample(most_x, most_y, 0, most_beta)

            
            select_index = available_indices[int(torch.argmin(score))]
            sub_sample_index[i].append(select_index)
            sub_sample_list[i].append(sample_list[select_index])
            # print("======= least: ", (sample_list[select_index].beta - sample_list[i*(n_total-1)].beta).mean())
            # print("======= mean: ", (mean_beta - sample_list[i*(n_total-1)].beta).mean())
            #temp_x[i], temp_y[i] = tensor_add_sample(temp_x[i], temp_y[i], sample_list[select_index])
            #sub_sample_list[i].append(new_sample)
            temp_x[i], temp_y[i] = tensor_add_sample(temp_x[i], temp_y[i], new_sample)

            # print(sub_sample_index[i])

        if k in set_size_list:
            # get the input tensor
            query_xs, query_ys = [], []
            for _ in task_index:

                _x, _y = tensor_add_xy(temp_x[_], temp_y[_], xs[i, -1, :], ys[i, -1])
                query_xs.append(_x)
                query_ys.append(_y)
            query_xs = torch.cat(query_xs, dim=0)
            query_ys = torch.cat(query_ys, dim=0)
            with torch.no_grad():
                pred = model(query_xs, query_ys)
            metric = task.get_metric()
            loss = metric(pred, query_ys).cpu().numpy()
            loss_query = loss.mean(axis=0)[-1]
            loss_list.append(loss_query)

    return loss_list

def beta_select(
    model, xs, ys, beta, sample_list, n_labeled, set_size_list, candidate_pool_size=200, candidate_indices=None
):
    loss_list = []
    if candidate_indices is None:
        n_all = len(sample_list)
        n_candidates = min(candidate_pool_size, n_all)
        fixed_candidate_indices = np.random.choice(n_all, size=n_candidates, replace=False).tolist()
    else:
        fixed_candidate_indices = [int(i) for i in candidate_indices]
    for set_size in set_size_list:
        seq_list = []
        for i in range(xs.shape[0]):
            seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, beta[i], i))
        for i in range(xs.shape[0]):
            beta_dist = []
            for g_idx in fixed_candidate_indices:
                d = torch.norm(sample_list[g_idx].beta - beta[i]).item()
                beta_dist.append((d, g_idx))
            beta_dist.sort(key=lambda x: x[0])
            for _, select_index in beta_dist[: min(set_size, len(beta_dist))]:
                seq_list[i].add_sample(sample_list[select_index])

        # get least loss samples
        for seq in seq_list:
            seq.get_input()
        #seq_list[0].add_sample(sample_list[2])

        xs, ys = sequence_to_tensor(seq_list)

        
        with torch.no_grad():
            #pred = model.seq_inference(xs, ys)
            pred = model(xs, ys)
        metric = task.get_metric()
        loss = metric(pred, ys).cpu().numpy()
        loss_query = loss.mean(axis=0)[-1]
        print(seq_list[0].get_prompt_length())
        loss_list.append(loss_query)
    
    return loss_list

def random_select(
    model, xs, ys, beta, sample_list, n_labeled, set_size_list, candidate_pool_size=200, candidate_indices=None
):
    loss_list = []
    if candidate_indices is None:
        n_all = len(sample_list)
        n_candidates = min(candidate_pool_size, n_all)
        fixed_candidate_indices = np.random.choice(n_all, size=n_candidates, replace=False).tolist()
    else:
        fixed_candidate_indices = [int(i) for i in candidate_indices]
    for set_size in set_size_list:
        seq_list = []
        for i in range(xs.shape[0]):
            seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, beta[i], i))
        for i in range(xs.shape[0]):
            beta_list = []
            for j in range(set_size):
                select_index = int(np.random.choice(fixed_candidate_indices))
                seq_list[i].add_sample(sample_list[select_index])
                beta_list.append(sample_list[select_index].beta[0, 0])

        # get least loss samples
        for seq in seq_list:
            seq.get_input(query_index=-2)
        #seq_list[0].add_sample(sample_list[2])

        xs, ys = sequence_to_tensor(seq_list)

        
        with torch.no_grad():
            #pred = model.seq_inference(xs, ys)
            pred = model(xs, ys)
        metric = task.get_metric()
        loss = metric(pred, ys).cpu().numpy()
        loss_query = loss.mean(axis=0)[-1]
        loss_list.append(loss_query)
    
    return loss_list

def random_ensemble_select(
    model,
    xs,
    ys,
    beta,
    sample_list,
    n_labeled,
    set_size_list,
    subset_multiplier=1,
    candidate_pool_size=200,
    candidate_indices=None,
):
    loss_list = []
    if candidate_indices is None:
        n_all = len(sample_list)
        n_candidates = min(candidate_pool_size, n_all)
        fixed_candidate_indices = np.random.choice(n_all, size=n_candidates, replace=False).tolist()
    else:
        fixed_candidate_indices = [int(i) for i in candidate_indices]

    for set_size in set_size_list:
        seq_list = []
        for i in range(xs.shape[0]):
            seq = Input_sequence(xs[i], ys[i], n_labeled, beta[i], i)

            if set_size > 0:
                candidate_indices = fixed_candidate_indices
                n_candidates = len(candidate_indices)
                m = max(1, int(np.ceil(subset_multiplier * n_candidates)))
                global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(candidate_indices)}

                sampled_subsets = []
                for _ in range(m):
                    subset = np.random.choice(n_candidates, size=min(set_size, n_candidates), replace=False)
                    sampled_subsets.append(subset.tolist())

                sub_seq_list = []
                subset_global_indices = []
                for subset in sampled_subsets:
                    sub_seq = Input_sequence(xs[i], ys[i], n_labeled, beta[i], i)
                    global_subset = [candidate_indices[idx] for idx in subset]
                    for g_idx in global_subset:
                        sub_seq.add_sample(sample_list[g_idx])
                    sub_seq.get_input(query_index=-2)
                    sub_seq_list.append(sub_seq)
                    subset_global_indices.append(global_subset)

                sub_xs, sub_ys = sequence_to_tensor(sub_seq_list)
                with torch.no_grad():
                    sub_pred = model(sub_xs, sub_ys)
                metric = task.get_metric()
                subset_losses = metric(sub_pred, sub_ys)[:, -1].detach().cpu().numpy()

                loss_sum = np.zeros(n_candidates, dtype=np.float64)
                loss_count = np.zeros(n_candidates, dtype=np.int64)
                for s_idx, global_subset in enumerate(subset_global_indices):
                    curr_loss = subset_losses[s_idx]
                    for g_idx in global_subset:
                        local_idx = global_to_local[g_idx]
                        loss_sum[local_idx] += curr_loss
                        loss_count[local_idx] += 1

                avg_loss = np.full(n_candidates, np.inf, dtype=np.float64)
                valid_mask = loss_count > 0
                avg_loss[valid_mask] = loss_sum[valid_mask] / loss_count[valid_mask]
                selected_local = np.argsort(avg_loss)[: min(set_size, n_candidates)]
                selected_global = [candidate_indices[idx] for idx in selected_local]

                for g_idx in selected_global:
                    seq.add_sample(sample_list[g_idx])

            seq.get_input(query_index=-2)
            seq_list.append(seq)

        eval_xs, eval_ys = sequence_to_tensor(seq_list)
        with torch.no_grad():
            pred = model(eval_xs, eval_ys)
        metric = task.get_metric()
        loss = metric(pred, eval_ys).cpu().numpy()
        loss_query = loss.mean(axis=0)[-1]
        loss_list.append(loss_query)

    return loss_list

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

def _subset_loss_with_kv_cache(model, metric, prompt_x, prompt_y, query_xs, query_ys):
    with torch.no_grad():
        prompt_x_b = prompt_x.unsqueeze(0)
        prompt_y_b = prompt_y.unsqueeze(0)
        prefix_zs = model._combine(prompt_x_b, prompt_y_b)
        prefix_embeds = model._read_in(prefix_zs)
        prefix_out = model._backbone(inputs_embeds=prefix_embeds, use_cache=True)
        past_key_values = prefix_out.past_key_values

        query_x_b = query_xs.unsqueeze(1)
        query_y_b = query_ys.view(-1, 1)
        query_zs = model._combine(query_x_b, query_y_b)
        query_embeds = model._read_in(query_zs)
        expanded_past = _expand_past_key_values(past_key_values, query_embeds.shape[0])

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

        subset_loss = _subset_loss_with_kv_cache(
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

def kv_cache_re_select(
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
):
    loss_list = []
    if candidate_indices is None:
        n_all = len(sample_list)
        print("n_all: ", n_all)
        n_candidates = min(candidate_pool_size, n_all)
        fixed_candidate_indices = np.random.choice(n_all, size=n_candidates, replace=False).tolist()
    else:
        fixed_candidate_indices = [int(i) for i in candidate_indices]

    for set_size in set_size_list:
        seq_list = []
        for i in range(xs.shape[0]):
            seq = Input_sequence(xs[i], ys[i], n_labeled, beta[i], i)

            if set_size > 0:
                candidate_indices = fixed_candidate_indices
                n_candidates = len(candidate_indices)
                avg_loss = _candidate_avg_loss_by_kv_cache(
                    model,
                    xs,
                    ys,
                    sample_list,
                    n_labeled,
                    i,
                    set_size,
                    candidate_indices,
                    subset_multiplier=subset_multiplier,
                    query_set_size=query_set_size,
                )
                selected_idx = np.argsort(avg_loss)[: min(set_size, n_candidates)]
                selected_global = [candidate_indices[idx] for idx in selected_idx]

                for g_idx in selected_global:
                    seq.add_sample(sample_list[g_idx])

            seq.get_input(query_index=-2)
            seq_list.append(seq)

        eval_xs, eval_ys = sequence_to_tensor(seq_list)
        with torch.no_grad():
            pred = model(eval_xs, eval_ys)
        metric = task.get_metric()
        loss = metric(pred, eval_ys).cpu().numpy()
        loss_query = loss.mean(axis=0)[-1]
        loss_list.append(loss_query)

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
):
    loss_list = []
    if candidate_indices is None:
        n_all = len(sample_list)
        n_candidates = min(candidate_pool_size, n_all)
        fixed_candidate_indices = np.random.choice(n_all, size=n_candidates, replace=False).tolist()
    else:
        fixed_candidate_indices = [int(i) for i in candidate_indices]

    for set_size in set_size_list:
        seq_list = []
        n_candidates = len(fixed_candidate_indices)
        freq_count = np.zeros(n_candidates, dtype=np.float64)
        order_signatures = []

        for i in range(xs.shape[0]):
            seq = Input_sequence(xs[i], ys[i], n_labeled, beta[i], i)

            if set_size > 0:
                avg_loss = _candidate_avg_loss_by_kv_cache(
                    model,
                    xs,
                    ys,
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

                print("affinity_norm: ", affinity_norm)
                print("freq_norm: ", freq_norm)
                print("score: ", score)
                print("--------------------------------")
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

            seq.get_input(query_index=-2)
            seq_list.append(seq)

        # Lexicographic query order to mimic Trie traversal.
        ordered_idx = sorted(range(len(seq_list)), key=lambda idx: order_signatures[idx])
        ordered_seq_list = [seq_list[idx] for idx in ordered_idx]
        pred_last = _eval_ordered_last_query_with_kv_cache(model, ordered_seq_list)
        target_last = torch.stack([seq.input_y[-1] for seq in ordered_seq_list], dim=0).to(pred_last.device)
        pred_last = pred_last.unsqueeze(1)
        target_last = target_last.unsqueeze(1)
        metric = task.get_metric()
        loss = metric(pred_last, target_last).cpu().numpy()
        loss_query = loss.mean(axis=0)[-1]
        loss_list.append(loss_query)

    return loss_list

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


runs = 3
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

    pred_full_label, loss_full_label = predict_full_label(model, xs, ys=labels, labels=labels)
    loss_full_label_list.append(loss_full_label)
    print("loss_full_label_list: ", loss_full_label_list)
    
    
    loss_fs_inference = fs_select(
        model,
        xs,
        ys,
        beta,
        sample_list,
        b_size,
        n_labeled,
        set_size_list,
        candidate_pool_size=candidate_pool_size,
        candidate_indices=fixed_candidate_indices,
    )
    print("loss_fs_inference: ", loss_fs_inference)
    loss_fs_inference_list.append(loss_fs_inference)
    
    loss_random = random_select(
        model,
        xs,
        ys,
        beta,
        sample_list,
        n_labeled,
        set_size_list,
        candidate_pool_size=candidate_pool_size,
        candidate_indices=fixed_candidate_indices,
    )
    print("loss_random: ", loss_random)
    loss_random_list.append(loss_random)

    # loss_random_ensemble = random_ensemble_select(model, xs, ys, beta, sample_list, n_labeled, set_size_list)
    # print("loss_random_ensemble: ", loss_random_ensemble)
    # loss_random_ensemble_list.append(loss_random_ensemble)

    # loss_kv_cache_re = kv_cache_re_select(
    #     model,
    #     xs,
    #     ys,
    #     beta,
    #     sample_list,
    #     n_labeled,
    #     set_size_list,
    #     candidate_pool_size=candidate_pool_size,
    #     candidate_indices=fixed_candidate_indices,
    # )
    # print("loss_kv_cache_re: ", loss_kv_cache_re)
    # loss_kv_cache_re_list.append(loss_kv_cache_re)

    loss_kv_final = kv_final_select(
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
    )
    print("loss_kv_final: ", loss_kv_final)
    loss_kv_final_list.append(loss_kv_final)

    loss_beta = beta_select(
        model,
        xs,
        ys,
        beta,
        sample_list,
        n_labeled,
        set_size_list,
        candidate_pool_size=candidate_pool_size,
        candidate_indices=fixed_candidate_indices,
    )
    print("loss_beta: ", loss_beta)
    loss_beta_list.append(loss_beta)



x = np.arange(n_total)
plt.plot(x, np.mean(loss_full_label_list, axis=0), lw=2, label="Full label")
plt.fill_between(x, np.mean(loss_full_label_list, axis=0)-np.std(loss_full_label_list, axis=0), np.mean(loss_full_label_list, axis=0)+np.std(loss_full_label_list, axis=0), alpha=0.2)

x = []
for i in set_size_list:
    x.append(n_labeled + i)
x = np.array(x)
print("x: ", x)

plt.plot(x, np.mean(loss_random_list, axis=0), lw=2, label="random")
plt.fill_between(x, np.mean(loss_random_list, axis=0)-np.std(loss_random_list, axis=0), np.mean(loss_random_list, axis=0)+np.std(loss_random_list, axis=0), alpha=0.2)

# plt.plot(x, np.mean(loss_random_ensemble_list, axis=0), lw=2, label="random ensemble")
# plt.fill_between(
#     x,
#     np.mean(loss_random_ensemble_list, axis=0)-np.std(loss_random_ensemble_list, axis=0),
#     np.mean(loss_random_ensemble_list, axis=0)+np.std(loss_random_ensemble_list, axis=0),
#     alpha=0.2,
# )

# plt.plot(x, np.mean(loss_kv_cache_re_list, axis=0), lw=2, label="kv_cache_re")
# plt.fill_between(
#     x,
#     np.mean(loss_kv_cache_re_list, axis=0)-np.std(loss_kv_cache_re_list, axis=0),
#     np.mean(loss_kv_cache_re_list, axis=0)+np.std(loss_kv_cache_re_list, axis=0),
#     alpha=0.2,
# )
plt.plot(x, np.mean(loss_kv_final_list, axis=0), lw=2, label="kv_final")
plt.fill_between(
    x,
    np.mean(loss_kv_final_list, axis=0)-np.std(loss_kv_final_list, axis=0),
    np.mean(loss_kv_final_list, axis=0)+np.std(loss_kv_final_list, axis=0),
    alpha=0.2,
)

plt.plot(x, np.mean(loss_fs_inference_list, axis=0), lw=2, label="inference")
plt.fill_between(x, np.mean(loss_fs_inference_list, axis=0)-np.std(loss_fs_inference_list, axis=0), np.mean(loss_fs_inference_list, axis=0)+np.std(loss_fs_inference_list, axis=0), alpha=0.2)

plt.plot(x, np.mean(loss_beta_list, axis=0), lw=2, label="beta")
plt.fill_between(x, np.mean(loss_beta_list, axis=0)-np.std(loss_beta_list, axis=0), np.mean(loss_beta_list, axis=0)+np.std(loss_beta_list, axis=0), alpha=0.2)

np.savez(
    "./results/noisy_LR.npz",
    x=x,
    loss_full_label_list=loss_full_label_list,
    loss_fs_inference_list=loss_fs_inference_list,
    loss_random_list=loss_random_list,
    loss_random_ensemble_list=loss_random_ensemble_list,
    loss_kv_cache_re_list=loss_kv_cache_re_list,
    loss_kv_final_list=loss_kv_final_list,
    loss_beta_list=loss_beta_list,
)

plot_line_means = {
    "x_full_label": np.arange(n_total),
    "x_prompt": x,
    "full_label_mean": np.mean(loss_full_label_list, axis=0),
    "random_mean": np.mean(loss_random_list, axis=0),
    "kv_final_mean": np.mean(loss_kv_final_list, axis=0),
    "inference_mean": np.mean(loss_fs_inference_list, axis=0),
    "beta_mean": np.mean(loss_beta_list, axis=0),
}
np.save("./results/noisy_LR_plot_lines.npy", plot_line_means, allow_pickle=True)

#plt.plot(loss_full_label, lw=2, label="Full label")
#plt.plot(loss_unlabeled_once, lw=2, label="Unlabeled once")
#plt.plot(loss_unlabeled_iter, lw=2, label="Unlabeled iter")
#plt.plot(loss_unlabeled_stepbystep, lw=2, label="Unlabeled step by step")
plt.xlabel("# in-context examples")
plt.ylabel("squared error")
plt.xlim(20, 30)
plt.legend()
plt.savefig("noisy_LR.png")