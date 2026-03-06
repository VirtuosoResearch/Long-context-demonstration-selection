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
        #pred = model.seq_inference(xs, ys)
        pred = model(xs, ys)
    #print(pred[0])
    metric = task.get_metric()
    loss = metric(pred, labels).cpu().numpy()

    return pred, loss.mean(axis=0)

def predict_full_label2(model, xs, ys, labels):
    n_labeled = 20
    n_total = xs.shape[1] - 1
    loss_list = []
    for i in range(10, n_total):
        query_x = xs[:, -1, :].unsqueeze(1)
        query_y = ys[:, -1].unsqueeze(1)
        print(query_y.shape)
        temp_xs = torch.cat([xs[:, :i, :], query_x], dim=1)
        temp_ys = torch.cat([ys[:, :i], query_y], dim=1)
        with torch.no_grad():
            pred = model(temp_xs, temp_ys)
        print(pred.shape)
        metric = task.get_metric()
        loss = metric(pred, temp_ys).cpu().numpy()
        loss_list.append(loss.mean(axis=0)[-1])

    return loss_list

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
loss_beta_list = []
loss_loss_list = []
loss_contrastive_list = []
loss_fs_inference_list = []

same_distribution = True


add_set_size = 5

def distance_score(model, seq, sample_list, new_sample):
    n = len(sample_list)
    #print(sample_list[0].x.shape)
    #print(sample_list[0])
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
    #print(embedding.mean(dim=0).shape)
    #print(new_embedding[0].shape)
    score = torch.norm(embedding.mean(dim=0) - new_embedding[0])
    score = 1 / score
    return score

def x_distance_score(model, sample_list, new_sample):
    n = len(sample_list)
    xs = torch.stack([s.x for s in sample_list], dim=0)
    #print(xs.shape)
    #print(new_sample.x.shape)
    score = torch.norm(xs.mean(dim=0) - new_sample.x, dim=0)
    #score = 1/score
    return score


def fs_inference_select(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list):
    n_points = xs.shape[1] - 1
    n = len(sample_list)
    print(n)
    pos_index = torch.zeros(n, n)
    seq_list = []
    for i in range(xs.shape[0]):
        seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, beta[i], i))

    # init select index for query in each sequence
    loss_list = []
    max_points = np.max(set_size_list)
    sub_sample_list = [[] for _ in range(b_size)]
    sub_sample_index = [[] for _ in range(b_size)]
    for k in range(1, max_points+1):
        for i in range(b_size):
            
            same_label_index = [i * n_points + j for j in range(n_points)]

            score = loss_score(model, seq_list[i], n_labeled, sample_list, sub_sample_list[i], task)

            select_index = torch.argmin(score)
            sub_sample_index[i].append(select_index)
            sub_sample_list[i].append(sample_list[select_index])
            seq_list[i].add_sample(sample_list[select_index])
            score[select_index] = 1e9
            print(sub_sample_index[i])
            if k in set_size_list:
                seq_list[i].get_input()
        if k in set_size_list:
            # get the input tensor
            xs, ys = sequence_to_tensor(seq_list)
            with torch.no_grad():
                pred = model(xs, ys)
            metric = task.get_metric()
            loss = metric(pred, ys).cpu().numpy()
            loss_query = loss.mean(axis=0)[-1]
            #print(loss_query)
            loss_list.append(loss_query)

    return loss_list

def fs_select(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list):
    n_points = xs.shape[1] - 1
    n = len(sample_list)
    print(n)
    pos_index = torch.zeros(n, n)

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
            n_val = 5
            i_x = temp_x[i]
            i_y = temp_y[i]
            pred_0, embeds_0 = model.forward_with_embeds(i_x, i_y)
            grad = torch.autograd.grad(pred_0[0, -1], embeds_0, retain_graph=True, create_graph=True)[0]
            grad = grad[:, 0::2, :]
            grad_0 = grad[:, -1, :]
            embeds_0 = embeds_0[:, 0::2, :]

            metric = task.get_metric()

            for j in range(n):
                if j in sub_sample_index[i]:
                    continue
                sub_i_x, sub_i_y = tensor_add_sample(i_x, i_y, sample_list[j])
                sub_seq_x_list.append(sub_i_x)
                sub_seq_y_list.append(sub_i_y)

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
            #print(score.sort())
            
            #score = score_infer
            
            threshold = 1e-3
            filtered_score = score.clone()
            #score[filtered_score < threshold] = 1e9
            topk = 10
            _, topk_indices = torch.topk(score, topk, largest=False)
            
            mean_x = 0
            mean_y = 0
            mean_beta = 0
            x_candidate = []
            y_candidate = []
            beta_candidate = []
            for idx in topk_indices:
                mean_x+=sample_list[idx].x
                mean_y+=sample_list[idx].y
                mean_beta+=sample_list[idx].beta
                beta_candidate.append(sample_list[idx].beta.unsqueeze(0))
                x_candidate.append(sample_list[idx].x.unsqueeze(0))
                y_candidate.append(sample_list[idx].y.unsqueeze(0))
                #print(sample_list[idx].x)
                #print(sample_list[idx].y)
                #print(sample_list[idx].beta[0,0])
            
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
            new_x = sample_list[topk_indices[0]].x
            #new_y = new_x @ mean_beta
            new_y = (new_x @ most_beta)
            #new_sample = Sample(mean_x, mean_y, 0, 0)
            #new_sample = Sample(new_x, new_y, 0, 0)
            new_sample = Sample(most_x, most_y, 0, most_beta)

            
            select_index = torch.argmin(score)
            sub_sample_index[i].append(select_index)
            sub_sample_list[i].append(sample_list[select_index])
            print("======= least: ", (sample_list[select_index].beta - sample_list[i*(n_total-1)].beta).mean())
            print("======= mean: ", (mean_beta - sample_list[i*(n_total-1)].beta).mean())
            #temp_x[i], temp_y[i] = tensor_add_sample(temp_x[i], temp_y[i], sample_list[select_index])
            #sub_sample_list[i].append(new_sample)
            temp_x[i], temp_y[i] = tensor_add_sample(temp_x[i], temp_y[i], new_sample)

            print(sub_sample_index[i])

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
            #print(loss_query)
            loss_list.append(loss_query)

    return loss_list

def fs_select2(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list):
    n_points = xs.shape[1] - 1
    n = len(sample_list)
    print(n)
    pos_index = torch.zeros(n, n)
    seq_list = []
    for i in range(xs.shape[0]):
        seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, beta[i], i))

    # init select index for query in each sequence
    loss_list = []
    loss_list_infer = []
    max_points = np.max(set_size_list)
    sub_sample_list = [[] for _ in range(b_size)]
    sub_sample_index = [[] for _ in range(b_size)]
    sub_sample_list_infer = [[] for _ in range(b_size)]
    sub_sample_index_infer = [[] for _ in range(b_size)]
    for k in range(1, max_points+1):
        for i in range(b_size):
            sub_seq_list = []
            n_val = 3
            i_seq = copy.deepcopy(seq_list[i])
            #i_seq.get_input(last_prompt=True)
            
            i_seq.get_input(query_range=n_val)
            xs, ys = sequence_to_tensor([i_seq])
            pred_0, embeds_0 = model.forward_with_embeds(xs, ys)
            print(pred_0[0, -n_val-1])
            grad = torch.autograd.grad(pred_0[0, -n_val:].mean(dim=-1), embeds_0, retain_graph=True, create_graph=True)[0]
            grad = grad[:, 0::2, :]
            grad_0 = grad[:, :, :]
            print(grad_0)
            embeds_0 = embeds_0[:, 0::2, :]


            metric = task.get_metric()

            delta_x = []
            for j in range(n):
                sub_seq = copy.deepcopy(seq_list[i])
                # becausre sample_list contains the last sample
                #delta_x.append(sample_list[j].x - sub_seq.prompt_x[-1])
                sub_seq.add_sample(sample_list[j])
                #sub_seq.get_input(last_prompt=True)
                #sub_seq.get_input(query_index=-2)
                sub_seq.get_input(query_range=n_val)
                #sub_seq.get_input(query_sample=sample_list[j])
                sub_seq_list.append(sub_seq)
            #delta_x = torch.stack(delta_x, dim=0)
            #delta_y = torch.zeros(delta_x.shape[0], delta_x.shape[-1]).to(device)
            #print(delta_x.shape)
            #print(delta_y.shape)
            sub_xs, sub_ys = sequence_to_tensor(sub_seq_list)
            embeds = model.embed_x(sub_xs)
            delta_embeds = embeds[:, -n_val-1, :] - embeds[:, -n_val-2, :]
            print(delta_embeds.shape)
            print(grad_0.shape)
            #print(embeds_0)
            #print(embeds[:, -2, :])
            #delta_term = torch.sum(grad_0 * delta_embeds, dim=1, keepdim=True)

            delta_term = delta_embeds @ grad_0[0, :, :].T
            print(delta_term.shape)
            print(pred_0.shape)
            #print(delta_embeds_real)
            #print(delta_embeds)
            #approx_pred = delta_term + pred_0[0, -1]
            approx_pred = pred_0
            #approx_pred[:, -n_val:] = approx_pred.expand(delta_term.shape[0], delta_term.shape[1])[:, -n_val:] + delta_term[:, -n_val:]
            approx_pred = approx_pred + delta_term
            #approx_pred = (delta_term + pred_0)
            print(delta_term)
            print('---------pred 0', pred_0[0, -n_val:])
            print(approx_pred.shape)
            #print("x0 pred", pred_0[0, -1])
            #print("delta ", delta_term)
            #print("approx pred", approx_pred)

            
            xs, ys = sequence_to_tensor(sub_seq_list)
            with torch.no_grad():
                pred = model(xs, ys)
            #print('---------real pred', pred[:5, -n_val:])
            #print('---------approx pred', approx_pred[:5, -n_val:])
            metric = task.get_metric()
            loss_infer = metric(pred, ys)
            #loss_approx = metric(approx_pred, ys)
            #print(loss_infer[:5, -1])
            #print(loss_approx[:5, -1])            
            score = loss_infer[:, -1]

            #score = score_infer
            select_index = torch.argmin(score)
            sub_sample_index[i].append(select_index)
            sub_sample_list[i].append(sample_list[select_index])
            seq_list[i].add_sample(sample_list[select_index])
            #print(a)
            print(sub_sample_index[i])
            if k in set_size_list:
                seq_list[i].get_input()
        if k in set_size_list:
            # get the input tensor
            xs, ys = sequence_to_tensor(seq_list)
            with torch.no_grad():
                pred = model(xs, ys)
            metric = task.get_metric()
            loss = metric(pred, ys).cpu().numpy()
            loss_query = loss.mean(axis=0)[-1]
            #print(loss_query)
            loss_list.append(loss_query)

    return loss_list

def fs_select3(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list):
    mse_loss = nn.MSELoss(reduction='none')
    n_points = xs.shape[1] - 1
    n = len(sample_list)
    print(n)
    pos_index = torch.zeros(n, n)
    seq_list = []
    for i in range(xs.shape[0]):
        seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, beta[i], i))

    # init select index for query in each sequence
    loss_list = []
    loss_list_infer = []
    max_points = np.max(set_size_list)
    sub_sample_list = [[] for _ in range(b_size)]
    sub_sample_index = [[] for _ in range(b_size)]
    sub_sample_list_infer = [[] for _ in range(b_size)]
    sub_sample_index_infer = [[] for _ in range(b_size)]
    select_index = 0
    for k in range(1, max_points+1):
        for i in range(b_size):
            sub_seq_list = []
            n_val = 5
            i_seq = copy.deepcopy(seq_list[i])
            #i_seq.get_input(last_prompt=True)
            i_seq.get_input(query_range=n_val)
            xs, ys = sequence_to_tensor([i_seq])
            pred_0, embeds_0 = model.forward_with_embeds(xs, ys)
            grad = torch.autograd.grad(pred_0[0, -1], embeds_0, retain_graph=True, create_graph=True)[0]
            grad = grad[:, 0::2, :]
            grad_0 = grad[:, -1, :]
            embeds_0 = embeds_0[:, 0::2, :]

            metric = task.get_metric()

            delta_x = []
            for j in range(n):
                if j in sub_sample_index[i]:
                    continue
                sub_seq = copy.deepcopy(seq_list[i])

                sub_seq.add_sample(sample_list[j])
                #sub_seq.get_input(last_prompt=True)
                sub_seq.get_input(query_range=n_val)
                sub_seq_list.append(sub_seq)

            sub_xs, sub_ys = sequence_to_tensor(sub_seq_list)
            embeds = model.embed_x(sub_xs)
            delta_embeds = embeds[:, -1, :] - embeds[:, -2, :]

            delta_term = torch.sum(grad_0 * delta_embeds, dim=1, keepdim=True)

            print(delta_term.shape)
            approx_pred = delta_term + pred_0[0, -1]
            xs, ys = sequence_to_tensor(sub_seq_list)
            with torch.no_grad():
                pred = model(xs, ys)
            print('---------real pred', pred[:5, -1])
            print('---------approx pred', approx_pred[:5, -1])
            metric = task.get_metric()
            #loss_infer = metric(pred, ys)
            #loss_approx = metric(approx_pred, ys)
            label = (xs @ seq_list[i].beta)[:, -1, 0]
            #print(label.shape)
            loss_infer = mse_loss(pred[:, -1], label)
            loss_approx = mse_loss(approx_pred[:, -1], label)
            #print(loss_infer.shape)
            print(loss_infer)
            print(loss_approx)
            score = loss_infer

            #score = score_infer
            select_index = torch.argmin(score)
            sub_sample_index[i].append(select_index)
            sub_sample_list[i].append(sample_list[select_index])
            seq_list[i].add_sample(sample_list[select_index])

            print(sub_sample_index[i])
            if k in set_size_list:
                seq_list[i].get_input()
        if k in set_size_list:
            # get the input tensor
            xs, ys = sequence_to_tensor(seq_list)
            with torch.no_grad():
                pred = model(xs, ys)
            metric = task.get_metric()
            loss = metric(pred, ys).cpu().numpy()
            loss_query = loss.mean(axis=0)[-1]
            #print(loss_query)
            loss_list.append(loss_query)

    return loss_list

def fs_select4(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list):
    n_points = xs.shape[1] - 1
    n = len(sample_list)
    n = 5
    print(n)
    pos_index = torch.zeros(n, n)

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
    set_size_list = [3]
    for i in range(b_size):
        temp_x.append(xs[i, :n_labeled, :].unsqueeze(0))
        temp_y.append(ys[i, :n_labeled].unsqueeze(0))

    task_index = range(1)
    for k in range(1, max_points+1):
        for i in task_index:
            ground_beta = sample_list[i*(n_total-1)].beta
            #print(ground_beta[:5,0])
            #if i > 0:
            #    continue
            sub_seq_x_list = []
            sub_seq_y_list = []
            n_val = 5
            i_x = temp_x[i]
            i_y = temp_y[i]
            pred_0, embeds_0 = model.forward_with_embeds(i_x, i_y)
            grad = torch.autograd.grad(pred_0[0, -1], embeds_0, retain_graph=True, create_graph=True)[0]
            grad = grad[:, 0::2, :]
            grad_0 = grad[:, -1, :]
            embeds_0 = embeds_0[:, 0::2, :]

            metric = task.get_metric()
            i_sample_list = []
            for j in range(n):
                sub_x = XS[i, k+n_labeled-1, :]
                sub_x = torch.randn(sub_x.shape).to(device)
                i_class = torch.randint(0, n_classes, (1,))
                sub_beta = beta[i_class]
                #sub_beta = torch.randn(sub_x.shape[0], 1).to(device)
                sub_y = sub_x @ sub_beta
                sub_sample = Sample(sub_x, sub_y, i_class, sub_beta)
                sub_i_x, sub_i_y = tensor_add_sample(i_x, i_y, sub_sample)
                #sub_i_x, sub_i_y = tensor_add_xy(i_x, i_y, temp_x, temp_y)
                sub_seq_x_list.append(sub_i_x)
                sub_seq_y_list.append(sub_i_y)
                i_sample_list.append(sub_sample)

            sub_xs = torch.cat(sub_seq_x_list, dim=0)
            sub_ys = torch.cat(sub_seq_y_list, dim=0)
            sub_label = copy.deepcopy(sub_ys)
            sub_label[:, -1] = sub_label[:, -2]
            embeds = model.embed_x(sub_xs)
            delta_embeds = embeds[:, -1, :] - embeds[:, -2, :]

            delta_term = torch.sum(grad_0 * delta_embeds, dim=1, keepdim=True)

            with torch.no_grad():
                pred = model(sub_xs, sub_ys)

            metric = task.get_metric()
            loss_infer = metric(pred, sub_ys)
            score = loss_infer[:, -1]
            #print(score.sort())

            #score = score_infer
            
            threshold = 1e-3
            filtered_score = score.clone()
            #score[filtered_score < threshold] = 1e9
            topk = 3
            _, topk_indices = torch.topk(score, topk, largest=False)
            #print(topk_indices)
            #print(score[topk_indices])
            #print(sub_xs[topk_indices])
            #print(sub_ys[topk_indices])
            #print("ground beta: ", sample_list[i*n_total].beta[0,0])
            
            mean_x = 0
            mean_y = 0
            mean_beta = 0
            for idx in topk_indices:
                #print("============")
                mean_x+=i_sample_list[idx].x
                mean_y+=i_sample_list[idx].y
                mean_beta+=i_sample_list[idx].beta
                #print(i_sample_list[idx].x)
                #print(i_sample_list[idx].y)
                #print(i_sample_list[idx].beta[0,0])
            mean_x /= topk
            mean_y /= topk
            #print(a)
            #print(mean_x.shape)
            #print(mean_y.shape)

            #new_sample = Sample(mean_x, mean_y, 0, 0)
            #new_sample = sample_list[i*n_labeled+k]
            #print(new_sample.beta[0,0])
            

            select_index = torch.argmin(score)
            sub_sample_index[i].append(select_index)
            #sub_sample_list[i].append(i_sample_list[select_index])
            #print((i_sample_list[select_index].beta - sample_list[i*n_total].beta).mean())
            #print(sample_list[select_index].x.shape)
            temp_x[i], temp_y[i] = tensor_add_sample(temp_x[i], temp_y[i], i_sample_list[select_index])
            print((new_sample.beta - ground_beta).mean())
            #sub_sample_list[i].append(new_sample)
            #temp_x[i], temp_y[i] = tensor_add_sample(temp_x[i], temp_y[i], new_sample)
            #print(temp_x[i][0,-5:,0])

            #print(sub_sample_index[i])

        if k in set_size_list:
            # get the input tensor
            query_xs, query_ys = [], []
            for _ in task_index:
                #if _ >0:
                #    continue
                _x, _y = tensor_add_xy(temp_x[_], temp_y[_], xs[i, -1, :], ys[i, -1])
                #_x, _y = tensor_add_xy(temp_x[_], temp_y[_], xs[i, n_labeled+k, :], ys[i, n_labeled+k])
                query_xs.append(_x)
                query_ys.append(_y)
            test_y = xs[i, -1, : ] @ ground_beta
            #print(test_y - ys[i, -1])
            query_xs = torch.cat(query_xs, dim=0)
            query_ys = torch.cat(query_ys, dim=0)
            #print(query_xs[0, -6:, 0])
            #print(a)
            with torch.no_grad():
                pred = model(query_xs, query_ys)
            metric = task.get_metric()
            loss = metric(pred, query_ys).cpu().numpy()
            loss_query = loss.mean(axis=0)[-1]
            #print(loss_query)
            loss_list.append(loss_query)

    return loss_list


def loss_select(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list):
    loss_list = []
    metric = task.get_metric()
    seq_list = []
    for i in range(b_size):
        seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, beta[i], i))
    
    for set_size in set_size_list:
        
        for temp_set_size in range(set_size):
            for outer_i in range(b_size):
                min_j = outer_i
                min_loss = 1e9
                for inner_j in range(b_size):
                    if outer_i != inner_j:
                        temp_seq = seq_list[outer_i]
                        temp_seq.add_sample(sample_list[inner_j * n_total + temp_set_size])
                        temp_seq.get_input()
                        temp_xs = temp_seq.input_x.unsqueeze(0)
                        temp_ys = temp_seq.input_y.unsqueeze(0)
                        with torch.no_grad():
                            #pred = model.seq_inference(xs, ys)
                            pred = model(temp_xs, temp_ys)
                        #print(pred[0])
                        
                        loss = metric(pred, temp_ys).numpy()
                        loss_query = loss.mean(axis=0)[-1]
                        print(f"loss for {outer_i} {inner_j}: {loss_query}")
                        if loss_query < min_loss:
                            min_loss = loss_query
                            min_j = inner_j
                        temp_seq.del_sample()
                print(f"min j for {outer_i}: {min_j}, {min_loss}")
                seq_list[outer_i].add_sample(sample_list[min_j * n_total + temp_set_size])
        for seq in seq_list:
            print(seq.get_prompt_length())
            seq.get_input(query_index=-2)

        xs, ys = sequence_to_tensor(seq_list)
        with torch.no_grad():
            #pred = model.seq_inference(xs, ys)
            pred = model(xs, ys)
        #print(pred[0])
        
        loss = metric(pred, ys).numpy()
        loss_query = loss.mean(axis=0)[-1]
        #print(loss_query)
        print(seq_list[0].get_prompt_length())
        loss_list.append(loss_query)

    return loss_list

def beta_select(model, xs, ys, beta, sample_list, n_labeled, set_size_list):
    n_total = xs.shape[1]-1
    loss_list = []
    # find nearest beta
    beta_distance = torch.cdist(beta.view(b_size, n_dims), beta.view(b_size, n_dims))
    beta_distance = beta_distance + torch.eye(beta.shape[0], device=beta.device) * 1e9
    #print(torch.min(beta_distance, dim=1))
    nearest_beta = torch.argmin(beta_distance, dim=1)
    #print(nearest_beta)
    for set_size in set_size_list:
        seq_list = []
        for i in range(xs.shape[0]):
            seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, beta[i], i))
        # select prompt samples
        # get random samples
        for i in range(xs.shape[0]):
            for j in range(set_size):
                select_index = nearest_beta[i] * n_total + j + n_labeled
                seq_list[i].add_sample(sample_list[select_index])

        #print(seq_list)
        # get least loss samples
        for seq in seq_list:
            seq.get_input()
        #seq_list[0].add_sample(sample_list[2])
        #print(seq_list[0])

        xs, ys = sequence_to_tensor(seq_list)
        #print(xs.shape)

        
        with torch.no_grad():
            #pred = model.seq_inference(xs, ys)
            pred = model(xs, ys)
        #print(pred[0])
        metric = task.get_metric()
        loss = metric(pred, ys).cpu().numpy()
        loss_query = loss.mean(axis=0)[-1]
        #print(loss_query)
        print(seq_list[0].get_prompt_length())
        loss_list.append(loss_query)
    
    return loss_list

def random_select(model, xs, ys, beta, sample_list, n_labeled, set_size_list):
    loss_list = []
    for set_size in set_size_list:
        seq_list = []
        for i in range(xs.shape[0]):
            seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, beta[i], i))
        # select prompt samples
        # get random samples
        for i in range(xs.shape[0]):
            beta_list = []
            #print(seq_list[i].beta[0,0])
            for j in range(set_size):
                select_index = torch.randint(0, len(sample_list), (1,))
                #select_index = i * n_total + j + n_labeled
                #select_index = nearest_beta[i] * n_total + j + n_labeled
                #print(select_index)
                seq_list[i].add_sample(sample_list[select_index])
                beta_list.append(sample_list[select_index].beta[0, 0])
            #print(beta_list)

        #print(seq_list)
        # get least loss samples
        for seq in seq_list:
            seq.get_input(query_index=-2)
        #seq_list[0].add_sample(sample_list[2])
        #print(seq_list[0])

        xs, ys = sequence_to_tensor(seq_list)
        #print(xs.shape)

        
        with torch.no_grad():
            #pred = model.seq_inference(xs, ys)
            pred = model(xs, ys)
        #print(pred[0])
        metric = task.get_metric()
        loss = metric(pred, ys).cpu().numpy()
        loss_query = loss.mean(axis=0)[-1]
        #print(loss_query)
        #print(seq_list[0].get_prompt_length())
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
    #print(xs_b[0])
    return xs_b

def fs_select_debug3(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list):
    n_points = xs.shape[1] - 1
    n = len(sample_list)
    print(n)
    pos_index = torch.zeros(n, n)
    seq_list = []
    for i in range(xs.shape[0]):
        seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, beta[i], i))

    # init select index for query in each sequence
    loss_list = []
    loss_list_infer = []
    max_points = np.max(set_size_list)
    sub_sample_list = [[] for _ in range(b_size)]
    sub_sample_index = [[] for _ in range(b_size)]

    test_i = 0
    test_j = n_points
    n_val = 1
    i_seq = copy.deepcopy(seq_list[test_i])
    i_seq.pad()
    #i_seq.get_input(query_range=n_val)
    i_seq.get_input(last_prompt=True)
    xs, ys = sequence_to_tensor([i_seq])
    xs_variable = xs.clone()
    ys_variable = ys.clone()
    seq_variable = model._combine(xs_variable, ys_variable).detach()[0,:,:].unsqueeze(0)
    print(seq_variable[:, :, 0])
    seq_variable.requires_grad = True
    pred = model.forward_by_seq(seq_variable)
    #embeds_0.retain_grad()
    #embeds_0.requires_grad = True
    print("pred: ", pred[0, -1])
    metric = task.get_metric()
    loss = F.mse_loss(pred[0, -2], ys[0, -2])
    loss.backward()
    #print(embeds_0.grad.shape)
    #grad = embeds_0.grad.detach()[:, 0::2, :]
    #grad = embeds_0.grad.detach()
    grad = seq_variable.grad.detach()

    print("grad shape", grad.shape)
    #print("--beta", sample_list[test_j].beta)
    #embeds_0 = embeds_0[:, -n_val-1, :]
    new_x = i_seq.prompt_x[-1]
    beta = torch.randn(n_dims).to(device)
    new_y = torch.dot(new_x, beta)
    test_sample = Sample(new_x, new_y, test_i, beta)
    test_sample = sample_list[test_j]
    new_seq = copy.deepcopy(seq_list[test_i])
    new_seq.add_sample(test_sample)
    #new_seq.get_input(query_range=n_val)
    new_seq.get_input(last_prompt=True)
    new_xs, new_ys = sequence_to_tensor([new_seq])
    print((new_xs - xs)[0, :, 0])
    print((new_ys - ys))
    new_seq_variable = model._combine(new_xs, new_ys)
    #delta_embeds = new_embeds - embeds_0
    print(new_seq_variable.shape)
    print(seq_variable.shape)
    delta_seq = new_seq_variable - seq_variable
    delta_seq = delta_seq[0, :, :]

    #delta_term = torch.sum(grad * delta_embeds)
    delta_term = torch.sum(grad * delta_seq)
    print(delta_term.shape)
    #delta_embeds = model.embed_x(delta_x)
    #print(delta_embeds_real)
    #print(delta_embeds)
    approx_loss = delta_term + loss
    print("x0 loss", loss)
    print("delta ", delta_term)
    print("approx loss", approx_loss)
    #print(approx_loss)
    #taylor_approx = y1 + torch.sum(grad_h1 * delta_h)
    pred = model(new_xs, new_ys)
    real_loss = F.mse_loss(pred[0, -2], ys[0, -2]).mean()
    print("real loss", real_loss)
    #print(real_loss[:, -1])
    
    print(a)
    return loss_list

def generate_orthogonal_matrix(n, m):
    if n > m:
        raise ValueError("The number of rows (n) should not be greater than the number of columns (m) for an orthogonal set.")
    
    # Generate a random matrix
    A = torch.randn(n, m)
    
    # Apply Gram-Schmidt process
    Q, _ = torch.linalg.qr(A.T)  # QR decomposition on the transpose
    return Q.T  # Transpose back to get row-wise orthogonality


runs = 3
for run in range(runs):
    task = task_sampler()
    #xs = generate_synthetic_data(num_sequences=b_size, n_points=n_total, x_dim=n_dims, diff_diftribution=False, alpha=0.0)
    x_single = torch.randn(n_total, n_dims)
    #x_single = (x_single - 0.5) * 2
    #xs = x_single.unsqueeze(0).expand(b_size, -1, -1)
    xs = torch.randn(b_size, n_total, n_dims)
    xs = xs.to(device)
    XS = xs
    n_classes = b_size // 10
    n_classes = 5
    #anchor_points = torch.randn(n_classes, n_dims).to(device)
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
    

    set_size_list = [1,2,3,4,5,7,10]

    pred_full_label, loss_full_label = predict_full_label(model, xs, ys=labels, labels=labels)
    #loss_full_label =  predict_full_label2(model, xs, ys=labels, labels=labels)
    loss_full_label_list.append(loss_full_label)
    print(loss_full_label_list)
    
    
    loss_fs_inference = fs_select(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list)
    print(loss_fs_inference)
    loss_fs_inference_list.append(loss_fs_inference)
    
    loss_random = random_select(model, xs, ys, beta, sample_list, n_labeled, set_size_list)
    print(loss_random)
    loss_random_list.append(loss_random)

    loss_beta = beta_select(model, xs, ys, beta, sample_list, n_labeled, set_size_list)
    print(loss_beta)
    loss_beta_list.append(loss_beta)



x = np.arange(n_total)
plt.plot(x, np.mean(loss_full_label_list, axis=0), lw=2, label="Full label")
plt.fill_between(x, np.mean(loss_full_label_list, axis=0)-np.std(loss_full_label_list, axis=0), np.mean(loss_full_label_list, axis=0)+np.std(loss_full_label_list, axis=0), alpha=0.2)

x = []
for i in set_size_list:
    x.append(n_labeled + i)
x = np.array(x)
print(x)

plt.plot(x, np.mean(loss_random_list, axis=0), lw=2, label="random")
plt.fill_between(x, np.mean(loss_random_list, axis=0)-np.std(loss_random_list, axis=0), np.mean(loss_random_list, axis=0)+np.std(loss_random_list, axis=0), alpha=0.2)

plt.plot(x, np.mean(loss_fs_inference_list, axis=0), lw=2, label="inference")
plt.fill_between(x, np.mean(loss_fs_inference_list, axis=0)-np.std(loss_fs_inference_list, axis=0), np.mean(loss_fs_inference_list, axis=0)+np.std(loss_fs_inference_list, axis=0), alpha=0.2)

np.savez("./results/noisy_LR.npz", x=x, loss_full_label_list=loss_full_label_list, loss_fs_inference_list=loss_fs_inference_list, loss_random_list=loss_random_list)

#plt.plot(loss_full_label, lw=2, label="Full label")
#plt.plot(loss_unlabeled_once, lw=2, label="Unlabeled once")
#plt.plot(loss_unlabeled_iter, lw=2, label="Unlabeled iter")
#plt.plot(loss_unlabeled_stepbystep, lw=2, label="Unlabeled step by step")
plt.xlabel("# in-context examples")
plt.ylabel("squared error")
plt.legend()
plt.savefig("noisy_LR.png")