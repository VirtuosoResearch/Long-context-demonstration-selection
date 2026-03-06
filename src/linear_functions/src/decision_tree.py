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



sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')

run_dir = "../models"

from samplers import get_data_sampler
from tasks import get_task_sampler
#task = "linear_regression"
#task = "sparse_linear_regression"
task = "decision_tree"
#task = "relu_2nn_regression"

run_id = "pretrained"  # if you train more models, replace with the run_id from the table above

run_path = os.path.join(run_dir, task, run_id)
recompute_metrics = False
print(run_path)
if recompute_metrics:
    get_run_metrics(run_path)  # these are normally precomputed at the end of training

model, conf = get_model_from_run(run_path)

n_dims = conf.model.n_dims
#batch_size = conf.training.batch_size
batch_size = 40

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

def predict_unlabled_once(model, xs, ys, labels, n_labeled):
    with torch.no_grad():
        #pred = model.seq_inference(xs, ys)
        pred = model(xs, ys)
    #print(pred[0])
    metric = task.get_metric()
    loss = metric(pred, labels).cpu().numpy()
    loss_labeled = loss[:, :n_labeled]
    loss_unlabeled = loss[:, n_labeled:]

    return pred, loss.mean(axis=0), loss_labeled.mean(axis=0), loss_unlabeled.mean(axis=0)

def predict_unlabled_iteration(model, xs, ys, labels, n_labeled, n_unlabeled, n_iter, ratio=1):
    #ratio = math.ceil((n_labeled + n_unlabeled) // n_unlabeled)
    #ratio = 2
    # find the most similar unlabeled data
    for b in range(xs.shape[0]):
        for i in range(n_unlabeled):
            l2_dist = torch.norm(xs[b, :] - xs[b, n_labeled+i].unsqueeze(0), dim=1)
            best_match = torch.argmin(l2_dist)
            #ys[b, n_labeled+i] = ys[b, best_match]


    print(f"ratio: {ratio}")
    loss_unlabeled_list = []
    for i in range(n_iter):
        with torch.no_grad():
            #pred = model.seq_inference(xs, ys)
            pred = model(xs, ys)
        #print(pred[0])
        metric = task.get_metric()
        loss = metric(pred, labels).cpu().numpy()
        loss_labeled = loss[:, :n_labeled]
        loss_unlabeled = loss[:, n_labeled:]
        #print(loss.shape)
        #print(loss_unlabeled.shape)
        loss_unlabeled_list.append(loss_unlabeled)
        
        #print(loss_unlabeled.mean())

        # update ys by random choice
        ys[:, n_labeled:] = pred[:, n_labeled:]
        
        index = np.random.choice(n_unlabeled, n_unlabeled//ratio, replace=False)
        print(f"iter {i}: labeled loss {loss_labeled.mean()}, unlabeled loss {loss_unlabeled.mean()}, mask {index}")
        if len(loss_unlabeled_list) > 1 and np.abs(loss_unlabeled_list[-1].mean() - loss_unlabeled_list[-2].mean()) < 1e-3:
            break
        #print(index)
        #print(ys[:, n_labeled:].shape)
        #print(ys[:, n_labeled:][index+ n_labeled].shape)
        for j in range(n_unlabeled):
            if j not in index:
                ys[:, n_labeled+j] = torch.zeros_like(ys[:, n_labeled+j])
        #print(ys[0, n_labeled:])
    
    return pred, loss.mean(axis=0), loss_labeled.mean(axis=0), loss_unlabeled.mean(axis=0)

def predict_unlabled_stepbysetp(model, xs, ys, labels, n_labeled, n_unlabeled):
    for i in range(n_unlabeled):
        with torch.no_grad():
            #pred = model.seq_inference(xs, ys)
            pred = model(xs, ys)
        #print(pred[0])
        metric = task.get_metric()
        loss = metric(pred, labels).cpu().numpy()
        loss_labeled = loss[:, :n_labeled]
        loss_unlabeled = loss[:, n_labeled:]
        #print(loss.shape)
        #print(loss_unlabeled.shape)
        #print(loss_unlabeled.mean())

        # update ys by random choice
        ys[:, n_labeled+i] = pred[:, n_labeled+i]
        print(f"iter {i}: labeled loss {loss_labeled.mean()}, unlabeled loss {loss_unlabeled.mean()}")

    return pred, loss.mean(axis=0), loss_labeled.mean(axis=0), loss_unlabeled.mean(axis=0)

import torch.nn.functional as F
import copy
from datetime import datetime

device = 'cuda:0'
model = model.to(device)

model.eval()
use_checkpoint = True
if not use_checkpoint:
    n_total = 5
    n_labeled = 2
    n_dims = 2
    b_size = 3
else:
    n_total = 80
    n_labeled = 40
    n_dims = conf.model.n_dims
    #b_size = conf.training.batch_size
    b_size = batch_size
ratio = n_total // n_labeled

n_unlabeled = n_total - n_labeled
runs = 1
loss_full_label_list = []
loss_random_list = []
loss_beta_list = []
loss_loss_list = []
loss_contrastive_list = []
loss_fs_inference_list = []
loss_fs_estimate_list = []

same_distribution = True

noise = 1
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

def contrastive_score(model, seq, n_labeled, sub_sample_list, new_sample):
    n_sub_prompt = n_labeled - 1
    sample_list = copy.deepcopy(sub_sample_list)
    sample_list.append(new_sample)
    n = len(sample_list)
    #print(sample_list[0].x.shape)
    #print(sample_list[0])
    # concat sample_list to seq
    sub_seq_list = []
    for i in range(n):
        sub_seq = copy.deepcopy(seq)
        # becausre sample_list contains the last sample
        sub_seq.del_sample()
        sub_seq.add_query(sample_list[i])
        sub_seq.get_input()
        sub_seq_list.append(sub_seq)

    xs, ys = sequence_to_tensor(sub_seq_list)

    with torch.no_grad():
        embedding = model.encoder(xs, ys)[:, 0::2, :]
    #print(embedding.shape)
    sample_embs = F.normalize(embedding[:, -1, :], dim=-1)
    #print(sample_embs.shape)

    pos_index = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if sample_list[i].c == sample_list[j].c:
                pos_index[i].append(j)

    print(pos_index)

    tau = 1
    contrastive_loss_each_sample = []
    for i in range(n):
        item = 0.0
        neg_sim = torch.sum(torch.exp(sample_embs[i] @ sample_embs.T / tau))
        if len(pos_index[i]) > 0:
            #print(neg_sim.shape)
            for j in pos_index[i]:
                pos_sim = torch.exp(sample_embs[i] @ sample_embs[j].T / tau)
                item += -torch.log(pos_sim / neg_sim)
            item = item / len(pos_index[i])
            item = item.float().item()
        else:
            item = -torch.log(torch.tensor(1.0 / neg_sim))
            item = item.float().item()
        contrastive_loss_each_sample.append(item)
    print(contrastive_loss_each_sample)

    contrastive_loss = torch.mean(torch.tensor(contrastive_loss_each_sample))
    print("loss:", contrastive_loss)

    return contrastive_loss

def estimate_score(model, seq, n_labeled, all_sample_list, sub_sample_list, task):
    n_sub_prompt = n_labeled - 1
    n_val = 5
    n = len(all_sample_list)
    sub_seq_list = []
    start_time = datetime.now()
    var_seq = copy.deepcopy(seq)
    var_seq.get_input(query_range=n_val)
    xs, ys = sequence_to_tensor([var_seq])
    xs_variable = xs.clone()
    xs_variable.requires_grad = True

    pred = model(xs_variable, ys)
    metric = task.get_metric()
    loss = metric(pred, ys)

    #embeds_0 = model.embed(xs, ys)
    #print(embeds_0.requires_grad)

    grad = torch.autograd.grad(loss.mean(), xs_variable, allow_unused=True)
    #print("grad: ", len(grad))
    #print(xs_variable.shape)
    #print(grad[0].shape)
    delta_x = []
    for i in range(n):
        sub_seq = copy.deepcopy(seq)
        # becausre sample_list contains the last sample
        sub_seq.add_sample(all_sample_list[i])
        delta_x.append(sub_seq.prompt_x[-1] - sub_seq.prompt_x[-2])
        #sub_seq.get_input(query_index=-2)
        sub_seq.get_input(query_range=n_val)
        sub_seq_list.append(sub_seq)
    print(datetime.now() - start_time)
    start_time = datetime.now()
    xs, ys = sequence_to_tensor(sub_seq_list)

    #with torch.no_grad():
    #embeds = model.embed(xs, ys)
    
    #delta_embeds = embeds[:, -1] - embeds[:, -2]
    #delta_x = xs[:, -n_val-1] - xs[:, -n_val-2]
    delta_x = torch.stack(delta_x, dim=0)
    #print(delta_x.shape)
    #print(delta_x.shape)
    #print(grad[0].shape)
    pred = delta_x @ grad[0][0, -1, :].T
    #print(pred.shape)
    #print(a)
    #print(xs.shape)
    #print(pred.shape)
    #print(ys.shape)
    metric = task.get_metric()
    #print(pred[:, -1])
    loss = metric(pred, ys[:, -1])
    #print(loss.shape)
    
    score = loss

    print("model time", datetime.now() - start_time)
    #print(loss_sample)
    return score

def loss_score(model, seq, n_labeled, all_sample_list, sub_sample_list, task):
    n_sub_prompt = n_labeled - 1
    n_val = 5
    n = len(all_sample_list)
    sub_seq_list = []
    start_time = datetime.now()
    for i in range(n):
        sub_seq = copy.deepcopy(seq)
        # becausre sample_list contains the last sample
        
        #sub_seq.get_input(query_index=-2)
        #sub_seq.add_sample(all_sample_list[i])
        #sub_seq.get_input(query_range=n_val)
        sub_seq.get_input(query_sample=all_sample_list[i])
        sub_seq_list.append(sub_seq)
    print(datetime.now() - start_time)
    start_time = datetime.now()
    xs, ys = sequence_to_tensor(sub_seq_list)
    with torch.no_grad():
        pred = model(xs, ys)
    metric = task.get_metric()
    loss = metric(pred, ys)
    loss_val = loss[:, -n_val:].mean(-1)
    #score = loss[:, -1]
    score = loss_val
    print("model time", datetime.now() - start_time)
    #print(loss_sample)
    return score

def contrastive_select(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list):
    n_points = xs.shape[1] - 1
    n = len(sample_list)
    print(n)
    pos_index = torch.zeros(n, n)
    seq_list = []
    for i in range(xs.shape[0]):
        seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, beta[i], i))

    # init select index for query in each sequence
    loss_list = []
    for set_size in set_size_list:
        for i in range(b_size):
            # use the first sample as the initilization
            first_index = i * n_points
            init_index = np.arange(i*n_points, i*n_points+n_labeled)
            #sub_sample_list  = [sample_list[j] for j in init_index]
            sub_sample_list = [sample_list[init_index[-1]]]
            #sub_sample_list.append(sample_list[first_index])
            # traverse all the samples except the first one
            #sub_sample_index = init_index.tolist()
            sub_sample_index = [init_index[-1]]
            #sub_sample_index.append(first_index)
            score = torch.zeros(n)
            #score[first_index] = 1e9
            for _ in init_index:
                score[_] = 1e9
            for k in range(set_size):
                for j in range(n):
                    if j in sub_sample_index:
                        continue
                    # compute the contrastive score for each sample
                    #score[j] = distance_score(model, seq_list[i], sub_sample_list, sample_list[j])
                    #score[j] = x_distance_score(model, sub_sample_list, sample_list[j])
                    #score[j] = contrastive_score(model, seq_list[i], n_labeled, sub_sample_list, sample_list[j])
                    score[j] = loss_score(model, seq_list[i], n_labeled, sub_sample_list, sample_list[j], task)
                select_index = torch.argmin(score)
                sub_sample_index.append(select_index)
                sub_sample_list.append(sample_list[select_index])
                seq_list[i].add_sample(sample_list[select_index])
                score[select_index] = 1e9
            print(sub_sample_index)
            # get prompt
            seq_list[i].get_input()
        # get the input tensor
        xs, ys = sequence_to_tensor(seq_list)
        with torch.no_grad():
            pred = model(xs, ys)
        metric = task.get_metric()
        loss = metric(pred, ys).numpy()
        loss_query = loss.mean(axis=0)[-1]
        #print(loss_query)
        loss_list.append(loss_query)

    return loss_list

def fs_inference_select(model, xs, ys, sample_list, b_size, n_labeled, set_size_list):
    n_points = xs.shape[1] - 1
    n = len(sample_list)
    print(n)
    pos_index = torch.zeros(n, n)
    

    # init select index for query in each sequence
    loss_list = []
    max_points = np.max(set_size_list)
    sub_sample_list = [[] for _ in range(b_size)]
    sub_sample_index = [[] for _ in range(b_size)]
    task_index = range(1)
    seq_list = []
    for i in task_index:
        seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, i))
    for k in range(1, max_points+1):
        for i in task_index:
            
            same_label_index = [i * n_points + j for j in range(n_points)]

            score = loss_score(model, seq_list[i], n_labeled, sample_list, sub_sample_list[i], task)
            #score = estimate_score(model, seq_list[i], n_labeled, sample_list, sub_sample_list[i], task)

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

def fs_estimate_select(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list):
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

            #score = loss_score(model, seq_list[i], n_labeled, sample_list, sub_sample_list[i], task)
            score = estimate_score(model, seq_list[i], n_labeled, sample_list, sub_sample_list[i], task)

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

def fs_select(model, xs, ys, sample_list, b_size, n_labeled, set_size_list, use_approx=True):
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
    task_index = range(1)
    seq_list = []
    for i in task_index:
        seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, i))
    for k in range(1, max_points+1):
        for i in task_index:
            sub_seq_list = []
            n_val = 5
            i_seq = copy.deepcopy(seq_list[i])
            i_seq.get_input(last_prompt=True)
            #sub_seq.get_input(query_range=n_val)
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
                sub_seq.get_input(last_prompt=True)

                sub_seq_list.append(sub_seq)

            sub_xs, sub_ys = sequence_to_tensor(sub_seq_list)
            embeds = model.embed_x(sub_xs)
            delta_embeds = embeds[:, -1, :] - embeds[:, -2, :]

            delta_term = torch.sum(grad_0 * delta_embeds, dim=1, keepdim=True)

            approx_pred = delta_term + pred_0[0, -1]

            xs, ys = sequence_to_tensor(sub_seq_list)
            with torch.no_grad():
                pred = model(xs, ys)

            metric = task.get_metric()
            loss_infer = metric(pred, ys)
            loss_approx = metric(approx_pred, ys)

            if use_approx:
                score = loss_approx[:, -1]
            else:
                score = loss_infer[:, -1]

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
            print('---------real pred', pred[:5, -n_val:])
            print('---------approx pred', approx_pred[:5, -n_val:])
            metric = task.get_metric()
            loss_infer = metric(pred, ys)
            loss_approx = metric(approx_pred, ys)
            print(loss_infer[:5, -1])
            print(loss_approx[:5, -1])            
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

def random_select(model, xs, ys, sample_list, n_labeled, set_size_list):
    loss_list = []
    for set_size in set_size_list:
        seq_list = []
        for i in range(xs.shape[0]):
            seq_list.append(Input_sequence(xs[i], ys[i], n_labeled, i))
        # select prompt samples
        # get random samples
        for i in range(xs.shape[0]):
            for j in range(set_size):
                select_index = torch.randint(0, len(sample_list), (1,))
                #select_index = i * n_total + j + n_labeled
                #select_index = nearest_beta[i] * n_total + j + n_labeled
                #print(select_index)
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


class Sample():
    def __init__(self, x, y, c):
        self.x = x
        self.y = y
        self.c = c
    
    def __repr__(self):
        return f"x: {self.x}, y: {self.y}, c: {self.c}\n"

class Input_sequence():
    def __init__(self, xs, ys, length, c):
        self.xs = xs
        self.ys = ys
        self.prompt_x = xs[:length]
        self.prompt_y = ys[:length]
        self.query_x = xs[-1]
        self.query_y = ys[-1]
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
    xs = data_sampler.sample_xs(b_size=batch_size, n_points=n_total)
    xs = xs.to(device)
    ys = task.evaluate(xs)
    ys = ys.to(device)
    #print(ys.shape)
    labels = ys.clone()
    #ys[:, n_labeled:] = torch.zeros_like(ys[:, n_labeled:])
    #print(labels[0])
    #print(ys[0])
    with torch.no_grad():
        embedding = model.encoder(xs, ys)
    print(embedding.shape)

    # split sequences
    sample_list = []
    for i in range(xs.shape[0]):
        # -1 for remaining the last sample as query
        for j in range(xs.shape[1]-1):
            s = Sample(xs[i, j], ys[i, j], i)
            sample_list.append(s)
    
    #print(sample_list)

    set_size_list = [1,5,10,15,20]
    #print(xs)
    #print(ys)
    pred_full_label, loss_full_label = predict_full_label(model, xs, ys=labels, labels=labels)
    loss_full_label_list.append(loss_full_label)
    
    
    loss_fs_estimate = fs_select(model, xs, ys, sample_list, b_size, n_labeled, set_size_list, use_approx=True)
    #loss_fs_estimate = fs_select_debug3(model, xs, ys, beta, sample_list, b_size, n_labeled, set_size_list)
    print(loss_fs_estimate)
    loss_fs_estimate_list.append(loss_fs_estimate)

    #loss_fs_inference = fs_inference_select(model, xs, ys, sample_list, b_size, n_labeled, set_size_list)
    loss_fs_inference = fs_select(model, xs, ys, sample_list, b_size, n_labeled, set_size_list, use_approx=False)
    print(loss_fs_inference)
    loss_fs_inference_list.append(loss_fs_inference)
    
    loss_random = random_select(model, xs, ys, sample_list, n_labeled, set_size_list)
    print(loss_random)
    loss_random_list.append(loss_random)

    #loss_beta = beta_select(model, xs, ys, beta, sample_list, n_labeled, set_size_list)
    #print(loss_beta)
    #loss_beta_list.append(loss_beta)



x = np.arange(n_total)
plt.plot(x, np.mean(loss_full_label_list, axis=0), lw=2, label="Full label")
plt.fill_between(x, np.mean(loss_full_label_list, axis=0)-np.std(loss_full_label_list, axis=0), np.mean(loss_full_label_list, axis=0)+np.std(loss_full_label_list, axis=0), alpha=0.2)

x = []
for i in set_size_list:
    x.append(n_labeled + i)
x = np.array(x)
print(x)

#plt.plot(x, np.mean(loss_loss_list, axis=0), lw=2, label="loss")
#plt.fill_between(x, np.mean(loss_loss_list, axis=0)-np.std(loss_loss_list, axis=0), np.mean(loss_loss_list, axis=0)+np.std(loss_loss_list, axis=0), alpha=0.2)

plt.plot(x, np.mean(loss_random_list, axis=0), lw=2, label="random")
plt.fill_between(x, np.mean(loss_random_list, axis=0)-np.std(loss_random_list, axis=0), np.mean(loss_random_list, axis=0)+np.std(loss_random_list, axis=0), alpha=0.2)


plt.plot(x, np.mean(loss_fs_inference_list, axis=0), lw=2, label="inference")
plt.fill_between(x, np.mean(loss_fs_inference_list, axis=0)-np.std(loss_fs_inference_list, axis=0), np.mean(loss_fs_inference_list, axis=0)+np.std(loss_fs_inference_list, axis=0), alpha=0.2)
plt.plot(x, np.mean(loss_fs_estimate_list, axis=0), lw=2, label="estimate")
plt.fill_between(x, np.mean(loss_fs_estimate_list, axis=0)-np.std(loss_fs_estimate_list, axis=0), np.mean(loss_fs_estimate_list, axis=0)+np.std(loss_fs_estimate_list, axis=0), alpha=0.2)


np.savez("./results/decision_tree.npz", x=x, loss_full_label_list=loss_full_label_list, loss_fs_estimate_list=loss_fs_estimate_list, loss_fs_inference_list=loss_fs_inference_list, loss_random_list=loss_random_list)

#plt.plot(loss_full_label, lw=2, label="Full label")
#plt.plot(loss_unlabeled_once, lw=2, label="Unlabeled once")
#plt.plot(loss_unlabeled_iter, lw=2, label="Unlabeled iter")
#plt.plot(loss_unlabeled_stepbystep, lw=2, label="Unlabeled step by step")
plt.xlabel("# in-context examples")
plt.ylabel("squared error")
plt.legend()
plt.savefig("decision_tree.png")