# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import pickle as pkl
import random
import torch
import math
import json
import string
import logging
import numpy as np

from tqdm import tqdm
from collections import Counter, defaultdict

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import GPT2Tokenizer, AutoTokenizer
from transformers.cache_utils import DynamicCache

from metaicl.data import MetaICLData, prepro_sentence_pair_single
from metaicl.model import MetaICLModel

from utils.data import load_data


def _parse_datasets_arg(dataset_arg):
    if dataset_arg is None:
        return None
    datasets = [d.strip() for d in dataset_arg.split(",") if d.strip()]
    return datasets if len(datasets) > 0 else None


def _split_multitask_retrieval(data, candidate_size, validation_size):
    grouped = defaultdict(list)
    for dp in data:
        grouped[dp["task"]].append(dp)

    retrieval_data = []
    validation_data = []
    for task, samples in grouped.items():
        required = candidate_size + validation_size
        if len(samples) < required:
            raise ValueError(
                f"Task {task} has only {len(samples)} samples in retrieval split, "
                f"but needs at least {required} for candidate/validation split"
            )
        retrieval_data.extend(samples[:candidate_size])
        validation_data.extend(samples[-validation_size:])
    return retrieval_data, validation_data


def _to_preview_text(value):
    if isinstance(value, list):
        return " | ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _format_demo_preview(dp, max_chars=220):
    text = f"input={_to_preview_text(dp.get('input', ''))}; output={_to_preview_text(dp.get('output', ''))}"
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _log_selected_demos(logger, selection_name, task, selected_demo_info):
    if not selected_demo_info:
        return
    logger.info("[%s] selected demos for task=%s (queries=%d)", selection_name, task, len(selected_demo_info))
    for query_info in selected_demo_info:
        eval_index = int(query_info.get("eval_index", -1))
        eval_task = query_info.get("eval_task", task)
        logger.info(
            "[%s][%s] eval_index=%d selected_count=%d",
            selection_name,
            eval_task,
            eval_index,
            len(query_info.get("selected_demos", [])),
        )
        for demo in query_info.get("selected_demos", []):
            logger.info(
                "[%s][%s] eval_index=%d rank=%d source=%s pool_idx=%d candidate_task=%s content=%s",
                selection_name,
                eval_task,
                eval_index,
                int(demo.get("rank", -1)),
                demo.get("selection_source", selection_name),
                int(demo.get("candidate_index_in_pool", -1)),
                demo.get("candidate_task"),
                demo.get("content", ""),
            )


def _mean_gt_loss_for_eval(metaicl_data, losses):
    losses = np.asarray(losses)
    gt_losses = []
    for md in metaicl_data.metadata:
        answer = md["answer"]
        if isinstance(answer, list):
            gt_option_idx = int(answer[0])
        else:
            gt_option_idx = int(answer)
        opt_indices = md["indices"][gt_option_idx]
        gt_losses.append(float(np.sum(losses[opt_indices])))
    if len(gt_losses) == 0:
        return 0.0
    return float(np.mean(gt_losses))


def _normalize_scores(arr):
    arr = np.asarray(arr, dtype=np.float32)
    amin, amax = float(arr.min()), float(arr.max())
    if amax > amin:
        return (arr - amin) / (amax - amin)
    return np.zeros_like(arr, dtype=np.float32)


def _precompute_candidate_demo_tokens(metaicl_data, candidates_task, add_newlines):
    token_seqs = []
    for dp in candidates_task:
        inp, out = metaicl_data._prepro_each_datapoint(
            dp, is_first=True, for_demonstrations=True, add_newlines=add_newlines
        )
        token_seqs.append(
            metaicl_data._strip_leading_special_tokens(inp) +
            metaicl_data._strip_leading_special_tokens(out)
        )
    return token_seqs


def _precompute_eval_items(metaicl_data, eval_task_data, add_newlines):
    rows = []
    for dp in eval_task_data:
        inputs, outputs, answer = metaicl_data._prepro_each_datapoint(
            dp, is_first=not metaicl_data.use_demonstrations, add_newlines=add_newlines
        )
        rows.append(
            {
                "dp": dp,
                "inputs": inputs,
                "outputs": outputs,
                "answer": answer,
            }
        )
    return rows


def _tensorize_with_cached_tokens(metaicl_data, eval_rows, candidate_demo_tokens, selected_indices):
    selected_indices = selected_indices[: metaicl_data.k]
    demonstrations = []
    if metaicl_data.use_demonstrations and len(selected_indices) > 0:
        demonstrations = metaicl_data._demo_prompt_prefix_tokens(len(selected_indices))
        sep_tokens = metaicl_data._demo_example_separator_tokens()
        for i, idx in enumerate(selected_indices):
            if i > 0:
                demonstrations += sep_tokens
            demonstrations += candidate_demo_tokens[idx]

    input_ids, attention_mask, token_type_ids, demo_lens = [], [], [], []
    metadata = []
    for row in eval_rows:
        inputs = row["inputs"]
        outputs = row["outputs"]
        answer = row["answer"]
        dp = row["dp"]
        indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]
        metadata.append({"indices": indices, "answer": answer, "options": dp["options"], "task": dp.get("task")})

        for inputs_, outputs_ in zip(inputs, outputs):
            if metaicl_data.use_demonstrations:
                demo_len = len(demonstrations)
                left = demonstrations + metaicl_data._query_prefix_tokens() + metaicl_data._strip_leading_special_tokens(inputs_)
            else:
                demo_len = 0
                left = inputs_
            enc = prepro_sentence_pair_single(
                left,
                outputs_,
                metaicl_data.max_length,
                metaicl_data.tokenizer,
                metaicl_data.tokenizer.bos_token_id,
                metaicl_data.tokenizer.eos_token_id,
                allow_truncation=metaicl_data.use_demonstrations,
            )
            input_ids.append(enc[0])
            attention_mask.append(enc[1])
            token_type_ids.append(enc[2])
            demo_lens.append(demo_len)

    metaicl_data.tensorized_inputs = dict(
        input_ids=torch.LongTensor(input_ids),
        attention_mask=torch.LongTensor(attention_mask),
        token_type_ids=torch.LongTensor(token_type_ids),
        demo_lens=torch.LongTensor(demo_lens),
    )
    metaicl_data.metadata = metadata


def _first_label_pos(token_type_ids, valid_len):
    for i in range(1, valid_len):
        if int(token_type_ids[i].item()) == 1:
            return i
    return valid_len


def _kv_cached_row_losses(metaicl_model, metaicl_data, progress_desc=None):
    model = metaicl_model.model
    inputs = metaicl_data.tensorized_inputs
    input_ids_all = inputs["input_ids"].to(metaicl_model.device)
    attention_mask_all = inputs["attention_mask"].to(metaicl_model.device)
    token_type_all = inputs["token_type_ids"].to(metaicl_model.device)
    demo_lens_all = inputs.get("demo_lens", torch.zeros(input_ids_all.size(0), dtype=torch.long)).to(metaicl_model.device)

    losses = np.zeros(input_ids_all.size(0), dtype=np.float32)
    demo_cache = {}
    prefix_cache = {}
    ce = torch.nn.CrossEntropyLoss(reduction="none")

    row_iter = range(input_ids_all.size(0))
    if progress_desc is not None:
        row_iter = tqdm(row_iter, desc=progress_desc, leave=False)

    with torch.no_grad():
        for row_idx in row_iter:
            ids = input_ids_all[row_idx]
            attn = attention_mask_all[row_idx]
            ttype = token_type_all[row_idx]
            valid_len = int(attn.sum().item())
            if valid_len <= 1:
                losses[row_idx] = 0.0
                continue

            label_pos = _first_label_pos(ttype, valid_len)
            if label_pos >= valid_len:
                losses[row_idx] = 0.0
                continue

            demo_len = int(demo_lens_all[row_idx].item())
            demo_end = min(1 + demo_len, label_pos)
            demo_prefix_ids = ids[:demo_end]
            prefix_ids = ids[:label_pos]
            option_ids = ids[label_pos:valid_len][ttype[label_pos:valid_len] == 1]
            if option_ids.numel() == 0:
                losses[row_idx] = 0.0
                continue

            demo_key = tuple(demo_prefix_ids.tolist())
            if demo_key not in demo_cache:
                out_demo = model(input_ids=demo_prefix_ids.unsqueeze(0), use_cache=True)
                demo_cache[demo_key] = {
                    "past": out_demo.past_key_values,
                    "last_logits": out_demo.logits[:, -1, :],
                    "len": demo_prefix_ids.shape[0],
                }
            demo_obj = demo_cache[demo_key]

            prefix_key = tuple(prefix_ids.tolist())
            if prefix_key not in prefix_cache:
                query_ids = ids[demo_end:label_pos]
                past_for_query = demo_obj["past"]
                if isinstance(past_for_query, tuple):
                    past_for_query = DynamicCache.from_legacy_cache(past_for_query)
                if query_ids.numel() > 0:
                    out_q = model(
                        input_ids=query_ids.unsqueeze(0),
                        past_key_values=past_for_query,
                        use_cache=True,
                    )
                    prefix_cache[prefix_key] = {
                        "past": out_q.past_key_values,
                        "first_logits": out_q.logits[:, -1, :],
                        "len": demo_obj["len"] + query_ids.shape[0],
                    }
                else:
                    prefix_cache[prefix_key] = {
                        "past": past_for_query,
                        "first_logits": demo_obj["last_logits"],
                        "len": demo_obj["len"],
                    }
            pref = prefix_cache[prefix_key]

            target = option_ids.unsqueeze(0)
            if target.shape[1] == 1:
                token_losses = ce(pref["first_logits"], target[:, 0])
                losses[row_idx] = float(token_losses.mean().item())
                continue

            past_for_opt = pref["past"]
            if isinstance(past_for_opt, tuple):
                past_for_opt = DynamicCache.from_legacy_cache(past_for_opt)
            out_opt = model(
                input_ids=target,
                past_key_values=past_for_opt,
                use_cache=False,
            )
            pred_logits = torch.cat([pref["first_logits"].unsqueeze(1), out_opt.logits[:, :-1, :]], dim=1)
            token_losses = ce(pred_logits.reshape(-1, pred_logits.size(-1)), target.reshape(-1))
            losses[row_idx] = float(token_losses.mean().item())

    return losses

def main(logger, args):
    assert (args.dataset is not None and args.task is None) or (args.dataset is None and args.task is not None)

    if args.model_name.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    elif any(name in args.model_name for name in ["Llama", "deepseek", "Qwen"]):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.padding_side=="left":
        tokenizer.padding_side = "right"
    add_newlines = True
    if tokenizer.eos_token_id is None and tokenizer.sep_token is not None:
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token
    ### checkpoint ...

    add_newlines = not args.model_name.startswith("gpt2")
    checkpoint = None

    metaicl_model = MetaICLModel(args.device, logger, args.out_dir, model_name=args.model_name)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # setup hyperparams for data

    # Treat --max_length as per-example budget and scale total context budget by k.
    max_length_per_example = args.max_length
    max_length = args.max_length
    if args.k > 0:
        max_length = args.max_length * args.k
    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))

    metaicl_data = MetaICLData(args.device ,logger, tokenizer, args.method, True, args.k,
                               max_length=max_length, max_length_per_example=max_length_per_example, seed=args.seed)
    # metaicl_data.to(device)
    results = []
    errors = []
    seeds = args.seed.split(",")
    config_split = "unseen_domain_test" if args.unseen_domain_only else "test"
    datasets = _parse_datasets_arg(args.dataset)
    is_multitask = datasets is not None and len(datasets) > 1

    for seed in seeds:

        retrieval_data = load_data(
            args.task, args.retrieval_split, args.k, seed=seed, config_split=config_split,
            datasets=datasets, is_null=args.is_null
        )
        for gid, dp in enumerate(retrieval_data):
            dp["__global_id"] = gid
        eval_data = load_data(
            args.task, args.eval_split, args.k, seed=seed, config_split=config_split,
            datasets=datasets, is_null=args.is_null
        )
        tensorize_eval_split = args.eval_split

        validation_data = []
        if args.multitask_filter_candidate_set:
            retrieval_data, validation_data = _split_multitask_retrieval(
                retrieval_data,
                candidate_size=args.multitask_candidate_size,
                validation_size=args.multitask_validation_size,
            )

        print("*"*20)
        print(f"retrieval_split: {args.retrieval_split}, eval_split: {args.eval_split}")
        logger.info(
            "seed=%s, multitask=%s, #candidate=%d, #validation=%d, #eval(test)=%d",
            seed, is_multitask, len(retrieval_data), len(validation_data), len(eval_data)
        )

        eval_by_task = defaultdict(list)
        for dp in eval_data:
            eval_by_task[dp["task"]].append(dp)

        seed_task_metrics = {}
        seed_total_correct = 0
        seed_total_count = 0

        if args.kv_final:
            if len(validation_data) == 0:
                raise ValueError("kv_final requires non-empty validation_data from retrieval split.")

            if metaicl_model.is_none():
                metaicl_model.load(checkpoint, model_name=args.model_name, is_quant=args.is_quant)
                metaicl_model.cuda()
                metaicl_model.eval()
            if "Llama" in args.model_name:
                metaicl_model.resize(tokenizer)

            candidate_by_task = defaultdict(list)
            for dp in retrieval_data:
                candidate_by_task[dp["task"]].append(dp)
            task_local_idx_by_gid = {}
            for task_name, rows in candidate_by_task.items():
                for task_idx, row in enumerate(rows):
                    gid = int(row.get("__global_id", -1))
                    task_local_idx_by_gid[gid] = (task_name, task_idx)
            validation_by_task = defaultdict(list)
            for dp in validation_data:
                validation_by_task[dp["task"]].append(dp)

            # Global query pool for frequency statistics (all tasks).
            eval_all_features = np.asarray(
                metaicl_data._load_features_for_datapoints(eval_data, tensorize_eval_split), dtype=np.float32
            )
            retrieval_features_all = np.asarray(
                metaicl_data._load_features_for_datapoints(retrieval_data, args.retrieval_split), dtype=np.float32
            )
            feat_by_gid = {
                int(dp["__global_id"]): retrieval_features_all[i]
                for i, dp in enumerate(retrieval_data)
            }

            task_demo_bank = {}
            task_selected_local = {}
            task_candidates_ref = {}
            task_score_artifacts = {}
            for test_task, eval_data_task in eval_by_task.items():
                candidates_task = retrieval_data if args.kv_final_global_pool else candidate_by_task[test_task]
                validation_task = validation_by_task[test_task]
                if len(candidates_task) == 0:
                    raise ValueError(f"No candidate data for task={test_task}")
                if len(validation_task) == 0:
                    raise ValueError(f"No validation data for task={test_task}")

                candidate_demo_tokens = _precompute_candidate_demo_tokens(metaicl_data, candidates_task, add_newlines)
                validation_rows = _precompute_eval_items(metaicl_data, validation_task, add_newlines)

                n_candidates = len(candidates_task)
                subset_k = min(args.k, n_candidates)
                m = max(1, int(math.ceil(args.kv_final_subset_multiplier * n_candidates)))
                loss_sum = np.zeros(n_candidates, dtype=np.float64)
                loss_count = np.zeros(n_candidates, dtype=np.int64)

                for _ in tqdm(range(m), desc=f"affinity-subsets-{test_task}", leave=False):
                    subset = np.random.choice(n_candidates, size=subset_k, replace=False).tolist()
                    _tensorize_with_cached_tokens(metaicl_data, validation_rows, candidate_demo_tokens, subset)
                    val_losses = _kv_cached_row_losses(metaicl_model, metaicl_data)
                    subset_loss = _mean_gt_loss_for_eval(metaicl_data, val_losses)
                    for idx in subset:
                        loss_sum[idx] += subset_loss
                        loss_count[idx] += 1

                affinity = np.full(n_candidates, -1e9, dtype=np.float32)
                valid_mask = loss_count > 0
                affinity[valid_mask] = -(loss_sum[valid_mask] / loss_count[valid_mask]).astype(np.float32)
                affinity_norm = _normalize_scores(affinity)

                candidate_features = np.asarray(
                    metaicl_data._load_features_for_datapoints(candidates_task, args.retrieval_split), dtype=np.float32
                )
                if candidate_features.shape[1] != eval_all_features.shape[1]:
                    raise ValueError(
                        f"Feature dim mismatch for task={test_task}: "
                        f"candidate_dim={candidate_features.shape[1]}, eval_dim={eval_all_features.shape[1]}"
                    )

                dists = np.linalg.norm(
                    eval_all_features[:, None, :] - candidate_features[None, :, :], axis=2
                )  # [n_eval_all, n_candidates]
                freq_count = np.zeros(n_candidates, dtype=np.float32)
                for i in range(dists.shape[0]):
                    nearest = np.argpartition(dists[i], subset_k - 1)[:subset_k]
                    freq_count[nearest] += 1.0
                freq_norm = _normalize_scores(freq_count)
                score = args.kv_final_lambda * affinity_norm + (1.0 - args.kv_final_lambda) * freq_norm
                selected = np.argsort(score)[-subset_k:][::-1].tolist()
                task_selected_local[test_task] = selected
                task_candidates_ref[test_task] = candidates_task
                task_score_artifacts[test_task] = {
                    "affinity_norm": affinity_norm,
                    "freq_norm": freq_norm,
                    "freq_count": freq_count,
                    "score": score,
                }
                print("--------------------------------")
                print("test_task: ", test_task)
                print("affinity_norm: ", affinity_norm)
                print("freq_norm: ", freq_norm)
                print("freq_count: ", freq_count)
                print("score: ", score)
                print("--------------------------------")

            # Stage2: global recompute over union of selected demos (all tasks), then task-wise internal sort.
            selected_gid_set = set()
            for test_task, selected in task_selected_local.items():
                cands = task_candidates_ref[test_task]
                for idx in selected:
                    selected_gid_set.add(int(cands[idx]["__global_id"]))

            selected_gids = sorted(selected_gid_set)
            selected_feats_global = np.asarray([feat_by_gid[g] for g in selected_gids], dtype=np.float32)
            d2_global = np.linalg.norm(eval_all_features[:, None, :] - selected_feats_global[None, :, :], axis=2)
            nearest_global = np.argmin(d2_global, axis=1)
            second_freq_global = {g: 0.0 for g in selected_gids}
            for loc in nearest_global:
                second_freq_global[selected_gids[int(loc)]] += 1.0

            for test_task, selected in task_selected_local.items():
                cands = task_candidates_ref[test_task]
                selected_sorted = sorted(
                    selected,
                    key=lambda idx: (-second_freq_global.get(int(cands[idx]["__global_id"]), 0.0), int(cands[idx]["__global_id"]))
                )
                task_demo_bank[test_task] = [cands[idx] for idx in selected_sorted]

                score_path = os.path.join(
                    args.out_dir,
                    f"{test_task}-{tensorize_eval_split}-ret={args.retrieval_split}-kv-final-scores-s={seed}.json",
                )
                score_rows = []
                art = task_score_artifacts[test_task]
                for rank, idx in enumerate(selected_sorted):
                    dp = cands[idx]
                    gid = int(dp["__global_id"])
                    score_rows.append(
                        {
                            "rank": rank,
                            "candidate_index_in_pool": int(idx),
                            "candidate_global_id": gid,
                            "candidate_task": dp.get("task"),
                            "candidate_feature_idx": int(dp.get("__feature_idx", -1)),
                            "affinity_norm": float(art["affinity_norm"][idx]),
                            "freq_norm": float(art["freq_norm"][idx]),
                            "freq_count": float(art["freq_count"][idx]),
                            "freq_count_stage2": float(second_freq_global.get(gid, 0.0)),
                            "score": float(art["score"][idx]),
                        }
                    )
                with open(score_path, "w") as f:
                    json.dump(score_rows, f, ensure_ascii=False, indent=2)
                logger.info("Saved kv_final task scores to %s", score_path)
                logger.info("[kv_final] selected demos for test_task=%s (count=%d)", test_task, len(selected_sorted))
                for rank, idx in enumerate(selected_sorted):
                    dp = cands[idx]
                    gid = int(dp["__global_id"])
                    candidate_task, task_local_idx = task_local_idx_by_gid.get(
                        gid, (dp.get("task"), -1)
                    )
                    logger.info(
                        "[kv_final][%s] rank=%d candidate_task=%s task_local_idx=%d pool_idx=%d global_id=%d content=%s",
                        test_task,
                        rank,
                        candidate_task,
                        int(task_local_idx),
                        int(idx),
                        gid,
                        _format_demo_preview(dp),
                    )

            pairs = [(dp, task_demo_bank[dp["task"]]) for dp in eval_data]
            pairs.sort(
                key=lambda x: tuple(int(d.get("__global_id", -1)) for d in x[1])
            )
            eval_data_sorted = [x[0] for x in pairs]
            selected_demo_examples_per_query = [x[1] for x in pairs]
            metaicl_data.tensorize_with_selected_demo_examples(
                eval_data_sorted, selected_demo_examples_per_query, add_newlines=add_newlines
            )
            losses = _kv_cached_row_losses(metaicl_model, metaicl_data, progress_desc="kv-final-test")
            preds = metaicl_model.do_predict(metaicl_data, losses=losses)

            cache_path = os.path.join(
                args.out_dir,
                f"multitask-{tensorize_eval_split}-ret={args.retrieval_split}-kv-final-k={args.k}-s={seed}.pkl",
            )
            with open(cache_path, "wb") as f:
                pkl.dump(losses, f)
            logger.info("Saved kv_final losses to %s", cache_path)

            pred_by_task = defaultdict(list)
            gt_by_task = defaultdict(list)
            for pred, dp in zip(preds, eval_data_sorted):
                pred_by_task[dp["task"]].append(pred)
                gt_by_task[dp["task"]].append(dp["output"])

            for test_task in sorted(pred_by_task.keys()):
                correct = 0
                gts = gt_by_task[test_task]
                for pred, gt in zip(pred_by_task[test_task], gts):
                    p = pred.strip()
                    if isinstance(gt, list):
                        ok = p in [x.strip() for x in gt]
                    else:
                        ok = p == gt.strip()
                    correct += int(ok)
                total = len(gts)
                acc = correct / total if total > 0 else 0.0
                seed_task_metrics[test_task] = {
                    "task": test_task,
                    "accuracy": acc,
                    "correct": correct,
                    "total": total,
                }
                seed_total_correct += correct
                seed_total_count += total
                results.append(seed_task_metrics[test_task])

            if seed_task_metrics:
                per_task_msg = ", ".join(
                    [
                        f"{task}: {100.0 * metric['accuracy']:.2f}% ({metric['correct']}/{metric['total']})"
                        for task, metric in sorted(seed_task_metrics.items())
                    ]
                )
                overall_acc = seed_total_correct / seed_total_count if seed_total_count > 0 else 0.0
                logger.info("Seed %s per-task accuracy => %s", seed, per_task_msg)
                logger.info(
                    "Seed %s overall accuracy => %.2f%% (%d/%d)",
                    seed, 100.0 * overall_acc, seed_total_correct, seed_total_count
                )
            continue

        for test_task, eval_data_task in eval_by_task.items():
            validation_data_task = [dp for dp in validation_data if dp["task"] == test_task] if validation_data else []
            result = run(logger, test_task, metaicl_data, metaicl_model,
                         retrieval_data, eval_data_task, validation_data_task, seed, checkpoint,
                         add_newlines, tokenizer, tensorize_eval_split)
            if result is None:
                errors.append("%s/%s" % (test_task, seed))
            else:
                seed_task_metrics[test_task] = result
                seed_total_correct += result["correct"]
                seed_total_count += result["total"]
                results.append(result)

        if seed_task_metrics:
            per_task_msg = ", ".join(
                [
                    f"{task}: {100.0 * metric['accuracy']:.2f}% ({metric['correct']}/{metric['total']})"
                    for task, metric in sorted(seed_task_metrics.items())
                ]
            )
            overall_acc = seed_total_correct / seed_total_count if seed_total_count > 0 else 0.0
            logger.info("Seed %s per-task accuracy => %s", seed, per_task_msg)
            logger.info(
                "Seed %s overall accuracy => %.2f%% (%d/%d)",
                seed, 100.0 * overall_acc, seed_total_correct, seed_total_count
            )

    if args.is_null:
        return

    if results:
        aggregate_by_task = defaultdict(lambda: {"correct": 0, "total": 0})
        aggregate_correct = 0
        aggregate_total = 0
        for metric in results:
            task = metric["task"]
            aggregate_by_task[task]["correct"] += metric["correct"]
            aggregate_by_task[task]["total"] += metric["total"]
            aggregate_correct += metric["correct"]
            aggregate_total += metric["total"]

        per_task_msg = ", ".join(
            [
                f"{task}: {100.0 * v['correct'] / v['total']:.2f}% ({v['correct']}/{v['total']})"
                for task, v in sorted(aggregate_by_task.items())
                if v["total"] > 0
            ]
        )
        overall_acc = aggregate_correct / aggregate_total if aggregate_total > 0 else 0.0
        logger.info("Aggregate per-task accuracy => %s", per_task_msg)
        logger.info(
            "Aggregate overall accuracy => %.2f%% (%d/%d)",
            100.0 * overall_acc, aggregate_correct, aggregate_total
        )

    if len(errors)>0:
        logger.info("You had errors with datasets:", ",".join(errors))
        logger.info("Please see the error messages")


def run(logger, task, metaicl_data, metaicl_model, retrieval_data, eval_data, validation_data, seed,
        checkpoint, add_newlines, tokenizer, tensorize_eval_split):

    split_name = f"{tensorize_eval_split}-ret={args.retrieval_split}"
    if args.is_null:
        split_name += "-null"
    cache_suffix = "".join([
        "-topk" if args.topk else "",
        "-randomk" if args.randomk else "",
        "-kv-final" if args.kv_final else "",
        "-bm25" if args.bm25 else "",
        "-k={}".format(args.k),
        "-s={}".format(seed),
        "" if add_newlines else "-no-newlines",
    ])
    cache_path = os.path.join(
        args.out_dir,
        f"{task}-{split_name}-{metaicl_data.method}{cache_suffix}.pkl",
    )

    if args.topk:
        metaicl_data.tensorize_topk(
            retrieval_data, eval_data, options=None, add_newlines=add_newlines,
            retrieval_split=args.retrieval_split, eval_split=tensorize_eval_split,
        )
        _log_selected_demos(
            logger,
            "topk",
            task,
            getattr(metaicl_data, "topk_last_selected_demo_info", []),
        )
    elif args.kv_final:
        metaicl_data.tensorize_kv_final(
            retrieval_data, eval_data, options=None, add_newlines=add_newlines,
            retrieval_split=args.retrieval_split, eval_split=tensorize_eval_split,
            validation_data=validation_data, validation_split=args.retrieval_split,
            lambda_weight=args.kv_final_lambda,
        )
        score_path = os.path.join(
            args.out_dir,
            f"{task}-{tensorize_eval_split}-ret={args.retrieval_split}-kv-final-scores-s={seed}.json",
        )
        with open(score_path, "w") as f:
            json.dump(getattr(metaicl_data, "kv_final_last_scores", []), f, ensure_ascii=False, indent=2)
        logger.info("Saved kv_final task scores to %s", score_path)
    elif args.randomk:
        metaicl_data.tensorize_randomk(
            retrieval_data, eval_data, options=None, add_newlines=add_newlines,
            retrieval_split=args.retrieval_split, eval_split=tensorize_eval_split,
        )
        _log_selected_demos(
            logger,
            "randomk",
            task,
            getattr(metaicl_data, "randomk_last_selected_demo_info", []),
        )
    elif args.bm25:
        metaicl_data.tensorize_bm25(retrieval_data, eval_data, options=None,  add_newlines=add_newlines)
    else:
        raise ValueError("Please choose one selection method: --topk, --kv_final, --randomk, or --bm25")

    metaicl_data.print_tensorized_example()
    logger.info(cache_path)
    prediction_path = cache_path.replace(".pkl", ".txt")
    if args.use_calibration:
        prediction_path = prediction_path.replace(".txt", "-calibrated.txt")

    if metaicl_model.is_none():
        metaicl_model.load(checkpoint, model_name=args.model_name, is_quant=args.is_quant)
        metaicl_model.cuda()
        metaicl_model.eval()
    if "Llama" in args.model_name:
        metaicl_model.resize(tokenizer)

    losses = []
    n = 0

    losses = metaicl_model.do_inference(metaicl_data, args.test_batch_size)

    with open(cache_path, "wb") as f:
        pkl.dump(losses, f)

    logger.info(f"len(losses): {len(losses)}; len(metaicl_data): {len(metaicl_data)}")

    if args.is_null:
        return None

    if args.use_calibration:
        key = "/" + task + "-" + f"{tensorize_eval_split}-ret={args.retrieval_split}"
        bias_path = cache_path.replace(key, key + "-null")
        assert os.path.exists(bias_path), bias_path
        with open(bias_path, "rb") as f:
            bias_losses = pkl.load(f)

        losses = np.array(losses)
        bias_losses = np.array(bias_losses)
        assert losses.shape == bias_losses.shape
        losses -= bias_losses
    predictions = metaicl_model.do_predict(metaicl_data, losses=losses)
    groundtruths = [dp["output"] for dp in eval_data]
    correct = 0
    for prediction, groundtruth in zip(predictions, groundtruths):
        prediction = prediction.strip()
        if isinstance(groundtruth, list):
            gt = [item.strip() for item in groundtruth]
            is_correct = prediction in gt
        else:
            is_correct = prediction == groundtruth.strip()
        correct += int(is_correct)
    total = len(groundtruths)
    perf = correct / total if total > 0 else 0.0
    logger.info("Task=%s Accuracy= %.4f (%d/%d)", task, perf, correct, total)

    with open(prediction_path, "w") as f:
        for prediction in predictions:
            f.write(prediction)
            f.write("\n")

    return {
        "task": task,
        "accuracy": perf,
        "correct": correct,
        "total": total,
    }

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_calibration", default=False, action="store_true")
    parser.add_argument("--unseen_domain_only", default=False, action="store_true")

    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=str, default="0")

    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--global_step", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--use_random_english_words", default=False, action="store_true")

    parser.add_argument("--out_dir", type=str, required=True, default="out/llama-3.2-1b-instruct")

    parser.add_argument("--eval_split", type=str, default="dev")
    parser.add_argument("--retrieval_split", type=str, default="test")
    parser.add_argument("--is_null", default=False, action="store_true")
    parser.add_argument("--method", type=str, default="direct", choices=["direct", "channel"])
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")

    parser.add_argument("--topk",default=False, action="store_true")
    parser.add_argument("--kv_final", default=False, action="store_true")
    parser.add_argument("--kv_final_lambda", type=float, default=0.7)
    parser.add_argument("--kv_final_subset_multiplier", type=float, default=1.0)
    parser.add_argument("--kv_final_global_pool", dest="kv_final_global_pool", action="store_true")
    parser.add_argument("--no_kv_final_global_pool", dest="kv_final_global_pool", action="store_false")
    parser.add_argument("--randomk", default=False, action="store_true")
    parser.add_argument("--bm25", default=False, action="store_true")
    
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--is_quant", default=False, action="store_true")
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--multitask_filter_candidate_set", dest="multitask_filter_candidate_set", action="store_true")
    parser.add_argument("--no_multitask_filter_candidate_set", dest="multitask_filter_candidate_set", action="store_false")
    parser.add_argument("--multitask_candidate_size", type=int, default=90)
    parser.add_argument("--multitask_validation_size", type=int, default=50)
    parser.add_argument("--multitask_use_validation_as_eval", dest="multitask_use_validation_as_eval", action="store_true")
    parser.add_argument("--no_multitask_use_validation_as_eval", dest="multitask_use_validation_as_eval", action="store_false")
    parser.set_defaults(
        multitask_filter_candidate_set=True,
        multitask_use_validation_as_eval=False,
        kv_final_global_pool=True,
    )
    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)
