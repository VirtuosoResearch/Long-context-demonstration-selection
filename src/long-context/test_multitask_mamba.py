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


def _filter_candidates_by_topk(metaicl_data, candidates, eval_data, retrieval_split, eval_split, filter_k):
    if filter_k is None or filter_k <= 0 or len(candidates) <= filter_k or len(eval_data) == 0:
        return candidates

    candidate_features = np.asarray(
        metaicl_data._load_features_for_datapoints(candidates, retrieval_split), dtype=np.float32
    )
    eval_features = np.asarray(
        metaicl_data._load_features_for_datapoints(eval_data, eval_split), dtype=np.float32
    )
    if candidate_features.ndim != 2 or eval_features.ndim != 2:
        return candidates
    if candidate_features.shape[1] != eval_features.shape[1]:
        raise ValueError(
            f"Feature dim mismatch in pre-filter: "
            f"candidate_dim={candidate_features.shape[1]}, eval_dim={eval_features.shape[1]}"
        )

    dists = np.linalg.norm(eval_features[:, None, :] - candidate_features[None, :, :], axis=2)
    mean_dist = dists.mean(axis=0)
    keep_indices = np.argsort(mean_dist)[:filter_k].tolist()
    return [candidates[i] for i in keep_indices]


def _gradicl_greedy_select(metaicl_data, metaicl_model, validation_rows, candidate_demo_tokens, k, task_name):
    n_candidates = len(candidate_demo_tokens)
    target_k = min(k, n_candidates)
    selected = []
    remaining = set(range(n_candidates))
    selection_trace = []

    for step in range(target_k):
        best_idx = None
        best_loss = None
        for cand_idx in tqdm(
            sorted(remaining),
            desc=f"gradicl-greedy-{task_name}-step={step}",
            leave=False,
        ):
            trial_subset = selected + [cand_idx]
            _tensorize_with_cached_tokens(metaicl_data, validation_rows, candidate_demo_tokens, trial_subset)
            val_losses = _kv_cached_row_losses(metaicl_model, metaicl_data)
            subset_loss = _mean_gt_loss_for_eval(metaicl_data, val_losses)
            if best_loss is None or subset_loss < best_loss:
                best_loss = subset_loss
                best_idx = cand_idx

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        selection_trace.append({"step": int(step), "picked_index": int(best_idx), "val_loss": float(best_loss)})

    return selected, selection_trace


def _precompute_candidate_demo_tokens(metaicl_data, candidates, add_newlines):
    token_seqs = []
    for dp in candidates:
        inp, out = metaicl_data._prepro_each_datapoint(
            dp, is_first=True, for_demonstrations=True, add_newlines=add_newlines
        )
        token_seqs.append(
            metaicl_data._strip_leading_special_tokens(inp) +
            metaicl_data._strip_leading_special_tokens(out)
        )
    return token_seqs


def _precompute_eval_items(metaicl_data, eval_data, add_newlines):
    rows = []
    for dp in eval_data:
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
    """
    Compute per-row option-token NLL without KV cache.

    This mirrors the "Full-KV: concatenated forward (use_cache=False)" eval path
    in `mamba_inference_multi.py` to avoid long-context KV-cache memory/compat issues.
    """
    model = metaicl_model.model
    inputs = metaicl_data.tensorized_inputs
    input_ids_all = inputs["input_ids"].to(metaicl_model.device)
    attention_mask_all = inputs["attention_mask"].to(metaicl_model.device)
    token_type_all = inputs["token_type_ids"].to(metaicl_model.device)

    losses = np.zeros(input_ids_all.size(0), dtype=np.float32)
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

            # Forward the whole (demo+query+option) sequence without caching.
            seq = ids[:valid_len].unsqueeze(0)
            out = model(input_ids=seq, use_cache=False)
            logits = out.logits  # [1, L, V]

            # Teacher-forcing NLL: predict token t using logits at t-1.
            pred = logits[:, :-1, :].contiguous()          # [1, L-1, V]
            target = seq[:, 1:].contiguous()               # [1, L-1]

            # Only score option tokens (token_type==1) within valid region.
            opt_mask = (ttype[1:valid_len] == 1).unsqueeze(0)  # [1, L-1]
            if opt_mask.sum().item() == 0:
                losses[row_idx] = 0.0
                continue

            token_losses = ce(pred.view(-1, pred.size(-1)), target.view(-1)).view(1, -1)
            losses[row_idx] = float(token_losses[opt_mask].mean().item())

    return losses


def _split_candidate_validation(data, candidate_size, validation_size):
    """Split a single task's retrieval data into candidate and validation sets."""
    required = candidate_size + validation_size
    if len(data) < required:
        raise ValueError(
            f"Only {len(data)} samples in retrieval split, "
            f"but needs at least {required} for candidate/validation split"
        )
    candidate_data = data[:candidate_size]
    validation_data = data[-validation_size:]
    return candidate_data, validation_data


def main(logger, args):
    assert args.dataset is not None, "Single-task mode requires --dataset"

    if args.model_name.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    elif any(name in args.model_name for name in ["Llama", "deepseek", "Qwen"]):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.padding_side == "left":
        tokenizer.padding_side = "right"
    if tokenizer.eos_token_id is None and tokenizer.sep_token is not None:
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token

    add_newlines = not args.model_name.startswith("gpt2")
    checkpoint = None

    metaicl_model = MetaICLModel(args.device, logger, args.out_dir, model_name=args.model_name)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    max_length_per_example = args.max_length
    max_length = args.max_length
    if args.k > 0:
        max_length = args.max_length * args.k
    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))

    metaicl_data = MetaICLData(args.device, logger, tokenizer, args.method, True, args.k,
                               max_length=max_length, max_length_per_example=max_length_per_example, seed=args.seed)

    results = []
    errors = []
    seeds = args.seed.split(",")
    config_split = "unseen_domain_test" if args.unseen_domain_only else "test"

    for seed in seeds:

        retrieval_data = load_data(
            args.dataset, args.retrieval_split, args.k, seed=seed, config_split=config_split,
            datasets=None, is_null=args.is_null
        )
        for gid, dp in enumerate(retrieval_data):
            dp["__global_id"] = gid

        eval_data = load_data(
            args.dataset, args.eval_split, args.k, seed=seed, config_split=config_split,
            datasets=None, is_null=args.is_null
        )
        tensorize_eval_split = args.eval_split

        # Split retrieval data into candidates and validation for the single task.
        validation_data = []
        if args.filter_candidate_set:
            retrieval_data, validation_data = _split_candidate_validation(
                retrieval_data,
                candidate_size=args.candidate_size,
                validation_size=args.validation_size,
            )

        print("*" * 20)
        print(f"retrieval_split: {args.retrieval_split}, eval_split: {args.eval_split}")
        logger.info(
            "seed=%s, dataset=%s, #candidate=%d, #validation=%d, #eval(test)=%d",
            seed, args.dataset, len(retrieval_data), len(validation_data), len(eval_data)
        )

        if args.kv_final:
            if len(validation_data) == 0:
                raise ValueError("kv_final requires non-empty validation_data. Use --filter_candidate_set.")

            if metaicl_model.is_none():
                metaicl_model.load(checkpoint, model_name=args.model_name, is_quant=args.is_quant)
                metaicl_model.cuda()
                metaicl_model.eval()
            if "Llama" in args.model_name:
                metaicl_model.resize(tokenizer)

            candidates = retrieval_data
            candidates_before_filter = len(candidates)
            candidates = _filter_candidates_by_topk(
                metaicl_data, candidates, eval_data,
                args.retrieval_split, tensorize_eval_split, args.filter,
            )
            if args.filter > 0:
                logger.info(
                    "[kv_final] pre-filter topk=%d candidates: %d -> %d",
                    args.filter, candidates_before_filter, len(candidates),
                )

            if len(candidates) == 0:
                raise ValueError("No candidate data after filtering")
            if len(validation_data) == 0:
                raise ValueError("No validation data")

            candidate_demo_tokens = _precompute_candidate_demo_tokens(metaicl_data, candidates, add_newlines)
            validation_rows = _precompute_eval_items(metaicl_data, validation_data, add_newlines)

            n_candidates = len(candidates)
            subset_k = min(args.k, n_candidates)
            m = max(1, int(math.ceil(args.kv_final_subset_multiplier * n_candidates)))
            loss_sum = np.zeros(n_candidates, dtype=np.float64)
            loss_count = np.zeros(n_candidates, dtype=np.int64)

            for _ in tqdm(range(m), desc=f"affinity-subsets-{args.dataset}", leave=False):
                subset = np.random.choice(n_candidates, size=subset_k, replace=False).tolist()
                _tensorize_with_cached_tokens(metaicl_data, validation_rows, candidate_demo_tokens, subset)
                val_losses = _kv_cached_row_losses(metaicl_model, metaicl_data)
                subset_loss = _mean_gt_loss_for_eval(metaicl_data, val_losses)
                for idx in subset:
                    loss_sum[idx] += subset_loss
                    loss_count[idx] += 1

            # Affinity score: negative mean loss (higher = better).
            affinity = np.full(n_candidates, -1e9, dtype=np.float32)
            valid_mask = loss_count > 0
            affinity[valid_mask] = -(loss_sum[valid_mask] / loss_count[valid_mask]).astype(np.float32)
            affinity_norm = _normalize_scores(affinity)

            # Frequency score: how often each candidate is nearest to an eval example.
            candidate_features = np.asarray(
                metaicl_data._load_features_for_datapoints(candidates, args.retrieval_split), dtype=np.float32
            )
            eval_features = np.asarray(
                metaicl_data._load_features_for_datapoints(eval_data, tensorize_eval_split), dtype=np.float32
            )
            if candidate_features.shape[1] != eval_features.shape[1]:
                raise ValueError(
                    f"Feature dim mismatch: "
                    f"candidate_dim={candidate_features.shape[1]}, eval_dim={eval_features.shape[1]}"
                )

            dists = np.linalg.norm(
                eval_features[:, None, :] - candidate_features[None, :, :], axis=2
            )  # [n_eval, n_candidates]
            freq_count = np.zeros(n_candidates, dtype=np.float32)
            for i in range(dists.shape[0]):
                nearest = np.argpartition(dists[i], subset_k - 1)[:subset_k]
                freq_count[nearest] += 1.0
            freq_norm = _normalize_scores(freq_count)

            # Combined score and selection.
            score = args.kv_final_lambda * affinity_norm + (1.0 - args.kv_final_lambda) * freq_norm
            selected = np.argsort(score)[-subset_k:][::-1].tolist()

            print("--------------------------------")
            print("dataset: ", args.dataset)
            print("affinity_norm: ", affinity_norm)
            print("freq_norm: ", freq_norm)
            print("freq_count: ", freq_count)
            print("score: ", score)
            print("--------------------------------")

            # Sort selected demos by score (descending).
            selected_demos = [candidates[idx] for idx in selected]

            # Save score artifacts.
            score_path = os.path.join(
                args.out_dir,
                f"{args.dataset}-{tensorize_eval_split}-ret={args.retrieval_split}-kv-final-scores-s={seed}.json",
            )
            score_rows = []
            for rank, idx in enumerate(selected):
                dp = candidates[idx]
                gid = int(dp["__global_id"])
                score_rows.append(
                    {
                        "rank": rank,
                        "candidate_index_in_pool": int(idx),
                        "candidate_global_id": gid,
                        "affinity_norm": float(affinity_norm[idx]),
                        "freq_norm": float(freq_norm[idx]),
                        "freq_count": float(freq_count[idx]),
                        "score": float(score[idx]),
                    }
                )
            with open(score_path, "w") as f:
                json.dump(score_rows, f, ensure_ascii=False, indent=2)
            logger.info("Saved kv_final scores to %s", score_path)
            logger.info("[kv_final] selected demos (count=%d)", len(selected))
            for rank, idx in enumerate(selected):
                dp = candidates[idx]
                logger.info(
                    "[kv_final] rank=%d pool_idx=%d global_id=%d content=%s",
                    rank, int(idx), int(dp["__global_id"]), _format_demo_preview(dp),
                )

            # Tensorize and evaluate.
            selected_demo_examples_per_query = [selected_demos] * len(eval_data)
            metaicl_data.tensorize_with_selected_demo_examples(
                eval_data, selected_demo_examples_per_query, add_newlines=add_newlines
            )
            losses = _kv_cached_row_losses(metaicl_model, metaicl_data, progress_desc="kv-final-test")
            preds = metaicl_model.do_predict(metaicl_data, losses=losses)

            cache_path = os.path.join(
                args.out_dir,
                f"{args.dataset}-{tensorize_eval_split}-ret={args.retrieval_split}-kv-final-k={args.k}-s={seed}.pkl",
            )
            with open(cache_path, "wb") as f:
                pkl.dump(losses, f)
            logger.info("Saved kv_final losses to %s", cache_path)

            # Compute accuracy.
            correct = 0
            groundtruths = [dp["output"] for dp in eval_data]
            for pred, gt in zip(preds, groundtruths):
                p = pred.strip()
                if isinstance(gt, list):
                    ok = p in [x.strip() for x in gt]
                else:
                    ok = p == gt.strip()
                correct += int(ok)
            total = len(groundtruths)
            acc = correct / total if total > 0 else 0.0
            logger.info("Dataset=%s Accuracy=%.4f (%d/%d)", args.dataset, acc, correct, total)
            results.append({"dataset": args.dataset, "accuracy": acc, "correct": correct, "total": total})
            continue

        if args.gradicl:
            if len(validation_data) == 0:
                raise ValueError("gradicl requires non-empty validation_data. Use --filter_candidate_set.")

            if metaicl_model.is_none():
                metaicl_model.load(checkpoint, model_name=args.model_name, is_quant=args.is_quant)
                metaicl_model.cuda()
                metaicl_model.eval()
            if "Llama" in args.model_name:
                metaicl_model.resize(tokenizer)

            candidates = retrieval_data
            candidates_before_filter = len(candidates)
            candidates = _filter_candidates_by_topk(
                metaicl_data, candidates, eval_data,
                args.retrieval_split, tensorize_eval_split, args.filter,
            )
            if args.filter > 0:
                logger.info(
                    "[gradicl] pre-filter topk=%d candidates: %d -> %d",
                    args.filter, candidates_before_filter, len(candidates),
                )

            if len(candidates) == 0:
                raise ValueError("No candidate data after filtering")
            if len(validation_data) == 0:
                raise ValueError("No validation data")

            candidate_demo_tokens = _precompute_candidate_demo_tokens(metaicl_data, candidates, add_newlines)
            validation_rows = _precompute_eval_items(metaicl_data, validation_data, add_newlines)
            selected, trace = _gradicl_greedy_select(
                metaicl_data, metaicl_model,
                validation_rows, candidate_demo_tokens,
                args.k, args.dataset,
            )

            selected_demos = [candidates[idx] for idx in selected]

            # Save score artifacts.
            score_path = os.path.join(
                args.out_dir,
                f"{args.dataset}-{tensorize_eval_split}-ret={args.retrieval_split}-gradicl-scores-s={seed}.json",
            )
            score_rows = []
            for rank, idx in enumerate(selected):
                dp = candidates[idx]
                gid = int(dp["__global_id"])
                score_rows.append(
                    {
                        "rank": rank,
                        "candidate_index_in_pool": int(idx),
                        "candidate_global_id": gid,
                        "selection_step": int(trace[rank]["step"]) if rank < len(trace) else int(rank),
                        "validation_loss_after_pick": float(trace[rank]["val_loss"]) if rank < len(trace) else None,
                    }
                )
            with open(score_path, "w") as f:
                json.dump(score_rows, f, ensure_ascii=False, indent=2)
            logger.info("Saved gradicl scores to %s", score_path)
            logger.info("[gradicl] selected demos (count=%d)", len(selected))
            for rank, idx in enumerate(selected):
                dp = candidates[idx]
                logger.info(
                    "[gradicl] rank=%d pool_idx=%d global_id=%d content=%s",
                    rank, int(idx), int(dp["__global_id"]), _format_demo_preview(dp),
                )

            # Tensorize and evaluate.
            selected_demo_examples_per_query = [selected_demos] * len(eval_data)
            metaicl_data.tensorize_with_selected_demo_examples(
                eval_data, selected_demo_examples_per_query, add_newlines=add_newlines
            )
            losses = _kv_cached_row_losses(metaicl_model, metaicl_data, progress_desc="gradicl-test")
            preds = metaicl_model.do_predict(metaicl_data, losses=losses)

            cache_path = os.path.join(
                args.out_dir,
                f"{args.dataset}-{tensorize_eval_split}-ret={args.retrieval_split}-gradicl-k={args.k}-s={seed}.pkl",
            )
            with open(cache_path, "wb") as f:
                pkl.dump(losses, f)
            logger.info("Saved gradicl losses to %s", cache_path)

            # Compute accuracy.
            correct = 0
            groundtruths = [dp["output"] for dp in eval_data]
            for pred, gt in zip(preds, groundtruths):
                p = pred.strip()
                if isinstance(gt, list):
                    ok = p in [x.strip() for x in gt]
                else:
                    ok = p == gt.strip()
                correct += int(ok)
            total = len(groundtruths)
            acc = correct / total if total > 0 else 0.0
            logger.info("Dataset=%s Accuracy=%.4f (%d/%d)", args.dataset, acc, correct, total)
            results.append({"dataset": args.dataset, "accuracy": acc, "correct": correct, "total": total})
            continue

        # Fallback: topk / randomk / bm25 via the run() function.
        result = run(logger, args.dataset, metaicl_data, metaicl_model,
                     retrieval_data, eval_data, validation_data, seed, checkpoint,
                     add_newlines, tokenizer, tensorize_eval_split)
        if result is None:
            errors.append("%s/%s" % (args.dataset, seed))
        else:
            results.append(result)

    if args.is_null:
        return

    if results:
        agg_correct = sum(r["correct"] for r in results)
        agg_total = sum(r["total"] for r in results)
        overall_acc = agg_correct / agg_total if agg_total > 0 else 0.0
        logger.info(
            "Aggregate accuracy => %.2f%% (%d/%d)",
            100.0 * overall_acc, agg_correct, agg_total
        )

    if len(errors) > 0:
        logger.info("You had errors with datasets: %s", ",".join(errors))
        logger.info("Please see the error messages")


def run(logger, dataset, metaicl_data, metaicl_model, retrieval_data, eval_data, validation_data, seed,
        checkpoint, add_newlines, tokenizer, tensorize_eval_split):

    split_name = f"{tensorize_eval_split}-ret={args.retrieval_split}"
    if args.is_null:
        split_name += "-null"
    cache_suffix = "".join([
        "-topk" if args.topk else "",
        "-randomk" if args.randomk else "",
        "-kv-final" if args.kv_final else "",
        "-gradicl" if args.gradicl else "",
        "-bm25" if args.bm25 else "",
        "-k={}".format(args.k),
        "-s={}".format(seed),
        "" if add_newlines else "-no-newlines",
    ])
    cache_path = os.path.join(
        args.out_dir,
        f"{dataset}-{split_name}-{metaicl_data.method}{cache_suffix}.pkl",
    )

    if args.topk:
        metaicl_data.tensorize_topk(
            retrieval_data, eval_data, options=None, add_newlines=add_newlines,
            retrieval_split=args.retrieval_split, eval_split=tensorize_eval_split,
        )
        _log_selected_demos(
            logger, "topk", dataset,
            getattr(metaicl_data, "topk_last_selected_demo_info", []),
        )
    elif args.randomk:
        metaicl_data.tensorize_randomk(
            retrieval_data, eval_data, options=None, add_newlines=add_newlines,
            retrieval_split=args.retrieval_split, eval_split=tensorize_eval_split,
        )
        _log_selected_demos(
            logger, "randomk", dataset,
            getattr(metaicl_data, "randomk_last_selected_demo_info", []),
        )
    elif args.bm25:
        metaicl_data.tensorize_bm25(retrieval_data, eval_data, options=None, add_newlines=add_newlines)
    else:
        raise ValueError("Please choose one selection method: --topk, --kv_final, --gradicl, --randomk, or --bm25")

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

    losses = metaicl_model.do_inference(metaicl_data, args.test_batch_size)

    with open(cache_path, "wb") as f:
        pkl.dump(losses, f)

    logger.info(f"len(losses): {len(losses)}; len(metaicl_data): {len(metaicl_data)}")

    if args.is_null:
        return None

    if args.use_calibration:
        key = "/" + dataset + "-" + f"{tensorize_eval_split}-ret={args.retrieval_split}"
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
    logger.info("Dataset=%s Accuracy= %.4f (%d/%d)", dataset, perf, correct, total)

    with open(prediction_path, "w") as f:
        for prediction in predictions:
            f.write(prediction)
            f.write("\n")

    return {
        "dataset": dataset,
        "accuracy": perf,
        "correct": correct,
        "total": total,
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_calibration", default=False, action="store_true")
    parser.add_argument("--unseen_domain_only", default=False, action="store_true")

    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--dataset", type=str, required=True)
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

    parser.add_argument("--topk", default=False, action="store_true")
    parser.add_argument("--kv_final", default=False, action="store_true")
    parser.add_argument("--gradicl", default=False, action="store_true")
    parser.add_argument("--kv_final_lambda", type=float, default=0.9)
    parser.add_argument("--kv_final_subset_multiplier", type=float, default=1.0)
    parser.add_argument("--filter", type=int, default=100,
                        help="Top-k pre-filter size for kv_final/gradicl candidates. 0 disables pre-filter.")
    parser.add_argument("--randomk", default=False, action="store_true")
    parser.add_argument("--bm25", default=False, action="store_true")

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--is_quant", default=False, action="store_true")
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--filter_candidate_set", dest="filter_candidate_set", action="store_true")
    parser.add_argument("--no_filter_candidate_set", dest="filter_candidate_set", action="store_false")
    parser.add_argument("--candidate_size", type=int, default=90)
    parser.add_argument("--validation_size", type=int, default=50)
    parser.set_defaults(
        filter_candidate_set=True,
    )
    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)