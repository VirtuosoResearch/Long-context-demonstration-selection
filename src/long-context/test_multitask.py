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

from metaicl.data import MetaICLData
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
        validation_data.extend(samples[candidate_size:required])
    return retrieval_data, validation_data

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
    if not args.do_zeroshot:
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
            assert args.global_step is None
        else:
            assert args.global_step is not None
            checkpoint = os.path.join(args.out_dir, "model-{}.pt".format(args.global_step))
        assert os.path.exists(checkpoint)
    else:
        add_newlines = not args.model_name.startswith("gpt2")
        if False: #args.model_name=="gpt-j-6B":
            # we are using the HF veresion where GPT-J-6B checkpoint is not officially registered
            # so need to download the model checkpoint and specify checkpoint
            assert args.checkpoint is not None and os.path.exists(args.checkpoint)
            args.model_name = args.checkpoint
        checkpoint = None
    
    metaicl_model = MetaICLModel(args.device, logger, args.out_dir, gpt2=args.model_name)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # setup hyperparams for data

    max_length_per_example = 128
    max_length = 128
    if args.use_demonstrations:
        orig_max_length = max_length
        if args.do_zeroshot:
            # max_length = min(max_length_per_example * args.k, 1024)
            max_length = args.max_length * args.k
            # max_length = max_length * args.k
        else:
            max_length = min(max_length * args.k, 1024)

    if args.k==0: max_length = args.max_length
    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))

    metaicl_data = MetaICLData(args.device ,logger, tokenizer, args.method,args.use_demonstrations, args.k,
                               max_length=max_length, seed=args.seed)
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
        eval_data = load_data(
            args.task, args.eval_split, args.k, seed=seed, config_split=config_split,
            datasets=datasets, is_null=args.is_null
        )
        tensorize_eval_split = args.eval_split

        if is_multitask and args.multitask_filter_candidate_set:
            retrieval_data, validation_data = _split_multitask_retrieval(
                retrieval_data,
                candidate_size=args.multitask_candidate_size,
                validation_size=args.multitask_validation_size,
            )
            if args.multitask_use_validation_as_eval:
                eval_data = validation_data
                tensorize_eval_split = args.retrieval_split

        print("*"*20)
        print(f"retrieval_split: {args.retrieval_split}, eval_split: {args.eval_split}")
        logger.info(
            "seed=%s, multitask=%s, #retrieval=%d, #eval=%d",
            seed, is_multitask, len(retrieval_data), len(eval_data)
        )

        eval_by_task = defaultdict(list)
        for dp in eval_data:
            eval_by_task[dp["task"]].append(dp)

        seed_task_metrics = {}
        seed_total_correct = 0
        seed_total_count = 0

        for test_task, eval_data_task in eval_by_task.items():
            config_file = "config/tasks/{}.json".format(test_task)
            assert os.path.exists(config_file), config_file
            with open(config_file, "r") as f:
                config = json.load(f)
            is_classification = config["task_type"]=="classification"
            if is_classification:
                options = eval_data_task[0]["options"]
                assert np.all([d["options"]==options for d in eval_data_task])

            result = run(logger, test_task, metaicl_data, metaicl_model,
                         retrieval_data, eval_data_task, seed, checkpoint, is_classification,
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


def run(logger, task, metaicl_data, metaicl_model, retrieval_data, eval_data, seed,
        checkpoint, is_classification, add_newlines, tokenizer, tensorize_eval_split):

    if args.do_zeroshot:
        split_name = f"{tensorize_eval_split}-ret={args.retrieval_split}"
        if args.is_null:
            split_name += "-null"
        cache_suffix = "".join([
            "-topk" if args.topk else "",
            "-randomk" if args.randomk else "",
            "-bm25" if args.bm25 else "",
            "-k={}".format(args.k) if args.use_demonstrations else "",
            "-s={}".format(seed) if args.use_demonstrations or args.use_random_english_words else "",
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
    elif args.randomk:
        metaicl_data.tensorize_randomk(
            retrieval_data, eval_data, options=None, add_newlines=add_newlines,
            retrieval_split=args.retrieval_split, eval_split=tensorize_eval_split,
        )
    elif args.bm25:
        metaicl_data.tensorize_bm25(retrieval_data, eval_data, options=None,  add_newlines=add_newlines)
    else:
        raise ValueError("Please choose one selection method: --topk, --randomk, or --bm25")

    metaicl_data.print_tensorized_example()
    logger.info(cache_path)
    prediction_path = cache_path.replace(".pkl", ".txt")
    if args.use_calibration:
        prediction_path = prediction_path.replace(".txt", "-calibrated.txt")

    if metaicl_model.is_none():
        metaicl_model.load(checkpoint, gpt2=args.model_name, is_quant=args.is_quant)
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
        assert args.do_zeroshot
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
    parser.add_argument("--do_zeroshot", default=False, action="store_true")
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument("--use_calibration", default=False, action="store_true")
    parser.add_argument("--unseen_domain_only", default=False, action="store_true")

    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=str, default="0")

    parser.add_argument("--test_batch_size", type=int, default=64)
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
    parser.set_defaults(multitask_filter_candidate_set=True, multitask_use_validation_as_eval=True)
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
