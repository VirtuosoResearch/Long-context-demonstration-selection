# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv
import json
import logging
import string
import numpy as np
import pickle as pkl
import math
import torch
import random
from itertools import combinations
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer, OPTForCausalLM
from tqdm import tqdm

from rank_bm25 import BM25Okapi

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
# from forward_path import Forward

# import nltk
# from nltk.corpus import wordnet
import random

# from data_augment import DataAugment

from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from sklearn.linear_model import LinearRegression


from utils.data import load_data
from metaicl.model import MetaICLModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

class OpenLLMEvaluator:
    def __init__(self, model_name="deepseek-ai/deepseek-llm-7b-chat"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.generation_config = GenerationConfig.from_pretrained(model_name)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

    def query(self, question: str, examples: list = []) -> str:
        messages = []
        messages_text = ""
        for inp, out in examples:
            messages_text+=inp+out
            messages.append({"role": "user", "content": inp})
            messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user", "content": question})

        input_tensor = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        input_ids = self.tokenizer(messages_text)["input_ids"]
        outputs = self.model.generate(input_tensor, max_new_tokens=100)
        result = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return result.strip()

def _get_embedding_loss(model, tokenizer, input_texts, pad_to_length):
    model = model.model
    tokenizer.padding_side = "right"

    inputs = tokenizer(input_texts, padding="max_length", return_tensors='pt', truncation=True, max_length=pad_to_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        embedding = model.model.embed_tokens(inputs['input_ids'])
    embedding.requires_grad = True 
    embedding = embedding.to(model.dtype)

    outputs = model(inputs_embeds=embedding)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())

    lens = (inputs["input_ids"] != tokenizer.pad_token_id).sum(-1)
    ce_loss = loss.sum(-1) / lens

    ce_loss.backward()
    embedding_grad = embedding.grad

    effective_embedding_grad = embedding_grad[:, :-1, :]

    return ce_loss, effective_embedding_grad

def _get_embedding_loss_(model, tokenizer, input_texts, pad_to_length):
    model = model.model
    tokenizer.padding_side = "right"
    inputs = tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True, max_length=pad_to_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        embedding = model.model.embed_tokens(inputs['input_ids'])
    embedding.requires_grad = True 
    embedding = embedding.to(model.dtype)

    outputs = model(inputs_embeds=embedding)

    #outputs = model(**inputs)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
        shift_labels.size())

    lens = (inputs["input_ids"] != tokenizer.pad_token_id).sum(-1)

    #ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
    #ce_loss = loss.sum(-1).cpu().detach().numpy()
    ce_loss = loss.sum(-1) / lens

    ce_loss.backward()
    embedding_grad = embedding.grad
    return ce_loss, embedding_grad

class MetaICLData(object):

    def __init__(self, device=0, logger=None, tokenizer=None, method="channel", use_demonstrations=True, k=16,
                 max_length=1024, max_length_per_example=256,
                 do_tensorize=False, tensorize_dir=None, seed=0, n_process=None, n_gpu=None, local_rank=-1):

        self.logger = logger
        self.tokenizer = tokenizer
        self.method = method
        self.use_demonstrations = use_demonstrations
        self.k = k
        self.max_length = max_length
        self.max_length_per_example = max_length_per_example

        self.do_tensorize = do_tensorize
        self.tensorize_dir = tensorize_dir
        self.n_process = n_process
        self.n_gpu = n_gpu
        self.local_rank = local_rank

        self.tensorized_inputs = None
        self.metadata = None
        self.device = device
        self.is_null = False
        self.seed =seed


        #print(tokenizer)

        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def __len__(self):
        if self.tensorized_inputs is None:
            return 0
        return len(self.tensorized_inputs["input_ids"])

    @staticmethod
    def _normalize_feature_split(split_name):
        if split_name == "val":
            return "dev"
        return split_name

    def _get_feature_path(self, task, split_name):
        normalized_split = self._normalize_feature_split(split_name)
        candidate_paths = [
            f"./features/{task}_{normalized_split}_features.json",
            # backward compatibility with older files
            f"./features/{task}_val_features.json" if normalized_split == "dev" else f"./features/{task}_dev_features.json",
            f"./features/{task}_features.json",
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"Cannot find feature file for task={task}, split={split_name}. "
            f"Tried: {candidate_paths}"
        )

    def _load_features_for_datapoints(self, data, split_name):
        if len(data) == 0:
            return []

        features_by_task = {}
        counters_by_task = defaultdict(int)
        selected_features = []

        for dp in data:
            task = dp["task"]
            if task not in features_by_task:
                feature_path = self._get_feature_path(task, split_name)
                with open(feature_path, "r") as file:
                    features_by_task[task] = json.load(file)
            # Prefer explicit source index so subsets (e.g. first 90 / last 50)
            # map to the correct feature rows from the original split file.
            idx_in_task = dp.get("__feature_idx", counters_by_task[task])
            task_features = features_by_task[task]
            if idx_in_task >= len(task_features):
                raise ValueError(
                    f"features size mismatch for task={task}: "
                    f"need index {idx_in_task}, but only {len(task_features)} features found"
                )
            selected_features.append(task_features[idx_in_task])
            if "__feature_idx" not in dp:
                counters_by_task[task] += 1

        return selected_features

    def __str__(self):
        text = "[MetaICL Data]: method=%d, "
        if self.use_demonstrations:
            text += "%d demonstrations\n" % self.k
        else:
            text += "no demonstrations\n"
        if self.metadata is None:
            text += "Currently not containing any examples"
        else:
            text += "Currently containing %d examples with %d tensors to be fed in\n" % (len(self.metadata), len(self))
            text += "\n"
            text += self.print_tensorized_example(return_string=True)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def _select_top_k_neighbors_bm25(self, query_text, candidate_data, k):
        corpus = [dp["input"] for dp in candidate_data]
        tokenized_corpus = [doc.split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query_text.split()
        scores = bm25.get_scores(tokenized_query)
        topk_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [candidate_data[i] for i in topk_indices], topk_indices, scores

    def _select_top_k_neighbors(self, test_sample_embedding, test_embeddings, test_data, k, exclude_idx=None):
        similarities = []
        query_dim = len(test_sample_embedding)
        for idx, dp in enumerate(test_embeddings):
            if idx == len(test_data): break
            if exclude_idx is not None and idx == exclude_idx:
                similarities.append(-1.0)
                continue
            cand_dim = len(dp)
            if cand_dim != query_dim:
                raise ValueError(
                    f"Feature dimension mismatch: query dim={query_dim}, candidate dim={cand_dim} "
                    f"at candidate index={idx}, task={test_data[idx].get('task', 'unknown')}. "
                    "Please regenerate all feature files with the same embedding model."
                )
            similarity = 1 - cosine(test_sample_embedding, dp)
            similarities.append(similarity)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [test_data[i] for i in top_k_indices], top_k_indices , similarities

    def _kv_final_plan(self, candidate_features, eval_features, k, lambda_weight=0.7):
        n_candidates = len(candidate_features)
        n_eval = len(eval_features)
        if k <= 0 or n_candidates == 0 or n_eval == 0:
            return [[] for _ in range(n_eval)]

        cand = np.asarray(candidate_features, dtype=np.float32)
        ev = np.asarray(eval_features, dtype=np.float32)
        if cand.ndim != 2 or ev.ndim != 2:
            raise ValueError("Features for kv_final must be 2D arrays.")
        if cand.shape[1] != ev.shape[1]:
            raise ValueError(
                f"Feature dim mismatch in kv_final: candidate_dim={cand.shape[1]}, eval_dim={ev.shape[1]}"
            )

        dists = np.linalg.norm(ev[:, None, :] - cand[None, :, :], axis=2)  # [n_eval, n_candidates]
        topk = min(k, n_candidates)

        # Stage 1: aggregate query-wise nearest top-k to global frequency.
        freq_count = np.zeros(n_candidates, dtype=np.float32)
        for i in range(n_eval):
            nearest = np.argpartition(dists[i], topk - 1)[:topk]
            freq_count[nearest] += 1.0

        fmin, fmax = float(freq_count.min()), float(freq_count.max())
        if fmax > fmin:
            freq_norm = (freq_count - fmin) / (fmax - fmin)
        else:
            freq_norm = np.zeros_like(freq_count)

        # Stage 2: per-query blend of affinity and frequency.
        selected_indices_per_query = []
        for i in range(n_eval):
            affinity = -dists[i]  # lower distance => higher affinity
            amin, amax = float(affinity.min()), float(affinity.max())
            if amax > amin:
                affinity_norm = (affinity - amin) / (amax - amin)
            else:
                affinity_norm = np.zeros_like(affinity)

            score = lambda_weight * affinity_norm + (1.0 - lambda_weight) * freq_norm
            selected = np.argsort(score)[-topk:][::-1].tolist()
            selected = sorted(selected, key=lambda idx: (-freq_count[idx], idx))
            selected_indices_per_query.append(selected)

        return selected_indices_per_query

    def _kv_final_task_scores(self, candidate_features, validation_features, test_features, k, lambda_weight=0.7):
        n_candidates = len(candidate_features)
        n_val = len(validation_features)
        n_test = len(test_features)
        if k <= 0 or n_candidates == 0 or n_val == 0 or n_test == 0:
            return {
                "selected_indices": [],
                "score": np.zeros(n_candidates, dtype=np.float32),
                "affinity_norm": np.zeros(n_candidates, dtype=np.float32),
                "freq_norm": np.zeros(n_candidates, dtype=np.float32),
                "freq_count": np.zeros(n_candidates, dtype=np.float32),
            }

        cand = np.asarray(candidate_features, dtype=np.float32)
        val = np.asarray(validation_features, dtype=np.float32)
        tst = np.asarray(test_features, dtype=np.float32)
        if cand.ndim != 2 or val.ndim != 2:
            raise ValueError("Features for kv_final must be 2D arrays.")
        if cand.shape[1] != val.shape[1] or cand.shape[1] != tst.shape[1]:
            raise ValueError(
                f"Feature dim mismatch in kv_final: candidate_dim={cand.shape[1]}, "
                f"validation_dim={val.shape[1]}, test_dim={tst.shape[1]}"
            )

        # 1) Affinity is estimated ONLY from validation set.
        dists_val = np.linalg.norm(val[:, None, :] - cand[None, :, :], axis=2)  # [n_val, n_candidates]
        mean_dist_val = dists_val.mean(axis=0)
        affinity = -mean_dist_val
        amin, amax = float(affinity.min()), float(affinity.max())
        if amax > amin:
            affinity_norm = (affinity - amin) / (amax - amin)
        else:
            affinity_norm = np.zeros_like(affinity)

        # 2) Final selection/ranking happens on test set.
        dists_test = np.linalg.norm(tst[:, None, :] - cand[None, :, :], axis=2)  # [n_test, n_candidates]
        topk = min(k, n_candidates)

        freq_count = np.zeros(n_candidates, dtype=np.float32)
        for i in range(n_test):
            nearest = np.argpartition(dists_test[i], topk - 1)[:topk]
            freq_count[nearest] += 1.0

        fmin, fmax = float(freq_count.min()), float(freq_count.max())
        if fmax > fmin:
            freq_norm = (freq_count - fmin) / (fmax - fmin)
        else:
            freq_norm = np.zeros_like(freq_count)

        score = lambda_weight * affinity_norm + (1.0 - lambda_weight) * freq_norm
        selected = np.argsort(score)[-topk:][::-1].tolist()
        selected = sorted(selected, key=lambda idx: (-freq_count[idx], idx))

        return {
            "selected_indices": selected,
            "score": score,
            "affinity_norm": affinity_norm,
            "freq_norm": freq_norm,
            "freq_count": freq_count,
        }
    
    def get_dataloader(self, batch_size, is_training):
        inputs = self.tensorized_inputs
        for k, v in inputs.items():
            if type(v)==list:
                inputs[k] = torch.LongTensor(v)
        shape = inputs["input_ids"].shape
        self.logger.info(shape)
        for k, v in inputs.items():
            if k in ["demo_lens"]:
                continue
            assert v.shape==shape
        if "labels" in inputs:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], inputs["labels"])
        else:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
        if is_training:
            sampler=RandomSampler(dataset)
        else:
            sampler=SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader

    def evaluate(self, predictions, groundtruths, is_classification):
        assert len(predictions)==len(self.metadata)
        accs = []
        for prediction, groundtruth in zip(predictions, groundtruths):
            prediction = prediction.strip()
            groundtruth = [gt.strip() for gt in groundtruth] if type(groundtruth)==list else groundtruth.strip()
            is_correct = prediction in groundtruth if type(groundtruth)==list else prediction==groundtruth
            accs.append(is_correct)
        return np.mean(accs)

    def _tokenize_prompt_text(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _demo_prompt_prefix_tokens(self, n_examples):
        prompt = f"Use the following {n_examples} examples to answer the final query.\n"
        return self._tokenize_prompt_text(prompt)

    def _demo_example_separator_tokens(self):
        return self._tokenize_prompt_text("\n")

    def _query_prefix_tokens(self):
        return self._tokenize_prompt_text("\nFinal query:\n")

    def _strip_leading_special_tokens(self, token_ids):
        special_ids = set(self.tokenizer.all_special_ids)
        i = 0
        while i < len(token_ids) and token_ids[i] in special_ids:
            i += 1
        return token_ids[i:]

    @staticmethod
    def _to_preview_text(value):
        if isinstance(value, list):
            return " | ".join(str(v) for v in value)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _format_demo_preview(self, dp, max_chars=220):
        text = (
            f"input={self._to_preview_text(dp.get('input', ''))}; "
            f"output={self._to_preview_text(dp.get('output', ''))}"
        )
        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text

    def _prepro_each_datapoint(self, dp, is_first=True, is_training=False, for_demonstrations=False,
                               add_newlines=True):
        dp = dp.copy()
        if add_newlines:
            no_label = np.all([option=="" for option in dp["options"]])
            no_input = dp["input"]==""
            if self.method=="direct":
                if not is_first:
                    if no_input:
                        dp["input"] = "\n" + dp["input"]
                    else:
                        dp["input"] = "\n" + dp["input"]
                if not no_label:
                    dp["output"] = "\n" + dp["output"]
                    if "options" in dp:
                        dp["options"] = ["\n" + opt for opt in dp["options"]]
            else:
                raise NotImplementedError()

        input_tokens = self.tokenizer(dp["input"])["input_ids"]

        if is_training or for_demonstrations:
            output_tokens = self.tokenizer(dp["output"])["input_ids"]
            if "task" in dp:
                if (dp["task"].startswith("inst:piqa") or dp["task"].startswith("inst:yahoo_answers_topics")) and \
                        len(input_tokens)+len(output_tokens)+2>self.max_length_per_example:
                    input_tokens = input_tokens[:self.max_length_per_example // 2]
                    output_tokens = output_tokens[:self.max_length_per_example // 2 - 2]

                elif len(input_tokens)>=self.max_length_per_example - 2 - len(output_tokens):
                    if dp["task"].startswith("inst:") and len(input_tokens)<len(output_tokens):
                        output_tokens = output_tokens[:self.max_length_per_example - 2 - len(input_tokens)]
                    else:
                        input_tokens = input_tokens[:self.max_length_per_example - 2 - len(output_tokens)]

            assert len(input_tokens)+len(output_tokens)+2<=self.max_length_per_example, \
                (dp.get("task", None), len(input_tokens), len(output_tokens), self.max_length_per_example)

            if self.method=="direct":
                return input_tokens, output_tokens
            elif self.method=="channel":
                return output_tokens, input_tokens
            else:
                raise NotImplementedError()

        else:
            assert len(dp["options"])>=2, dp
            assert dp["output"] in dp["options"]
            option_tokens = [self.tokenizer(option)["input_ids"] for option in dp["options"]]
            option_length = np.max([len(option) for option in option_tokens])

            if len(input_tokens)>=self.max_length_per_example - 2 - option_length:
                input_tokens = input_tokens[:self.max_length_per_example - 2 - option_length]

            input_tokens = [input_tokens for _ in option_tokens]
            output_tokens = option_tokens
            option_tokens = [dp["options"].index(dp["output"])]

            if self.method=="direct":
                return input_tokens, output_tokens, option_tokens
            elif self.method=="channel":
                return output_tokens, input_tokens, option_tokens
            else:
                raise NotImplementedError()

    def tensorize_topk(self, _test_data, _val_data, options=None, add_newlines=True,
                       retrieval_split="test", eval_split="dev"):
        if options is not None:
            for i, dp in enumerate(_test_data):
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                _val_data[i] = {"input": dp, "options": options}

        val_data, test_data =  [], []

        for dp in _test_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())
        for dp in _val_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            val_data.append(dp.copy())
        
        test_features = self._load_features_for_datapoints(test_data, retrieval_split)
        val_features = self._load_features_for_datapoints(val_data, eval_split)

        input_ids, attention_mask, token_type_ids, demo_lens = [], [], [], []
        metadata = []
        self.topk_last_selected_demo_info = []
        special_ids = set(self.tokenizer.all_special_ids)

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)
            if self.use_demonstrations:
                dp_feature = val_features[dp_idx]            

                top_k_neighbors, top_k_indices, __ = self._select_top_k_neighbors(
                    dp_feature, test_features, test_data, self.k, exclude_idx=None
                )
                prompt_order = list(reversed(top_k_neighbors))
                prompt_order_indices = list(reversed([int(i) for i in top_k_indices]))
                self.topk_last_selected_demo_info.append(
                    {
                        "eval_index": int(dp_idx),
                        "eval_task": dp.get("task"),
                        "selected_indices": prompt_order_indices,
                        "selected_demos": [
                            {
                                "rank": int(rank),
                                "candidate_index_in_pool": int(prompt_order_indices[rank]),
                                "candidate_task": demo_dp.get("task"),
                                "candidate_feature_idx": int(demo_dp.get("__feature_idx", -1)),
                                "content": self._format_demo_preview(demo_dp),
                            }
                            for rank, demo_dp in enumerate(prompt_order)
                        ],
                    }
                )

                demonstrations = self._demo_prompt_prefix_tokens(len(top_k_neighbors))
                sep_tokens = self._demo_example_separator_tokens()
                for i, neighbor_dp in enumerate(prompt_order):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    if i > 0:
                        demonstrations += sep_tokens
                    demonstrations += self._strip_leading_special_tokens(input_)
                    demonstrations += self._strip_leading_special_tokens(output_)
                #print(demonstrations)
            #print(a)
            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]
            # print("indices : ",indices)
            # print("inputs : ",inputs)
            # print("answer : ",answer)
            # print("demonstrations : ",demonstrations)

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    demo_len = 1 + sum(1 for tok in demonstrations if tok not in special_ids)
                    inputs_ = demonstrations + self._query_prefix_tokens() + self._strip_leading_special_tokens(inputs_)
                else:
                    demo_len = 0
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, self.tokenizer,self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, 
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])
                demo_lens.append(demo_len)


        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids),
                                      demo_lens=torch.LongTensor(demo_lens))
        self.metadata = metadata

    def tensorize_kv_final(self, _test_data, _val_data, options=None, add_newlines=True,
                           retrieval_split="test", eval_split="dev", validation_data=None,
                           validation_split="test", lambda_weight=0.7):
        if options is not None:
            for i, dp in enumerate(_test_data):
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                _val_data[i] = {"input": dp, "options": options}

        val_data, test_data = [], []
        for dp in _test_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]
            test_data.append(dp.copy())
        for dp in _val_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]
            val_data.append(dp.copy())

        if validation_data is None:
            validation_data = _val_data
            validation_split = eval_split

        validation_task_data = []
        for dp in validation_data:
            if "output" not in dp:
                dp = dp.copy()
                dp["output"] = dp["options"][0]
            validation_task_data.append(dp.copy())

        test_features = self._load_features_for_datapoints(test_data, retrieval_split)
        validation_features = self._load_features_for_datapoints(validation_task_data, validation_split)
        # Affinity comes from validation; final subset selection/ranking uses test features.
        test_eval_features = self._load_features_for_datapoints(val_data, eval_split)
        kv_info = self._kv_final_task_scores(
            candidate_features=test_features,
            validation_features=validation_features,
            test_features=test_eval_features,
            k=self.k,
            lambda_weight=lambda_weight,
        )
        selected_indices = kv_info["selected_indices"]

        self.kv_final_last_scores = []
        for rank, idx in enumerate(selected_indices):
            dp = test_data[idx]
            self.kv_final_last_scores.append(
                {
                    "rank": rank,
                    "candidate_index_in_pool": int(idx),
                    "candidate_task": dp.get("task"),
                    "candidate_feature_idx": int(dp.get("__feature_idx", -1)),
                    "score": float(kv_info["score"][idx]),
                    "affinity_norm": float(kv_info["affinity_norm"][idx]),
                    "freq_norm": float(kv_info["freq_norm"][idx]),
                    "freq_count": float(kv_info["freq_count"][idx]),
                }
            )

        input_ids, attention_mask, token_type_ids, demo_lens = [], [], [], []
        metadata = []

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            demonstrations = []
            if self.use_demonstrations and self.k > 0:
                selected_local_indices = selected_indices
                selected_neighbors = [test_data[idx] for idx in selected_local_indices]
                demonstrations = self._demo_prompt_prefix_tokens(len(selected_neighbors))
                sep_tokens = self._demo_example_separator_tokens()
                for i, neighbor_dp in enumerate(selected_neighbors):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    if i > 0:
                        demonstrations += sep_tokens
                    demonstrations += self._strip_leading_special_tokens(input_)
                    demonstrations += self._strip_leading_special_tokens(output_)

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]
            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + self._query_prefix_tokens() + self._strip_leading_special_tokens(inputs_)
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, self.tokenizer,
                    self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(
            input_ids=torch.LongTensor(input_ids),
            attention_mask=torch.LongTensor(attention_mask),
            token_type_ids=torch.LongTensor(token_type_ids)
        )
        self.metadata = metadata

    def tensorize_with_selected_demo_examples(self, eval_data, selected_demo_examples_per_query, add_newlines=True):
        if len(eval_data) != len(selected_demo_examples_per_query):
            raise ValueError(
                f"Length mismatch: eval_data={len(eval_data)} vs selected_demo_examples_per_query={len(selected_demo_examples_per_query)}"
            )

        val_data = []
        for dp in eval_data:
            if "output" not in dp:
                dp = dp.copy()
                dp["output"] = dp["options"][0]
            val_data.append(dp.copy())

        input_ids, attention_mask, token_type_ids, demo_lens = [], [], [], []
        metadata = []

        special_ids = set(self.tokenizer.all_special_ids)

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines
            )

            demonstrations = []
            if self.use_demonstrations and self.k > 0:
                selected_neighbors = selected_demo_examples_per_query[dp_idx][: self.k]
                demonstrations = self._demo_prompt_prefix_tokens(len(selected_neighbors))
                sep_tokens = self._demo_example_separator_tokens()
                for i, neighbor_dp in enumerate(selected_neighbors):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines
                    )
                    if i > 0:
                        demonstrations += sep_tokens
                    demonstrations += self._strip_leading_special_tokens(input_)
                    demonstrations += self._strip_leading_special_tokens(output_)

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]
            metadata.append({"indices": indices, "answer": answer, "options": dp["options"], "task": dp.get("task")})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    # Must match prepro_sentence_pair_single() behavior that drops all special tokens.
                    # +1 for BOS that prepro_sentence_pair_single() prepends.
                    demo_len = 1 + sum(1 for tok in demonstrations if tok not in special_ids)
                    inputs_ = demonstrations + self._query_prefix_tokens() + self._strip_leading_special_tokens(inputs_)
                else:
                    demo_len = 0
                encoded = prepro_sentence_pair_single(
                    inputs_,
                    outputs_,
                    self.max_length,
                    self.tokenizer,
                    self.tokenizer.bos_token_id,
                    self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations,
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])
                demo_lens.append(demo_len)

        self.tensorized_inputs = dict(
            input_ids=torch.LongTensor(input_ids),
            attention_mask=torch.LongTensor(attention_mask),
            token_type_ids=torch.LongTensor(token_type_ids),
            demo_lens=torch.LongTensor(demo_lens),
        )
        self.metadata = metadata


    def _forward_selection(self, embeddings, top_k_indices, m, candidate_labels, test_data, similarities, seed, temperature=0.1):
        assert m <= len(top_k_indices), "Error: m must less than k"

        for idx in range(len(embeddings)):
            embeddings[idx] = torch.tensor(embeddings[idx], dtype=torch.float32)
            embeddings[idx] = embeddings[idx] / torch.norm(embeddings[idx])

        top_indice = np.argsort(similarities)[-1:][::-1]

        selected_indices = set(top_indice)
        top_k_indices = [item for item in top_k_indices if item != top_indice]
        remaining_indices = set(top_k_indices)

        while len(selected_indices) < m:
            best_candidate = None
            best_loss = float('inf')

            for candidate in remaining_indices:
                temp_selected = list(selected_indices.union({candidate}))

                simloss = sum(similarities[idx] for idx in temp_selected) / len(temp_selected)

                con_loss = 0.0
                for idx1 in temp_selected:
                    cnt_pos = 0
                    pos_loss = 0.0
                    # print("len(embeddings) : ", len(embeddings),"idx1 : ", idx1, "len(candidate_labels) : ",len(candidate_labels))
                    idx1_embedding, idx1_label = embeddings[idx1], candidate_labels[idx1]
                    logits = []
                    for idx2 in temp_selected:
                        if idx1 == idx2:
                            continue
                        idx2_embedding, idx2_label = embeddings[idx2], candidate_labels[idx2]
                        similarity = torch.matmul(idx1_embedding, idx2_embedding) / temperature
                        logits.append(similarity)
                        if idx1_label == idx2_label:
                            pos_loss += torch.exp(similarity)
                            cnt_pos += 1

                    logits = torch.tensor(logits)
                    logits_max = torch.max(logits)
                    logits = logits - logits_max.detach()
                    exp_logits = torch.exp(logits)
                    exp_logits_sum = exp_logits.sum()

                    if cnt_pos > 0:
                        pos_prob = pos_loss / exp_logits_sum
                        idx_loss = pos_prob / cnt_pos
                    else:
                        idx_loss = 1e-6
                    idx_loss = torch.tensor(idx_loss, dtype=torch.float32)
                    con_loss += -1.0 * torch.log(idx_loss)

                lam = 0.05
                total_loss = -simloss + lam * con_loss

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_candidate = candidate

            selected_indices.add(best_candidate)
            remaining_indices.remove(best_candidate)

        real_id = [idx for idx in selected_indices]
        return [test_data[idx] for idx in real_id], real_id

    def _select_random_k_neighbors(self, test_sample_embedding, test_embeddings, test_data, k, exclude_idx=None):
        length = len(test_data)
        candidates = [i for i in range(length) if i != exclude_idx]
        random_indices = random.sample(candidates, k)

        return [test_data[i] for i in random_indices]
    
    def tensorize_randomk(self, _test_data, _val_data, options=None, add_newlines=True,
                          retrieval_split="test", eval_split="dev"):
        if options is not None:
            for i, dp in enumerate(_test_data):
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                _val_data[i] = {"input": dp, "options": options}

        val_data, test_data =  [], []

        for dp in _test_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())
        for dp in _val_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]  # randomly choose one (we don't need it anyways)
            val_data.append(dp.copy())
        test_index_by_obj = {id(dp): idx for idx, dp in enumerate(test_data)}
        
        test_features = self._load_features_for_datapoints(test_data, retrieval_split)
        val_features = self._load_features_for_datapoints(val_data, eval_split)

        input_ids, attention_mask, token_type_ids, demo_lens = [], [], [], []
        metadata = []
        self.randomk_last_selected_demo_info = []
        special_ids = set(self.tokenizer.all_special_ids)

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            demonstrations = []
            if self.use_demonstrations and self.k>0:
                dp_feature = val_features[dp_idx]            

                random_k_neighbors = self._select_random_k_neighbors(
                    dp_feature, test_features, test_data, self.k-self.k//4, exclude_idx=None
                )

                top_k_neighbors, top_k_indices, __ = self._select_top_k_neighbors(
                    dp_feature, test_features, test_data, self.k//4, exclude_idx=None
                )
                top_k_indices = [int(i) for i in top_k_indices]
                random_indices = [int(test_index_by_obj[id(neighbor_dp)]) for neighbor_dp in random_k_neighbors]
                selected_demos = []
                for rank, (neighbor_dp, candidate_idx) in enumerate(zip(random_k_neighbors, random_indices)):
                    selected_demos.append(
                        {
                            "rank": int(rank),
                            "selection_source": "random",
                            "candidate_index_in_pool": int(candidate_idx),
                            "candidate_task": neighbor_dp.get("task"),
                            "candidate_feature_idx": int(neighbor_dp.get("__feature_idx", -1)),
                            "content": self._format_demo_preview(neighbor_dp),
                        }
                    )
                base_rank = len(selected_demos)
                for offset, (neighbor_dp, candidate_idx) in enumerate(zip(top_k_neighbors, top_k_indices)):
                    selected_demos.append(
                        {
                            "rank": int(base_rank + offset),
                            "selection_source": "topk_tail",
                            "candidate_index_in_pool": int(candidate_idx),
                            "candidate_task": neighbor_dp.get("task"),
                            "candidate_feature_idx": int(neighbor_dp.get("__feature_idx", -1)),
                            "content": self._format_demo_preview(neighbor_dp),
                        }
                    )
                self.randomk_last_selected_demo_info.append(
                    {
                        "eval_index": int(dp_idx),
                        "eval_task": dp.get("task"),
                        "selected_indices": random_indices + top_k_indices,
                        "selected_demos": selected_demos,
                    }
                )

                demonstrations = self._demo_prompt_prefix_tokens(len(random_k_neighbors) + len(top_k_neighbors))
                sep_tokens = self._demo_example_separator_tokens()
                for i, neighbor_dp in enumerate(random_k_neighbors):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    if i > 0:
                        demonstrations += sep_tokens
                    demonstrations += input_ + output_
                for i, neighbor_dp in enumerate(top_k_neighbors):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += sep_tokens
                    demonstrations += input_ + output_

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    demo_len = 1 + sum(1 for tok in demonstrations if tok not in special_ids)
                    inputs_ = demonstrations + self._query_prefix_tokens() + inputs_
                else:
                    demo_len = 0
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, self.tokenizer, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])
                demo_lens.append(demo_len)

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids),
                                      demo_lens=torch.LongTensor(demo_lens))
        self.metadata = metadata

    def _random_datasource(self, task, datapath, m):
        with open(datapath, "r") as file:
            data = [json.loads(line) for line in file]
        
        candidates = [item for item in data if item['task']!=task]
        output = random.sample(candidates, m)
        return output

    def tensorize_bm25(self, _test_data, _val_data, options=None, add_newlines=True):
        if options is not None:
            for i, dp in enumerate(_test_data):
                _test_data[i] = {"input": dp, "options": options}
            for i, dp in enumerate(_val_data):
                _val_data[i] = {"input": dp, "options": options}

        val_data, test_data = [], []
        for dp in _test_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]
            test_data.append(dp.copy())
        for dp in _val_data:
            if "output" not in dp:
                dp["output"] = dp["options"][0]
            val_data.append(dp.copy())

        task = _test_data[0]["task"]
        input_ids, attention_mask, token_type_ids, demo_lens = [], [], [], []
        metadata = []
        special_ids = set(self.tokenizer.all_special_ids)

        test_inputs = [dp["input"].split() for dp in test_data]
        bm25 = BM25Okapi(test_inputs)

        instructions = f"Use the following {self.k} examples to answer the final query.\n"
        init_tokens = self._tokenize_prompt_text(instructions)

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            if self.use_demonstrations:
                query_terms = dp["input"].split()
                scores = bm25.get_scores(query_terms)
                scores[dp_idx] = -1e9

                topk_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.k]
                top_k_neighbors = [test_data[i] for i in topk_indices]

                demonstrations = init_tokens
                sep_tokens = self._demo_example_separator_tokens()
                for i, neighbor_dp in enumerate(top_k_neighbors):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    if i > 0:
                        demonstrations += sep_tokens
                    demonstrations += self._strip_leading_special_tokens(input_)
                    demonstrations += self._strip_leading_special_tokens(output_)

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})
            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    demo_len = 1 + sum(1 for tok in demonstrations if tok not in special_ids)
                    inputs_ = demonstrations + self._query_prefix_tokens() + self._strip_leading_special_tokens(inputs_)
                else:
                    demo_len = 0
                encoded = prepro_sentence_pair_single(
                    inputs_, [outputs_[2]], self.max_length, self.tokenizer,
                    self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations)
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])
                demo_lens.append(demo_len)

        self.tensorized_inputs = dict(
            input_ids=torch.LongTensor(input_ids),
            attention_mask=torch.LongTensor(attention_mask),
            token_type_ids=torch.LongTensor(token_type_ids),
            demo_lens=torch.LongTensor(demo_lens))
        self.metadata = metadata
    
    def tensorize(self, _train_data, _test_data, options=None,
                  add_newlines=True):

        if options is not None:
            for i, dp in enumerate(_test_data):
                assert "options" not in dp
                assert type(dp)==str
                _test_data[i] = {"input": dp, "options": options}

        train_data, test_data = [], []

        for dp in _test_data:
            assert type(dp)==dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"])==list, \
                ("Test example should contain input and options in a list format", dp)
            if "output" not in dp:
                dp["output"] = dp["options"][0] # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())

        # each datapoint: passage, question, options, output
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        if self.use_demonstrations:
            assert len(train_data)==self.k
            demonstrations = []
            for i, dp in enumerate(train_data):
                input_, output_ = self._prepro_each_datapoint(
                    dp, is_first=i==0, for_demonstrations=True,
                    add_newlines=add_newlines)
                demonstrations += input_ + output_

        for dp_idx, dp in enumerate(test_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            indices = [[i] for i in range(len(input_ids), len(input_ids)+len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            print("inputs: ",inputs)
            print("outputs: ",outputs)

            for inputs_, outputs_ in zip(inputs, outputs):
                print("inputs_ : ",inputs_)
                print("outputs_ : ",outputs_)
                
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_

                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, self.tokenizer, bos_token_id, eos_token_id,
                    allow_truncation=self.use_demonstrations)

                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata

    def print_tensorized_example(self, return_string=False):
        assert self.tensorized_inputs is not None

        idx = 0
        text = "Checking the first example..."
        input_ids = self.tensorized_inputs["input_ids"][idx]
        token_type_ids = self.tensorized_inputs["token_type_ids"][idx]
        if type(input_ids)!=list:
            input_ids = input_ids.numpy().tolist()
        if type(token_type_ids)!=list:
            token_type_ids = token_type_ids.numpy().tolist()

        text += "\nInput:\n"
        text += self.tokenizer.decode(input_ids[:token_type_ids.index(1)])
        text += "\nOutput:\n"
        text += self.tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id==1])

        if return_string:
            return text

        if self.local_rank<=0:
            self.logger.info(text)



def prepro_sentence_pair_single_(ids1, ids2, max_length, tokenizer,
                                bos_token_id, eos_token_id,
                                allow_truncation=False):

    if allow_truncation and len(ids1)+len(ids2) > max_length:
        ids1 = ids1[len(ids1)+len(ids2)-max_length:] # len = max_length-len(ids2)
        assert len(ids1)+len(ids2)==max_length

    n_mask = max_length-len(ids1)-len(ids2)
    assert n_mask>=0, (max_length, len(ids1), len(ids2))
    input_ids = ids1+ids2 + [eos_token_id for _ in range(n_mask)]
    #print("input_ids : ",len(input_ids))
    attention_mask = [1 for _ in ids1+ids2] + [eos_token_id for _ in range(n_mask)]
    token_type_ids = [0 for _ in ids1] + [1 for _ in ids2] + [eos_token_id for _ in range(n_mask)]

    return input_ids, attention_mask, token_type_ids

def prepro_sentence_pair_single(ids1, ids2, max_length,
                                tokenizer, bos_token_id, eos_token_id,
                                allow_truncation=False):
    # Remove special tokens
    #print(tokenizer.all_special_ids)
    special_ids = set(tokenizer.all_special_ids)
    #special_ids.extend([128000, 128001])
    ids1 = [i for i in ids1 if i not in special_ids]
    ids2 = [i for i in ids2 if i not in special_ids]

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is None; cannot build fixed-length input.")

    # Add bos and eos tokens later, so leave space for them
    total_len = len(ids1) + len(ids2) + 2  # +2 for bos and eos

    if allow_truncation and total_len > max_length:
        # Truncate from the beginning of ids1
        overflow = total_len - max_length
        ids1 = ids1[overflow:]
        total_len = len(ids1) + len(ids2) + 2
        assert total_len == max_length

    # Add bos at start, eos at end
    input_ids = [bos_token_id] + ids1 + ids2 + [eos_token_id]

    # Padding if needed
    n_pad = max_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * n_pad

    # Attention mask: 1 for tokens, 0 for padding (if padding is eos_token)
    attention_mask = [1] * (len(input_ids) - n_pad) + [0] * n_pad

    # Token type ids: 0 for ids1, 1 for ids2, 0 for bos and eos (you can adjust this)
    token_type_ids = [0] + [0] * len(ids1) + [1] * len(ids2) + [0] + [0] * n_pad

    return input_ids, attention_mask, token_type_ids


def prepro_sentence_pair(train_inputs, test_inputs, max_length, tokenizer, 
                         bos_token_id, eos_token_id,
                         allow_truncation=False):
    input_ids, attention_mask, token_type_ids = [], [], []
    for test_input in test_inputs:
        for train_input in train_inputs:
            _input_ids, _attention_mask, _token_type_ids = \
                prepro_sentence_pair_single(train_input, test_input, max_length, tokenizer, 
                                            bos_token_id, eos_token_id,
                                            allow_truncation=allow_truncation)
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
            token_type_ids.append(_token_type_ids)

    return {"input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "token_type_ids": torch.LongTensor(token_type_ids)}
