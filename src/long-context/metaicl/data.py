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
from thop import profile

from rank_bm25 import BM25Okapi
logging.getLogger("thop").setLevel(logging.WARNING)

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

    def query(self, question: str, examples: list = [], is_flops=False) -> str:
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
        flops=0
        if is_flops:
            flops, params = profile(self.model, inputs=(input_ids,))

        outputs = self.model.generate(input_tensor, max_new_tokens=100)
        result = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return result.strip(), flops

def _get_embedding_loss(model, tokenizer, input_texts, pad_to_length, is_flops=False):
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

    flops=0
    if is_flops:
        flops, params = profile(model, inputs=(inputs['input_ids'],))

    return ce_loss, effective_embedding_grad, flops

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
                 do_tensorize=False, tensorize_dir=None, seed=0, n_process=None, n_gpu=None, local_rank=-1, is_flops=False):

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
        self.is_flops = is_flops
        self.seed =seed
        self.total_flops=0


        #print(tokenizer)

        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def __len__(self):
        if self.tensorized_inputs is None:
            return 0
        return len(self.tensorized_inputs["input_ids"])

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

    def _select_top_k_neighbors(self, test_sample_embedding, test_embeddings, test_data, k, dp_idx):
        similarities = []
        for idx, dp in enumerate(test_embeddings):
            if idx == len(test_data): break
            if idx == dp_idx:
                similarities.append(-1.0)
                continue
            similarity = 1 - cosine(test_sample_embedding, dp)
            similarities.append(similarity)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [test_data[i] for i in top_k_indices], top_k_indices , similarities
    
    def get_dataloader(self, batch_size, is_training):
        inputs = self.tensorized_inputs
        for k, v in inputs.items():
            if type(v)==list:
                inputs[k] = torch.LongTensor(v)
        shape = inputs["input_ids"].shape
        self.logger.info(shape)
        for v in inputs.values():
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
        precisions = defaultdict(list)
        recalls = defaultdict(list)
        for prediction, groundtruth in zip(predictions, groundtruths):
            prediction = prediction.strip()
            groundtruth = [gt.strip() for gt in groundtruth] if type(groundtruth)==list else groundtruth.strip()
            is_correct = prediction in groundtruth if type(groundtruth)==list else prediction==groundtruth
            accs.append(is_correct)
            if is_classification:
                recalls[groundtruth].append(is_correct)
                precisions[prediction].append(is_correct)

        if not is_classification:
            return np.mean(accs)

        f1s = []
        for key in recalls:
            precision = np.mean(precisions[key]) if key in precisions else 1.0
            recall = np.mean(recalls[key])
            if precision+recall==0:
                f1s.append(0)
            else:
                f1s.append(2*precision*recall / (precision+recall))

        return np.mean(f1s)

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

    def tensorize_topk(self, _test_data, _val_data, options=None, add_newlines=True):
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
        
        task = _test_data[0]["task"]
        test_features_path = f"./features/{task}_test_features.json"
        with open(test_features_path, "r") as file:
            test_features = json.load(file)
        val_features_path = f"./features/{task}_val_features.json"
        with open(val_features_path, "r") as file:
            val_features = json.load(file)

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)
            if self.use_demonstrations:
                dp_feature = val_features[dp_idx]            

                top_k_neighbors, _, __ = self._select_top_k_neighbors(
                    dp_feature, test_features, test_data, self.k, dp_idx
                )

                demonstrations = []
                for i, neighbor_dp in enumerate(reversed(top_k_neighbors)):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_[1:] + output_[1:]
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
                    inputs_ = demonstrations + inputs_[1:]
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, self.tokenizer,self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, 
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])


        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
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

    def _select_random_k_neighbors(self, test_sample_embedding, test_embeddings, test_data, k, dp_idx):
        length = len(test_data)
        candidates = [i for i in range(length) if i!= dp_idx]
        random_indices = random.sample(candidates, k)

        return [test_data[i] for i in random_indices]
    
    def tensorize_randomk(self, _test_data, _val_data, options=None, add_newlines=True):
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
        
        task = _test_data[0]["task"]
        test_features_path = f"./features/{task}_test_features.json"
        with open(test_features_path, "r") as file:
            test_features = json.load(file)
        val_features_path = f"./features/{task}_val_features.json"
        with open(val_features_path, "r") as file:
            val_features = json.load(file)

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        for dp_idx, dp in enumerate(val_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            demonstrations = []
            if self.use_demonstrations and self.k>0:
                dp_feature = val_features[dp_idx]            

                random_k_neighbors = self._select_random_k_neighbors(
                    dp_feature, test_features, test_data, self.k-self.k//4, dp_idx
                )

                top_k_neighbors, _, __ = self._select_top_k_neighbors(
                    dp_feature, test_features, test_data, self.k//4, dp_idx
                )

                demonstrations = []
                for i, neighbor_dp in enumerate(random_k_neighbors):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_ + output_
                for i, neighbor_dp in enumerate(top_k_neighbors):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_ + output_

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_
                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, self.tokenizer, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations
                )
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
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
        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        test_inputs = [dp["input"].split() for dp in test_data]
        bm25 = BM25Okapi(test_inputs)

        instructions = f"Here are {len(test_data[0]['options'])} options: "
        for option in test_data[0]["options"]:
            instructions += option + ", "
        instructions += f"You should choose one of them to answer at the end. \nHere are {self.k} samples for your reference. \n"
        init_tokens = self.tokenizer(instructions)["input_ids"][1:]

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
                for i, neighbor_dp in enumerate(top_k_neighbors):
                    input_, output_ = self._prepro_each_datapoint(
                        neighbor_dp, is_first=i == 0, for_demonstrations=True, add_newlines=add_newlines)
                    demonstrations += input_[1:] + output_[1:]

            indices = [[i] for i in range(len(input_ids), len(input_ids) + len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})
            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_[1:]
                encoded = prepro_sentence_pair_single(
                    inputs_, [outputs_[2]], self.max_length, self.tokenizer,
                    self.tokenizer.bos_token_id, self.tokenizer.eos_token_id,
                    allow_truncation=self.use_demonstrations)
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(
            input_ids=torch.LongTensor(input_ids),
            attention_mask=torch.LongTensor(attention_mask),
            token_type_ids=torch.LongTensor(token_type_ids))
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
