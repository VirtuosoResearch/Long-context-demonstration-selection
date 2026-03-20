# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv
import json
import string
import numpy as np
import torch

def load_data(task, split, k, seed=0, config_split=None, datasets=None,
              is_null=False):
    if config_split is None:
        config_split = split

    if datasets is None:
        with open(os.path.join("config", task+".json"), "r") as f:
            config = json.load(f)
        datasets = config[config_split]

    data = []
    for dataset in datasets:
        data_path = os.path.join("data", dataset,
                                 "{}_{}.jsonl".format(dataset, split))
        with open(data_path, "r") as f:
            for feature_idx, line in enumerate(f):
                dp = json.loads(line)
                if is_null:
                    dp["input"] = "N/A"
                # Keep source index so feature lookup stays correct
                # after candidate/validation/test slicing.
                dp["__feature_idx"] = feature_idx
                dp["__feature_task"] = dataset
                dp["__feature_split"] = split
                data.append(dp)
    return data

