import json
import logging
import os
from typing import Any

import pandas as pd
import polars as pl
import torch

logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: list[dict[str, Any]], name: str | None = None, split: str | None = None):
        super().__init__()
        self.data = data
        self.name = name
        self.split = split

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]

    def get_data(self) -> list[dict[str, Any]]:
        return self.data

    def repeat(self, n: int) -> "Dataset":
        if n <= 0:
            raise ValueError("Repeat count must be positive")

        repeated_data = []
        for item in self.data:
            repeated_data.extend([item.copy() for _ in range(n)])

        return Dataset(data=repeated_data, name=self.name, split=self.split)

    def get_data_path(self) -> str | None:
        if self.name is None or self.split is None:
            return None

        registry = DatasetRegistry._load_registry()
        if self.name not in registry or self.split not in registry[self.name]:
            return None

        return registry[self.name][self.split]

    def get_verl_data_path(self) -> str | None:
        data_path = self.get_data_path()
        if data_path is None:
            return None

        verl_path = data_path.replace(".parquet", "_verl.parquet")
        return verl_path if os.path.exists(verl_path) else None

    @classmethod
    def load_data(cls, path: str) -> "Dataset":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found at {path}")

        file_ext = os.path.splitext(path)[1].lower()

        if file_ext == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        elif file_ext == ".jsonl":
            data = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        elif file_ext == ".csv":
            data = pd.read_csv(path).to_dict("records")
        elif file_ext == ".parquet":
            data = pd.read_parquet(path).to_dict("records")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        return cls(data=data)


class DatasetRegistry:
    _REGISTRY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "registry")
    _REGISTRY_FILE = os.path.join(_REGISTRY_DIR, "dataset_registry.json")
    _DATASET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets")

    @classmethod
    def _ensure_directories(cls) -> None:
        os.makedirs(cls._REGISTRY_DIR, exist_ok=True)
        os.makedirs(cls._DATASET_DIR, exist_ok=True)

    @classmethod
    def _load_registry(cls) -> dict[str, dict[str, str]]:
        cls._ensure_directories()
        if not os.path.exists(cls._REGISTRY_FILE):
            return {}

        try:
            with open(cls._REGISTRY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON format in registry file. Creating a new registry.")
            return {}

    @classmethod
    def _save_registry(cls, registry: dict[str, dict[str, str]]) -> None:
        cls._ensure_directories()
        with open(cls._REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

    @classmethod
    def register_dataset(cls, name: str, data: list[dict[str, Any]] | Any, split: str = "default") -> Dataset:
        cls._ensure_directories()

        dataset_dir = os.path.join(cls._DATASET_DIR, name)
        os.makedirs(dataset_dir, exist_ok=True)

        if hasattr(data, "to_pandas") and callable(data.to_pandas):
            data_df = data.to_pandas()
            data_list = data_df.to_dict("records")
        else:
            data_list = data
            data_df = pd.DataFrame(data_list)

        dataset_path = os.path.join(dataset_dir, f"{split}.parquet")
        data_df.to_parquet(dataset_path)

        verl_data = cls.apply_verl_postprocessing(data_list)
        verl_dataset_path = os.path.join(dataset_dir, f"{split}_verl.parquet")
        verl_data_df = pd.DataFrame(verl_data)
        verl_data_df.to_parquet(verl_dataset_path)

        registry = cls._load_registry()
        if name not in registry:
            registry[name] = {}
        registry[name][split] = dataset_path
        cls._save_registry(registry)

        logger.info(
            "Registered dataset '%s' split '%s' with %s examples. Verl-processed version saved at %s.",
            name,
            split,
            len(data_list),
            verl_dataset_path,
        )

        return Dataset(data=data_list, name=name, split=split)

    @classmethod
    def load_dataset(cls, name: str, split: str = "default") -> Dataset | None:
        registry = cls._load_registry()
        if name not in registry:
            logger.warning("Dataset '%s' not found in registry.", name)
            return None

        dataset_info = registry[name]
        if split not in dataset_info:
            logger.warning("Split '%s' not found in dataset '%s'.", split, name)
            return None

        dataset_path = dataset_info[split]
        if not os.path.exists(dataset_path):
            logger.warning("Dataset file not found: %s", dataset_path)
            return None

        data = pl.read_parquet(dataset_path).to_dicts()
        logger.info("Loaded dataset '%s' split '%s' with %s examples.", name, split, len(data))
        return Dataset(data=data, name=name, split=split)

    @classmethod
    def get_dataset_names(cls) -> list[str]:
        return list(cls._load_registry().keys())

    @classmethod
    def get_dataset_splits(cls, name: str) -> list[str]:
        registry = cls._load_registry()
        if name not in registry:
            return []
        return list(registry[name].keys())

    @classmethod
    def dataset_exists(cls, name: str, split: str | None = None) -> bool:
        registry = cls._load_registry()
        if name not in registry:
            return False
        if split is not None:
            return split in registry[name]
        return True

    @classmethod
    def remove_dataset_split(cls, name: str, split: str) -> bool:
        registry = cls._load_registry()
        if name not in registry or split not in registry[name]:
            logger.warning("Dataset '%s' split '%s' not found in registry.", name, split)
            return False

        dataset_path = registry[name][split]

        if dataset_path and os.path.exists(dataset_path):
            os.remove(dataset_path)

        verl_path = dataset_path.replace(".parquet", "_verl.parquet")
        if os.path.exists(verl_path):
            os.remove(verl_path)

        del registry[name][split]

        if not registry[name]:
            del registry[name]
            dataset_dir = os.path.join(cls._DATASET_DIR, name)
            if os.path.exists(dataset_dir) and not os.listdir(dataset_dir):
                os.rmdir(dataset_dir)

        cls._save_registry(registry)

        logger.info("Removed dataset '%s' split '%s' from registry.", name, split)
        return True

    @classmethod
    def remove_dataset(cls, name: str) -> bool:
        registry = cls._load_registry()
        if name not in registry:
            logger.warning("Dataset '%s' not found in registry.", name)
            return False

        dataset_info = registry[name]
        for _, path in dataset_info.items():
            if path and os.path.exists(path):
                os.remove(path)
            verl_path = path.replace(".parquet", "_verl.parquet")
            if os.path.exists(verl_path):
                os.remove(verl_path)

        dataset_dir = os.path.join(cls._DATASET_DIR, name)
        if os.path.exists(dataset_dir) and not os.listdir(dataset_dir):
            os.rmdir(dataset_dir)

        del registry[name]
        cls._save_registry(registry)

        logger.info("Removed dataset '%s' from registry.", name)
        return True

    @classmethod
    def apply_verl_postprocessing(cls, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        processed_data = []
        for entry in data:
            processed_entry = {
                "prompt": [{"role": "user", "content": "placeholder"}],
                "reward_model": {"style": "rule", "ground_truth": None},
                "extra_info": entry,
            }
            processed_data.append(processed_entry)
        return processed_data
