"""Utility to download and register SWE datasets for the mta package."""

from __future__ import annotations

from datasets import load_dataset

from mta.data import DatasetRegistry

SWE_DATASETS = [
    "R2E-Gym/R2E-Gym-Subset",
    "R2E-Gym/R2E-Gym-Lite",
    "R2E-Gym/R2E-Gym-V1",
    "R2E-Gym/SWE-Bench-Lite",
    "R2E-Gym/SWE-Bench-Verified",
    "r2e-edits/SweSmith-RL-Dataset",
]


def prepare_swe_data():
    def make_process_fn():
        def process_fn(row):
            return dict(row)

        return process_fn

    process_fn = make_process_fn()
    train_datasets = []
    test_datasets = []

    for dataset_name in SWE_DATASETS:
        print(f"Processing dataset: {dataset_name}")
        try:
            dataset_splits = load_dataset(dataset_name)
        except Exception as exc:
            print(f"Failed to load dataset {dataset_name}: {exc}")
            continue

        dataset_key = dataset_name.split("/")[-1].replace("-", "_")

        if "train" in dataset_splits:
            print(f"Processing 'train' split for {dataset_name}")
            train_data = [process_fn(row) for row in dataset_splits["train"]]
            train_dataset = DatasetRegistry.register_dataset(f"{dataset_key}", train_data, "train")
            train_datasets.append(train_dataset)
            print(f"Registered train dataset with {len(train_data)} examples")

        if "test" in dataset_splits:
            print(f"Processing 'test' split for {dataset_name}")
            test_data = [process_fn(row) for row in dataset_splits["test"]]
            test_dataset = DatasetRegistry.register_dataset(f"{dataset_key}", test_data, "test")
            test_datasets.append(test_dataset)
            print(f"Registered test dataset with {len(test_data)} examples")

        if "train" not in dataset_splits and "test" not in dataset_splits:
            available_splits = list(dataset_splits.keys())
            if available_splits:
                split_name = available_splits[0]
                print(f"Using '{split_name}' split as train data for {dataset_name}")
                train_data = [process_fn(row) for row in dataset_splits[split_name]]
                train_dataset = DatasetRegistry.register_dataset(f"{dataset_key}", train_data, "train")
                train_datasets.append(train_dataset)
                print(f"Registered train dataset with {len(train_data)} examples")

    return train_datasets, test_datasets


if __name__ == "__main__":
    train_datasets, test_datasets = prepare_swe_data()
    print("\nSummary:")
    print(f"Total train datasets: {len(train_datasets)}")
    print(f"Total test datasets: {len(test_datasets)}")

    if train_datasets:
        print("Sample train example from first dataset:")
        print(train_datasets[0].get_data()[0])

    if test_datasets:
        print("Sample test example from first dataset:")
        print(test_datasets[0].get_data()[0])
