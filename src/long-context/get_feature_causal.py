import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class ICLDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {"input": item["input"], "label": item.get("output", None)}


def extract_features_lasttoken(model, tokenizer, dataloader, device, output_file, split_name, max_length):
    features = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"extract-{split_name}"):
            input_texts = []
            for i in range(len(batch["input"])):
                text = batch["input"][i]
                # Keep old behavior: for test split, append gold output.
                if split_name == "test" and batch["label"][i] is not None:
                    text = text + batch["label"][i]
                input_texts.append(text)

            inputs = tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {key: val.to(device) for key, val in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)
            if outputs.hidden_states is None:
                raise ValueError("Model did not return hidden states. Ensure `output_hidden_states=True`.")

            hidden_states = outputs.hidden_states[-1]
            attention_mask = inputs["attention_mask"]

            for i in range(hidden_states.size(0)):
                last_non_pad_idx = attention_mask[i].nonzero(as_tuple=True)[0].max().item()
                last_hidden_state = hidden_states[i, last_non_pad_idx, :]
                features.append(last_hidden_state.float().cpu().numpy().tolist())

    with open(output_file, "w") as f:
        json.dump(features, f)

    print(f"saved: {output_file} ({len(features)} rows)")


def _discover_tasks(data_root):
    tasks = []
    for task_dir in sorted(Path(data_root).iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        test_path = task_dir / f"{task}_test.jsonl"
        dev_path = task_dir / f"{task}_dev.jsonl"
        if test_path.exists() and dev_path.exists():
            tasks.append(task)
    return tasks


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    os.makedirs(args.feature_dir, exist_ok=True)

    if args.task is not None:
        tasks = [args.task]
    elif args.tasks is not None:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = _discover_tasks(args.data_dir)

    if not tasks:
        raise ValueError("No tasks found to process.")

    for task in tasks:
        test_file_path = os.path.join(args.data_dir, task, f"{task}_test.jsonl")
        dev_file_path = os.path.join(args.data_dir, task, f"{task}_dev.jsonl")
        if not os.path.exists(test_file_path) or not os.path.exists(dev_file_path):
            print(f"skip {task}: missing dev/test jsonl")
            continue

        print(f"\n=== task: {task} ===")
        test_output_file = os.path.join(args.feature_dir, f"{task}_test_features.json")
        dev_output_file = os.path.join(args.feature_dir, f"{task}_dev_features.json")

        test_dataset = ICLDataset(test_file_path)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        extract_features_lasttoken(
            model,
            tokenizer,
            test_dataloader,
            device,
            test_output_file,
            "test",
            args.max_length,
        )

        dev_dataset = ICLDataset(dev_file_path)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
        extract_features_lasttoken(
            model,
            tokenizer,
            dev_dataloader,
            device,
            dev_output_file,
            "dev",
            args.max_length,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None, help="single task name, e.g. cr")
    parser.add_argument("--tasks", type=str, default=None, help="comma-separated task names")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--feature_dir", type=str, default="./features")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    main(args)