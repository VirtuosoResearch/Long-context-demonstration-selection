from datasets import load_dataset

swebench = load_dataset('princeton-nlp/SWE-bench_oracle', split="test")


for idx, item in enumerate(swebench):
    with open(f"sample_{idx}_input.txt", "w") as f:
        f.write(f"==== Sample {idx} ====\n")
        f.write(item["text"] + "\n\n")
    with open(f"sample_{idx}_output.txt", "w") as f:
        f.write(f"==== Sample {idx} ====\n")
        f.write(item["patch"] + "\n\n")
    if idx >= 4:
        break