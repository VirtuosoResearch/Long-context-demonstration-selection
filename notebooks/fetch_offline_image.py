import subprocess as sp
from datasets import load_dataset

ARCH = "x86_64"
GHCR_PREFIX = "ghcr.io/epoch-research/swe-bench.eval"
LOCAL_PREFIX = "swe-bench/eval"

DATASET = "princeton-nlp/SWE-bench_Lite"
SPLIT = "test"

def sh(cmd):
    print("$", cmd)
    return sp.call(cmd, shell=True)

def main():
    ds = load_dataset(DATASET, split=SPLIT)
    ids = list(ds["instance_id"])
    print(f"Total instances: {len(ids)}")

    for iid in ids:
        ghcr = f"{GHCR_PREFIX}.{ARCH}.{iid}:latest"
        local = f"{LOCAL_PREFIX}/{ARCH}/{iid}:latest"

        if sh(f"docker pull {ghcr}") != 0:
            print(f"[warn] pull failed: {ghcr}")
            continue

        if sh(f"docker tag {ghcr} {local}") != 0:
            print(f"[warn] tag failed: {ghcr} -> {local}")

if __name__ == "__main__":
    main()
