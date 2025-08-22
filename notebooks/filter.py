from datasets import load_dataset, DatasetDict

RETRIEVAL_DATASET = "princeton-nlp/SWE-bench_oracle" 
LITE_DATASET = "princeton-nlp/SWE-bench_Lite"
LITE_SPLIT = "test" 

OUT_DIR = f"./datasets/oracle_lite_{LITE_SPLIT}"

def main():

    lite = load_dataset(LITE_DATASET, split=LITE_SPLIT)
    lite_ids = set(lite["instance_id"])

    retrieval = load_dataset(RETRIEVAL_DATASET, split="test")

    retrieval_lite = retrieval.filter(lambda x: x["instance_id"] in lite_ids)

    ds = DatasetDict({LITE_SPLIT: retrieval_lite})
    ds.save_to_disk(OUT_DIR)
    print(f"Saved to {OUT_DIR}. Instances: {len(retrieval_lite)}")

if __name__ == "__main__":
    main()