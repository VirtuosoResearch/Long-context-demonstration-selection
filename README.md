# Long-Context Demonstration Selection Using State Space Models

* Authors: [Ziniu Zhang](https://ziniuzhang.github.io), [Zhenshuo Zhang](https://zhenshuozhang.github.io), [Ruoxuan Xiong](https://www.ruoxuanxiong.com/), [Gene Cooperman](https://www.ccs.neu.edu/home/gene/), and [Hongyang R. Zhang](https://www.hongyangzhang.com).

This repository contains code for reproducing the main experiments of **Long-Context Demonstration Selection Using State Space Models**. The method uses layer-grouped state space models (SSMs) to compress long demonstration prefixes, and evaluates demonstration-selection strategies for long-context inference.

The main experiments cover:

- SSMs for compressing long demonstration prefixes into compact virtual KV states while preserving full-context inference behavior.
- Demonstration-selection methods implemented through `test_multitask.py`, including Top-K, Random-K, BM25, KV-final, GradCE, GradICL, and BRIDGE.

## Usage

Here is the procedure to reproduce the main results. First, we introduce how to construct the environment and prepare data. Then, we give the two main bash entry points used for experiments: `scripts/run_mamba_inference_multi.sh` and `scripts/run_test_multitask.sh`.

### Environment reconstruction

The long-context experiments use the exported `metaicl` package set in `requirements.txt`. From the repository root, reconstruct the environment with:

```bash
conda create -n metaicl python=3.10 -y
conda activate metaicl
pip install -r requirements.txt
```

If you use gated Hugging Face models such as Llama, log in before running experiments:

```bash
huggingface-cli login
```

### Data preparation

Run all experiment scripts from `src/long-context`:

```bash
cd src/long-context
```

The data loader expects local task files in the following JSONL format:

```text
data/<dataset>/<dataset>_<split>.jsonl
```

Each row should contain `input`, `output`, and optionally `options`, `task`, and `dataset`. The default split names are `train`, `test`, and `dev`. The loader also accepts common aliases used in this repository, such as `gsm8k_judge`.

Feature-based selection methods need precomputed feature files under:

```text
features/<dataset>_<split>_features.json
```

The repository already contains data and feature directories for the main tasks used in the experiments, including SST-2, addition, coin flip, edge existence, MMLU, poem sentiment, and GSM8K judge.

### Experiment 1: SSM long-context compression

The main compression experiment is wrapped by `scripts/run_mamba_inference_multi.sh`. This script runs `mamba_inference_multi.py` for two-stage sidecar training and optional evaluation.

Run the default SST-2 experiment:

```bash
bash scripts/run_mamba_inference_multi.sh
```

Run on another dataset or model by overriding environment variables:

```bash
DATASET=addition K=50 MODEL_NAME=Qwen/Qwen2.5-3B-Instruct bash scripts/run_mamba_inference_multi.sh
```

Skip evaluation and only train checkpoints with:

```bash
RUN_EVAL=0 bash scripts/run_mamba_inference_multi.sh
```

Enable 4-bit quantization with:

```bash
USE_QUANT=1 bash scripts/run_mamba_inference_multi.sh
```

### Experiment 2: Demonstration selection

The demonstration-selection experiment is wrapped by `scripts/run_test_multitask.sh`. This script runs `test_multitask.py` with one selected method flag.

Run the default KV-final selection experiment:

```bash
bash scripts/run_test_multitask.sh
```

## Project structure

```text
MTL-SWE-agents/
├── README.md
├── requirements.txt
└── src/
    └── long-context/
        ├── mamba_inference_multi.py             # grouped SSM sidecar training/evaluation
        ├── test_multitask.py                    # demonstration selection and evaluation
        ├── scripts/
        │   ├── run_mamba_inference_multi.sh     # bash entry point for SSM compression experiments
        │   └── run_test_multitask.sh            # bash entry point for demonstration-selection experiments
        ├── get_feature_causal.py                # feature extraction used by selection methods
        ├── metaicl/                             # MetaICL data/model wrappers
        ├── utils/                               # data loading helpers
        ├── data/                                # local JSONL task data
        ├── features/                            # precomputed task features
        ├── checkpoints/                         # trained sidecar checkpoints
        └── out/                                 # predictions, losses, traces, and logs
```

## Main metrics

`scripts/run_mamba_inference_multi.sh` reports metrics from `mamba_inference_multi.py`:

- Full-KV accuracy and SSM accuracy.
- Accuracy gap between full-context inference and compressed SSM inference.
- Cross-entropy and per-sample MSE fidelity between full-KV and SSM predictions.
- Peak CUDA memory for model weights, full-KV inference, sidecar compression, and per-query SSM inference.
- FLOPs totals and reduction ratios when `FLOPS=1`.

`scripts/run_test_multitask.sh` reports metrics from `test_multitask.py`:

- Accuracy over the evaluation split.
- Correct and total example counts.
- Selected demonstrations and score traces for the chosen selection method.
- Optional FLOPs logs when `FLOPS=1`.

Outputs are written under the configured `OUT_DIR`, `CKPT_DIR`, and repository `features/` directories.

## Reference

If you find this repository useful or use it in a research paper, please cite our work with the following BibTeX entry.

```bibtex
@inproceedings{zhang2026longcontext,
  title={Long-Context Demonstration Selection Using State Space Models},
  author={Zhang, Ziniu and Zhang, Zhenshuo and Xiong, Ruoxuan and Cooperman, Gene and Zhang, Hongyang R.},
  booktitle={Findings of Empirical Methods in Natural Language Processing},
  year={2026}
}
```
