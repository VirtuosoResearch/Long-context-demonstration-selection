import os
import sys
import json
import glob
import random

# Parse device argument FIRST, before any CUDA initialization
device_arg = "0"  # default
for i, arg in enumerate(sys.argv):
    if arg == "--device" and i + 1 < len(sys.argv):
        device_arg = sys.argv[i + 1]
        break

# Set CUDA_VISIBLE_DEVICES before importing any libraries that use CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = device_arg

import logging
from datetime import datetime
from envs.webshop_env import WebshopEnv
from llm_utils.prompts_webshop import prompt1, prompt1_actonly
from llm_utils.language_models import HF_LLM

# Setup logging
def setup_logging(log_dir="logs", args=None):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_str = args.model_name.replace("/", "_") if args else "default_model"
    log_file = os.path.join(log_dir, f"webshop_eval_model_{model_name_str}_{timestamp}.log")

    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    logging.info(f"Logging to {log_file}")
    return log_file


def _load_dialog_trajectory(path):
    turns = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            turns.append(
                {
                    "generated_action": obj.get("generated_action", "").strip(),
                    "action": obj.get("action", "").strip(),
                    "observation": obj.get("observation", "").strip(),
                }
            )
    return turns


def _format_demo_turns(turns):
    chunks = []
    for turn in turns:
        obs = turn["observation"]
        gen_act = turn["generated_action"]
        act = turn["action"]
        chunks.append(
            "\n".join(
                [
                    f"Observation: {obs}",
                    f"Generated action: {gen_act}",
                    f"Action: {act}",
                ]
            )
        )
    return "\n\n".join(chunks)


def build_few_shot_prompt(k, trajectories_dir, seed):
    # Keep existing behavior for 1-shot.
    if k <= 1:
        return prompt1

    all_files = sorted(glob.glob(os.path.join(trajectories_dir, "*.jsonl")))
    if not all_files:
        logging.warning(
            f"No trajectory files found under {trajectories_dir}. Falling back to prompt1."
        )
        return prompt1

    # k includes prompt1; we add k-1 extra demonstrations from trajectory files.
    n_extra = min(k - 1, len(all_files))
    random.seed(seed)
    selected_files = random.sample(all_files, n_extra)

    demo_blocks = []
    for i, path in enumerate(selected_files, start=1):
        turns = _load_dialog_trajectory(path)
        if not turns:
            continue
        demo_text = _format_demo_turns(turns)
        demo_blocks.append(f"Demo {i}:\n{demo_text}")

    if not demo_blocks:
        return prompt1

    few_shot_prefix = "\n\n".join(demo_blocks) + "\n\n"
    logging.info(
        f"Using few-shot prompt: k={k} (prompt1 + {len(demo_blocks)} sampled trajectories)"
    )
    return few_shot_prefix + prompt1

class WebshopAgent:   
    def __init__(self, env, llm):
        self.env = env
        self.llm = llm

    def run_one_example(self, idx, prompt, to_print=True):
        logging.info(f'Initial prompt: {prompt}')
        action = 'reset'
        init_prompt = prompt
        prompt = ''
        for i in range(15):
            try:
                res = self.env.step(idx, action)
                observation = res[0]
            except AssertionError:
                observation = 'Invalid action!'

            if action.startswith('think'):
                observation = 'OK.'

            if to_print:
                logging.info(f'Action: {action}\nObservation: {observation}\n')
                logging.info('--------------------------------')
            #     sys.stdout.flush()
            if i:
                prompt += f' {action}\nObservation: {observation}\n\nAction:'
            else:
                prompt += f'{observation}\n\nAction:'

            if res[2]:
                return res[1]

            action = self.llm(
                init_prompt + prompt[-(12800-len(init_prompt)):]).lstrip(' ')
            logging.info("===============================")
            logging.info(f'Generated action: {action}')
            logging.info("===============================")

        return 0


    def evaluate(self, prompt, n=50):
        rewards = []
        counts = 0
        for i in range(n):
            logging.info(f'----------------- Episode {i} -----------------')
            try:
                r = self.run_one_example(f'fixed_{i}', prompt, to_print=True)
            except AssertionError:
                r = 0
                counts += 1
            rewards.append(r)
            if (i+1) % 1 == 0:
                r, sr, fr = sum(
                    rewards) / len(rewards), len([_ for _ in rewards if _ == 1]) / len(rewards), counts / len(rewards)
                logging.info(f'{i+1} Avg Reward: {r}, Success Rate: {sr}, Failure Rate: {fr}')
                logging.info('-------------')
        r, sr, fr = sum(rewards) / len(rewards), len([_ for _ in rewards if _ == 1]) / n, counts / n
        logging.info(f"Average Reward: {r}")
        logging.info(f"Success Rate: {sr}")
        logging.info(f"Failure Rate: {fr}")
        return rewards


def main(args):
    # Device is already set at module import time
    # (see top of file where CUDA_VISIBLE_DEVICES is set)
    
    # Setup logging
    log_file = setup_logging(log_dir=args.log_dir, args=args)
    logging.info(f"Starting evaluation with model: {args.model_name}")
    logging.info(f"Using device: {args.device} (CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    logging.info(f"Number of episodes: {args.n_eval}")
    logging.info(f"Few-shot k: {args.k}")
    
    llm = HF_LLM(
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        generation_kwargs={
            "temperature": 0.3,
            "top_p": 0.9,
            "do_sample": True,
            "stop": "\n",
            "format_guard": ""
        }
    )

    env = WebshopEnv()
    agent = WebshopAgent(env, llm)
    eval_prompt = build_few_shot_prompt(
        k=args.k,
        trajectories_dir=args.trajectories_dir,
        seed=args.few_shot_seed,
    )
    rewards = agent.evaluate(eval_prompt, n=args.n_eval)
    
    logging.info(f"Evaluation complete. Results saved to {log_file}") 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Name of the language model to use.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate.")
    
    parser.add_argument("--n_eval", type=int, default=500, help="Number of evaluation episodes.")
    
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save log files.")
    
    parser.add_argument("--device", type=str, default="0", help="CUDA device ID to use (e.g., '0', '1', '2', or '0,1' for multiple GPUs).")
    parser.add_argument("--k", type=int, default=1, help="Number of few-shot demonstrations. k=1 uses only prompt1; k>1 adds k-1 sampled trajectories.")
    parser.add_argument(
        "--trajectories_dir",
        type=str,
        default="cmd_results/scored_trajectories",
        help="Directory containing extracted trajectory jsonl files for few-shot prompting.",
    )
    parser.add_argument(
        "--few_shot_seed",
        type=int,
        default=42,
        help="Random seed for sampling few-shot trajectories.",
    )
    
    args = parser.parse_args()
    
    main(args)
    