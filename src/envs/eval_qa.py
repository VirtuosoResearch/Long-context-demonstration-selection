import os
import sys

# Parse device argument FIRST, before any CUDA initialization
device_arg = "0"  # default
for i, arg in enumerate(sys.argv):
    if arg == "--device" and i + 1 < len(sys.argv):
        device_arg = sys.argv[i + 1]
        break

# Set CUDA_VISIBLE_DEVICES before importing any libraries that use CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = device_arg

import time
import json
import requests
import random
import logging
from datetime import datetime
from llm_utils.language_models import HF_LLM
from envs.wikienv import WikiEnv
from envs.wrappers import HotPotQAWrapper, LoggingWrapper

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


class WikiAgent:
    def __init__(self, env, llm):
        self.env = env
        self.llm = llm
        
        folder = './prompts/'
        prompt_file = 'prompts_naive.json'
        with open(folder + prompt_file, 'r') as f:
            prompt_dict = json.load(f)

        webthink_examples = prompt_dict['webthink_simple6']
        instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
        (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
        (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
        (3) Finish[answer], which returns the answer and finishes the task.
        Here are some examples.
        """
        self.prompt = instruction + webthink_examples

        
    def step(self, action):
        attempts = 0
        while attempts < 10:
            try:
                return self.env.step(action)
            except requests.exceptions.Timeout:
                attempts += 1

    def run_one_example(self, idx, to_print=True):
        question = self.env.reset(idx=idx)
        if to_print:
            logging.info(idx, question)
        prompt = self.prompt + question + "\n"
        n_calls, n_badcalls = 0, 0
        for i in range(1, 8):
            n_calls += 1
            thought_action = self.llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
            try:
                thought, action = thought_action.strip().split(f"\nAction {i}: ")
            except:
                print('ohh...', thought_action)
                n_badcalls += 1
                n_calls += 1
                thought = thought_action.strip().split('\n')[0]
                action = self.llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
            obs, r, done, info = self.step(env, action[0].lower() + action[1:])
            obs = obs.replace('\\n', '')
            step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            prompt += step_str
            if to_print:
                logging.info(step_str)
            if done:
                break
        if not done:
            obs, r, done, info = self.step(self.env, "finish[]")
        if to_print:
            logging.info(info, '\n')
        info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
        return r, info
    
    def evaluate(self, n=50):
        idxs = list(range(7405))
        random.Random(233).shuffle(idxs)

        rewards = []
        infos = []
        old_time = time.time()
        for i in idxs[:n]:
            reward, info = self.run_one_example(i, to_print=True)
            rewards.append(info['em'])
            infos.append(info)
            logging.info(f"Example Info: {info}")
            logging.info(f"Average Reward: {sum(rewards) / len(rewards)}")
            logging.info(f"Average Time per Example: {(time.time() - old_time) / len(rewards)}")
            logging.info('-----------')

        return rewards

def main(args):
    # Setup logging
    log_file = setup_logging(log_dir=args.log_dir, args=args)
    logging.info(f"Starting evaluation with model: {args.model_name}")
    logging.info(f"Using device: {args.device} (CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    logging.info(f"Number of episodes: {args.n_eval}")
    
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
    
    env = WikiEnv()
    env = HotPotQAWrapper(env, split="dev")
    env = LoggingWrapper(env)
    agent = WikiAgent(env, llm)
    rewards = agent.evaluate(n=args.n_eval)
    
    logging.info(f"Evaluation complete. Results saved to {log_file}") 
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Name of the language model to use.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate.")
    
    parser.add_argument("--n_eval", type=int, default=500, help="Number of evaluation episodes.")
    
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save log files.")
    
    parser.add_argument("--device", type=str, default="0", help="CUDA device ID to use (e.g., '0', '1', '2', or '0,1' for multiple GPUs).")
    
    args = parser.parse_args()
    
    main(args)