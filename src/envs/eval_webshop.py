import os
import logging
from datetime import datetime
from webshop_env import WebshopEnv
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

class WebshopAgent:   
    def __init__(self, env, llm):
        self.env = env
        self.llm = llm

    def run_one_example(self, idx, prompt, to_print=True):
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
                init_prompt + prompt[-(6400-len(init_prompt)):]).lstrip(' ')
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
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    # Setup logging
    log_file = setup_logging(log_dir=args.log_dir, args=args)
    logging.info(f"Starting evaluation with model: {args.model_name}")
    logging.info(f"Using device: {args.device}")
    logging.info(f"Number of episodes: {args.n_eval}")
    
    llm = HF_LLM(
        model_name=args.model_name,
        max_new_tokens=160,
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
    rewards = agent.evaluate(prompt1, n=args.n_eval)
    
    logging.info(f"Evaluation complete. Results saved to {log_file}") 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Name of the language model to use.")
    
    parser.add_argument("--n_eval", type=int, default=500, help="Number of evaluation episodes.")
    
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save log files.")
    
    parser.add_argument("--device", type=str, default="0", help="CUDA device ID to use (e.g., '0', '1', '2', or '0,1' for multiple GPUs).")
    
    args = parser.parse_args()
    
    main(args)
    