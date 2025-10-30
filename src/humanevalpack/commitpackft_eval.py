"""
Agent-style evaluation loop for bigcode/humanevalpack (Python).
- Loads tasks
- Generates solution with an open-source HF model
- Runs the official tests in a subprocess with a timeout
- If tests fail, feeds traceback back to the model for self-repair (N rounds)
- Reports pass rate
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Tuple

from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class Generator:
    def generate(self, prompt: str, max_new_tokens: int = 256, temp: float = 0.2, top_p: float = 0.95) -> str:
        raise NotImplementedError

class TransformersGenerator(Generator):
    def __init__(self, model_name: str, device: Optional[str] = None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.is_chat = getattr(self.tokenizer, "chat_template", None) not in (None, "", False)

    def _make_inputs(self, task_prompt: str) -> str:
        sys_inst = (
            "You are a meticulous software engineer. "
            "Complete the Python function below so that it passes the provided tests. "
            "Return ONLY valid Python code. Do not use markdown fences."
        )
        user_inst = (
            "Fill in the function by writing correct, efficient Python. "
            "Do not include explanations—only executable code that defines the function."
        )
        code = task_prompt
        if self.is_chat:
            messages = [
                {"role": "system", "content": sys_inst},
                {"role": "user", "content": f"{user_inst}\n\n{code}"},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            preface = sys_inst + "\n\n" + user_inst + "\n\n"
            return preface + code

    def _make_repair_inputs(self, task_prompt: str, prev_code: str, error_msg: str) -> str:
        repair_inst = (
            "The previous attempt failed tests. Read the traceback and produce a corrected version. "
            "Return ONLY valid Python code for the function(s); no comments; no markdown fences."
        )
        # \n\n### TRACEBACK\n{error_msg}
        # \n\n### PREVIOUS CODE\n{prev_code}
        context = f"### PROMPT\n{task_prompt}\n\n### TRACEBACK\n{error_msg}\n\n### FIXED CODE"
        if self.is_chat:
            messages = [
                {"role": "system", "content": "You are a senior Python engineer who fixes code using unit test feedback."},
                {"role": "user", "content": repair_inst + "\n\n" + context},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return repair_inst + "\n\n" + context + "\n"

    @staticmethod
    def _strip_fences(s: str) -> str:
        # Remove ```python ... ``` and ``` ... ``` if present
        s = re.sub(r"^```(?:python)?\s*", "", s.strip())
        s = re.sub(r"\s*```$", "", s.strip())
        return s.strip()

    def generate(self, prompt: str, max_new_tokens: int = 256, temp: float = 0.2, top_p: float = 0.95) -> str:
        inputs = self._make_inputs(prompt)
        import torch
        input_ids = self.tokenizer(inputs, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **input_ids,
                do_sample=True if temp > 0 else False,
                temperature=temp,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Return only the newly generated tail
        tail = text[len(self.tokenizer.decode(input_ids["input_ids"][0], skip_special_tokens=True)) :]
        return self._strip_fences(tail)

    def repair(self, task_prompt: str, prev_code: str, error_msg: str, max_new_tokens: int = 256,
               temp: float = 0.2, top_p: float = 0.95) -> str:
        inputs = self._make_repair_inputs(task_prompt, prev_code, error_msg)
        import torch
        input_ids = self.tokenizer(inputs, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **input_ids,
                do_sample=True if temp > 0 else False,
                temperature=temp,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        tail = text[len(self.tokenizer.decode(input_ids["input_ids"][0], skip_special_tokens=True)) :]
        return self._strip_fences(tail)

# -----------------------------
# Utilities
# -----------------------------
def write_solution_file(task_prompt: str, completion: str, imports: str, test_setup: str, test: str, out_path: str):
    """
    Compose a single runnable Python file:
    [imports]
    [prompt + completion]
    [test_setup]
    [test]
    """
    code = []
    if imports and imports.strip():
        code.append(imports.strip())
    # If the model reproduced the signature, joining prompt+completion is fine.
    # If it returned only a body, this still works because prompt ends with 'def ...:\n    ...'
    code.append(task_prompt.rstrip() + "\n" + completion.strip() + "\n")
    if test_setup and test_setup.strip():
        code.append(test_setup.strip())
    if test and test.strip():
        code.append(test.strip())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(code))
    # print(code)

def run_with_timeout(pyfile: str, timeout_sec: int = 5) -> Tuple[bool, str]:
    """
    Run `python pyfile` in a fresh subprocess. Returns (passed, stderr_or_empty).
    We redirect stdout; if any assertion fails or exception occurs, we capture it.
    """
    try:
        cp = subprocess.run(
            [sys.executable, pyfile],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
            text=True,
        )
        # HumanEval-style tests typically raise AssertionError when failing.
        if cp.returncode == 0:
            return True, ""
        else:
            err = (cp.stderr or "") + "\n" + (cp.stdout or "")
            return False, err.strip()
    except subprocess.TimeoutExpired as e:
        return False, f"Timeout after {timeout_sec}s\n{str(e)}"

def strip_non_code(text: str) -> str:
    # Remove accidental markdown fences or extraneous prose
    text = re.sub(r"^```(?:python)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()

@dataclass
class TaskResult:
    task_id: str
    passed: bool
    attempts: int
    last_error: Optional[str]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="google/codegemma-7b")
    ap.add_argument("--engine", type=str, choices=["transformers", "vllm"], default="transformers") # no vllm currently
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-iter", type=int, default=6, help="Max self-repair rounds per task (including first attempt).")
    ap.add_argument("--timeout", type=int, default=10, help="Seconds per attempt.")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only evaluate first N tasks.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print("Model: ",args.model)
    ds = load_dataset("bigcode/humanevalpack", "python", split="test")
    gen = TransformersGenerator(args.model)
    tmp_root = tempfile.mkdtemp(prefix="humanevalpack_eval_")
    results: List[TaskResult] = []

    total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    print(f"Running {total} HumanEvalPack(Python) tasks with {args.model} ({args.engine})")
    print(f"Max iters per task: {args.num_iter}, timeout: {args.timeout}s\n")

    for i, ex in enumerate(ds):
        if i >= total:
            break
        task_id = ex["task_id"]
        prompt = ex["prompt"] or ex["declaration"] or ""
        imports = ex.get("import", "") or ""
        test_setup = ex.get("test_setup", "") or ""
        test = ex.get("test", "") or ""

        print(f"[{i+1}/{total}] {task_id}")

        # First attempt
        completion = gen.generate(prompt, max_new_tokens=args.max_new_tokens, temp=args.temp, top_p=args.top_p)
        completion = strip_non_code(completion)

        passed = False
        attempts = 0
        last_error = ""

        for attempt in range(1, args.num_iter + 1):
            attempts = attempt
            work_dir = tempfile.mkdtemp(prefix=f"task_{i:03d}_", dir=tmp_root)
            main_py = os.path.join(work_dir, "main.py")
            write_solution_file(prompt, completion, imports, test_setup, test, main_py)
            print("main: ",main_py)

            ok, err = run_with_timeout(main_py, timeout_sec=args.timeout)
            if ok:
                passed = True
                print(f"Passed on attempt {attempt}")
                shutil.rmtree(work_dir, ignore_errors=True)
                break
            else:
                last_error = err[-2000:] if err else ""
                print(f" Failed attempt {attempt}. Retrying…" if attempt < args.num_iter else " Failed. Giving up.")
                # Self-repair for next round
                if attempt < args.num_iter:
                    completion = gen.repair(prompt, completion, last_error,
                                            max_new_tokens=args.max_new_tokens, temp=args.temp, top_p=args.top_p)
                    completion = strip_non_code(completion)

            shutil.rmtree(work_dir, ignore_errors=True)

        results.append(TaskResult(task_id=task_id, passed=passed, attempts=attempts, last_error=None if passed else last_error))
        print(results)

    # Summary
    passed_count = sum(r.passed for r in results)
    print("\n=== Summary ===")
    print(f"Passed: {passed_count}/{len(results)}  ({passed_count/len(results)*100:.2f}%)")
    # Save raw results
    out_json = os.path.join(tmp_root, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in results], f, ensure_ascii=False, indent=2)
    print(f"Per-task results saved to: {out_json}")


if __name__ == "__main__":
    main()
