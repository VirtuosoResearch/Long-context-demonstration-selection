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
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from datasets import load_dataset


def seed_everything(seed: int):
    import torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Generator:
    def generate(self, prompt: str, max_new_tokens: int = 256, temp: float = 0.2, top_p: float = 0.95) -> str:
        raise NotImplementedError
    def repair(self, task_prompt: str, prev_code: str, error_msg: str, max_new_tokens: int = 256,
               temp: float = 0.2, top_p: float = 0.95) -> str:
        raise NotImplementedError

class TransformersGenerator(Generator):
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=getattr(torch, "bfloat16", None)
        )
        # ensure a pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.is_chat = hasattr(self.tokenizer, "apply_chat_template")

    def _make_inputs(self, task_prompt: str, entry_point: Optional[str]) -> str:
        sys_inst = (
            "You are a meticulous software engineer. "
            "Complete ONLY the missing Python function implementation so that it passes the unit tests. "
            "Return ONLY valid Python code that defines the function; no prose, no comments, no imports, no tests."
        )
        if entry_point:
            user_inst = (
                f"Implement the body of the function `{entry_point}` correctly and efficiently. "
                "Do not change its signature. Return only the function code."
            )
        else:
            user_inst = (
                "Fill in the function by writing correct, efficient Python. "
                "Do not include explanations—only executable code that defines the function."
            )
        if self.is_chat:
            messages = [
                {"role": "system", "content": sys_inst},
                {"role": "user", "content": f"{user_inst}\n\n{task_prompt}"},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return sys_inst + "\n\n" + user_inst + "\n\n" + task_prompt

    def _make_repair_inputs(self, task_prompt: str, prev_code: str, error_msg: str, entry_point: Optional[str]) -> str:
        repair_inst = (
            "The previous attempt failed tests. Read the traceback and produce a corrected version. "
            "Return ONLY valid Python code for the function(s); no comments, no imports, no tests."
        )
        if entry_point:
            repair_inst += f" Do not change the signature of `{entry_point}`."
        # keep traceback tail (already truncated by caller)
        context = f"### PROMPT\n{task_prompt}\n\n### PREVIOUS CODE\n{prev_code}\n\n### TRACEBACK\n{error_msg}\n\n### FIXED CODE"
        if self.is_chat:
            messages = [
                {"role": "system", "content": "You are a senior Python engineer who fixes code using unit test feedback."},
                {"role": "user", "content": repair_inst + "\n\n" + context},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return repair_inst + "\n\n" + context + "\n"

    @staticmethod
    def _strip_fences_and_prose(s: str) -> str:
        s = s.strip()
        s = re.sub(r"^```(?:python)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        # drop obvious prose lines
        lines = [ln for ln in s.splitlines() if not ln.strip().startswith(("#", "//"))]
        return "\n".join(lines).strip()

    @staticmethod
    def _only_function_block(text: str, entry_point: Optional[str]) -> str:
        """
        Try to keep only the target function definition(s).
        - If entry_point is known, extract the block starting at 'def entry_point(' up to the next 'def ' or EOF.
        - Otherwise, return the whole text (already cleaned).
        """
        if not entry_point:
            return text.strip()
        pat = re.compile(rf"(?ms)^\s*def\s+{re.escape(entry_point)}\s*\(.*?\):\s*(?:\n(?:[ \t].*|\s*)*)")
        m = pat.search(text)
        if m:
            return m.group(0).rstrip()
        # fallback: if model wrote multiple defs, try to grab the first def with that name (looser)
        pat2 = re.compile(rf"(?ms)^\s*def\s+{re.escape(entry_point)}\s*\(.*")
        m2 = pat2.search(text)
        if m2:
            # cut until next top-level def/class or EOF
            start = m2.start()
            after = text[m2.end():]
            m_next = re.search(r"(?m)^\s*(def|class)\s+\w+\s*\(", after)
            end = m2.end() + (m_next.start() if m_next else len(after))
            return text[start:end].rstrip()
        return text.strip()

    def _generate_raw(self, inputs: str, max_new_tokens: int, temp: float, top_p: float) -> str:
        import torch
        toks = self.tokenizer(inputs, return_tensors="pt").to(self.model.device)
        in_len = toks["input_ids"].shape[-1]
        g = torch.Generator(device=self.model.device)
        # Allow external seeding via torch.manual_seed; g uses default seed chain.
        out_ids = self.model.generate(
            **toks,
            do_sample=True if temp > 0 else False,
            temperature=temp,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_ids = out_ids[0, in_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def generate(self, prompt: str, entry_point: Optional[str] = None,
                 max_new_tokens: int = 256, temp: float = 0.2, top_p: float = 0.95) -> str:
        inputs = self._make_inputs(prompt, entry_point)
        text = self._generate_raw(inputs, max_new_tokens, temp, top_p)
        text = self._strip_fences_and_prose(text)
        text = self._only_function_block(text, entry_point)
        return text

    def repair(self, task_prompt: str, prev_code: str, error_msg: str, entry_point: Optional[str] = None,
               max_new_tokens: int = 256, temp: float = 0.2, top_p: float = 0.95) -> str:
        inputs = self._make_repair_inputs(task_prompt, prev_code, error_msg, entry_point)
        text = self._generate_raw(inputs, max_new_tokens, temp, top_p)
        text = self._strip_fences_and_prose(text)
        text = self._only_function_block(text, entry_point)
        return text


def sanitize_model_output(s: str) -> str:
    """Remove accidental imports/tests; keep defs only."""
    s = s.strip()
    # drop common accidental sections
    kill_blocks = [r"(?ms)^if\s+__name__\s*==\s*['\"]__main__['\"]:.*?$",
                   r"(?ms)^\s*import\s+.*?$",
                   r"(?ms)^\s*from\s+\S+\s+import\s+.*?$"]
    for kb in kill_blocks:
        s = re.sub(kb, "", s)
    # keep only function/class defs if present
    defs = re.findall(r"(?ms)^\s*(def|class)\s+\w+\s*\(.*?\):\s*(?:\n(?:[ \t].*|\s*)*)", s)
    if defs:
        s = "\n\n".join(defs).strip()
    return s.strip()

def write_solution_file(task_prompt: str, completion: str, imports: str, test_setup: str, test: str, out_path: str):
    """
    Compose a single runnable Python file:
    [imports]
    [prompt + completion]
    [test_setup]
    [test]
    """
    code_parts = []
    if imports and imports.strip():
        code_parts.append(imports.strip())
    body = task_prompt.rstrip() + "\n" + completion.strip() + "\n"
    code_parts.append(body)
    if test_setup and test_setup.strip():
        code_parts.append(test_setup.strip())
    if test and test.strip():
        code_parts.append(test.strip())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(code_parts))

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
        if cp.returncode == 0:
            return True, ""
        else:
            err = (cp.stderr or "") + "\n" + (cp.stdout or "")
            return False, err.strip()
    except subprocess.TimeoutExpired as e:
        return False, f"Timeout after {timeout_sec}s\n{str(e)}"

def strip_non_code(text: str) -> str:
    text = re.sub(r"^```(?:python)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()

def tail_traceback(msg: str, max_lines: int = 120) -> str:
    lines = (msg or "").splitlines()
    return "\n".join(lines[-max_lines:]).strip()

@dataclass
class TaskResult:
    task_id: str
    passed: bool
    attempts: int
    last_error: Optional[str]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-iter", type=int, default=3, help="Max self-repair rounds per task (including first attempt).")
    ap.add_argument("--timeout", type=int, default=8, help="Seconds per attempt.")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only evaluate first N tasks.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-artifacts", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)

    ds = load_dataset("bigcode/humanevalpack", "python", split="test")
    gen = TransformersGenerator(args.model)
    tmp_root = tempfile.mkdtemp(prefix="humanevalpack_eval_")
    results: List[TaskResult] = []

    try:
        total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
        print(f"Running {total} HumanEvalPack(Python) tasks with {args.model} (transformers)")
        print(f"Max iters per task: {args.num_iter}, timeout: {args.timeout}s, seed: {args.seed}\n")

        for i, ex in enumerate(ds):
            if i >= total:
                break
            task_id = ex["task_id"]
            # Prefer structured prompt fields if present
            prompt = (ex.get("prompt") or ex.get("declaration") or "").rstrip()
            entry_point = ex.get("entry_point", None)
            imports = ex.get("import", "") or ""
            test_setup = ex.get("test_setup", "") or ""
            test = ex.get("test", "") or ""

            print(f"[{i+1}/{total}] {task_id}")

            # First attempt
            completion = gen.generate(prompt, entry_point=entry_point,
                                      max_new_tokens=args.max_new_tokens, temp=args.temp, top_p=args.top_p)
            completion = sanitize_model_output(strip_non_code(completion))

            passed = False
            attempts = 0
            last_error = ""

            for attempt in range(1, args.num_iter + 1):
                attempts = attempt
                work_dir = tempfile.mkdtemp(prefix=f"task_{i:03d}_", dir=tmp_root)
                main_py = os.path.join(work_dir, "main.py")
                write_solution_file(prompt, completion, imports, test_setup, test, main_py)

                ok, err = run_with_timeout(main_py, timeout_sec=args.timeout)
                if ok:
                    passed = True
                    print(f" Passed on attempt {attempt}")
                    if not args.keep_artifacts:
                        shutil.rmtree(work_dir, ignore_errors=True)
                    break
                else:
                    last_error = tail_traceback(err, max_lines=160)
                    print(f" Failed attempt {attempt}.", "Retrying…" if attempt < args.num_iter else "Giving up.")
                    # Dump failing artifact for inspection
                    with open(os.path.join(work_dir, "traceback.txt"), "w", encoding="utf-8") as f:
                        f.write(last_error)
                    if attempt < args.num_iter:
                        completion = gen.repair(
                            prompt, completion, last_error, entry_point=entry_point,
                            max_new_tokens=args.max_new_tokens, temp=args.temp, top_p=args.top_p
                        )
                        completion = sanitize_model_output(strip_non_code(completion))
                    if not args.keep_artifacts and attempt < args.num_iter:
                        shutil.rmtree(work_dir, ignore_errors=True)

            results.append(TaskResult(task_id=task_id, passed=passed, attempts=attempts,
                                      last_error=None if passed else last_error))

        passed_count = sum(r.passed for r in results)
        print("\n=== Summary ===")
        print(f"Passed: {passed_count}/{len(results)}  ({passed_count/len(results)*100:.2f}%)")

        # Save raw results
        out_json = os.path.join(tmp_root, "results.jsonl")
        with open(out_json, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
        print(f"Per-task results saved to: {out_json}")
        if not args.keep_artifacts:
            print(f"(Temporary per-task dirs cleaned; pass --keep-artifacts to keep them.)")
        else:
            print(f"(Artifacts kept under: {tmp_root})")
    finally:
        # Keep tmp_root for artifacts (results.jsonl). Caller controls per-task dirs with --keep-artifacts.
        pass

if __name__ == "__main__":
    main()
