import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

from datasets import load_dataset


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
            "You are a meticulous C++ software engineer.\n"
            "Complete the requested C++ function so that it passes the provided tests.\n"
            "Return ONLY valid C++ code for the function body (and any necessary helper functions), "
            "without markdown fences or explanations."
        )
        user_inst = (
            "Write correct, efficient, standard C++17 code. Do not print debug info. "
            "Do not include a main() unless explicitly asked in the prompt."
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
            "The previous attempt failed to compile or failed tests. Read the error/trace and produce a corrected version.\n"
            "Return ONLY valid C++17 code for the function(s); no comments; no markdown fences."
        )
        context = (
            f"### PROMPT\n{task_prompt}\n\n"
            f"### PREVIOUS CODE\n{prev_code}\n\n"
            f"### COMPILER/RUNTIME OUTPUT\n{error_msg}\n\n"
            f"### FIXED CODE"
        )
        if self.is_chat:
            messages = [
                {"role": "system", "content": "You are a senior C++ engineer who fixes code using unit-test feedback."},
                {"role": "user", "content": repair_inst + "\n\n" + context},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return repair_inst + "\n\n" + context + "\n"

    @staticmethod
    def _strip_fences(s: str) -> str:
        # Remove ```cpp ... ``` / ```c++ ... ``` and generic ```
        s = re.sub(r"^```(?:cpp|c\+\+)?\s*", "", s.strip())
        s = re.sub(r"\s*```$", "", s.strip())
        return s.strip()

    def _tail_from_generated(self, inputs_ids, full_text: str) -> str:
        # Return only the newly generated tail beyond the prompt
        prompt_text = self.tokenizer.decode(inputs_ids["input_ids"][0], skip_special_tokens=True)
        return full_text[len(prompt_text):]

    def generate(self, prompt: str, max_new_tokens: int = 256, temp: float = 0.2, top_p: float = 0.95) -> str:
        import torch
        inputs = self._make_inputs(prompt)
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
        tail = self._tail_from_generated(input_ids, text)
        return self._strip_fences(tail)

    def repair(self, task_prompt: str, prev_code: str, error_msg: str, max_new_tokens: int = 256,
               temp: float = 0.2, top_p: float = 0.95) -> str:
        import torch
        inputs = self._make_repair_inputs(task_prompt, prev_code, error_msg)
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
        tail = self._tail_from_generated(input_ids, text)
        return self._strip_fences(tail)


def write_solution_file_cpp(task_prompt: str, completion: str, includes: str, test_setup: str, test: str, out_path: str):
    """
    Compose a single runnable C++ source file:
    [includes]
    [prompt + completion]
    [test_setup]
    [test]
    Assumptions:
    - 'includes' contains necessary #include / using directives (if any).
    - 'prompt' declares the function signature (e.g., `int foo(int x);` or a stub).
    - 'completion' provides its definition/implementation (and helpers).
    - 'test_setup' may define helpers, test harness utilities, etc.
    - 'test' should contain either a main() or assertions in a harness main() we provide.
    """
    parts = []
    if includes and includes.strip():
        parts.append(includes.strip())

    # Ensure tests can use standard lib; if dataset doesn't provide, we can default
    default_includes = "#include <bits/stdc++.h>\nusing namespace std;"
    if not re.search(r"#\s*include", "\n".join(parts), flags=re.IGNORECASE):
        parts.append(default_includes)

    parts.append(task_prompt.rstrip() + "\n" + completion.strip() + "\n")

    if test_setup and test_setup.strip():
        parts.append(test_setup.strip())

    parts.append(test.strip())

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts))


def compile_cpp(src_path: str, exe_path: str, timeout_sec: int = 15) -> Tuple[bool, str]:
    """
    Compile C++ with g++ -std=c++17. Return (ok, stderr_or_empty).
    """
    try:
        cp = subprocess.run(
            ["g++", "-std=c++17", "-O2", "-pipe", "-static-libstdc++", "-static-libgcc", src_path, "-o", exe_path],
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
        return False, f"Compile timeout after {timeout_sec}s\n{str(e)}"


def run_exe_with_timeout(exe_path: str, timeout_sec: int = 5) -> Tuple[bool, str]:
    try:
        cp = subprocess.run(
            [exe_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
            text=True,
        )
        if cp.returncode == 0:
            return True, (cp.stdout or "").strip()
        else:
            # include both stdout & stderr to give model richer feedback
            err = (cp.stderr or "") + "\n" + (cp.stdout or "")
            return False, err.strip()
    except subprocess.TimeoutExpired as e:
        return False, f"Runtime timeout after {timeout_sec}s\n{str(e)}"


def strip_non_code(text: str) -> str:
    # Remove accidental markdown fences or extraneous prose
    text = re.sub(r"^```(?:cpp|c\+\+)?\s*", "", text.strip())
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
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--engine", type=str, choices=["transformers", "vllm"], default="transformers")  # vllm not used here
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-iter", type=int, default=3, help="Max self-repair rounds per task (including first attempt).")
    ap.add_argument("--compile-timeout", type=int, default=20, help="Seconds for compilation.")
    ap.add_argument("--timeout", type=int, default=5, help="Seconds per run attempt.")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only evaluate first N tasks.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset-config", type=str, default="cpp", help="HumanEvalPack config name for C++.")
    args = ap.parse_args()

    print("Model:", args.model)
    ds = load_dataset("bigcode/humanevalpack", args.dataset_config, split="test")
    gen = TransformersGenerator(args.model)
    tmp_root = tempfile.mkdtemp(prefix="humanevalpack_cpp_eval_")
    results: List[TaskResult] = []

    total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    print(f"Running {total} HumanEvalPack({args.dataset_config}) tasks with {args.model} ({args.engine})")
    print(f"Max iters per task: {args.num_iter}, run-timeout: {args.timeout}s, compile-timeout: {args.compile-timeout if hasattr(args,'compile-timeout') else args.compile_timeout}s\n")

    for i, ex in enumerate(ds):
        if i >= total:
            break
        task_id = ex["task_id"]
        # Field names follow the Python split; for C++ we assume same keys:
        # "prompt": function declaration / stub, "import": headers/usings, "test_setup": harness helpers, "test": main/tests
        prompt = ex.get("prompt") or ex.get("declaration") or ""
        includes = ex.get("import", "") or ""     # e.g., "#include <vector>\nusing namespace std;"
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
            src_cpp = os.path.join(work_dir, "main.cpp")
            exe_path = os.path.join(work_dir, "a.out")

            write_solution_file_cpp(prompt, completion, includes, test_setup, test, src_cpp)
            print("source:", src_cpp)

            ok_compile, compile_msg = compile_cpp(src_cpp, exe_path, timeout_sec=args.compile_timeout)
            if not ok_compile:
                last_error = compile_msg[-2000:] if compile_msg else "Unknown compile error"
                print(f"  Compile failed (attempt {attempt})." + (" Retrying…" if attempt < args.num_iter else " Giving up."))
                if attempt < args.num_iter:
                    completion = gen.repair(prompt, completion, last_error,
                                            max_new_tokens=args.max_new_tokens, temp=args.temp, top_p=args.top_p)
                    completion = strip_non_code(completion)
                shutil.rmtree(work_dir, ignore_errors=True)
                continue

            # Run
            ok_run, run_msg = run_exe_with_timeout(exe_path, timeout_sec=args.timeout)
            if ok_run:
                passed = True
                print(f"  Passed on attempt {attempt}")
                shutil.rmtree(work_dir, ignore_errors=True)
                break
            else:
                last_error = run_msg[-2000:] if run_msg else ""
                print(f"  Failed attempt {attempt}." + (" Retrying…" if attempt < args.num_iter else " Giving up."))
                if attempt < args.num_iter:
                    completion = gen.repair(prompt, completion, last_error,
                                            max_new_tokens=args.max_new_tokens, temp=args.temp, top_p=args.top_p)
                    completion = strip_non_code(completion)
            shutil.rmtree(work_dir, ignore_errors=True)

        results.append(TaskResult(task_id=task_id, passed=passed, attempts=attempts, last_error=None if passed else last_error))

    passed_count = sum(r.passed for r in results)
    print("\n=== Summary ===")
    print(f"Passed: {passed_count}/{len(results)}  ({passed_count/len(results)*100:.2f}%)")

    out_json = os.path.join(tmp_root, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in results], f, ensure_ascii=False, indent=2)
    print(f"Per-task results saved to: {out_json}")


if __name__ == "__main__":
    main()