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
            "You are a meticulous Java software engineer.\n"
            "Complete the requested Java method/class so that it passes the provided tests.\n"
            "Return ONLY valid Java code (no markdown fences, no explanations)."
        )
        user_inst = (
            "Write correct, efficient Java 8+ (no external libs). "
            "If the prompt declares a class, implement inside it; otherwise provide method implementation (and helpers) only. "
            "Do NOT print debug logs."
        )
        if self.is_chat:
            messages = [
                {"role": "system", "content": sys_inst},
                {"role": "user", "content": f"{user_inst}\n\n{task_prompt}"},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return sys_inst + "\n\n" + user_inst + "\n\n" + task_prompt

    def _make_repair_inputs(self, task_prompt: str, prev_code: str, error_msg: str) -> str:
        repair_inst = (
            "The previous attempt failed to compile or failed tests. Read the error/trace and output a corrected version.\n"
            "Return ONLY valid Java code; no comments; no markdown fences."
        )
        context = (
            f"### PROMPT\n{task_prompt}\n\n"
            f"### PREVIOUS CODE\n{prev_code}\n\n"
            f"### COMPILER/RUNTIME OUTPUT\n{error_msg}\n\n"
            f"### FIXED CODE"
        )
        if self.is_chat:
            messages = [
                {"role": "system", "content": "You are a senior Java engineer who fixes code using unit-test feedback."},
                {"role": "user", "content": repair_inst + "\n\n" + context},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return repair_inst + "\n\n" + context + "\n"

    @staticmethod
    def _strip_fences(s: str) -> str:
        s = re.sub(r"^```(?:java)?\s*", "", s.strip())
        s = re.sub(r"\s*```$", "", s.strip())
        return s.strip()

    def _tail_from_generated(self, inputs_ids, full_text: str) -> str:
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


# -----------------------------
# Java utilities
# -----------------------------


def choose_public_class_and_filename(includes: str, prompt: str, completion: str, test_setup: str, test: str):
    """
    Strategy:
      1) If any snippet declares `public class X`, use X.java and main class X.
      2) Else default to public class Main in Main.java.
    """

    PUB_CLASS_RE = re.compile(r"\bpublic\s+class\s+([A-Za-z_]\w*)")
    blob = "\n".join([includes or "", prompt or "", completion or "", test_setup or "", test or ""])
    m = PUB_CLASS_RE.search(blob)
    if m:
        cls = m.group(1)
        return cls, f"{cls}.java"
    else:
        return "Main", "Main.java"

def ensure_wrapped_classes_for_default(prompt: str, completion: str, test_setup: str, test: str) -> str:

    parts = []
    ANY_CLASS_RE = re.compile(r"\bclass\s+([A-Za-z_]\w*)")
    has_class_in_prompt = ANY_CLASS_RE.search(prompt or "") is not None
    code_impl = (prompt or "").rstrip() + "\n" + (completion or "").strip()
    if has_class_in_prompt:
        code_impl = re.sub(r"\bpublic\s+class\b", "class", code_impl)
        parts.append(code_impl.strip())
    else:
        body = code_impl.strip()
        if not body:
            body = ""
        wrapped = "class Solution {\n" + indent_block(body) + "\n}"
        parts.append(wrapped)

    if test_setup and test_setup.strip():
        ts = re.sub(r"\bpublic\s+class\b", "class", test_setup.strip())
        parts.append(ts)

    if test and test.strip():
        t = test.strip()
        if "class " in t:
            t = re.sub(r"\bpublic\s+class\b", "class", t)
            parts.append(t)
        else:
            main_wrapped = (
                "public class Main {\n"
                "    public static void main(String[] args) throws Exception {\n"
                + indent_block(t, 2) + "\n"
                "    }\n"
                "}"
            )
            parts.append(main_wrapped)
    else:
        parts.append("public class Main { public static void main(String[] args) {} }")

    return "\n\n".join(parts)

def indent_block(src: str, level: int = 1, width: int = 4) -> str:
    pad = " " * (level * width)
    return "\n".join(pad + line if line.strip() != "" else "" for line in src.splitlines())

def write_solution_file_java(includes: str, prompt: str, completion: str, test_setup: str, test: str, out_dir: str) -> Tuple[str, str]:
    """
    Returns (public_class_name, java_filename)
    - Decides the public class name & file name
    - Writes a single .java file under out_dir
    """
    public_class, filename = choose_public_class_and_filename(includes, prompt, completion, test_setup, test)

    PACKAGE_RE = re.compile(r"^\s*package\s+([A-Za-z_][\w\.]*);", re.MULTILINE)
    # If a package statement exists anywhere, drop it (we compile in a flat temp dir)
    def strip_package(s: str) -> str:
        return PACKAGE_RE.sub("", s or "")

    includes = strip_package(includes or "")
    prompt   = strip_package(prompt or "")
    completion = strip_package(completion or "")
    test_setup = strip_package(test_setup or "")
    test = strip_package(test or "")

    # If a specific public class was found (e.g., from test), we respect it and just concatenate pieces.
    # Otherwise, we fabricate a public Main and neutralize other 'public class' to 'class'.
    if public_class == "Main":
        # Build a single file with only one public class: Main
        src_body = []
        # includes first
        if includes.strip():
            src_body.append(includes.strip())
        src_body.append(ensure_wrapped_classes_for_default(prompt, completion, test_setup, test))
        source = "\n\n".join(src_body)
    else:
        # Use the found public class and file name:
        # Neutralize any other 'public class' occurrences to avoid multiple-public-class error
        concat = []
        if includes.strip():
            concat.append(includes.strip())
        # Keep prompt+completion as-is
        concat.append((prompt or "").rstrip() + "\n" + (completion or "").strip())
        if test_setup.strip():
            concat.append(test_setup.strip())
        if test.strip():
            concat.append(test.strip())

        source = "\n\n".join(concat)
        # If there are multiple 'public class' and file name won't match, try to demote all but the chosen one
        source = demote_other_public_classes(source, chosen=public_class)

    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(source)
    return public_class, out_path

def demote_other_public_classes(src: str, chosen: str) -> str:
    """
    Replace `public class X` with `class X` for all X != chosen.
    """
    def repl(m):
        name = m.group(1)
        if name == chosen:
            return f"public class {name}"
        else:
            return f"class {name}"
    return re.sub(r"\bpublic\s+class\s+([A-Za-z_]\w*)", repl, src)

def compile_java(java_file: str, timeout_sec: int = 20) -> Tuple[bool, str]:
    try:
        cp = subprocess.run(
            ["javac", "-J-Dfile.encoding=UTF-8", "-encoding", "UTF-8", os.path.basename(java_file)],
            cwd=os.path.dirname(java_file),
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

def run_java(main_class: str, work_dir: str, timeout_sec: int = 5) -> Tuple[bool, str]:
    try:
        cp = subprocess.run(
            ["java", "-Dfile.encoding=UTF-8", main_class],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
            text=True,
        )
        if cp.returncode == 0:
            return True, (cp.stdout or "").strip()
        else:
            err = (cp.stderr or "") + "\n" + (cp.stdout or "")
            return False, err.strip()
    except subprocess.TimeoutExpired as e:
        return False, f"Runtime timeout after {timeout_sec}s\n{str(e)}"

def strip_non_code(text: str) -> str:
    text = re.sub(r"^```(?:java)?\s*", "", text.strip())
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
    ap.add_argument("--engine", type=str, choices=["transformers", "vllm"], default="transformers")  # vllm not used
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-iter", type=int, default=3, help="Max self-repair rounds per task (including first attempt).")
    ap.add_argument("--compile-timeout", type=int, default=20, help="Seconds for javac.")
    ap.add_argument("--timeout", type=int, default=5, help="Seconds per run attempt.")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only evaluate first N tasks.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset-config", type=str, default="java", help="HumanEvalPack config name for Java.")
    args = ap.parse_args()

    print("Model:", args.model)
    ds = load_dataset("bigcode/humanevalpack", args.dataset_config, split="test")
    gen = TransformersGenerator(args.model)
    tmp_root = tempfile.mkdtemp(prefix="humanevalpack_java_eval_")
    results: List[TaskResult] = []

    total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    print(f"Running {total} HumanEvalPack({args.dataset_config}) tasks with {args.model} ({args.engine})")
    print(f"Max iters per task: {args.num_iter}, run-timeout: {args.timeout}s, compile-timeout: {args.compile_timeout}s\n")

    for i, ex in enumerate(ds):
        if i >= total:
            break
        task_id = ex["task_id"]
        prompt = ex.get("prompt") or ex.get("declaration") or ""
        includes = ex.get("import", "") or ""       # e.g., imports; no package
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

            # Write Main.java or X.java
            main_class, java_path = write_solution_file_java(includes, prompt, completion, test_setup, test, work_dir)
            print("source:", java_path, " main_class:", main_class)

            # Compile
            okc, cerr = compile_java(java_path, timeout_sec=args.compile_timeout)
            if not okc:
                last_error = cerr[-2000:] if cerr else "Unknown compile error"
                print(f"  Compile failed (attempt {attempt})." + (" Retrying…" if attempt < args.num_iter else " Giving up."))
                if attempt < args.num_iter:
                    completion = gen.repair(prompt, completion, last_error,
                                            max_new_tokens=args.max_new_tokens, temp=args.temp, top_p=args.top_p)
                    completion = strip_non_code(completion)
                shutil.rmtree(work_dir, ignore_errors=True)
                continue

            # Run
            okr, rmsg = run_java(main_class, work_dir, timeout_sec=args.timeout)
            if okr:
                passed = True
                print(f"  Passed on attempt {attempt}")
                shutil.rmtree(work_dir, ignore_errors=True)
                break
            else:
                last_error = rmsg[-2000:] if rmsg else ""
                print(f"  Failed attempt {attempt}." + (" Retrying…" if attempt < args.num_iter else " Giving up."))
                if attempt < args.num_iter:
                    completion = gen.repair(prompt, completion, last_error,
                                            max_new_tokens=args.max_new_tokens, temp=args.temp, top_p=args.top_p)
                    completion = strip_non_code(completion)
            shutil.rmtree(work_dir, ignore_errors=True)

        results.append(TaskResult(task_id=task_id, passed=passed, attempts=attempts, last_error=None if passed else last_error))

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
