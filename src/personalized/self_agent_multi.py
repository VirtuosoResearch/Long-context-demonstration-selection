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
from generator import Generator, TransformersGenerator
from language_utils import write_solution_file_go, write_solution_file_js, write_solution_file_rust, write_solution_python, write_solution_java, write_solution_cpp
from language_utils import run_node, compile_go, run_exe, compile_rust, run_with_timeout, compile_java, run_java, compile_cpp, run_cpp_exe_with_timeout
from datasets import load_dataset

def strip_non_code(text: str) -> str:
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()

@dataclass
class TaskResult:
    task_id: str
    passed: bool
    attempts: int
    last_error: Optional[str]

def main(args):
    cfg = args.lang
    print("Model:", args.model)
    ds = load_dataset("bigcode/humanevalpack", cfg, split="test")
    gen = TransformersGenerator(args.model)
    tmp_root = tempfile.mkdtemp(prefix=f"humanevalpack_{args.lang}_eval_")
    results: List[TaskResult] = []

    total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    print(f"Running {total} HumanEvalPack({cfg}) tasks with {args.model} ({args.engine})")
    print(f"Max iters per task: {args.num_iter}, run-timeout: {args.timeout}s, compile-timeout: {args.compile_timeout}s\n")

    for i, ex in enumerate(ds):
        if i >= total:
            break
        task_id = ex["task_id"]
        prompt = ex.get("prompt") or ex.get("declaration") or ""
        includes = ex.get("import", "") or "" 
        imports = ex.get("import", "") or ""
        test_setup = ex.get("test_setup", "") or ""
        test = ex.get("test", "") or ""

        print(f"[{i+1}/{total}] {task_id}")

        # First attempt
        completion = gen.generate(prompt, args.lang, max_new_tokens=args.max_new_tokens, temp=args.temp, top_p=args.top_p)
        completion = strip_non_code(completion)

        passed = False
        attempts = 0
        last_error = ""

        for attempt in range(1, args.num_iter + 1):
            attempts = attempt
            work_dir = tempfile.mkdtemp(prefix=f"task_{i:03d}_", dir=tmp_root)
            if args.lang == "python":
                main_py = os.path.join(work_dir, "main.py")
                write_solution_python(prompt, completion, imports, test_setup, test, main_py)
                print("main: ",main_py)
                ok_run, msg = run_with_timeout(main_py, timeout_sec=args.timeout)

            elif args.lang == "js":
                src_path = write_solution_file_js(imports, prompt, completion, test_setup, test, work_dir)
                ok_run, msg = run_node(src_path, timeout_sec=args.timeout)

            elif args.lang == "go":
                src_path = write_solution_file_go(imports, prompt, completion, test_setup, test, work_dir)
                exe_path = os.path.join(work_dir, "main_go_exec")
                okc, cerr = compile_go(src_path, exe_path, timeout_sec=args.compile_timeout)
                if not okc:
                    last_error = cerr[-2000:] if cerr else "Unknown compile error"
                    print(f"  Compile failed (attempt {attempt})." + (" Retrying…" if attempt < args.num_iter else " Giving up."))
                    if attempt < args.num_iter:
                        completion = gen.repair(prompt, completion, last_error, args.lang,
                                                max_new_tokens=args.max_new_tokens, temp=args.temp, top_p=args.top_p)
                        completion = strip_non_code(completion)
                    shutil.rmtree(work_dir, ignore_errors=True)
                    continue
                ok_run, msg = run_exe(exe_path, timeout_sec=args.timeout)

            elif args.lang == "rust":
                src_path = write_solution_file_rust(imports, prompt, completion, test_setup, test, work_dir)
                exe_path = os.path.join(work_dir, "main_rs_exec")
                okc, cerr = compile_rust(src_path, exe_path, timeout_sec=args.compile_timeout)
                if not okc:
                    last_error = cerr[-2000:] if cerr else "Unknown compile error"
                    print(f"  Compile failed (attempt {attempt})." + (" Retrying…" if attempt < args.num_iter else " Giving up."))
                    if attempt < args.num_iter:
                        completion = gen.repair(prompt, completion, last_error, args.lang,
                                                max_new_tokens=args.max_new_tokens, temp=args.temp, top_p=args.top_p)
                        completion = strip_non_code(completion)
                    shutil.rmtree(work_dir, ignore_errors=True)
                    continue
                ok_run, msg = run_exe(exe_path, timeout_sec=args.timeout)
            
            elif args.lang == "java":
                main_class, java_path = write_solution_java(includes, prompt, completion, test_setup, test, work_dir)
                print("source:", java_path, " main_class:", main_class)

                # Compile
                okc, cerr = compile_java(java_path, timeout_sec=args.compile_timeout)
                if not okc:
                    last_error = cerr[-2000:] if cerr else "Unknown compile error"
                    print(f"  Compile failed (attempt {attempt})." + (" Retrying…" if attempt < args.num_iter else " Giving up."))
                    if attempt < args.num_iter:
                        completion = gen.repair(prompt, completion, last_error, args.lang,
                                                max_new_tokens=args.max_new_tokens, temp=args.temp, top_p=args.top_p)
                        completion = strip_non_code(completion)
                    shutil.rmtree(work_dir, ignore_errors=True)
                    continue
                # Run
                ok_run, msg = run_java(main_class, work_dir, timeout_sec=args.timeout)
            
            elif args.lang == "cpp":
                src_cpp = os.path.join(work_dir, "main.cpp")
                exe_path = os.path.join(work_dir, "a.out")

                write_solution_cpp(prompt, completion, includes, test_setup, test, src_cpp)
                print("source:", src_cpp)

                ok_compile, compile_msg = compile_cpp(src_cpp, exe_path, timeout_sec=args.compile_timeout)
                if not ok_compile:
                    last_error = compile_msg[-2000:] if compile_msg else "Unknown compile error"
                    print(f"  Compile failed (attempt {attempt})." + (" Retrying…" if attempt < args.num_iter else " Giving up."))
                    if attempt < args.num_iter:
                        completion = gen.repair(prompt, completion, last_error, args.lang,
                                                max_new_tokens=args.max_new_tokens, temp=args.temp, top_p=args.top_p)
                        completion = strip_non_code(completion)
                    shutil.rmtree(work_dir, ignore_errors=True)
                    continue

                # Run
                ok_run, msg = run_cpp_exe_with_timeout(exe_path, timeout_sec=args.timeout)

            if ok_run:
                passed = True
                print(f"  Passed on attempt {attempt}")
                shutil.rmtree(work_dir, ignore_errors=True)
                break
            else:
                last_error = msg[-2000:] if msg else ""
                print(f"  Failed attempt {attempt}." + (" Retrying…" if attempt < args.num_iter else " Giving up."))
                if attempt < args.num_iter:
                    completion = gen.repair(prompt, completion, last_error, args.lang,
                                            max_new_tokens=args.max_new_tokens, temp=args.temp, top_p=args.top_p)
                    completion = strip_non_code(completion)
            shutil.rmtree(work_dir, ignore_errors=True)

        results.append(TaskResult(task_id=task_id, passed=passed, attempts=attempts, last_error=None if passed else last_error))

    passed_count = sum(r.passed for r in results)
    print("\n=== Summary ===")
    print(f"Passed: {passed_count}/{len(results)}  ({passed_count/len(results)*100:.2f}%)")

    # Save raw results
    out_json = os.path.join(tmp_root, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in results], f, ensure_ascii=False, indent=2)
    print(f"Per-task results saved to: {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, choices=["js", "go", "rust", "python", "java", "cpp"], required=True, help="Target language")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--engine", type=str, choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_iter", type=int, default=10, help="Max self-repair rounds per task (including first attempt).")
    parser.add_argument("--compile_timeout", type=int, default=25, help="Seconds for compilation (Go/Rust).")
    parser.add_argument("--timeout", type=int, default=5, help="Seconds per run attempt.")
    parser.add_argument("--limit", type=int, default=0, help="If >0, only evaluate first N tasks.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
