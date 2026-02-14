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


def _safe_task_dirname(task_id: str, index: int) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", task_id).strip("_")
    if not safe:
        safe = f"task_{index:03d}"
    return f"{index:03d}_{safe}"


def _lang_extension(lang: str) -> str:
    return {
        "python": "py",
        "js": "js",
        "go": "go",
        "rust": "rs",
        "java": "java",
        "cpp": "cpp",
    }.get(lang, "txt")

def build_demonstrations(lang_counts: dict) -> Tuple[str, List[Tuple[int, int, str]]]:
    demo_text = ""
    demo_spans: List[Tuple[int, int, str]] = []
    order = ["python", "cpp", "java", "go"]
    demo_index = 1
    for lang in order:
        count = min(lang_counts.get(lang, 0), 10)
        if count <= 0:
            continue
        ds = load_dataset("bigcode/humanevalpack", lang, split="test")
        total = len(ds)
        start = max(0, total - 10)
        candidates = [ds[i] for i in range(start, total)]
        chosen = candidates[-count:]
        for ex in chosen:
            task_id = ex.get("task_id", "")
            prompt = ex.get("prompt") or ex.get("declaration") or ""
            solution = ex.get("canonical_solution", "") or ""
            block = (
                f"### DEMONSTRATION {demo_index} ({lang}) {task_id}\n"
                f"{prompt}\n"
                f"### SOLUTION\n"
                f"{solution}\n"
            )
            start_pos = len(demo_text)
            demo_text += block + "\n"
            end_pos = len(demo_text)
            demo_spans.append((start_pos, end_pos, f"demo_{demo_index}_{lang}_{task_id}".strip("_")))
            demo_index += 1
    return demo_text.strip(), demo_spans


def build_demo_instruction() -> str:
    return (
        "Below are solved reference examples from related tasks. "
        "Use them only as guidance for style, decomposition, and edge-case handling. "
        "Do not copy them verbatim; prioritize correctness for the current task."
    )

def map_demo_spans_to_input(input_text: str, demo_text: str, demo_spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    if not demo_text or not demo_spans:
        return []
    base = input_text.find(demo_text)
    if base < 0:
        return []
    return [(base + start, base + end, label) for start, end, label in demo_spans]

def print_input_token_attribution(
    gen: TransformersGenerator,
    input_text: str,
    completion_text: str,
    method: str = "grad_x_input",
    ig_steps: int = 20,
    show_sentence_level: bool = False,
    demo_spans: Optional[List[Tuple[int, int, str]]] = None,
) -> None:
    import torch
    import torch.nn.functional as F

    model = gen.model
    tokenizer = gen.tokenizer
    device = model.device

    model.eval()
    with torch.enable_grad():
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        completion_ids = tokenizer(
            completion_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        max_pos = getattr(model.config, "max_position_embeddings", None)
        if max_pos is not None:
            allowed = max_pos - input_ids.shape[1]
            if allowed <= 0:
                print("  Attribution skipped: input too long for model context.")
                return
            if completion_ids.shape[1] > allowed:
                completion_ids = completion_ids[:, :allowed]

        full_ids = torch.cat([input_ids, completion_ids], dim=1)
        attention_mask = torch.ones_like(full_ids)

        embed_layer = model.get_input_embeddings()
        input_embeds = embed_layer(input_ids).detach()
        completion_embeds = embed_layer(completion_ids).detach()

        def compute_loss(embeds: torch.Tensor) -> torch.Tensor:
            outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
            logits = outputs.logits
            labels = full_ids.clone()
            labels[:, : input_ids.shape[1]] = -100
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            return F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )

        if method == "integrated_gradients":
            # Keep completion embeddings fixed at true values; interpolate input only.
            baseline_input = torch.zeros_like(input_embeds)
            total_grads_input = torch.zeros_like(input_embeds)
            steps = max(1, ig_steps)
            for step in range(1, steps + 1):
                alpha = step / steps
                scaled_input = (
                    baseline_input + alpha * (input_embeds - baseline_input)
                ).detach().requires_grad_(True)
                full_embeds = torch.cat([scaled_input, completion_embeds], dim=1)
                model.zero_grad(set_to_none=True)
                loss = compute_loss(full_embeds)
                loss.backward()
                if scaled_input.grad is not None:
                    total_grads_input += scaled_input.grad.detach()
            avg_grads_input = total_grads_input / steps
            attributions = (
                (input_embeds - baseline_input) * avg_grads_input
            ).sum(dim=-1)[0].detach().cpu().tolist()
        else:
            model.zero_grad(set_to_none=True)
            full_embeds = torch.cat([input_embeds, completion_embeds], dim=1)
            full_embeds.requires_grad_(True)
            full_embeds.retain_grad()
            loss = compute_loss(full_embeds)
            loss.backward()
            grads = full_embeds.grad[:, : input_ids.shape[1], :]
            attributions = (
                grads * full_embeds[:, : input_ids.shape[1], :]
            ).sum(dim=-1)[0].detach().cpu().tolist()
        token_ids = input_ids[0].detach().cpu().tolist()
        decoded = [tokenizer.decode([tid]) for tid in token_ids]

        print("  Input text:")
        print(input_text)
        offsets = None
        if show_sentence_level or demo_spans:
            try:
                offsets = tokenizer(input_text, return_offsets_mapping=True).offset_mapping
            except Exception:
                offsets = None

        if demo_spans:
            print("  Demonstration-level attribution:")
            if offsets is None:
                print("    (offset mapping unavailable; demo-level attribution skipped)")
            else:
                demo_scores = [0.0 for _ in demo_spans]
                for idx, (tok_start, tok_end) in enumerate(offsets):
                    if tok_end <= tok_start:
                        continue
                    for d_idx, (d_start, d_end, _) in enumerate(demo_spans):
                        if tok_start >= d_end or tok_end <= d_start:
                            continue
                        demo_scores[d_idx] += attributions[idx]
                        break
                order = sorted(range(len(demo_spans)), key=lambda i: demo_scores[i], reverse=True)
                for rank, d_idx in enumerate(order, start=1):
                    label = demo_spans[d_idx][2]
                    print(f"    {rank:03d} | influence={demo_scores[d_idx]:.6f} | label={label}")
            return

        if show_sentence_level:
            print("  Sentence-level attribution:")
            sentence_units = []
            cursor = 0
            for line in input_text.splitlines(keepends=True):
                stripped = line.strip()
                is_code_like = (
                    stripped.startswith(("def ", "class ", "from ", "import ", ">>>"))
                    or line.startswith((" ", "\t"))
                )
                if "." in line and not is_code_like:
                    start = 0
                    for match in re.finditer(r"(?<!\d)\.(?!\d)", line):
                        end = match.start() + 1
                        sent = line[start:end]
                        if sent.strip():
                            sentence_units.append((cursor + start, cursor + end, sent))
                        start = end
                    tail = line[start:]
                    if tail.strip():
                        sentence_units.append((cursor + start, cursor + len(line), tail))
                else:
                    if stripped:
                        sentence_units.append((cursor, cursor + len(line), line))
                cursor += len(line)

            if offsets is None:
                print("    (offset mapping unavailable; sentence-level attribution skipped)")
            else:
                sent_scores = [0.0 for _ in sentence_units]
                for idx, (tok_start, tok_end) in enumerate(offsets):
                    if tok_end <= tok_start:
                        continue
                    for s_idx, (s_start, s_end, _) in enumerate(sentence_units):
                        if tok_start >= s_end or tok_end <= s_start:
                            continue
                        sent_scores[s_idx] += attributions[idx]
                        break
                order = sorted(range(len(sentence_units)), key=lambda i: sent_scores[i], reverse=True)
                for rank, s_idx in enumerate(order, start=1):
                    sent = sentence_units[s_idx][2]
                    print(f"    {rank:03d} | influence={sent_scores[s_idx]:.6f} | text={sent!r}")

        if not show_sentence_level:
            top_k = min(30, len(attributions))
            top_indices = sorted(range(len(attributions)), key=lambda i: attributions[i], reverse=True)[:top_k]
            print(f"  Top {top_k} input tokens by influence:")
            for idx in top_indices:
                print(f" {idx:04d} | text={decoded[idx]!r} | influence={attributions[idx]:.6f}")

            print(f"  Input token attribution (text + influence) [{method}]:")
            for idx, (dec, score) in enumerate(zip(decoded, attributions)):
                print(f" {idx:04d} | text={dec!r} | influence={score:.6f}")

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
    demo_text, demo_spans = build_demonstrations(
        {
            "python": args.demo_python,
            "cpp": args.demo_cpp,
            "java": args.demo_java,
            "go": args.demo_go,
        }
    )
    cache_root = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_root, exist_ok=True)
    tmp_root = tempfile.mkdtemp(prefix=f"humanevalpack_{args.lang}_eval_", dir=cache_root)
    results: List[TaskResult] = []

    total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    print(f"Running {total} HumanEvalPack({cfg}) tasks with {args.model} ({args.engine})")
    print(f"Max iters per task: {args.num_iter}, run-timeout: {args.timeout}s, compile-timeout: {args.compile_timeout}s\n")

    for i, ex in enumerate(ds):
        if i >= total:
            break
        task_id = ex["task_id"]
        prompt = ex.get("prompt") or ex.get("declaration") or ""
        if demo_text:
            demo_instruction = build_demo_instruction()
            prompt_with_demos = f"{demo_instruction}\n\n{demo_text}\n\n{prompt}"
        else:
            prompt_with_demos = prompt
        includes = ex.get("import", "") or "" 
        imports = ex.get("import", "") or ""
        test_setup = ex.get("test_setup", "") or ""
        test = ex.get("test", "") or ""

        print(f"[{i+1}/{total}] {task_id}")

        # First attempt
        input_text_for_completion = gen._make_inputs(prompt_with_demos, args.lang)
        completion = gen.generate(prompt_with_demos, args.lang, max_new_tokens=args.max_new_tokens, temp=args.temp, top_p=args.top_p)
        completion = strip_non_code(completion)

        passed = False
        attempts = 0
        last_error = ""
        attempt_history: List[str] = []

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
                        attempt_history.append(
                            f"Attempt {attempt}\n"
                            f"Code:\n{completion}\n\n"
                            f"Feedback:\n{last_error}"
                        )
                        history_text = "\n\n".join(attempt_history)
                        input_text_for_completion = gen._make_inputs(
                            prompt_with_demos, args.lang, history=history_text, latest_error=last_error
                        )
                        completion = gen.generate(
                            prompt_with_demos,
                            args.lang,
                            max_new_tokens=args.max_new_tokens,
                            temp=args.temp,
                            top_p=args.top_p,
                            history=history_text,
                            latest_error=last_error,
                        )
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
                        attempt_history.append(
                            f"Attempt {attempt}\n"
                            f"Code:\n{completion}\n\n"
                            f"Feedback:\n{last_error}"
                        )
                        history_text = "\n\n".join(attempt_history)
                        input_text_for_completion = gen._make_inputs(
                            prompt_with_demos, args.lang, history=history_text, latest_error=last_error
                        )
                        completion = gen.generate(
                            prompt_with_demos,
                            args.lang,
                            max_new_tokens=args.max_new_tokens,
                            temp=args.temp,
                            top_p=args.top_p,
                            history=history_text,
                            latest_error=last_error,
                        )
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
                        attempt_history.append(
                            f"Attempt {attempt}\n"
                            f"Code:\n{completion}\n\n"
                            f"Feedback:\n{last_error}"
                        )
                        history_text = "\n\n".join(attempt_history)
                        input_text_for_completion = gen._make_inputs(
                            prompt_with_demos, args.lang, history=history_text, latest_error=last_error
                        )
                        completion = gen.generate(
                            prompt_with_demos,
                            args.lang,
                            max_new_tokens=args.max_new_tokens,
                            temp=args.temp,
                            top_p=args.top_p,
                            history=history_text,
                            latest_error=last_error,
                        )
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
                        attempt_history.append(
                            f"Attempt {attempt}\n"
                            f"Code:\n{completion}\n\n"
                            f"Feedback:\n{last_error}"
                        )
                        history_text = "\n\n".join(attempt_history)
                        input_text_for_completion = gen._make_inputs(
                            prompt_with_demos, args.lang, history=history_text, latest_error=last_error
                        )
                        completion = gen.generate(
                            prompt_with_demos,
                            args.lang,
                            max_new_tokens=args.max_new_tokens,
                            temp=args.temp,
                            top_p=args.top_p,
                            history=history_text,
                            latest_error=last_error,
                        )
                        completion = strip_non_code(completion)
                    shutil.rmtree(work_dir, ignore_errors=True)
                    continue

                # Run
                ok_run, msg = run_cpp_exe_with_timeout(exe_path, timeout_sec=args.timeout)

            if ok_run:
                passed = True
                print(f"  Passed on attempt {attempt}")
                if args.print_attribution:
                    mapped_spans = map_demo_spans_to_input(input_text_for_completion, demo_text, demo_spans)
                    print_input_token_attribution(
                        gen,
                        input_text_for_completion,
                        completion,
                        method=args.attribution_method,
                        ig_steps=args.ig_steps,
                        show_sentence_level=args.sentence_level,
                        demo_spans=mapped_spans,
                    )
                shutil.rmtree(work_dir, ignore_errors=True)
                break
            else:
                last_error = msg[-2000:] if msg else ""
                print(f"  Failed attempt {attempt}." + (" Retrying…" if attempt < args.num_iter else " Giving up."))
                if attempt < args.num_iter:
                    attempt_history.append(
                        f"Attempt {attempt}\n"
                        f"Code:\n{completion}\n\n"
                        f"Feedback:\n{last_error}"
                    )
                    history_text = "\n\n".join(attempt_history)
                    input_text_for_completion = gen._make_inputs(
                        prompt_with_demos, args.lang, history=history_text, latest_error=last_error
                    )
                    completion = gen.generate(
                        prompt_with_demos,
                        args.lang,
                        max_new_tokens=args.max_new_tokens,
                        temp=args.temp,
                        top_p=args.top_p,
                        history=history_text,
                        latest_error=last_error,
                    )
                    completion = strip_non_code(completion)
            shutil.rmtree(work_dir, ignore_errors=True)

        task_cache_dir = os.path.join(tmp_root, _safe_task_dirname(task_id, i))
        os.makedirs(task_cache_dir, exist_ok=True)
        final_code_path = os.path.join(task_cache_dir, f"final_code.{_lang_extension(args.lang)}")
        with open(final_code_path, "w", encoding="utf-8") as f:
            f.write(completion)

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
    parser.add_argument("--print_attribution", action="store_true", help="Print input token attribution after a pass.")
    parser.add_argument("--sentence_level", action="store_true", help="Also print sentence-level attribution.")
    parser.add_argument("--attribution_method", type=str, choices=["grad_x_input", "integrated_gradients"], default="grad_x_input")
    parser.add_argument("--ig_steps", type=int, default=20, help="Steps for Integrated Gradients.")
    parser.add_argument("--demo_python", type=int, default=0, help="Number of Python demonstrations to add (from last 10).")
    parser.add_argument("--demo_cpp", type=int, default=0, help="Number of C++ demonstrations to add (from last 10).")
    parser.add_argument("--demo_java", type=int, default=0, help="Number of Java demonstrations to add (from last 10).")
    parser.add_argument("--demo_go", type=int, default=0, help="Number of Go demonstrations to add (from last 10).")
    args = parser.parse_args()
    main(args)
