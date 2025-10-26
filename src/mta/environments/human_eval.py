from __future__ import annotations

import copy
import os
import re
import shutil
import tempfile
from typing import Any, Dict

from humanevalpack.language_utils import (
    compile_cpp,
    compile_go,
    compile_java,
    compile_rust,
    run_cpp_exe_with_timeout,
    run_exe,
    run_java,
    run_node,
    run_with_timeout,
    write_solution_cpp,
    write_solution_file_go,
    write_solution_file_js,
    write_solution_file_rust,
    write_solution_java,
    write_solution_python,
)

from mta.environments.base import BaseEnv


class HumanEvalEnv(BaseEnv):
    """Environment wrapper that evaluates HumanEval-style problems across multiple languages."""

    DEFAULT_RUNTIME_TIMEOUT = 5.0
    DEFAULT_COMPILE_TIMEOUT = 25.0
    DEFAULT_MAX_ATTEMPTS = 3

    LANGUAGE_ALIASES = {
        "py": "python",
        "python": "python",
        "js": "js",
        "javascript": "js",
        "typescript": "js",
        "ts": "js",
        "go": "go",
        "golang": "go",
        "rust": "rust",
        "rs": "rust",
        "java": "java",
        "cpp": "cpp",
        "c++": "cpp",
        "cxx": "cpp",
    }

    LANGUAGE_DISPLAY_NAMES = {
        "python": "Python",
        "js": "JavaScript",
        "go": "Go",
        "rust": "Rust",
        "java": "Java",
        "cpp": "C++",
    }

    LANGUAGE_COMPILE_TIMEOUTS = {
        "go": 25.0,
        "rust": 30.0,
        "java": 20.0,
        "cpp": 20.0,
    }

    def __init__(
        self,
        *,
        problem: Dict[str, Any],
        timeout: float | None = None,
        max_attempts: int | None = None,
        compile_timeout: float | None = None,
        print_submission: bool | None = None,
    ):
        if problem is None:
            raise ValueError("A HumanEval problem dictionary must be provided.")

        self.problem = copy.deepcopy(problem)
        raw_language = str(self.problem.get("language") or "python").lower()
        self.language = self.LANGUAGE_ALIASES.get(raw_language, raw_language)
        self.entry_point = self.problem.get("entry_point")
        self.prompt = (self.problem.get("prompt") or self.problem.get("declaration") or "").rstrip("\n")
        self.imports = (self.problem.get("imports") or self.problem.get("import") or "").strip()
        self.includes = (self.problem.get("includes") or self.imports).strip()
        self.test_setup = (self.problem.get("test_setup") or "").strip()
        self.test = (self.problem.get("test") or "").strip()
        self.instructions_override = self.problem.get("instructions")
        embedded_flag = self.problem.pop("print_submission", None)
        if print_submission is None:
            print_setting = embedded_flag
        else:
            print_setting = print_submission
        self.print_submission = bool(print_setting)

        runtime_timeout = self.problem.get("timeout")
        if runtime_timeout is None:
            runtime_timeout = timeout
        if runtime_timeout is None:
            runtime_timeout = self.DEFAULT_RUNTIME_TIMEOUT
        self.timeout = float(runtime_timeout)

        compile_timeout_value = self.problem.get("compile_timeout")
        if compile_timeout_value is None:
            compile_timeout_value = compile_timeout
        if compile_timeout_value is None:
            compile_timeout_value = self.LANGUAGE_COMPILE_TIMEOUTS.get(self.language)
        if compile_timeout_value is not None:
            self.compile_timeout = float(compile_timeout_value)
        else:
            self.compile_timeout = None

        attempts = self.problem.get("max_attempts")
        if attempts is None:
            attempts = max_attempts
        if attempts is None:
            attempts = self.DEFAULT_MAX_ATTEMPTS
        self.max_attempts = int(attempts)
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer.")

        self._attempts = 0
        self._done = False
        self._last_reward = 0.0
        self._tmp_root = tempfile.mkdtemp(prefix=f"humaneval_{self.language}_")

    # ------------------------------------------------------------------ #
    # BaseEnv implementation
    # ------------------------------------------------------------------ #

    def reset(self) -> tuple[str, dict]:
        self._attempts = 0
        self._done = False
        self._last_reward = 0.0

        instructions = self.instructions_override or self._build_default_instructions()
        info = {
            "task_id": self.problem.get("task_id"),
            "entry_point": self.entry_point,
            "max_attempts": self.max_attempts,
            "language": self.language,
        }
        if self.timeout is not None:
            info["timeout"] = self.timeout
        if self.compile_timeout is not None:
            info["compile_timeout"] = self.compile_timeout

        return instructions, info

    def step(self, action: Any) -> tuple[str, float, bool, dict]:
        if self._done:
            info = {
                "task_id": self.problem.get("task_id"),
                "entry_point": self.entry_point,
                "attempt": self._attempts,
                "max_attempts": self.max_attempts,
                "language": self.language,
                "passed": self._last_reward > 0,
                "result": "completed",
            }
            return "Task already completed. No further submissions are required.", self._last_reward, True, info

        submission = self._extract_completion(str(action) if action is not None else "")
        if not submission.strip():
            feedback = (
                f"No {self._language_display_name()} implementation detected in your response. "
                f"Please reply with a {self._language_display_name()} code block that completes the task."
            )
            self._attempts += 1
            self._update_state(passed=False)
            info = {
                "task_id": self.problem.get("task_id"),
                "entry_point": self.entry_point,
                "attempt": self._attempts,
                "max_attempts": self.max_attempts,
                "language": self.language,
                "passed": False,
                "result": "empty submission",
            }
            return feedback, self._last_reward, self._done, info

        if self.print_submission:
            attempt_no = self._attempts + 1
            header = f"=== Submission attempt {attempt_no} ({self._language_display_name()}) ==="
            print(header)
            print(submission.rstrip("\n"))
            print("=" * len(header))

        passed, message = self._evaluate_submission(submission)
        self._attempts += 1
        self._update_state(passed=passed)

        status = "PASSED" if passed else "FAILED"
        message_text = message.strip() or ("All tests passed." if passed else "Evaluation failed.")
        feedback_lines = [f"Evaluation result ({status}) on attempt {self._attempts}: {message_text}"]
        if passed:
            feedback_lines.append("All unit tests passed. You may stop responding.")
        else:
            if self._done:
                feedback_lines.append("Maximum attempts reached. No further submissions will be evaluated.")
            else:
                feedback_lines.append(f"Please submit a revised {self._language_display_name()} code block containing your solution.")

        info = {
            "task_id": self.problem.get("task_id"),
            "entry_point": self.entry_point,
            "attempt": self._attempts,
            "max_attempts": self.max_attempts,
            "language": self.language,
            "passed": passed,
            "result": message_text,
        }

        return "\n".join(feedback_lines), self._last_reward, self._done, info

    def compute_final_reward(self):
        return self._last_reward

    def close(self):
        shutil.rmtree(self._tmp_root, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_dict(extra_info: dict | str) -> "HumanEvalEnv":
        if isinstance(extra_info, str):
            import json

            problem_dict = json.loads(extra_info)
        else:
            problem_dict = extra_info

        problem = copy.deepcopy(problem_dict)
        timeout = problem.pop("timeout", None)
        max_attempts = problem.pop("max_attempts", None)
        compile_timeout = problem.pop("compile_timeout", None)
        print_submission = problem.pop("print_submission", None)

        return HumanEvalEnv(
            problem=problem,
            timeout=timeout,
            max_attempts=max_attempts,
            compile_timeout=compile_timeout,
            print_submission=print_submission,
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return True

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _language_display_name(self) -> str:
        return self.LANGUAGE_DISPLAY_NAMES.get(self.language, self.language.capitalize())

    def _build_default_instructions(self) -> str:
        task_id = self.problem.get("task_id", "unknown")
        entry_point = self.entry_point or "the target entry point"
        prompt = (self.problem.get("prompt") or "").rstrip()
        language_name = self._language_display_name()

        header = [
            f"Task: {task_id}",
            f"Implement the body of the provided {language_name} stub.",
        ]
        if entry_point:
            header.append(f"Return your answer as a {language_name} code block that completes `{entry_point}`.")
        else:
            header.append(f"Return your answer as a {language_name} code block implementing the described behaviour.")
        header.append(f"You may attempt up to {self.max_attempts} submission(s); feedback will be provided after each attempt.")

        if prompt:
            header.extend(["---", prompt])

        return "\n\n".join(header)

    def _update_state(self, *, passed: bool):
        self._last_reward = 1.0 if passed else 0.0
        if passed or self._attempts >= self.max_attempts:
            self._done = True
        else:
            self._done = False

    def _extract_completion(self, action_text: str) -> str:
        code = self._extract_code_block(action_text)
        if not code:
            code = action_text.strip()
        if not code:
            return ""
        if self.language == "python":
            normalized = self._normalize_python_definition(code)
            return normalized.rstrip("\n")
        return code.strip()

    def _extract_code_block(self, text: str) -> str:
        pattern = re.compile(r"```(?:[\w#+-]+)?\s*(.*?)```", re.DOTALL)
        match = pattern.search(text)
        if match:
            return match.group(1).strip("\n")
        return ""

    def _normalize_python_definition(self, code: str) -> str:
        entry_point = self.entry_point
        if not entry_point:
            return code.strip()

        definition_pattern = re.compile(rf"def\s+{re.escape(entry_point)}\s*\(.*?\):\s*(?:#.*)?", re.DOTALL)
        match = definition_pattern.search(code)
        if not match:
            return self._format_python_body(code)

        body = code[match.end() :].lstrip("\n")
        if not body:
            return ""

        return self._format_python_body(body)

    def _format_python_body(self, body: str) -> str:
        trimmed = body.strip("\n")
        if not trimmed:
            return ""

        lines = trimmed.splitlines()
        indent_candidates = [
            len(line) - len(line.lstrip())
            for line in lines
            if line.strip() and (len(line) - len(line.lstrip())) > 0
        ]
        base_indent = min(indent_candidates) if indent_candidates else 0

        normalized_lines: list[str] = []
        for line in lines:
            if line.strip():
                stripped = line.lstrip()
                indent_len = len(line) - len(stripped)
                relative_indent = max(0, indent_len - base_indent)
                normalized_lines.append(" " * (4 + relative_indent) + stripped)
            else:
                normalized_lines.append("")

        normalized = "\n".join(normalized_lines).rstrip("\n")
        return normalized

    def _make_attempt_dir(self) -> str:
        return tempfile.mkdtemp(prefix=f"attempt_{self._attempts:03d}_", dir=self._tmp_root)

    def _runtime_timeout_seconds(self) -> int:
        return max(1, int(round(self.timeout)))

    def _compile_timeout_seconds(self) -> int:
        base = self.compile_timeout
        if base is None:
            base = self.LANGUAGE_COMPILE_TIMEOUTS.get(self.language, self.DEFAULT_COMPILE_TIMEOUT)
        return max(1, int(round(base)))

    def _evaluate_submission(self, completion: str) -> tuple[bool, str]:
        language = self.language
        if language == "python":
            return self._evaluate_python(completion)
        if language == "js":
            return self._evaluate_js(completion)
        if language == "go":
            return self._evaluate_go(completion)
        if language == "rust":
            return self._evaluate_rust(completion)
        if language == "java":
            return self._evaluate_java(completion)
        if language == "cpp":
            return self._evaluate_cpp(completion)
        raise ValueError(f"Unsupported language '{language}'")

    def _evaluate_python(self, completion: str) -> tuple[bool, str]:
        work_dir = self._make_attempt_dir()
        try:
            main_py = os.path.join(work_dir, "main.py")
            write_solution_python(self.prompt, completion, self.imports, self.test_setup, self.test, main_py)
            ok, message = run_with_timeout(main_py, timeout_sec=self._runtime_timeout_seconds())
            return ok, message or ""
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _evaluate_js(self, completion: str) -> tuple[bool, str]:
        work_dir = self._make_attempt_dir()
        try:
            src_path = write_solution_file_js(self.imports, self.prompt, completion, self.test_setup, self.test, work_dir)
            ok, message = run_node(src_path, timeout_sec=self._runtime_timeout_seconds())
            return ok, message or ""
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _evaluate_go(self, completion: str) -> tuple[bool, str]:
        work_dir = self._make_attempt_dir()
        try:
            src_path = write_solution_file_go(self.imports, self.prompt, completion, self.test_setup, self.test, work_dir)
            exe_path = os.path.join(work_dir, "main_go_exec")
            ok_compile, compile_msg = compile_go(src_path, exe_path, timeout_sec=self._compile_timeout_seconds())
            if not ok_compile:
                return False, compile_msg or "Compilation failed."
            ok_run, run_msg = run_exe(exe_path, timeout_sec=self._runtime_timeout_seconds())
            return ok_run, run_msg or ""
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _evaluate_rust(self, completion: str) -> tuple[bool, str]:
        work_dir = self._make_attempt_dir()
        try:
            src_path = write_solution_file_rust(self.imports, self.prompt, completion, self.test_setup, self.test, work_dir)
            exe_path = os.path.join(work_dir, "main_rs_exec")
            ok_compile, compile_msg = compile_rust(src_path, exe_path, timeout_sec=self._compile_timeout_seconds())
            if not ok_compile:
                return False, compile_msg or "Compilation failed."
            ok_run, run_msg = run_exe(exe_path, timeout_sec=self._runtime_timeout_seconds())
            return ok_run, run_msg or ""
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _evaluate_java(self, completion: str) -> tuple[bool, str]:
        work_dir = self._make_attempt_dir()
        try:
            main_class, java_path = write_solution_java(self.includes, self.prompt, completion, self.test_setup, self.test, work_dir)
            ok_compile, compile_msg = compile_java(java_path, timeout_sec=self._compile_timeout_seconds())
            if not ok_compile:
                return False, compile_msg or "Compilation failed."
            ok_run, run_msg = run_java(main_class, work_dir, timeout_sec=self._runtime_timeout_seconds())
            return ok_run, run_msg or ""
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _evaluate_cpp(self, completion: str) -> tuple[bool, str]:
        work_dir = self._make_attempt_dir()
        try:
            src_path = os.path.join(work_dir, "main.cpp")
            exe_path = os.path.join(work_dir, "main.out")
            write_solution_cpp(self.prompt, completion, self.includes, self.test_setup, self.test, src_path, entry_point=self.entry_point)
            ok_compile, compile_msg = compile_cpp(src_path, exe_path, timeout_sec=self._compile_timeout_seconds())
            if not ok_compile:
                return False, compile_msg or "Compilation failed."
            ok_run, run_msg = run_cpp_exe_with_timeout(exe_path, timeout_sec=self._runtime_timeout_seconds())
            return ok_run, run_msg or ""
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
