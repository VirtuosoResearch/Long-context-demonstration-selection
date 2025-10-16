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
# -----------------------------
# JavaScript (Node.js)
# -----------------------------
def tool_exists(cmd: str) -> bool:
    from shutil import which
    return which(cmd) is not None

def indent_block(src: str, level: int = 1, width: int = 4) -> str:
    pad = " " * (level * width)
    return "\n".join(pad + line if line.strip() != "" else "" for line in src.splitlines())

def write_solution_file_js(imports: str, prompt: str, completion: str, test_setup: str, test: str, out_dir: str) -> str:
    """
    Emit a single .js file. Node executes top-level code, so we just concatenate.
    If tests don't contain anything, create a noop at the end.
    """
    parts = []
    if imports and imports.strip():
        parts.append(imports.strip())
    # Join prompt + completion
    parts.append((prompt or "").rstrip() + "\n" + (completion or "").strip())
    if test_setup and test_setup.strip():
        parts.append(test_setup.strip())
    if test and test.strip():
        parts.append(test.strip())
    else:
        parts.append("// No tests provided\n")
    out_path = os.path.join(out_dir, "main.js")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts))
    return out_path

def run_node(js_file: str, timeout_sec: int = 5) -> Tuple[bool, str]:
    if not tool_exists("node"):
        return False, "Node.js not found: please install `node`."
    try:
        cp = subprocess.run(
            ["node", os.path.basename(js_file)],
            cwd=os.path.dirname(js_file),
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


# -----------------------------
# Go
# -----------------------------

def strip_package_decl(s: str) -> str:
    PKG_RE = re.compile(r"^\s*package\s+([A-Za-z_]\w*)", re.MULTILINE)
    return PKG_RE.sub("", s or "")

def ensure_go_main_source(imports: str, prompt: str, completion: str, test_setup: str, test: str) -> str:
    """
    Build a single file with package main.
    - Place user's function(s) (prompt+completion) at top-level (package main).
    - If test doesn't contain `func main()`, wrap test into one.
    - No external modules.
    """
    parts = ["package main"]
    if imports and imports.strip():
        # Keep only import blocks/lines; if dataset provided misc text here, pass-through
        parts.append(imports.strip())

    # User code (strip any package decls)
    code_impl = strip_package_decl((prompt or "")) + "\n" + strip_package_decl((completion or ""))
    parts.append(code_impl.strip())

    if test_setup and test_setup.strip():
        parts.append(strip_package_decl(test_setup.strip()))

    test_code = test.strip() if test else ""
    if test_code:
        if re.search(r"\bfunc\s+main\s*\(", test_code):
            parts.append(strip_package_decl(test_code))
        else:
            wrapped = "func main() {\n" + indent_block(test_code) + "\n}"
            parts.append(wrapped)
    else:
        parts.append("func main() {}")
    return "\n\n".join(parts)

def write_solution_file_go(imports: str, prompt: str, completion: str, test_setup: str, test: str, out_dir: str) -> str:
    source = ensure_go_main_source(imports, prompt, completion, test_setup, test)
    out_path = os.path.join(out_dir, "main.go")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(source)
    return out_path

def compile_go(go_file: str, exe_path: str, timeout_sec: int = 20) -> Tuple[bool, str]:
    if not tool_exists("go"):
        return False, "Go toolchain not found: please install `go`."
    try:
        cp = subprocess.run(
            ["go", "build", "-o", os.path.basename(exe_path), os.path.basename(go_file)],
            cwd=os.path.dirname(go_file),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
            text=True,
            env=dict(os.environ, GOFLAGS="-buildvcs=false"),
        )
        if cp.returncode == 0:
            return True, ""
        else:
            err = (cp.stderr or "") + "\n" + (cp.stdout or "")
            return False, err.strip()
    except subprocess.TimeoutExpired as e:
        return False, f"Compile timeout after {timeout_sec}s\n{str(e)}"

def run_exe(exe_path: str, timeout_sec: int = 5) -> Tuple[bool, str]:
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
            err = (cp.stderr or "") + "\n" + (cp.stdout or "")
            return False, err.strip()
    except subprocess.TimeoutExpired as e:
        return False, f"Runtime timeout after {timeout_sec}s\n{str(e)}"


# -----------------------------
# Rust
# -----------------------------
def write_solution_file_rust(imports: str, prompt: str, completion: str, test_setup: str, test: str, out_dir: str) -> str:
    """
    Build a single main.rs-like source:
    - Put functions first (prompt+completion)
    - Append test_setup
    - If test has `fn main()`, keep it; else wrap test body into `fn main() { ... }`
    """
    parts = []
    if imports and imports.strip():
        parts.append(imports.strip())  # Usually empty for Rust; pass-through if provided.

    # functions
    parts.append((prompt or "").rstrip() + "\n" + (completion or "").strip())

    if test_setup and test_setup.strip():
        parts.append(test_setup.strip())

    test_code = test.strip() if test else ""
    if test_code:
        if re.search(r"\bfn\s+main\s*\(", test_code):
            parts.append(test_code)
        else:
            wrapped = "fn main() {\n" + indent_block(test_code) + "\n}"
            parts.append(wrapped)
    else:
        parts.append("fn main() {}")

    out_path = os.path.join(out_dir, "main.rs")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts))
    return out_path

def compile_rust(rs_file: str, exe_path: str, timeout_sec: int = 30) -> Tuple[bool, str]:
    if not tool_exists("rustc"):
        return False, "Rust compiler not found: please install `rustc`."
    try:
        cp = subprocess.run(
            ["rustc", "-O", os.path.basename(rs_file), "-o", os.path.basename(exe_path)],
            cwd=os.path.dirname(rs_file),
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


# -----------------------------
# Python
# -----------------------------

def write_solution_python(task_prompt: str, completion: str, imports: str, test_setup: str, test: str, out_path: str):
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