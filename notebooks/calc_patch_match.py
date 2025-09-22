#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional, Any, Iterable

from datasets import load_dataset
from unidiff import PatchSet
import pandas as pd

DEF_NAME_RE = re.compile(r'\bdef\s+([A-Za-z_]\w*)\s*\(')
CLASS_NAME_RE = re.compile(r'\bclass\s+([A-Za-z_]\w*)\s*(?:\(|:)')
HUNK_HEADER_RE = re.compile(r'^@@\s*-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s*@@\s*(.*)$')

def extract_patch_text(raw: str) -> str:
    if raw is None:
        return ""
    m = re.search(r"<patch>\s*(.*?)\s*</patch>", raw, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else raw

def _iter_hunk_header_lines(hunk) -> Iterable[str]:
    sh = getattr(hunk, "section_header", None)
    if isinstance(sh, str) and sh.strip():
        yield f"@@ -{getattr(hunk, 'source_start', '')} +{getattr(hunk, 'target_start', '')} @@ {sh}".rstrip()
        return
    hunk_str = str(hunk)
    if hunk_str:
        yield hunk_str.splitlines()[0]

def _extract_name_from_header(header_line: str) -> Optional[str]:
    m = HUNK_HEADER_RE.match(header_line)
    if not m:
        return None
    ctx = m.group(1).strip()
    if not ctx:
        return None
    m_def = DEF_NAME_RE.search(ctx)
    if m_def:
        return m_def.group(1)
    m_cls = CLASS_NAME_RE.search(ctx)
    if m_cls:
        return m_cls.group(1)
    return None

def _get_source_line_no(line) -> Optional[int]:
    # unidiff PatchedLine has .source_line_no for removed/context, .target_line_no for added
    val = getattr(line, "source_line_no", None)
    return int(val) if isinstance(val, int) else None

def parse_patch(patch_text: str):
    """
    Returns:
      files: set[str]
      hunk_starts: set[(file, source_start)]
      funcs: dict[file] -> set[func_or_class_name]
      details: list[dict]
      changed_old_lines: set[(file, old_line_no)]
    """
    files: Set[str] = set()
    hunk_starts: Set[Tuple[str, int]] = set()
    funcs: Dict[str, Set[str]] = defaultdict(set)
    details: List[Dict] = []
    changed_old_lines: Set[Tuple[str, int]] = set()

    if not patch_text or not patch_text.strip():
        return files, hunk_starts, funcs, details, changed_old_lines

    try:
        ps = PatchSet(patch_text.splitlines(True))
    except Exception:
        return files, hunk_starts, funcs, details, changed_old_lines

    for f in ps:
        fpath = f.path
        files.add(fpath)
        for h in f:
            old_start = getattr(h, "source_start", None)
            if isinstance(old_start, int):
                hunk_starts.add((fpath, old_start))

            func_name: Optional[str] = None
            for header_line in _iter_hunk_header_lines(h):
                func_name = _extract_name_from_header(header_line)
                if func_name:
                    break
            if func_name:
                funcs[fpath].add(func_name)

            # Track actual changed old-line numbers (removed or replaced lines)
            for ln in h:
                if getattr(ln, "is_removed", False):
                    no = _get_source_line_no(ln)
                    if isinstance(no, int):
                        changed_old_lines.add((fpath, no))

            details.append({
                "file": fpath,
                "old_start": old_start,
                "new_start": getattr(h, "target_start", None),
                "func": func_name
            })

    return files, hunk_starts, funcs, details, changed_old_lines

def load_official_oracle(split: str = "test"):
    ds = load_dataset("princeton-nlp/SWE-bench_Lite_oracle", split=split)
    official = {}
    for row in ds:
        iid = row.get("instance_id")
        patch_text = extract_patch_text(row.get("patch") or "")
        official[iid] = parse_patch(patch_text)
    return official

def load_generated_jsonl(path: str):
    generated = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = obj.get("instance_id") or obj.get("id") or obj.get("issue_id")
            if not iid:
                continue
            raw = obj.get("model_patch") or obj.get("full_output") or obj.get("patch") or obj.get("s")
            if not raw:
                continue
            patch_text = extract_patch_text(raw)
            generated[iid] = parse_patch(patch_text)
    return generated

def choose_example_match(s):
    try:
        return next(iter(s))
    except StopIteration:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_jsonl", type=str, required=True, help="Path to your generated jsonl file")
    parser.add_argument("--split", type=str, default="test", help="SWE-bench_Lite_oracle split (default: test)")
    parser.add_argument("--out_csv", type=str, default="compare_patch_alignment.csv", help="Output CSV filename")
    args = parser.parse_args()

    print("Loading official oracle patches ...")
    official = load_official_oracle(split=args.split)

    print("Loading your generated patches ...")
    generated = load_generated_jsonl(args.gen_jsonl)

    rows = []
    all_iids = sorted(set(generated.keys()) & set(official.keys()))
    missing_in_official = sorted(set(generated.keys()) - set(official.keys()))
    missing_in_gen = sorted(set(official.keys()) - set(generated.keys()))

    if missing_in_official:
        print(f"[Warning] {len(missing_in_official)} instance(s) only in your JSONL but not in official split.")

    for iid in all_iids:
        g_files, g_hunks, g_funcs, _, g_changed = generated[iid]
        o_files, o_hunks, o_funcs, _, o_changed = official[iid]

        same_file = len(g_files & o_files) > 0

        same_line_hunkstart = len(g_hunks & o_hunks) > 0
        same_line_changed = len(g_changed & o_changed) > 0

        common_files = g_files & o_files
        func_match = False
        example_func = None
        found_any_name = False
        for f in common_files:
            gf = g_funcs.get(f, set())
            of = o_funcs.get(f, set())
            if gf or of:
                found_any_name = True
            inter = gf & of
            if inter:
                func_match = True
                example_func = next(iter(inter))
                break
        function_status = "match" if func_match else ("unknown" if not found_any_name else "mismatch")

        rows.append({
            "instance_id": iid,
            "same_file": same_file,
            "same_function": func_match,
            "function_status": function_status,
            "same_line_hunkstart": same_line_hunkstart,
            "same_line_changed": same_line_changed,
            "example_common_file": choose_example_match(common_files) or "",
            "example_common_function": example_func or "",
            "example_common_changed_line": f"{choose_example_match(g_changed & o_changed)}" if (g_changed & o_changed) else "",
        })

    df = pd.DataFrame(rows).sort_values(
        ["same_file", "same_function", "same_line_changed", "same_line_hunkstart", "instance_id"],
        ascending=[False, False, False, False, True]
    )
    print("\n=== Summary (top 30 rows) ===")
    print(df.head(30).to_string(index=False))

    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"\nSaved CSV -> {args.out_csv}")

    if missing_in_gen:
        print(f"[Info] {len(missing_in_gen)} instance(s) only in official split.")

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import argparse
# import json
# import re
# from collections import defaultdict
# from typing import Dict, Set, Tuple, List, Optional, Any, Iterable

# from datasets import load_dataset
# from unidiff import PatchSet
# import pandas as pd

# # Extract function/class name from hunk header trailing context:
# #   @@ -a,b +c,d @@ def foo(x):
# #   @@ -a +c @@ class Bar:
# DEF_NAME_RE = re.compile(r'\bdef\s+([A-Za-z_]\w*)\s*\(')
# CLASS_NAME_RE = re.compile(r'\bclass\s+([A-Za-z_]\w*)\s*(?:\(|:)')
# HUNK_HEADER_RE = re.compile(r'^@@\s*-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s*@@\s*(.*)$')

# def _call_or_value(obj: Any) -> Any:
#     return obj() if callable(obj) else obj

# def extract_patch_text(raw: str) -> str:
#     if raw is None:
#         return ""
#     m = re.search(r"<patch>\s*(.*?)\s*</patch>", raw, flags=re.DOTALL | re.IGNORECASE)
#     return m.group(1) if m else raw

# def _iter_hunk_header_lines(hunk) -> Iterable[str]:
#     """
#     Yield the first header line of a hunk in a version-agnostic way.
#     """
#     # Try unidiff's section_header (if present)
#     sh = getattr(hunk, "section_header", None)
#     if isinstance(sh, str) and sh.strip():
#         # Compose a synthetic header line like real unified diff first line
#         # Consumers only care about the trailing context, so this is fine.
#         yield f"@@ -{getattr(hunk, 'source_start', '')} +{getattr(hunk, 'target_start', '')} @@ {sh}".rstrip()
#         return

#     # Fallback: use the stringified hunk and take its first line
#     hunk_str = str(hunk)
#     if hunk_str:
#         first = hunk_str.splitlines()[0]
#         yield first

# def _extract_name_from_header(header_line: str) -> Optional[str]:
#     """
#     Parse trailing context after '@@ ... @@ ' and extract function/class name.
#     """
#     m = HUNK_HEADER_RE.match(header_line)
#     if not m:
#         return None
#     ctx = m.group(1).strip()
#     if not ctx:
#         return None
#     m_def = DEF_NAME_RE.search(ctx)
#     if m_def:
#         return m_def.group(1)
#     m_cls = CLASS_NAME_RE.search(ctx)
#     if m_cls:
#         return m_cls.group(1)
#     return None

# def parse_patch(patch_text: str):
#     """
#     Parse a unified diff and return:
#       - files: set of file paths changed
#       - lines: set of (file, source_start) per hunk
#       - funcs: dict[file] -> set of function/class names from hunk header context
#       - details: list of hunk summaries
#     """
#     files: Set[str] = set()
#     lines: Set[Tuple[str, int]] = set()
#     funcs: Dict[str, Set[str]] = defaultdict(set)
#     details: List[Dict] = []

#     if not patch_text or not patch_text.strip():
#         return files, lines, funcs, details

#     try:
#         ps = PatchSet(patch_text.splitlines(True))
#     except Exception:
#         return files, lines, funcs, details

#     for f in ps:
#         fpath = f.path
#         files.add(fpath)
#         for h in f:
#             old_start = getattr(h, "source_start", None)
#             if isinstance(old_start, int):
#                 lines.add((fpath, old_start))

#             func_name: Optional[str] = None
#             for header_line in _iter_hunk_header_lines(h):
#                 func_name = _extract_name_from_header(header_line)
#                 if func_name:
#                     break
#             if func_name:
#                 funcs[fpath].add(func_name)

#             details.append({
#                 "file": fpath,
#                 "old_start": old_start,
#                 "new_start": getattr(h, "target_start", None),
#                 "func": func_name
#             })

#     return files, lines, funcs, details

# def load_official_oracle(split: str = "test"):
#     ds = load_dataset("princeton-nlp/SWE-bench_Lite_oracle", split=split)
#     official = {}
#     for row in ds:
#         iid = row.get("instance_id")
#         patch_text = extract_patch_text(row.get("patch") or "")
#         official[iid] = parse_patch(patch_text)
#     return official

# def load_generated_jsonl(path: str):
#     generated = {}
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             if not line.strip():
#                 continue
#             try:
#                 obj = json.loads(line)
#             except json.JSONDecodeError:
#                 continue
#             iid = obj.get("instance_id") or obj.get("id") or obj.get("issue_id")
#             if not iid:
#                 continue
#             raw = obj.get("model_patch") or obj.get("full_output") or obj.get("patch") or obj.get("s")
#             if not raw:
#                 continue
#             patch_text = extract_patch_text(raw)
#             generated[iid] = parse_patch(patch_text)
#     return generated

# def choose_example_match(intersection_set):
#     try:
#         return next(iter(intersection_set))
#     except StopIteration:
#         return None

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--gen_jsonl", type=str, required=True,
#                         help="Path to your generated jsonl file")
#     parser.add_argument("--split", type=str, default="test",
#                         help="SWE-bench_Lite_oracle split (default: test)")
#     parser.add_argument("--out_csv", type=str, default="compare_patch_alignment.csv",
#                         help="Output CSV filename")
#     args = parser.parse_args()

#     print("Loading official oracle patches ...")
#     official = load_official_oracle(split=args.split)

#     print("Loading your generated patches ...")
#     generated = load_generated_jsonl(args.gen_jsonl)

#     rows = []
#     all_iids = sorted(set(generated.keys()) & set(official.keys()))
#     missing_in_official = sorted(set(generated.keys()) - set(official.keys()))
#     missing_in_gen = sorted(set(official.keys()) - set(generated.keys()))

#     if missing_in_official:
#         print(f"[Warning] {len(missing_in_official)} instance(s) only in your JSONL but not in official split.")

#     for iid in all_iids:
#         g_files, g_lines, g_funcs, _ = generated[iid]
#         o_files, o_lines, o_funcs, _ = official[iid]

#         same_file = len(g_files & o_files) > 0
#         same_line = len(g_lines & o_lines) > 0

#         common_files = g_files & o_files
#         func_match = False
#         example_func = None
#         found_any_name = False
#         for f in common_files:
#             gf = g_funcs.get(f, set())
#             of = o_funcs.get(f, set())
#             if gf or of:
#                 found_any_name = True
#             inter = gf & of
#             if inter:
#                 func_match = True
#                 example_func = next(iter(inter))
#                 break

#         function_status = (
#             "match" if func_match else
#             ("unknown" if not found_any_name else "mismatch")
#         )

#         example_file = choose_example_match(common_files)
#         example_line = choose_example_match(g_lines & o_lines)

#         rows.append({
#             "instance_id": iid,
#             "same_file": same_file,
#             "same_function": func_match,
#             "function_status": function_status,  # match / mismatch / unknown
#             "same_line": same_line,
#             "example_common_file": example_file or "",
#             "example_common_function": example_func or "",
#             "example_common_line": f"{example_line}" if example_line else "",
#             "g_files": ";".join(sorted(g_files)) or "",
#             "o_files": ";".join(sorted(o_files)) or "",
#         })

#     df = pd.DataFrame(rows).sort_values(
#         ["same_file", "same_function", "same_line", "instance_id"],
#         ascending=[False, False, False, True]
#     )
#     print("\n=== Summary (top 30 rows) ===")
#     print(df.head(30).to_string(index=False))

#     df.to_csv(args.out_csv, index=False, encoding="utf-8")
#     print(f"\nSaved CSV -> {args.out_csv}")

#     if missing_in_gen:
#         print(f"[Info] {len(missing_in_gen)} instance(s) only in official split.")

# if __name__ == "__main__":
#     main()
