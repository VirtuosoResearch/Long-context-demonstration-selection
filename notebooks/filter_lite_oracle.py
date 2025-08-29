import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd

try:
    from datasets import load_dataset, load_from_disk
except Exception as e:
    raise SystemExit("Please `pip install datasets pandas` before running this script.")

CATEGORY_PRIORITY = ["bugfix", "compat", "feature", "test", "docs", "refactor"]

KEYWORDS: List[Tuple[re.Pattern, Tuple[str, int]]] = [
    # Bug fix (crashes, exceptions, wrong results, regressions)
    (re.compile(r"\b(bug|regression|crash|segfault|hang|freeze|deadlock)\b", re.I), ("bugfix", 5)),
    (re.compile(r"\b(error|exception|traceback|stack ?trace|assertion ?error)\b", re.I), ("bugfix", 4)),
    (re.compile(r"\b(fail(?:ed|s)?|does not|not working|broken|incorrect|wrong|mismatch|inconsistent)\b", re.I), ("bugfix", 3)),
    (re.compile(r"\b(no ?convergence|converge|overflow|underflow|nan|inf)\b", re.I), ("bugfix", 3)),
    (re.compile(r"\b(fix|hotfix|patch)\b", re.I), ("bugfix", 2)),

    # Feature / enhancement
    (re.compile(r"\b(feature|enhancement|proposal|rfc|roadmap)\b", re.I), ("feature", 4)),
    (re.compile(r"\b(add|support|implement|expose|enable|allow|option|flag|parameter|api)\b", re.I), ("feature", 2)),
    (re.compile(r"\b(convert|conversion|helper|utility|adapter)\b", re.I), ("feature", 2)),

    # Compatibility / dependency / deprecation
    (re.compile(r"\b(compat(?:ibility)?|deprecat(?:e|ion)|deprecated)\b", re.I), ("compat", 4)),
    (re.compile(r"\b(python\s*\d+(\.\d+)?|numpy|pandas|matplotlib|scipy|torch|tensorflow|jax)\b", re.I), ("compat", 2)),
    (re.compile(r"\b(version|bump|pin|requirements?|setup\.py|pyproject\.toml)\b", re.I), ("compat", 2)),

    # Docs
    (re.compile(r"\b(docs?|documentation|docstring|tutorial|guide|readme|typo|spelling)\b", re.I), ("docs", 3)),
    (re.compile(r"\b(example|usage)\b", re.I), ("docs", 1)),

    # Tests
    (re.compile(r"\b(test(?:ing)?|pytest|unittest|ci|coverage|flaky|fixture|xpass|xfail)\b", re.I), ("test", 3)),
    (re.compile(r"\b(assert|parametrize)\b", re.I), ("test", 1)),

    # Refactor / cleanup / style
    (re.compile(r"\b(refactor|cleanup|restructure|simplify|rename)\b", re.I), ("refactor", 3)),
    (re.compile(r"\b(format|lint|black|isort|pep8|flake8|ruff)\b", re.I), ("refactor", 2)),
    (re.compile(r"\b(dead code|remove unused)\b", re.I), ("refactor", 2)),
]

def text_from_example(ex: Dict) -> str:
    """Concatenate available text fields robustly across SWE-bench variants."""
    fields_priority = [
        "title",
        "problem_statement",
        "issue_title",
        "issue_body",
        "text",
        "desc",
        "body",
        "description",
    ]
    parts = []
    for f in fields_priority:
        v = ex.get(f, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            v = " ".join(map(str, v))
        parts.append(str(v))

    for f in ["repo", "instance_id"]:
        if f in ex and ex[f]:
            parts.append(str(ex[f]))
    return "\n".join(parts).strip()

def score_categories(text: str) -> Tuple[str, Dict[str, int], Dict[str, List[str]]]:
    """Return (best_category, score_map, hits_map)."""
    scores = defaultdict(int)
    hits = defaultdict(list)
    for rx, (cat, w) in KEYWORDS:
        for m in rx.findall(text):
            scores[cat] += w

            if isinstance(m, tuple):
                m = " ".join([mm for mm in m if isinstance(mm, str)])
            if not m:
                m = rx.pattern
            hits[cat].append(m if isinstance(m, str) else str(m))
    if not scores:
        return ("bugfix", {}, {}) 

    max_score = max(scores.values())
    candidates = [cat for cat, s in scores.items() if s == max_score]
    if len(candidates) == 1:
        return candidates[0], dict(scores), hits
    for cat in CATEGORY_PRIORITY:
        if cat in candidates:
            return cat, dict(scores), hits
    return candidates[0], dict(scores), hits

def load_swebench_lite(dataset_path: str = None, split: str = "test"):
    if dataset_path:
        ds = load_from_disk(dataset_path)
        if isinstance(ds, dict):
            ds = ds.get(split, None)
        return ds
    return load_dataset("princeton-nlp/SWE-bench_Lite", split=split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=str, default=None)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--out", type=str, default=".")
    ap.add_argument("--jsonl", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    ds = load_swebench_lite(args.dataset_path, args.split)
    if ds is None:
        raise SystemExit(f"Could not load dataset with split='{args.split}' from {args.dataset_path or 'HF hub'}.")

    rows = []
    cat_counter = Counter()

    for i, ex in enumerate(ds):
        text = text_from_example(ex)
        cat, scores, hits = score_categories(text)
        cat_counter[cat] += 1

        row = {
            "idx": i,
            "instance_id": ex.get("instance_id", ""),
            "repo": ex.get("repo", ""),
            "title": ex.get("title", ex.get("issue_title", "")),
            "category": cat,
            "scores": scores,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, "swebench_lite_classification.csv")
    df.to_csv(csv_path, index=False)

    summary_rows = [{"category": c, "count": n} for c, n in cat_counter.most_common()]
    df_sum = pd.DataFrame(summary_rows)
    sum_path = os.path.join(args.out, "swebench_lite_classification_summary.csv")
    df_sum.to_csv(sum_path, index=False)

    if args.jsonl:
        jsonl_path = os.path.join(args.out, "swebench_lite_classification.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"- Per-instance CSV: {csv_path}")
    print(f"- Summary CSV:      {sum_path}")
    if args.jsonl:
        print(f"- JSONL:            {jsonl_path}")


if __name__ == "__main__":
    main()
