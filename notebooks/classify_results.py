#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize per-category success/failure for SWE-bench Lite runs.

Classification input can be EITHER:
  - CSV from classify_swebench_lite.py (must contain columns: instance_id, category)
  - JSONL from classify_swebench_lite.py (objects with fields: instance_id, category)

Logs root directory example:
  ./logs/run_evaluation/test_SWE-Llama-7b/princeton-nlp__SWE-Llama-7b
Each instance has a subfolder named by instance_id. If it contains report.json, it was run.
In that folder, run_instance.log includes a line like:
  "Results for <instance_id>: resolved: True" (or False)

Outputs (written to --out):
  - category_resolution_summary.csv: per-category counts of ran/resolved true/false/not_ran + accuracy
  - instance_resolution.csv: per-instance category and resolved status (for auditing)

Usage:
  python summarize_results_by_category.py \
      --cls /path/to/swebench_lite_classification.csv \
      --logs ./logs/run_evaluation/test_SWE-Llama-7b/princeton-nlp__SWE-Llama-7b \
      --out out_dir

  # or JSONL
  python summarize_results_by_category.py \
      --cls /path/to/swebench_lite_classification.jsonl \
      --logs ./logs/run_evaluation/test_SWE-Llama-7b/princeton-nlp__SWE-Llama-7b \
      --out out_dir
"""
import argparse
import json
import os
import re
from collections import defaultdict
from typing import Optional, Tuple

import pandas as pd

# Accept both "Result for" and "Results for"
RES_LINE_RE = re.compile(r"Results?\s+for\s+([^:]+):\s*resolved:\s*(True|False)", re.I)


def load_classification_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        # normalize columns
        if "instance_id" not in df.columns:
            raise SystemExit("CSV must contain column 'instance_id'.")
        if "category" not in df.columns:
            if "label" in df.columns:
                df = df.rename(columns={"label": "category"})
            else:
                raise SystemExit("CSV must contain column 'category' (or 'label').")
        return df[["instance_id", "category"]]
    elif ext in (".jsonl", ".json"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                iid = obj.get("instance_id") or obj.get("id") or obj.get("instance") or obj.get("meta", {}).get("instance_id")
                cat = obj.get("category") or obj.get("label")
                if iid:
                    rows.append({"instance_id": iid, "category": cat})
        if not rows:
            raise SystemExit("JSONL contained no usable rows with 'instance_id'.")
        return pd.DataFrame(rows)
    else:
        raise SystemExit(f"Unsupported classification file extension: {ext}. Use .csv or .jsonl")


def parse_run_status(instance_dir: str, expected_iid: str) -> Tuple[bool, Optional[bool]]:
    """Return (ran: bool, resolved: None/True/False) for the given expected instance_id."""
    report_path = os.path.join(instance_dir, "report.json")
    if not os.path.isfile(report_path):
        return False, None

    log_path = os.path.join(instance_dir, "run_instance.log")
    resolved = None
    if os.path.isfile(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = RES_LINE_RE.search(line)
                    if not m:
                        continue
                    iid_in_line = m.group(1).strip()
                    if iid_in_line == expected_iid:
                        val = m.group(2).strip().lower()
                        if val in ("true", "false"):
                            resolved = (val == "true")
        except Exception:
            pass

    if resolved is None:
        try:
            with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
                rep = json.load(f)
                # common keys in SWE-bench harness
                for k in ["resolved", "success", "passed", "is_resolved"]:
                    if k in rep:
                        resolved = bool(rep[k])
                        break
        except Exception:
            pass

    return True, resolved


def summarize(cls_path: str, logs_root: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df_cls = load_classification_any(cls_path)

    inst_rows = []
    for _, row in df_cls.iterrows():
        iid = str(row["instance_id"])
        cat = (row["category"] if pd.notna(row["category"]) else "unknown") or "unknown"
        inst_dir = os.path.join(logs_root, iid)
        ran, resolved = parse_run_status(inst_dir, iid)
        inst_rows.append({
            "instance_id": iid,
            "category": cat,
            "ran": ran,
            "resolved": resolved,
        })

    df_inst = pd.DataFrame(inst_rows)
    df_inst.to_csv(os.path.join(out_dir, "instance_resolution.csv"), index=False)

    agg = defaultdict(lambda: {"total": 0, "ran": 0, "resolved_true": 0, "resolved_false": 0, "not_ran": 0})
    for _, r in df_inst.iterrows():
        a = agg[r["category"]]
        a["total"] += 1
        if r["ran"]:
            a["ran"] += 1
            if r["resolved"] is True:
                a["resolved_true"] += 1
            elif r["resolved"] is False:
                a["resolved_false"] += 1
        else:
            a["not_ran"] += 1

    rows = []
    for cat, a in agg.items():
        acc = (a["resolved_true"] / a["ran"]) if a["ran"] else 0.0
        rows.append({
            "category": cat,
            "total_instances": a["total"],
            "ran_instances": a["ran"],
            "resolved_true": a["resolved_true"],
            "resolved_false": a["resolved_false"],
            "not_ran": a["not_ran"],
            "accuracy_on_ran": round(acc, 4),
        })

    df_sum = pd.DataFrame(rows).sort_values(by=["category"]).reset_index(drop=True)
    df_sum.to_csv(os.path.join(out_dir, "category_resolution_summary.csv"), index=False)

    print("Wrote:")
    print("-", os.path.join(out_dir, "instance_resolution.csv"))
    print("-", os.path.join(out_dir, "category_resolution_summary.csv"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cls", required=True, help="Path to classification CSV or JSONL.")
    p.add_argument("--logs", required=True, help="Path to logs root (contains per-instance subfolders).")
    p.add_argument("--out", default=".", help="Output directory.")
    args = p.parse_args()
    summarize(args.cls, args.logs, args.out)
