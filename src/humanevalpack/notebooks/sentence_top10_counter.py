#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import Counter
from pathlib import Path


def normalize_sentence(sentence: str) -> str:
    if re.match(r"^\s*>>>", sentence):
        return "sample"
    if re.match(r"^\s*def\b", sentence):
        return "function definition"
    return sentence


def parse_last_command_topk_sentences(log_path: Path, command_index: int = -1, top_k: int = 10):
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    cmd_indices = [i for i, line in enumerate(lines) if line.startswith("+ python")]
    if not cmd_indices:
        raise ValueError("No '+ python' command found in log.")

    start = cmd_indices[command_index]
    section = lines[start:]

    counter = Counter()
    blocks = 0
    i = 0
    while i < len(section):
        line = section[i]
        if "Sentence-level LOO attribution:" not in line:
            i += 1
            continue

        blocks += 1
        j = i + 1
        rank_count = 0
        while j < len(section):
            s = section[j].strip()
            if not s:
                j += 1
                continue
            if "Sentence-level LOO attribution:" in s:
                break
            if s.startswith("[") and "/" in s and "]" in s:
                break
            if "| delta=" in s and "| text=" in s:
                rank_count += 1
                if rank_count <= top_k:
                    text_part = s.split("| text=", 1)[1].strip()
                    try:
                        text = ast.literal_eval(text_part)
                    except Exception:
                        text = text_part
                    counter[normalize_sentence(text)] += 1
                j += 1
                continue
            if s.startswith("Top ") or s.startswith("Input text:") or s.startswith("Demonstration-level") or s.startswith("====="):
                break
            j += 1
        i = j

    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return {
        "blocks": blocks,
        "command_start_line": start + 1,
        "rows": sorted_items,
    }


def main():
    parser = argparse.ArgumentParser(description="Count sentence frequencies in top-k LOO attribution entries.")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path(
            "/home/michael/project/MTL-SWE-agents/src/humanevalpack/cmd_results/eval_python_loo_2_3_5_7_3.2_3.1.1_0.log"
        ),
        help="Path to log file.",
    )
    parser.add_argument("--command-index", type=int, default=-1, help="Which '+ python' command block to parse. -1 means last.")
    parser.add_argument("--top-k", type=int, default=10, help="Top k entries per sentence-level block.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path(
            "/home/michael/project/MTL-SWE-agents/src/humanevalpack/cmd_results/last_python_cmd_sentence_top10_counts_normalized_with_function_definition.csv"
        ),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    stats = parse_last_command_topk_sentences(args.log, args.command_index, args.top_k)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "count", "sentence"])
        for idx, (sentence, count) in enumerate(stats["rows"], start=1):
            writer.writerow([idx, count, sentence])

    print(f"log: {args.log}")
    print(f"command_start_line: {stats['command_start_line']}")
    print(f"blocks: {stats['blocks']}")
    print(f"unique_sentences: {len(stats['rows'])}")
    print(f"out_csv: {args.out_csv}")
    print("top10:")
    for sentence, count in stats["rows"][:10]:
        print(f"{count}\t{sentence!r}")


if __name__ == "__main__":
    main()
