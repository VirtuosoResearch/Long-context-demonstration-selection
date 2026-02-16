#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


CMD_RE = re.compile(r"^\+ python\b")
RANK_LINE_RE = re.compile(r"^\s*(\d{3})\s+\|\s+delta=.*\|\s+label=(.+)$")


def get_command_section(lines: list[str], command_index: int) -> tuple[list[str], int]:
    cmd_indices = [i for i, line in enumerate(lines) if CMD_RE.match(line)]
    if not cmd_indices:
        raise ValueError("No '+ python' command found in log.")

    if command_index < 0:
        command_index = len(cmd_indices) + command_index
    if command_index < 0 or command_index >= len(cmd_indices):
        raise IndexError(
            f"command-index out of range: {command_index}. valid=[0, {len(cmd_indices) - 1}]"
        )

    start = cmd_indices[command_index]
    end = cmd_indices[command_index + 1] if command_index + 1 < len(cmd_indices) else len(lines)
    return lines[start:end], start + 1


def parse_topk_python_ratio(section: list[str], top_k: int) -> dict:
    blocks = 0
    python_hits = 0
    total_slots = 0
    per_task_python_counts: list[int] = []

    i = 0
    while i < len(section):
        line = section[i]
        if "Demonstration-level LOO attribution:" not in line:
            i += 1
            continue

        blocks += 1
        j = i + 1
        taken = 0
        block_python = 0
        while j < len(section) and taken < top_k:
            s = section[j].strip()
            if not s:
                j += 1
                continue
            m = RANK_LINE_RE.match(s)
            if not m:
                break

            rank = int(m.group(1))
            label = m.group(2)
            if rank <= top_k:
                taken += 1
                total_slots += 1
                label_l = label.lower()
                if "_python_" in label_l or "python/" in label_l:
                    python_hits += 1
                    block_python += 1
            j += 1

        per_task_python_counts.append(block_python)
        i = j

    ratio = (python_hits / total_slots) if total_slots else 0.0
    return {
        "blocks": blocks,
        "python_hits": python_hits,
        "total_slots": total_slots,
        "ratio": ratio,
        "per_task_python_counts": per_task_python_counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute Python ratio in top-k demonstration attribution for one '+ python' command."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path(
            "/home/michael/project/MTL-SWE-agents/src/humanevalpack/cmd_results/eval_python_loo_2_3_5_7_3.2_3.1.1_0.log"
        ),
        help="Path to log file.",
    )
    parser.add_argument(
        "--command-index",
        type=int,
        default=5,
        help="Which '+ python' command block to parse (0-based). 5 means the 6th command.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top k demonstrations per task block.")
    args = parser.parse_args()

    lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
    section, command_start_line = get_command_section(lines, args.command_index)
    stats = parse_topk_python_ratio(section, args.top_k)

    print(f"log: {args.log}")
    print(f"command_index(0-based): {args.command_index}")
    print(f"command_start_line: {command_start_line}")
    print(f"blocks: {stats['blocks']}")
    print(f"top_k: {args.top_k}")
    print(f"python_hits: {stats['python_hits']}")
    print(f"total_slots: {stats['total_slots']}")
    print(f"python_ratio: {stats['ratio']:.6f} ({stats['ratio'] * 100:.2f}%)")


if __name__ == "__main__":
    main()
