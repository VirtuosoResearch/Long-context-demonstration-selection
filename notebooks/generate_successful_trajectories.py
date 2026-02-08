#!/usr/bin/env python
"""
Generate successful trajectories for POMDP MST tasks (fixed nodes=4).

Strategy:
- Sort edges by weight (tie-break by endpoints, then index).
- For each edge in order:
  1) query_edge(edge_idx)
  2) if it does not create a cycle, select_edge(edge_idx)
Stop when MST has num_nodes - 1 edges.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UnionFind:
    parent: list[int]
    rank: list[int]

    @classmethod
    def create(cls, size: int) -> "UnionFind":
        return cls(parent=list(range(size)), rank=[0] * size)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1
        return True


def complete_graph_edges(num_nodes: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            edges.append((u, v))
    return edges


def generate_weights(
    rng: random.Random, num_edges: int, weight_min: int, weight_max: int
) -> list[int]:
    return [rng.randint(weight_min, weight_max) for _ in range(num_edges)]


def build_successful_trajectory(
    num_nodes: int, edges: list[tuple[int, int]], weights: list[int]
) -> list[dict]:
    uf = UnionFind.create(num_nodes)
    ordered_edges = list(range(len(edges)))
    ordered_edges.sort(key=lambda i: (weights[i], edges[i][0], edges[i][1], i))

    trajectory: list[dict] = []
    selected_edges: list[int] = []
    queried: set[int] = set()

    for edge_idx in ordered_edges:
        if len(selected_edges) == num_nodes - 1:
            break
        if edge_idx in queried:
            continue
        queried.add(edge_idx)
        u, v = edges[edge_idx]
        trajectory.append(
            {"action": ["query_edge", edge_idx], "result": {"ok": True, "endpoints": [u, v]}}
        )
        if uf.union(u, v):
            selected_edges.append(edge_idx)
            trajectory.append(
                {"action": ["select_edge", edge_idx], "result": {"ok": True, "reason": "added"}}
            )

    return trajectory


def build_record(task_id: str, num_nodes: int, weights: list[int]) -> dict:
    edges = complete_graph_edges(num_nodes)
    traj = build_successful_trajectory(num_nodes, edges, weights)
    reward = 1.0
    return {
        "task_id": task_id,
        "observation": {
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "edge_weights": weights,
        },
        "hidden_graph": {
            "edge_endpoints": [[u, v] for u, v in edges],
            "edge_weights": weights,
        },
        "trajectory": traj,
        "reward": reward,
    }


def _compact_list(values: list) -> str:
    return "[" + ",".join(_format_value(item, 0, compact_lists=True) for item in values) + "]"


def _format_value(value, indent: int, *, compact_lists: bool) -> str:
    if isinstance(value, dict):
        if not value:
            return "{}"
        pieces = []
        for key, val in value.items():
            key_str = json.dumps(key, ensure_ascii=True)
            val_str = _format_value(val, indent + 2, compact_lists=compact_lists)
            pieces.append(" " * (indent + 2) + f"{key_str}: {val_str}")
        return "{\n" + ",\n".join(pieces) + "\n" + " " * indent + "}"
    if isinstance(value, list):
        if compact_lists:
            return _compact_list(value)
        if not value:
            return "[]"
        items = [
            " " * (indent + 2) + _format_value(item, indent + 2, compact_lists=True)
            for item in value
        ]
        return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"
    return json.dumps(value, ensure_ascii=True)


def format_dataset(records: list[dict]) -> str:
    return _format_value(records, 0, compact_lists=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate successful trajectories for POMDP MST tasks."
    )
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples.")
    parser.add_argument("--num-nodes", type=int, default=4, help="Fixed number of nodes.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--weight-min", type=int, default=1, help="Minimum edge weight.")
    parser.add_argument("--weight-max", type=int, default=9, help="Maximum edge weight.")
    parser.add_argument(
        "--output-path", default="pomdp_mst_success.json", help="Output JSON file path."
    )
    parser.add_argument("--log-level", default="INFO", help="Root logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    rng = random.Random(args.seed)
    records: list[dict] = []

    edges = complete_graph_edges(args.num_nodes)
    for idx in range(args.num_samples):
        weights = generate_weights(rng, len(edges), args.weight_min, args.weight_max)
        records.append(build_record(f"pomdp_mst_{idx:06d}", args.num_nodes, weights))

    output_path = os.path.abspath(args.output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(format_dataset(records))

    logger.info("Wrote %s successful trajectories to %s", len(records), output_path)


if __name__ == "__main__":
    main()
