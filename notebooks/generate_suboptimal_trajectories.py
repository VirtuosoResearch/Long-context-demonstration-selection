#!/usr/bin/env python
"""
Generate suboptimal trajectories for POMDP MST tasks (fixed nodes=4).

Strategy:
- Query edges by decreasing weight.
- After each query, select the edge if it does not create a cycle.
Stops after max_steps or when a spanning tree is complete.
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


def build_suboptimal_trajectory(
    num_nodes: int,
    edges: list[tuple[int, int]],
    weights: list[int],
    max_steps: int,
) -> list[dict]:
    uf = UnionFind.create(num_nodes)
    ordered_edges = list(range(len(edges)))
    ordered_edges.sort(key=lambda i: (-weights[i], edges[i][0], edges[i][1], i))

    trajectory: list[dict] = []
    selected_edges: list[int] = []
    steps = 0

    for edge_idx in ordered_edges:
        if steps >= max_steps or len(selected_edges) == num_nodes - 1:
            break

        u, v = edges[edge_idx]
        trajectory.append(
            {"action": ["query_edge", edge_idx], "result": {"ok": True, "endpoints": [u, v]}}
        )
        steps += 1
        if steps >= max_steps:
            break

        if uf.union(u, v):
            selected_edges.append(edge_idx)
            trajectory.append(
                {"action": ["select_edge", edge_idx], "result": {"ok": True, "reason": "added"}}
            )
            steps += 1

    return trajectory


def kruskal_mst_weight(
    num_nodes: int, edges: list[tuple[int, int]], weights: list[int]
) -> int:
    uf = UnionFind.create(num_nodes)
    ordered_edges = list(range(len(edges)))
    ordered_edges.sort(key=lambda i: (weights[i], edges[i][0], edges[i][1], i))
    total = 0
    chosen = 0
    for edge_idx in ordered_edges:
        u, v = edges[edge_idx]
        if uf.union(u, v):
            total += weights[edge_idx]
            chosen += 1
            if chosen == num_nodes - 1:
                break
    return total


def tree_weight_from_trajectory(weights: list[int], trajectory: list[dict]) -> int | None:
    selected: list[int] = []
    for step in trajectory:
        action = step.get("action")
        result = step.get("result", {})
        if action and isinstance(action, list) and action[0] == "select_edge":
            if result.get("ok") is True:
                selected.append(action[1])
    if not selected:
        return None
    if len(selected) != len(set(selected)):
        return None
    if len(selected) == 0:
        return None
    return sum(weights[i] for i in selected)


def build_record(
    task_id: str,
    num_nodes: int,
    edges: list[tuple[int, int]],
    weights: list[int],
    rng: random.Random,
    mistake_rate: float,
    max_steps: int,
) -> dict:
    _ = mistake_rate
    optimal_weight = kruskal_mst_weight(num_nodes, edges, weights)
    trajectory = build_suboptimal_trajectory(num_nodes, edges, weights, max_steps)
    tree_weight = tree_weight_from_trajectory(weights, trajectory)
    if tree_weight is None or tree_weight <= 0:
        reward = 0.0
    else:
        reward = float(optimal_weight) / float(tree_weight)
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
        "trajectory": trajectory,
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
        description="Generate suboptimal trajectories for POMDP MST tasks."
    )
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples.")
    parser.add_argument("--num-nodes", type=int, default=4, help="Fixed number of nodes.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--weight-min", type=int, default=1, help="Minimum edge weight.")
    parser.add_argument("--weight-max", type=int, default=9, help="Maximum edge weight.")
    parser.add_argument(
        "--mistake-rate",
        type=float,
        default=0.9,
        help="Probability of using a non-optimal spanning tree.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Max actions per trajectory.",
    )
    parser.add_argument(
        "--output-path",
        default="pomdp_mst_suboptimal.json",
        help="Output JSON file path.",
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
        records.append(
            build_record(
                f"pomdp_mst_{idx:06d}",
                args.num_nodes,
                edges,
                weights,
                rng,
                args.mistake_rate,
                args.max_steps,
            )
        )

    output_path = os.path.abspath(args.output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(format_dataset(records))

    logger.info("Wrote %s suboptimal trajectories to %s", len(records), output_path)


if __name__ == "__main__":
    main()
