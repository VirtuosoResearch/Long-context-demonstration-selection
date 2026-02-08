#!/usr/bin/env python
"""
Generate a synthetic dataset for a POMDP MST task on K5 graphs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Iterable

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


def generate_weights(rng: random.Random, num_edges: int, weight_min: int, weight_max: int) -> list[int]:
    return [rng.randint(weight_min, weight_max) for _ in range(num_edges)]


def kruskal_mst(
    num_nodes: int,
    edges: list[tuple[int, int]],
    weights: list[int],
) -> tuple[int, list[int]]:
    uf = UnionFind.create(num_nodes)
    indexed_edges = list(range(len(edges)))
    indexed_edges.sort(key=lambda i: (weights[i], edges[i][0], edges[i][1], i))

    total_weight = 0
    mst_edges: list[int] = []
    for edge_idx in indexed_edges:
        u, v = edges[edge_idx]
        if uf.union(u, v):
            mst_edges.append(edge_idx)
            total_weight += weights[edge_idx]
            if len(mst_edges) == num_nodes - 1:
                break
    return total_weight, mst_edges


def build_record(
    task_id: str,
    num_nodes: int,
    edges: list[tuple[int, int]],
    weights: list[int],
) -> dict:
    mst_weight, mst_edge_indices = kruskal_mst(num_nodes, edges, weights)
    mst_edge_endpoints = [[edges[i][0], edges[i][1]] for i in mst_edge_indices]

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
        "oracle": {
            "mst_weight": mst_weight,
            "mst_edges": mst_edge_indices,
            "mst_edge_endpoints": mst_edge_endpoints,
        },
        "metadata": {
            "edge_ordering": "lexicographic(u,v)",
            "action_space": {
                "select_edge": {"params": ["edge_index"]},
                "query_edge": {"params": ["edge_index"]},
            },
            "termination_signals": ["cycle", "tree_complete"],
            "metric": "score = optimal_weight / agent_weight",
        },
    }


def generate_records(
    *,
    num_samples: int,
    min_nodes: int,
    max_nodes: int,
    seed: int,
    weight_min: int,
    weight_max: int,
) -> Iterable[dict]:
    rng = random.Random(seed)
    for idx in range(num_samples):
        num_nodes = rng.randint(min_nodes, max_nodes)
        edges = complete_graph_edges(num_nodes)
        weights = generate_weights(rng, len(edges), weight_min, weight_max)
        yield build_record(f"pomdp_mst_{idx:06d}", num_nodes, edges, weights)


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
            " " * (indent + 2) + _format_value(item, indent + 2, compact_lists=True) for item in value
        ]
        return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"
    return json.dumps(value, ensure_ascii=True)


def format_dataset(records: list[dict]) -> str:
    return _format_value(records, 0, compact_lists=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic POMDP MST dataset for K5 graphs.")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of graph instances to generate.")
    parser.add_argument("--min-nodes", type=int, default=4, help="Minimum nodes per complete graph.")
    parser.add_argument("--max-nodes", type=int, default=5, help="Maximum nodes per complete graph.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--weight-min", type=int, default=1, help="Minimum integer edge weight.")
    parser.add_argument("--weight-max", type=int, default=9, help="Maximum integer edge weight.")
    parser.add_argument("--output-path", default="pomdp_mst_synthetic.json", help="Output JSON file path.")
    parser.add_argument("--log-level", default="INFO", help="Root logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    records = list(
        generate_records(
            num_samples=args.num_samples,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            seed=args.seed,
            weight_min=args.weight_min,
            weight_max=args.weight_max,
        )
    )

    output_path = os.path.abspath(args.output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(format_dataset(records))

    sample = records[0] if records else None
    logger.info("Wrote %s samples to %s", len(records), output_path)
    if sample:
        logger.info("Sample task_id: %s", sample["task_id"])


if __name__ == "__main__":
    main()
