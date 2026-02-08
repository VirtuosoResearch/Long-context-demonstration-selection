
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Callable


Action = tuple[str, int]


@dataclass
class AgentState:
    observation: dict
    task_id: str | None = None
    history: list[dict] = field(default_factory=list)
    selected_edges: list[int] = field(default_factory=list)
    known_endpoints: dict[int, tuple[int, int]] = field(default_factory=dict)
    steps: int = 0


class POMDPMSTAgent:
    def __init__(
        self,
        llm: Callable[[str], str] | None = None,
        seed: int = 7,
        trace: list[dict] | None = None,
        print_io: bool = False,
        trajectory_examples: list[str] | None = None,
    ) -> None:
        self.llm = llm
        self.rng = random.Random(seed)
        self.trace = trace
        self.print_io = print_io
        self.last_output: str | None = None
        self.trajectory_examples = trajectory_examples or []

    def action(self, state: AgentState) -> Action:
        self._log_step_start(state)
        prompt = self._build_prompt(state)
        llm_output = ""
        parsed: Action | None = None
        max_retries = 3
        for attempt in range(max_retries + 1):
            if attempt == 0:
                llm_output = self.llm(prompt) if self.llm is not None else ""
            else:
                allowed_actions = self._allowed_actions(state)
                forbidden_actions = self._history_actions(state)
                retry_prompt = (
                    prompt
                    + "\n\nYour previous action repeats an earlier action. "
                    + "Choose a DIFFERENT action that is NOT in the history. "
                    + "Return only one action using the required format.\n"
                    + f"Forbidden actions: {sorted(forbidden_actions)}\n"
                    + f"Allowed actions (choose one): {sorted(allowed_actions)}"
                )
                llm_output = self.llm(retry_prompt) if self.llm is not None else llm_output

            parsed = self._parse_action_with_forbidden(
                llm_output, forbidden=self._history_actions(state)
            )
            if parsed is not None:
                break

        self.last_output = llm_output
        chosen = parsed if parsed is not None else self._fallback_action(state)
        if self.trace is not None:
            self.trace.append(
                {
                    "task_id": state.task_id,
                    "step": state.steps,
                    "prompt": prompt,
                    "llm_output": llm_output,
                    "parsed_action": parsed,
                    "chosen_action": chosen,
                }
            )
        self._log_llm(prompt, llm_output, parsed, chosen)
        if parsed is not None:
            return parsed
        return chosen

    def log_transition(self, info: dict) -> None:
        if not self.print_io:
            return
        print("TRANSITION:")
        print(json.dumps(info, ensure_ascii=True))

    def log_done(self, reward: float, state: AgentState) -> None:
        if not self.print_io:
            return
        print("DONE:")
        print(json.dumps({"reward": reward, "steps": state.steps}, ensure_ascii=True))

    def _log_step_start(self, state: AgentState) -> None:
        if not self.print_io:
            return
        print(f"\n=== task {state.task_id} step {state.steps} ===")
        print("STATE:")
        print(
            json.dumps(
                {
                    "observation": state.observation,
                    "history": state.history,
                    "selected_edges": state.selected_edges,
                    "known_endpoints": {str(k): list(v) for k, v in state.known_endpoints.items()},
                },
                ensure_ascii=True,
            )
        )

    def _log_llm(
        self,
        prompt: str,
        llm_output: str,
        parsed: Action | None,
        chosen: Action,
    ) -> None:
        if not self.print_io:
            return
        print("PROMPT:")
        print(prompt)
        print("OUTPUT:")
        print(llm_output)
        print(f"PARSED: {parsed}")
        print(f"ACTION: {chosen}")

    def _build_prompt(self, state: AgentState) -> str:
        obs = state.observation
        known_endpoints = {str(k): list(v) for k, v in state.known_endpoints.items()}
        lines = [
            "You are solving a hidden-graph MST task.",
            "Goal: build a minimum spanning tree by selecting edges; endpoints are hidden unless queried.",
            "Current state:",
            f"- Nodes: {obs['num_nodes']}",
            f"- Edges: {obs['num_edges']}",
            f"- Edge weights (by index 0..{obs['num_edges'] - 1}): {obs['edge_weights']}",
            f"- Selected edges so far (by index): {state.selected_edges}",
            f"- Known endpoints (edge_index -> [u, v]): {known_endpoints}",
            "- Note: any edge index not listed in known endpoints is still unknown.",
            f"- History (actions/results): {state.history}",
            f"- Previous step output: {self.last_output}",
        ]
        if self.trajectory_examples:
            lines.append("Trajectory demonstrations:")
            for idx, example in enumerate(self.trajectory_examples, start=1):
                lines.append(f"{idx}. {example}")
        lines += [
            "Rule: You mustn't repeat an action that already appears in the history!!!",
            "Let's think step by step. Show your thinking process in the response and include the action you choose in 500 tokens. You should go through all previous steps and actions to make sure you don't repeat an action that already appears in the history.",
            "Choose one action:",
            "Format: Action: query_edge(<idx>) or Action: select_edge(<idx>)",
            "- query_edge(<idx>)",
            "- select_edge(<idx>)",
            "Include one action somewhere in your response; extra text is allowed.",
        ]
        return "\n".join(lines)

    def _parse_action(self, text: str) -> Action | None:
        if not text:
            return None
        return self._parse_action_with_forbidden(text, forbidden=set())

    def _parse_action_with_forbidden(
        self, text: str, *, forbidden: set[Action]
    ) -> Action | None:
        actions = self._parse_actions(text)
        for action in reversed(actions):
            if action not in forbidden:
                return action
        return None

    def _parse_actions(self, text: str) -> list[Action]:
        if not text:
            return []
        action_pattern = r"(query_edge|select_edge)\s*(?:\(\s*(\d+)\s*\)|\[\s*(\d+)\s*\]|(\d+))"
        tagged_pattern = rf"Action:\s*{action_pattern}"
        matches = list(re.finditer(tagged_pattern, text, flags=re.IGNORECASE))
        if not matches:
            matches = list(re.finditer(action_pattern, text))
        actions: list[Action] = []
        for match in matches:
            action_type = match.group(1)
            edge_idx = int(next(g for g in match.groups()[1:] if g is not None))
            actions.append((action_type, edge_idx))
        return actions

    def _action_in_history(self, state: AgentState, action: Action) -> bool:
        for entry in state.history:
            hist_action = entry.get("action")
            if isinstance(hist_action, list):
                hist_action = tuple(hist_action)
            if hist_action == action:
                return True
        return False

    def _history_actions(self, state: AgentState) -> set[Action]:
        actions: set[Action] = set()
        for entry in state.history:
            hist_action = entry.get("action")
            if isinstance(hist_action, list):
                hist_action = tuple(hist_action)
            if isinstance(hist_action, tuple) and len(hist_action) == 2:
                actions.add(hist_action)
        return actions

    def _allowed_actions(self, state: AgentState) -> set[Action]:
        num_edges = state.observation["num_edges"]
        actions: set[Action] = set()
        for idx in range(num_edges):
            actions.add(("query_edge", idx))
            actions.add(("select_edge", idx))
        return actions - self._history_actions(state)

    def _fallback_action(self, state: AgentState) -> Action:
        num_edges = state.observation["num_edges"]
        unknown_edges = [i for i in range(num_edges) if i not in state.known_endpoints]
        if unknown_edges:
            return "query_edge", self.rng.choice(unknown_edges)
        return "select_edge", self.rng.randrange(num_edges)


class POMDPMSTEnvironment:
    def __init__(self, entry: dict) -> None:
        self.entry = entry
        self.num_nodes = entry["observation"]["num_nodes"]
        self.num_edges = entry["observation"]["num_edges"]
        self.edge_endpoints = [tuple(x) for x in entry["hidden_graph"]["edge_endpoints"]]
        self.edge_weights = entry["observation"]["edge_weights"]
        self.mst_weight = entry["oracle"]["mst_weight"]

    def initial_state(self) -> AgentState:
        return AgentState(task_id=self.entry.get("task_id"), observation=self.entry["observation"])

    def state(self, state: AgentState) -> dict:
        return {
            "observation": state.observation,
            "history": state.history,
            "selected_edges": state.selected_edges,
            "known_endpoints": {str(k): list(v) for k, v in state.known_endpoints.items()},
            "steps": state.steps,
        }

    def transition(self, state: AgentState, action: Action) -> tuple[AgentState, dict]:
        action_type, edge_idx = action
        state.steps += 1

        if edge_idx < 0 or edge_idx >= self.num_edges:
            info = {"ok": False, "reason": "edge_index_out_of_range"}
            state.history.append({"action": action, "result": info})
            return state, info

        if action_type == "query_edge":
            endpoints = self.edge_endpoints[edge_idx]
            state.known_endpoints[edge_idx] = endpoints
            info = {"ok": True, "endpoints": endpoints}
            state.history.append({"action": action, "result": info})
            return state, info

        if action_type == "select_edge":
            success, reason = self._try_add_edge(state, edge_idx)
            info = {"ok": success, "reason": reason}
            state.history.append({"action": action, "result": info})
            return state, info

        info = {"ok": False, "reason": "unknown_action"}
        state.history.append({"action": action, "result": info})
        return state, info

    def reward(self, state: AgentState, done: bool) -> float:
        if not done:
            return 0.0
        if len(state.selected_edges) != self.num_nodes - 1:
            return 0.0
        tree_weight = sum(self.edge_weights[i] for i in state.selected_edges)
        if tree_weight <= 0:
            return 0.0
        return float(self.mst_weight) / float(tree_weight)

    def is_done(self, state: AgentState, max_steps: int) -> bool:
        return len(state.selected_edges) == self.num_nodes - 1 or state.steps >= max_steps

    def _try_add_edge(self, state: AgentState, edge_idx: int) -> tuple[bool, str]:
        if edge_idx in state.selected_edges:
            return False, "already_selected"
        endpoints = self.edge_endpoints[edge_idx]
        if self._creates_cycle(state.selected_edges, endpoints):
            return False, "cycle"
        state.selected_edges.append(edge_idx)
        return True, "added"

    def _creates_cycle(self, selected_edges: list[int], new_edge: tuple[int, int]) -> bool:
        parent = list(range(self.num_nodes))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            parent[rb] = ra
            return True

        for edge_idx in selected_edges:
            u, v = self.edge_endpoints[edge_idx]
            union(u, v)

        u, v = new_edge
        return not union(u, v)


def run_episode(entry: dict, agent: POMDPMSTAgent, max_steps: int = 30) -> dict:
    env = POMDPMSTEnvironment(entry)
    state = env.initial_state()
    while True:
        action = agent.action(state)
        state, info = env.transition(state, action)
        agent.log_transition(info)
        done = env.is_done(state, max_steps)
        if done:
            reward = env.reward(state, done=True)
            agent.log_done(reward, state)
            return {
                "task_id": entry["task_id"],
                "selected_edges": state.selected_edges,
                "steps": state.steps,
                "reward": reward,
                "done": True,
            }
