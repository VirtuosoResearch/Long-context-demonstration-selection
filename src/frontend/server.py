#!/usr/bin/env python3
"""Lightweight HTTP server providing a web UI for running MTA agents."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass(frozen=True)
class TaskParam:
    """Describes one CLI flag that accepts a string value."""

    id: str
    flag: str
    label: str
    default: str
    placeholder: str | None = None


@dataclass
class TaskConfig:
    """Container describing how to execute a specific agent task."""

    label: str
    command: List[str]
    params: List[TaskParam]
    pythonpath: Path | None = None
    extra_env: Dict[str, str] | None = None
    static_flags: List[str] = field(default_factory=list)

    def build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.update(self.extra_env or {})
        if self.pythonpath:
            current = env.get("PYTHONPATH", "")
            new_paths = [str(self.pythonpath)]
            if current:
                new_paths.append(current)
            env["PYTHONPATH"] = os.pathsep.join(new_paths)
        return env


TASKS: Dict[str, TaskConfig] = {
    "swe": TaskConfig(
        label="SWE Agent",
        command=["python", "-m", "mta.scripts.run_sweagent_vllm"],
        params=[
            TaskParam("engine", "--engine", "Engine", "openai"),
            TaskParam("model", "--model", "Model", "gpt-4o-mini"),
            TaskParam(
                "base_url",
                "--base-url",
                "Base URL",
                "https://api.openai.com/v1",
            ),
            TaskParam("api_key", "--api-key", "API Key", "<YOUR_API_KEY>"),
            TaskParam("dataset", "--dataset", "Dataset", "SWE_Bench_Verified"),
            TaskParam("split", "--split", "Split", "test"),
            TaskParam("limit", "--limit", "Limit", "1"),
            TaskParam("n_parallel", "--n-parallel", "N Parallel", "4"),
            TaskParam(
                "max_response_length",
                "--max-response-length",
                "Max Response Length",
                "10000",
            ),
            TaskParam(
                "max_prompt_length",
                "--max-prompt-length",
                "Max Prompt Length",
                "20000",
            ),
            TaskParam("max_steps", "--max-steps", "Max Steps", "4"),
            TaskParam(
                "agent_scaffold",
                "--agent-scaffold",
                "Agent Scaffold",
                "sweagent",
            ),
            TaskParam(
                "env_backend", "--env-backend", "Env Backend", "docker"
            ),
        ],
        static_flags=["--use-fn-calling"],
    ),
    "humaneval": TaskConfig(
        label="HumanEval",
        command=["python", "-m", "mta.scripts.run_humaneval_agent"],
        params=[
            TaskParam("engine", "--engine", "Engine", "openai"),
            TaskParam("model", "--model", "Model", "gpt-4o-mini"),
            TaskParam(
                "base_url",
                "--base-url",
                "Base URL",
                "https://api.openai.com/v1",
            ),
            TaskParam("api_key", "--api-key", "API Key", "<YOUR_API_KEY>"),
            TaskParam("limit", "--limit", "Limit", "1"),
            TaskParam("n_parallel", "--n-parallel", "N Parallel", "4"),
            TaskParam(
                "max_response_length",
                "--max-response-length",
                "Max Response Length",
                "10000",
            ),
            TaskParam(
                "max_prompt_length",
                "--max-prompt-length",
                "Max Prompt Length",
                "10000",
            ),
            TaskParam("max_steps", "--max-steps", "Max Steps", "4"),
            TaskParam("temperature", "--temperature", "Temperature", "1"),
        ],
        pythonpath=REPO_ROOT / "human-eval",
    ),
    "webarena": TaskConfig(
        label="WebArena",
        command=["python", "-m", "mta.scripts.run_webarena_agent"],
        params=[
            TaskParam("engine", "--engine", "Engine", "openai"),
            TaskParam("model", "--model", "Model", "gpt-4o-mini"),
            TaskParam(
                "base_url",
                "--base-url",
                "Base URL",
                "https://api.openai.com/v1",
            ),
            TaskParam("api_key", "--api-key", "API Key", "<YOUR_API_KEY>"),
            TaskParam("env_id", "--env-id", "Env ID", "browsergym/webarena"),
            TaskParam("limit", "--limit", "Limit", "1"),
            TaskParam("n_parallel", "--n-parallel", "N Parallel", "4"),
            TaskParam(
                "max_response_length",
                "--max-response-length",
                "Max Response Length",
                "10000",
            ),
            TaskParam(
                "max_prompt_length",
                "--max-prompt-length",
                "Max Prompt Length",
                "20000",
            ),
            TaskParam("max_steps", "--max-steps", "Max Steps", "4"),
            TaskParam(
                "agent_scaffold",
                "--agent-scaffold",
                "Agent Scaffold",
                "sweagent",
            ),
            TaskParam(
                "env_backend", "--env-backend", "Env Backend", "docker"
            ),
        ],
        static_flags=["--use-fn-calling"],
    ),
}


class FrontendHandler(SimpleHTTPRequestHandler):
    """Serve static assets and expose a /run endpoint to launch agents."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 (base class signature)
        if self.path == "/tasks":
            self._send_json(
                {
                    name: {
                        "label": config.label,
                        "commandPrefix": config.command,
                        "params": [
                            {
                                "id": param.id,
                                "flag": param.flag,
                                "label": param.label,
                                "default": param.default,
                                "placeholder": param.placeholder or "",
                            }
                            for param in config.params
                        ],
                        "staticFlags": config.static_flags,
                    }
                    for name, config in TASKS.items()
                }
            )
            return

        if self.path in {"/", ""}:
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 (base class signature)
        if self.path != "/run":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_payload = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self.send_error(
                HTTPStatus.BAD_REQUEST, f"Invalid JSON payload: {exc}"
            )
            return

        task_name = payload.get("task")
        params_payload = payload.get("params", {})

        if not task_name or task_name not in TASKS:
            self.send_error(HTTPStatus.BAD_REQUEST, "Unsupported task")
            return

        if isinstance(params_payload, dict):
            command, env = self._build_command_from_dict(
                task_name, params_payload
            )
        else:
            try:
                params_list = shlex.split(str(params_payload), posix=True)
            except ValueError as exc:
                self._send_json(
                    {"error": f"Could not parse parameters: {exc}"}, status=400
                )
                return
            command, env = self._build_command_from_list(task_name, params_list)

        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

        response = {
            "command": " ".join(shlex.quote(part) for part in command),
            "exitCode": result.returncode,
            "output": result.stdout,
        }
        self._send_json(response)

    def _build_command_from_dict(
        self, task_name: str, params_dict: Dict[str, str]
    ) -> Tuple[List[str], Dict[str, str]]:
        config = TASKS[task_name]
        command = [*config.command]
        for param in config.params:
            value = params_dict.get(param.id, param.default)
            if value is None or str(value).strip() == "":
                continue
            command.extend([param.flag, str(value).strip()])
        command.extend(config.static_flags)
        env = config.build_env()
        return command, env

    def _build_command_from_list(
        self, task_name: str, params_list: List[str]
    ) -> Tuple[List[str], Dict[str, str]]:
        config = TASKS[task_name]
        command = [*config.command, *params_list]
        command.extend(config.static_flags)
        env = config.build_env()
        return command, env

    def _send_json(self, payload: Dict[str, object], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the MTA web frontend."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to listen on"
    )
    args = parser.parse_args()

    if not STATIC_DIR.exists():
        raise FileNotFoundError(
            f"Static assets directory not found: {STATIC_DIR}"
        )

    server = ThreadingHTTPServer((args.host, args.port), FrontendHandler)
    print(
        f"Serving frontend on http://{args.host}:{args.port} "
        f"(root={REPO_ROOT})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
