import importlib
import logging
import multiprocessing as mp
from functools import lru_cache

import gymnasium as gym

from mta.environments.base import BaseEnv

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _try_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        logger.debug("Optional BrowserGym module '%s' could not be imported.", module_name)
        return False


def _maybe_register_webarena(env_id: str) -> None:
    if "webarena" not in env_id:
        return

    # Try the known BrowserGym registration entry points.
    registered = False
    for module_name in (
        "browsergym.webarena",  # v0.2+
        "browsergym.webarena.envs",  # legacy layouts
        "browsergym.benchmarks.webarena",  # experimental namespace
        "browsergym_async.webarena",  # async namespace if packaged separately
        "browsergym_async.webarena.envs",
    ):
        if _try_import(module_name):
            registered = True

    if not registered:
        logger.warning(
            "None of the expected BrowserGym WebArena entry points could be imported. "
            "Ensure the BrowserGym WebArena extras are installed."
        )
        return

    # BrowserGym sometimes registers versioned IDs such as 'browsergym/webarena.9'.
    # If the requested env_id is missing but a versioned variant exists, register an alias.
    try:
        from gymnasium.envs.registration import register, registry
    except Exception as exc:  # pragma: no cover - defensive for missing gymnasium
        logger.debug("Unable to access gymnasium registry for aliasing: %s", exc)
        return

    if env_id in registry:
        return

    alias_target = None
    for spec_id in list(registry.keys()):
        if spec_id.startswith("browsergym/webarena"):
            alias_target = spec_id
            if spec_id == env_id:
                alias_target = env_id
            break

    if not alias_target or alias_target not in registry:
        logger.debug("No BrowserGym WebArena specs found to alias for %s", env_id)
        return

    if env_id == alias_target:
        return

    spec = registry[alias_target]
    try:
        register_kwargs = {
            "id": env_id,
            "entry_point": spec.entry_point,
            "kwargs": dict(spec.kwargs or {}),
            "reward_threshold": spec.reward_threshold,
            "max_episode_steps": spec.max_episode_steps,
            "nondeterministic": spec.nondeterministic,
            "order_enforce": getattr(spec, "order_enforce", True),
            "autoreset": getattr(spec, "autoreset", False),
            "disable_env_checker": getattr(spec, "disable_env_checker", False),
        }
        max_episode_seconds = getattr(spec, "max_episode_seconds", None)
        if max_episode_seconds is not None:
            register_kwargs["max_episode_seconds"] = max_episode_seconds
        register(**register_kwargs)
        logger.info("Registered BrowserGym WebArena alias '%s' -> '%s'", env_id, alias_target)
    except Exception as exc:  # pragma: no cover - registry collisions
        logger.debug("Failed to register alias %s -> %s: %s", env_id, alias_target, exc)

class BrowserGymEnv(BaseEnv):
    def __init__(self, env_id="browsergym/openended", task=None, **env_kwargs):
        self.parent_conn, self.child_conn = mp.Pipe()
        timeout = env_kwargs.pop("timeout", None)
        self.process = mp.Process(target=self._worker, args=(self.child_conn, env_id, task, env_kwargs))
        self.timeout = None if timeout is None else timeout / 1000 if timeout > 10 else timeout
        self.process.start()

    def _worker(self, conn, env_id, task, env_kwargs):
        # Import browsergym modules in the worker process to register environments
        try:
            import browsergym.miniwob  # noqa: F401
        except ImportError:
            pass
        _maybe_register_webarena(env_id)

        env = (
            gym.make(
                env_id,
                task_kwargs=task,
                **env_kwargs,
                browser_args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-application-cache",
                    "--disk-cache-size=1",
                    "--media-cache-size=1",
                    "--disable-cache",
                    "--disable-gpu",
                    "--disable-software-rasterizer",
                    "--incognito",
                ],
                user_data_dir=None,  # Forces incognito
            )
            if task
            else gym.make(env_id, **env_kwargs)
        )
        try:
            while True:
                cmd, data = conn.recv()
                if cmd == "reset":
                    obs = env.reset()
                    conn.send(obs)
                elif cmd == "step":
                    action = data
                    obs, reward, terminated, truncated, extra_info = env.step(action)
                    conn.send((obs, reward, terminated or truncated, extra_info))
                elif cmd == "close":
                    env.close()
                    conn.close()
                    break
        except EOFError:
            env.close()

    def reset(self):
        self.parent_conn.send(("reset", None))
        if self.timeout is not None:
            if not self.parent_conn.poll(self.timeout):
                raise TimeoutError(f"Timeout after {self.timeout} seconds waiting for response.")
        return self.parent_conn.recv()

    def step(self, action):
        self.parent_conn.send(("step", action))
        if self.timeout is not None:
            if not self.parent_conn.poll(self.timeout):
                raise TimeoutError(f"Timeout after {self.timeout} seconds waiting for response.")
        return self.parent_conn.recv()

    def close(self):
        self.parent_conn.send(("close", None))
        self.process.join(60 * 2)
        if self.process.is_alive():
            print(f"Process still alive after {self.timeout} seconds. Killing it.")
            self.process.terminate()
            self.process.join()

    @staticmethod
    def from_dict(extra_info: dict) -> "BrowserGymEnv":
        env_kwargs = extra_info.copy()
        env_id = env_kwargs.pop("env_id")
        task = env_kwargs.pop("task", None)
        return BrowserGymEnv(env_id=env_id, task=task, **env_kwargs)

    @staticmethod
    def is_multithread_safe() -> bool:
        return True
