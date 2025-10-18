"""
Custom SWE environment helpers for the mta package.

`ConfigurableSWEEnv` extends the stock mta SWEEnv so that auxiliary kwargs
such as scaffold configuration or backend options can be supplied alongside
the raw dataset entry without mutating the entry itself.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from mta.environments.swe import SWEEnv as _BaseSWEEnv

ENV_KWARGS_KEY = "__env_kwargs__"
ENTRY_KEY = "entry"


class ConfigurableSWEEnv(_BaseSWEEnv):
    """Thin wrapper that respects the `__env_kwargs__` convention in tasks."""

    @staticmethod
    def from_dict(extra_info: dict | str) -> "ConfigurableSWEEnv":
        """
        Create an environment instance from configuration metadata.

        The metadata accepts either:
          * a raw dataset entry (same as the base class), or
          * a dictionary containing an `entry` key plus an optional
            `__env_kwargs__` dictionary that is forwarded as keyword arguments.
        """
        if isinstance(extra_info, str):
            metadata = json.loads(extra_info)
        else:
            metadata = copy.deepcopy(extra_info)

        env_kwargs: dict[str, Any] = metadata.pop(ENV_KWARGS_KEY, {})
        entry = metadata.pop(ENTRY_KEY, metadata)

        return ConfigurableSWEEnv(entry=entry, **env_kwargs)
