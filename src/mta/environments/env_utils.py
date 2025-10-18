from __future__ import annotations

from typing import Any

import numpy as np

from mta.agents.base import Trajectory


def compute_trajectory_reward(trajectory: Trajectory) -> Trajectory:
    if not trajectory:
        return trajectory
    trajectory_reward = np.sum([d.reward for d in trajectory.steps])
    trajectory.reward = trajectory_reward
    return trajectory


def compute_mc_return(trajectory: Trajectory, gamma: float = 0.95) -> Trajectory:
    G = 0.0
    for step in reversed(trajectory.steps):
        G = step.reward + gamma * G
        step.mc_return = G
    return trajectory
