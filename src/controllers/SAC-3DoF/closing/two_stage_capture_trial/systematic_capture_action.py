"""Action filtering for the isolated systematic capture experiment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class LowPassCaptureActionWrapper(gym.Wrapper):
    """Apply a first-order low-pass filter to normalized SAC commands.

    The wrapped Markov observation records the filtered action actually sent
    to the dynamics. Consequently the next filtered action is determined by
    the observed previous action and the policy's current command.
    """

    def __init__(self, env, alpha=0.25):
        super().__init__(env)
        self.alpha = float(alpha)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1].")
        if env.action_space.shape != (3,):
            raise ValueError("Capture action filtering requires a 3-action space.")
        self.executed_action = np.zeros(3, dtype=np.float32)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.executed_action = np.zeros(3, dtype=np.float32)
        info = dict(info)
        info["action_filter_alpha"] = self.alpha
        return observation, info

    def step(self, action):
        commanded_action = np.clip(
            np.asarray(action, dtype=np.float32), -1.0, 1.0
        )
        self.executed_action = (
            (1.0 - self.alpha) * self.executed_action
            + self.alpha * commanded_action
        ).astype(np.float32)
        observation, reward, terminated, truncated, info = self.env.step(
            self.executed_action
        )
        info = dict(info)
        info.update(
            {
                "action_filter_alpha": self.alpha,
                "commanded_normalized_action": commanded_action.copy(),
                "executed_normalized_action": self.executed_action.copy(),
            }
        )
        return observation, reward, terminated, truncated, info
