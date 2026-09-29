"""Fixed physical observation scaling for the systematic capture task."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class FixedCaptureObservationScale(gym.ObservationWrapper):
    """Scale each physical observation component by a fixed contract value.

    The underlying closing observation is ordered as port state followed by
    error to the sampled P3 target. Fixed scaling prevents failed trajectories
    from expanding adaptive statistics and hiding the small P3 tolerances.
    """

    labels = (
        "port_rx",
        "port_ry",
        "port_vx",
        "port_vy",
        "attitude",
        "angular_rate",
        "error_rx",
        "error_ry",
        "error_vx",
        "error_vy",
        "error_attitude",
        "error_angular_rate",
    )
    scales = np.array(
        [
            500.0,
            50.0,
            0.15,
            0.15,
            np.deg2rad(4.0),
            np.deg2rad(0.20),
            150.0,
            10.0,
            0.08,
            0.03,
            np.deg2rad(3.0),
            np.deg2rad(0.15),
        ],
        dtype=np.float64,
    )
    clip = 10.0

    def __init__(self, env):
        super().__init__(env)
        if env.observation_space.shape != self.scales.shape:
            raise ValueError(
                "Fixed capture observation scaling requires a 12-vector."
            )
        self.observation_space = gym.spaces.Box(
            low=-self.clip,
            high=self.clip,
            shape=self.scales.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        scaled = np.asarray(observation, dtype=np.float64) / self.scales
        return np.clip(scaled, -self.clip, self.clip).astype(np.float32)

    @classmethod
    def as_dict(cls):
        return {
            "type": "fixed_physical_component_scaling",
            "labels": list(cls.labels),
            "scales": cls.scales.tolist(),
            "clip": cls.clip,
        }


class MarkovCaptureObservationScale(gym.Wrapper):
    """Fixed physical scaling with the preceding normalized action appended.

    The v9 reward penalizes action slew. Appending the previous action makes
    that reward a function of the current observation and selected action,
    instead of introducing hidden history into SAC's transition model.
    """

    physical_labels = FixedCaptureObservationScale.labels
    action_labels = ("previous_action_x", "previous_action_y", "previous_action_tau")
    labels = physical_labels + action_labels
    physical_scales = FixedCaptureObservationScale.scales
    clip = FixedCaptureObservationScale.clip

    def __init__(self, env):
        super().__init__(env)
        if env.observation_space.shape != self.physical_scales.shape:
            raise ValueError(
                "Markov capture observation requires a physical 12-vector."
            )
        if env.action_space.shape != (3,):
            raise ValueError("Markov capture observation requires a 3-action space.")
        self.previous_normalized_action = np.zeros(3, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-self.clip,
            high=self.clip,
            shape=(15,),
            dtype=np.float32,
        )

    def _observation(self, observation):
        physical = np.asarray(observation, dtype=np.float64) / self.physical_scales
        physical = np.clip(physical, -self.clip, self.clip).astype(np.float32)
        return np.concatenate((physical, self.previous_normalized_action))

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_normalized_action = np.zeros(3, dtype=np.float32)
        return self._observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.previous_normalized_action = np.clip(
            np.asarray(action, dtype=np.float32), -1.0, 1.0
        ).copy()
        return (
            self._observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    @classmethod
    def as_dict(cls):
        return {
            "type": "fixed_physical_scaling_with_previous_action",
            "labels": list(cls.labels),
            "physical_scales": cls.physical_scales.tolist(),
            "previous_action_scales": [1.0, 1.0, 1.0],
            "clip": cls.clip,
        }
