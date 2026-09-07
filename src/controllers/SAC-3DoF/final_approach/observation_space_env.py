"""Final-approach environment variants for observation-space experiments."""

import os
import sys

import numpy as np
from gymnasium import spaces


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.gym_env import Docking3DoFGymEnv


OBSERVATION_MODES = ("com", "docking_port", "combined")


class FinalApproachObservationEnv(Docking3DoFGymEnv):
    """Keep the existing final-approach dynamics and reward, changing only obs."""

    _LABELS = {
        "com": ("rx", "ry", "vx", "vy", "psi_e", "omegaz"),
        "docking_port": ("rpx", "rpy", "vpx", "vpy", "psi_e", "omegaz"),
        "combined": (
            "rx",
            "ry",
            "vx",
            "vy",
            "rpx",
            "rpy",
            "vpx",
            "vpy",
            "psi_e",
            "omegaz",
        ),
    }

    def __init__(self, observation_mode="docking_port", **kwargs):
        if observation_mode not in OBSERVATION_MODES:
            choices = ", ".join(OBSERVATION_MODES)
            raise ValueError(f"observation_mode must be one of: {choices}.")

        super().__init__(phase="final", **kwargs)
        self.observation_mode = observation_mode
        observation_size = len(self._LABELS[observation_mode])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self.final_observation_labels = self._LABELS[observation_mode]

    def _get_obs(self):
        if self.observation_mode == "docking_port":
            return super()._get_obs()

        state = np.asarray(self.core_env.state, dtype=np.float64)
        q = self.core_env.get_derived_quantities(state)
        attitude_and_rate = np.array(
            [q["psi_e"], q["omega"]],
            dtype=np.float64,
        )
        com_state = np.concatenate((state[:4], attitude_and_rate))

        if self.observation_mode == "com":
            return com_state.astype(np.float32)

        port_state = np.concatenate(
            (
                np.asarray(q["r_p"], dtype=np.float64),
                np.asarray(q["v_p"], dtype=np.float64),
                attitude_and_rate,
            )
        )
        return np.concatenate((com_state[:4], port_state)).astype(np.float32)
