"""Isolated six-component observation variant for vanilla closing SAC."""

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


class ClosingDockingPortObservationEnv(Docking3DoFGymEnv):
    """Original closing environment with only the six port-relative values."""

    closing_observation_labels = (
        "rpx",
        "rpy",
        "vpx",
        "vpy",
        "psi_e",
        "omegaz",
    )

    def __init__(self, **kwargs):
        super().__init__(phase="closing", **kwargs)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )

    def _get_obs(self):
        q = self.core_env.get_derived_quantities(self.core_env.state)
        return np.asarray(
            [
                q["r_p"][0],
                q["r_p"][1],
                q["v_p"][0],
                q["v_p"][1],
                q["psi_e"],
                q["omega"],
            ],
            dtype=np.float32,
        )
