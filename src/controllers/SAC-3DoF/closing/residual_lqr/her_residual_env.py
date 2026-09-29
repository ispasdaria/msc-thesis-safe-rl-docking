"""Goal-conditioned HER environment with the safe V15 policy as baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.docking3dof.goal_conditioned_closing_env import (
    GoalConditionedClosingV3Env,
)
from envs.docking3dof.gym_env import Docking3DoFGymEnv


CLOSING_DIR = Path(__file__).resolve().parent
VANILLA_DIR = CLOSING_DIR.parent / "vanilla_sac"
SOURCE_STEM = (
    "sac_closing_p3augobs_rewardv15_optuna_trial16_"
    "400ksteps_seed0_buffer300k"
)
DEFAULT_SOURCE_MODEL = (
    VANILLA_DIR / "models" / "reward_shaping_trials" / f"{SOURCE_STEM}.zip"
)
DEFAULT_SOURCE_VECNORMALIZE = (
    VANILLA_DIR
    / "models"
    / "reward_shaping_trials"
    / f"{SOURCE_STEM}_vecnormalize.pkl"
)


class ResidualV15GoalConditionedClosingEnv(GoalConditionedClosingV3Env):
    """Execute ``V15 action + scale * HER residual action``.

    The frozen V15 policy remains in the control loop at every step.  The SAC
    actor therefore does not need to imitate a 12,000-step feedback law; it
    learns only a bounded correction in the same normalized action space.
    """

    reward_scheme = "goal_conditioned_p3_v4_residual_v15"

    def __init__(
        self,
        *,
        residual_scale=0.05,
        source_model_path=DEFAULT_SOURCE_MODEL,
        source_vecnormalize_path=DEFAULT_SOURCE_VECNORMALIZE,
        source_device="cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.residual_scale = float(residual_scale)
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError("residual_scale must lie in (0, 1].")
        self.source_model_path = Path(source_model_path)
        self.source_vecnormalize_path = Path(source_vecnormalize_path)
        if not self.source_model_path.is_file():
            raise FileNotFoundError(self.source_model_path)
        if not self.source_vecnormalize_path.is_file():
            raise FileNotFoundError(self.source_vecnormalize_path)

        self._source_dummy = DummyVecEnv(
            [
                lambda: Monitor(
                    Docking3DoFGymEnv(
                        phase="closing",
                        mode="testing",
                        dt=1.0,
                        max_time=12_000.0,
                        integrator="rk4",
                        normalize_action=True,
                        seed=999_998,
                        closing_reward_version="v15",
                    )
                )
            ]
        )
        self._source_vecnormalize = VecNormalize.load(
            str(self.source_vecnormalize_path),
            self._source_dummy,
        )
        self._source_vecnormalize.training = False
        self._source_vecnormalize.norm_reward = False
        self._source_model = SAC.load(
            str(self.source_model_path),
            device=source_device,
        )

    def source_action(self):
        base_observation = np.asarray(
            self.env._get_obs(),
            dtype=np.float32,
        ).reshape(1, -1)
        normalized = self._source_vecnormalize.normalize_obs(base_observation)
        action, _ = self._source_model.predict(
            normalized,
            deterministic=True,
        )
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def step(self, action):
        residual_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if residual_action.shape != (3,):
            raise ValueError("Residual action must have shape (3,).")
        source_action = self.source_action()
        executed_action = np.clip(
            source_action + self.residual_scale * residual_action,
            -1.0,
            1.0,
        ).astype(np.float32)
        observation, reward, terminated, truncated, info = super().step(
            executed_action
        )
        info = dict(info)
        info.update(
            {
                "residual_controller_enabled": True,
                "residual_scale": self.residual_scale,
                "source_normalized_action": source_action.copy(),
                "residual_normalized_action": residual_action.copy(),
                "executed_normalized_action": executed_action.copy(),
            }
        )
        return observation, reward, terminated, truncated, info

    def close(self):
        if getattr(self, "_source_dummy", None) is not None:
            self._source_dummy.close()
            self._source_dummy = None
        super().close()
