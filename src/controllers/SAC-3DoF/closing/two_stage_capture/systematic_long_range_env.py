"""Long-range side of the shared systematic handover contract."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from capture_safety_terminal_env import (
    CaptureSafetyTerminalCoreEnv,
    CaptureSafetyTerminalEnv,
)
from systematic_interface import DEFAULT_CONTRACT


class SystematicLongRangeCoreEnv(CaptureSafetyTerminalCoreEnv):
    """Require all six X_H component tolerances simultaneously."""

    def check_done(self, state):
        done, info = super().check_done(state)
        if info.get("success", False):
            info["terminal_set"] = "systematic_X_H"
            info["termination_reason"] = "success_systematic_X_H"
        return done, info


class SystematicLongRangeEnv(CaptureSafetyTerminalEnv):
    """P2-to-X_H environment with capture-consistent hard safety handling."""

    reward_scheme = "systematic_long_range_X_H_v1"
    negative_progress_clip = 0.50
    positive_progress_clip = 0.05
    progress_scale = 100.0
    state_cost_weight = 0.10
    warning_weight = 0.25
    action_weights = np.array([0.005, 0.005, 0.020], dtype=np.float64)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inherited_core = self.core_env
        self.core_env = SystematicLongRangeCoreEnv(
            dt=inherited_core.dt,
            max_time=self.max_time,
            integrator=inherited_core.integrator,
            phase="closing",
            p3_target_tolerance=DEFAULT_CONTRACT.handover_tolerance,
            seed=kwargs.get("seed", None),
        )

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        info = dict(info)
        info["success_set"] = "systematic_X_H_six_state"
        return observation, info

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        del physical_action, terminated
        capture = self._capture_cost(state)
        previous_cost = getattr(self, "previous_capture_cost", capture["cost"])
        raw_progress = float(previous_cost - capture["cost"])
        clipped_progress = float(
            np.clip(
                raw_progress,
                -self.negative_progress_clip,
                self.positive_progress_clip,
            )
        )
        self.previous_capture_cost = capture["cost"]
        bounded_state_cost = float(np.tanh(capture["cost"]))
        derived = self.core_env.get_derived_quantities(state)
        warning_depth = self._warning_depth(derived, info)
        action = np.asarray(normalized_action, dtype=np.float64)

        terms = {
            "reward_capture_progress": self.progress_scale * clipped_progress,
            "reward_state_cost": -self.state_cost_weight * bounded_state_cost,
            "reward_warning_margin": -self.warning_weight * warning_depth**2,
            "reward_action": -float(np.dot(self.action_weights, action**2)),
            "reward_success": self.success_bonus if info.get("success", False) else 0.0,
            "reward_safety_failure": (
                -self.safety_failure_penalty if info.get("failure", False) else 0.0
            ),
            "reward_timeout": -self.timeout_penalty if truncated else 0.0,
        }
        reward = float(sum(terms.values()))
        components = {
            "phase": self.phase,
            "reward_scheme": self.reward_scheme,
            "success_set": "systematic_X_H_six_state",
            "capture_cost": capture["cost"],
            "capture_cost_progress_raw": raw_progress,
            "capture_cost_progress_clipped": clipped_progress,
            "capture_normalized_error_max": float(np.max(capture["ratios"])),
            "warning_depth": warning_depth,
            **{name: float(value) for name, value in terms.items()},
            "reward_total": reward,
        }
        return reward, components


class SystematicLongRangeResetWrapper(gym.Wrapper):
    """Pair a sampled P2 start with an episode-specific six-state X_H goal."""

    def __init__(self, env, p2_mode="training", maximum_sampling_attempts=512):
        super().__init__(env)
        self.p2_mode = p2_mode
        self.maximum_sampling_attempts = int(maximum_sampling_attempts)
        if p2_mode not in ("nominal", "testing", "training"):
            raise ValueError("p2_mode must be nominal, testing, or training.")
        self.final_p3_reference = None
        self.handover_target = None

    @property
    def core_env(self):
        return self.env.unwrapped.core_env

    def reset(self, seed=None, options=None):
        if options and options.get("initial_state") is not None:
            return self.env.reset(seed=seed, options=options)

        self.env.reset(seed=seed, options={"mode": self.p2_mode})
        rng = self.core_env.rng
        for _ in range(self.maximum_sampling_attempts):
            p3_reference = DEFAULT_CONTRACT.sample_p3(rng)
            handover_target = DEFAULT_CONTRACT.intermediate_target(p3_reference)
            p2_state = self.core_env.sample_X_P2(self.p2_mode)
            if self.core_env.monitor_safety(p2_state)["unsafe"]:
                continue
            observation, info = self.env.reset(
                options={
                    "initial_state": p2_state,
                    # The closing observation and terminal predicate now point
                    # to X_H, not directly to P3.
                    "p3_target": handover_target,
                    "mode": self.p2_mode,
                }
            )
            if info.get("unsafe", False):
                continue
            self.final_p3_reference = p3_reference
            self.handover_target = handover_target
            info = dict(info)
            info.update(
                {
                    "long_range_reset_source": f"X_P2_{self.p2_mode}",
                    "final_p3_reference": p3_reference.copy(),
                    "handover_target": handover_target.copy(),
                    "handover_tolerance": DEFAULT_CONTRACT.handover_tolerance.copy(),
                }
            )
            return observation, info
        raise RuntimeError("Could not sample a non-unsafe P2/X_H pair.")

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["final_p3_reference"] = self.final_p3_reference.copy()
        info["handover_target"] = self.handover_target.copy()
        return observation, reward, terminated, truncated, info
