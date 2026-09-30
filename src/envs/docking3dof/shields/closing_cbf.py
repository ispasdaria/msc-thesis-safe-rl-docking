"""Closing Gym environment with the deployment CBF in the action path.

The SAC action remains the nominal action exposed to the learning algorithm.
Before every one-second simulator integration step, the action is converted to
physical units and projected by the closing CBF-QP.  The simulator, task
reward, and action-effort reward all receive the filtered action.  An optional
penalty on the normalized correction discourages the policy from relying on
the shield unnecessarily.
"""

from dataclasses import asdict, replace

import numpy as np

from cbf_3dof.closing_barriers import ClosingCBFConfig
from cbf_3dof.qp_filter import ClosingCBFFilter
from envs.docking3dof.gym_env import Docking3DoFGymEnv


CBF_TRAINING_VARIANTS = ("all", "full")


def build_closing_cbf_config(
    variant="all",
    *,
    solver_backend="osqp",
):
    """Return the frozen hard-filter configuration for a training variant."""
    if variant not in CBF_TRAINING_VARIANTS:
        raise ValueError(
            f"variant must be one of {CBF_TRAINING_VARIANTS}, got {variant!r}."
        )
    return replace(
        ClosingCBFConfig(),
        allow_slack=False,
        enable_cone=True,
        enable_port_speed=True,
        enable_axial_port_speed=True,
        enable_alignment=variant == "full",
        solver_backend=solver_backend,
    )


class ClosingCBFShieldedGymEnv(Docking3DoFGymEnv):
    """Closing environment whose transition dynamics include a CBF shield."""

    def __init__(
        self,
        *args,
        cbf_config=None,
        cbf_variant="all",
        intervention_penalty_weight=0.1,
        **kwargs,
    ):
        phase = kwargs.get("phase", args[0] if args else "final")
        if phase != "closing":
            raise ValueError(
                "ClosingCBFShieldedGymEnv requires phase='closing'."
            )
        if (
            not np.isfinite(intervention_penalty_weight)
            or intervention_penalty_weight < 0.0
        ):
            raise ValueError(
                "intervention_penalty_weight must be finite and nonnegative."
            )

        super().__init__(*args, **kwargs)
        if not self.normalize_action:
            raise ValueError(
                "CBF-aware SAC training requires normalized environment actions."
            )

        if cbf_config is None:
            cbf_config = build_closing_cbf_config(cbf_variant)
        self.cbf_config = cbf_config
        self.cbf_variant = str(cbf_variant)
        self.intervention_penalty_weight = float(
            intervention_penalty_weight
        )
        self.cbf_filter = ClosingCBFFilter(self.core_env, self.cbf_config)
        self._cbf_hold_records = None

    def cbf_metadata(self):
        """Return JSON-serializable shield settings for experiment records."""
        return {
            "variant": self.cbf_variant,
            "intervention_penalty_weight": self.intervention_penalty_weight,
            "penalty_definition": (
                "-weight * ||(u_filtered-u_nominal)/action_limits||_2^2"
            ),
            "config": asdict(self.cbf_config),
        }

    def _get_reset_info(self):
        info = super()._get_reset_info()
        info.update(
            {
                "cbf_filter_enabled": True,
                "cbf_variant": self.cbf_variant,
                "cbf_allow_slack": bool(self.cbf_config.allow_slack),
                "cbf_intervention_penalty_weight": (
                    self.intervention_penalty_weight
                ),
            }
        )
        return info

    def _step_once(self, action):
        nominal_normalized = np.asarray(action, dtype=np.float64)
        if nominal_normalized.shape != (3,):
            raise ValueError(
                "Nominal normalized action must have shape (3,), got "
                f"{nominal_normalized.shape}."
            )
        nominal_physical = self.normalized_action_to_physical(
            nominal_normalized
        )
        result = self.cbf_filter.filter(
            self.get_physical_state(),
            nominal_physical,
        )
        filtered_normalized = self.physical_action_to_normalized(
            result.action_filtered
        )

        obs, reward, terminated, truncated, info = super()._step_once(
            filtered_normalized
        )
        penalty = float(
            self.intervention_penalty_weight
            * result.intervention_scaled_norm**2
        )
        reward -= penalty

        components = dict(info["reward_components"])
        components["reward_cbf_intervention"] = -penalty
        components["reward_total"] = float(reward)
        info.update(
            {
                "reward_components": components,
                "cbf_filter_enabled": True,
                "cbf_variant": self.cbf_variant,
                "cbf_intervened": bool(result.intervened),
                "cbf_intervention_norm": float(result.intervention_norm),
                "cbf_intervention_scaled_norm": float(
                    result.intervention_scaled_norm
                ),
                "cbf_intervention_penalty": penalty,
                "cbf_qp_failed": bool(result.qp_failed),
                "cbf_solver_status": result.solver_status,
                "cbf_solver_backend": result.solver_backend,
                "cbf_barrier_values": dict(result.barrier_values),
                "cbf_residuals": dict(result.cbf_residuals),
                "cbf_slack_values": dict(result.slack_values),
                "nominal_normalized_action": nominal_normalized.astype(
                    np.float32
                ),
                "filtered_normalized_action": filtered_normalized.astype(
                    np.float32
                ),
                "nominal_physical_action": result.action_nominal.astype(
                    np.float32
                ),
                "filtered_physical_action": result.action_filtered.astype(
                    np.float32
                ),
            }
        )

        if self._cbf_hold_records is not None:
            self._cbf_hold_records.append(
                {
                    "intervened": bool(result.intervened),
                    "scaled_norm": float(result.intervention_scaled_norm),
                    "penalty": penalty,
                    "qp_failed": bool(result.qp_failed),
                }
            )
        return obs, float(reward), terminated, truncated, info

    def step(self, action):
        self._cbf_hold_records = []
        try:
            obs, reward, terminated, truncated, info = super().step(action)
            records = self._cbf_hold_records
        finally:
            self._cbf_hold_records = None

        count = len(records)
        intervention_count = sum(item["intervened"] for item in records)
        scaled_norms = [item["scaled_norm"] for item in records]
        info["cbf_intervention_count"] = int(intervention_count)
        info["cbf_intervention_fraction"] = float(
            intervention_count / max(count, 1)
        )
        info["cbf_qp_failure_count"] = int(
            sum(item["qp_failed"] for item in records)
        )
        info["cbf_intervention_penalty_sum"] = float(
            sum(item["penalty"] for item in records)
        )
        info["cbf_mean_intervention_scaled_norm"] = float(
            np.mean(scaled_norms) if scaled_norms else 0.0
        )
        info["cbf_max_intervention_scaled_norm"] = float(
            max(scaled_norms, default=0.0)
        )
        return obs, reward, terminated, truncated, info
