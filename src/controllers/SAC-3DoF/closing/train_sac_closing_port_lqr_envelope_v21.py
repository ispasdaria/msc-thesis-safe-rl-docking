"""Train the V21 port-level LQR-envelope vanilla SAC pilot."""

import train_sac_closing_p3reach_from_scratch as training

from envs.docking3dof.closing_safety_exploration_env import (
    ClosingSafetyTerminalV21GymEnv,
)


def checkpoint_rank_position_first_until_success(summary):
    """Use closest approach while success is zero, then exact task quality."""
    success_rate = float(summary["success_rate"])
    second_metric = (
        -float(summary["mean_minimum_position_error_m"])
        if success_rate == 0.0
        else -float(summary["mean_final_max_target_ratio"])
    )
    return (
        success_rate,
        second_metric,
        -float(summary["mean_final_position_error_m"]),
        -float(summary["mean_final_max_target_ratio"]),
        -float(summary["mean_unsafe_fraction"]),
        float(summary["mean_return"]),
    )


training.EXPERIMENT_ENV_CLASS = ClosingSafetyTerminalV21GymEnv
training.EXPERIMENT_REWARD_SCHEME = (
    ClosingSafetyTerminalV21GymEnv.reward_scheme
)
training.CHECKPOINT_RANK_FUNCTION = (
    checkpoint_rank_position_first_until_success
)
training.CHECKPOINT_SELECTION_ORDER = [
    "highest success rate",
    "if success is zero: lowest mean minimum P3 position error",
    "if success is nonzero: lowest mean final max target ratio",
    "lowest mean final P3 position error",
    "lowest mean final max target ratio",
    "lowest mean unsafe fraction",
    "highest mean return",
]

_build_base_config = training.build_config


def build_v21_config(args, checkpoints, experiment_name):
    config = _build_base_config(args, checkpoints, experiment_name)
    env_class = ClosingSafetyTerminalV21GymEnv
    config.update(
        {
            "purpose": (
                "100k decision pilot for terminal vanilla SAC with a "
                "coupled docking-port LQR envelope"
            ),
            "lqr_policy_or_actions_used": False,
            "cbf_filter_used_during_training": False,
            "safety_violation_termination_enabled": True,
            "terminal_safety_penalty": env_class.terminal_safety_penalty,
            "success_bonus": env_class.reward_weights["success"],
            "persistent_goal_state_cost_enabled": False,
            "persistent_time_cost_enabled": False,
            "scalar_axial_braking_reward_enabled": False,
            "group_weights": {
                "position": env_class.position_group_weight,
                "alignment_rate": env_class.attitude_group_weight,
                "velocity": env_class.velocity_group_weight,
            },
            "observation_size": len(env_class.closing_observation_labels),
            "additional_observations": list(
                env_class.closing_observation_labels[-7:]
            ),
            "port_lqr_envelope": {
                "enabled": True,
                "tracked_components": [
                    "docking-port axial velocity",
                    "docking-port lateral velocity",
                    "attitude",
                    "angular rate",
                ],
                "design_scales": {
                    "axial_velocity_mps": (
                        env_class.lqr_axial_velocity_design_scale_mps
                    ),
                    "lateral_velocity_mps": (
                        env_class.lqr_lateral_velocity_design_scale_mps
                    ),
                    "attitude_deg": 2.0,
                    "rate_degs": 0.05,
                },
                "aggregation": "worst normalized component",
                "tracking_cost": "log1p(worst normalized component)",
                "progress_clip": env_class.port_lqr_progress_clip,
                "outside_envelope_penalty_max": (
                    env_class.reward_weights["port_lqr_envelope"]
                    * env_class.port_lqr_envelope_violation_ratio_clip
                ),
                "axial_reference": (
                    "LQR -0.08 m/s beyond 100 m, linear blend to P3 "
                    "velocity by 10 m, mirrored after overshoot"
                ),
            },
            "closing_velocity_warning": {
                "enabled": True,
                "type": "bounded quadratic warning-band occupancy cost",
                "port_speed_band_mps": [0.25, 0.35],
                "axial_speed_band_mps": [0.15, 0.25],
                "penalty_max": env_class.reward_weights[
                    "closing_velocity_warning"
                ],
            },
            "pilot_decision_checks": {
                "maximum_training_steps": args.total_timesteps,
                "required_improvements": [
                    "lower velocity safety-failure rate than V20",
                    "lower port/cone exit occupancy than V20",
                    "lower alignment error than V20",
                    "continued reduction in minimum P3 distance",
                ],
            },
        }
    )
    config["goal_metric"]["group_weights"] = {
        "position": env_class.position_group_weight,
        "attitude_rate": env_class.attitude_group_weight,
        "terminal_velocity": env_class.velocity_group_weight,
    }
    config["near_target_capture_shaping"]["group_weights"] = {
        "position": env_class.position_group_weight,
        "attitude_rate": env_class.attitude_group_weight,
        "terminal_velocity": env_class.velocity_group_weight,
    }
    config["terminal_conditions"] = [
        "P3 success",
        "hard safety violation",
        "failed-approach reset truncation",
        "time limit",
        "non-finite state",
    ]
    return config


training.build_config = build_v21_config


if __name__ == "__main__":
    training.main()
