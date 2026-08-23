"""Run the V20 safety-terminal braking-envelope SAC pilot.

V20 corrects the return mismatch found in V19: hard safety failure receives a
-100 terminal reward, while persistent goal-state and time costs are disabled.
The policy remains vanilla SAC trained from scratch without a CBF filter, LQR
actions, demonstrations, or replay initialization.
"""

import train_sac_closing_p3reach_from_scratch as training

from envs.docking3dof.closing_safety_exploration_env import (
    ClosingSafetyTerminalV20GymEnv,
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


training.EXPERIMENT_ENV_CLASS = ClosingSafetyTerminalV20GymEnv
training.EXPERIMENT_REWARD_SCHEME = (
    ClosingSafetyTerminalV20GymEnv.reward_scheme
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


def build_v20_config(args, checkpoints, experiment_name):
    config = _build_base_config(args, checkpoints, experiment_name)
    env_class = ClosingSafetyTerminalV20GymEnv
    config.update(
        {
            "purpose": (
                "100k decision pilot for safety-terminal vanilla SAC with "
                "LQR-inspired braking-envelope progress"
            ),
            "prescribed_velocity_profile": True,
            "lqr_policy_or_actions_used": False,
            "cbf_filter_used_during_training": False,
            "safety_violation_termination_enabled": True,
            "terminal_safety_penalty": env_class.terminal_safety_penalty,
            "success_bonus": env_class.reward_weights["success"],
            "persistent_goal_state_cost_enabled": False,
            "persistent_time_cost_enabled": False,
            "retained_progress_terms": [
                "global grouped P3 progress",
                "bounded near-target capture progress",
                "bounded axial-position capture progress",
                "bounded LQR-reference braking progress",
            ],
            "observation_size": len(env_class.closing_observation_labels),
            "additional_observations": list(
                env_class.closing_observation_labels[-3:]
            ),
            "braking_envelope": {
                "enabled": True,
                "source": "successful tuned closing LQR design",
                "fast_axial_velocity_mps": (
                    env_class.lqr_fast_axial_velocity_mps
                ),
                "fast_distance_m": env_class.lqr_fast_distance_m,
                "slow_distance_m": env_class.lqr_slow_distance_m,
                "axial_velocity_design_scale_mps": (
                    env_class.lqr_axial_velocity_design_scale_mps
                ),
                "tracking_progress_clip": env_class.braking_progress_clip,
                "outside_envelope_penalty_max": (
                    env_class.reward_weights["braking_envelope"]
                    * env_class.braking_envelope_violation_ratio_clip
                ),
            },
            "pilot_decision_checks": {
                "maximum_training_steps": 100_000,
                "desired_mean_minimum_position_error_m": "below 1000",
                "desired_episode_survival_trend": "increasing",
                "continue_to_600k_only_if_checks_improve": True,
            },
        }
    )
    config["terminal_conditions"] = [
        "P3 success",
        "hard safety violation",
        "failed-approach reset truncation",
        "time limit",
        "non-finite state",
    ]
    return config


training.build_config = build_v20_config


if __name__ == "__main__":
    training.main()
