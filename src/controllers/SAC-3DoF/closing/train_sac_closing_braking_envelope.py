"""Train safety-terminal vanilla SAC with the V19 braking envelope.

This launcher reuses the established from-scratch V18 training and validation
pipeline while selecting the isolated safety-terminal V19 environment.  It
does not load an LQR policy, demonstrations, an existing SAC model, replay
data, or a CBF training filter.
"""

import train_sac_closing_p3reach_from_scratch as training

from envs.docking3dof.closing_safety_exploration_env import (
    ClosingSafetyTerminalV19GymEnv,
)


training.EXPERIMENT_ENV_CLASS = ClosingSafetyTerminalV19GymEnv
training.EXPERIMENT_REWARD_SCHEME = (
    ClosingSafetyTerminalV19GymEnv.reward_scheme
)

_build_base_config = training.build_config


def build_braking_envelope_config(args, checkpoints, experiment_name):
    config = _build_base_config(args, checkpoints, experiment_name)
    env_class = ClosingSafetyTerminalV19GymEnv
    config.update(
        {
            "purpose": (
                "safety-terminal closing vanilla SAC P3 capture with an "
                "LQR-inspired axial braking envelope"
            ),
            "prescribed_velocity_profile": True,
            "lqr_policy_or_actions_used": False,
            "cbf_filter_used_during_training": False,
            "safety_violation_termination_enabled": True,
            "safety_penalty_semantics": "bounded terminal failure penalty",
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
                "axial_position_design_scale_m": (
                    env_class.lqr_axial_position_design_scale_m
                ),
                "axial_velocity_design_scale_mps": (
                    env_class.lqr_axial_velocity_design_scale_mps
                ),
                "terminal_half_width_mps": (
                    "episode P3 axial-velocity tolerance"
                ),
                "overshoot_recovery": (
                    "mirror the P3-relative velocity reference"
                ),
                "tracking_progress": (
                    "bounded symmetric temporal difference of log1p "
                    "normalized reference error"
                ),
                "tracking_progress_clip": env_class.braking_progress_clip,
                "outside_envelope_penalty_max": (
                    env_class.reward_weights["braking_envelope"]
                    * env_class.braking_envelope_violation_ratio_clip
                ),
                "unsafe_step_penalty": (
                    env_class.reward_weights["unsafe_step"]
                ),
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


training.build_config = build_braking_envelope_config


if __name__ == "__main__":
    training.main()
