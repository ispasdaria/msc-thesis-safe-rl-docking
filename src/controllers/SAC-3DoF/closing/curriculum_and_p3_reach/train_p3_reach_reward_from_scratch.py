"""Train closing SAC from scratch with the isolated P3-reaching reward.

This experiment starts every rollout from the normal P2 training distribution.
It creates a new actor, new critics, an empty replay buffer, and new observation
normalization statistics. The selected isolated environment determines whether
a safety violation terminates an episode. Raw rewards are deliberately not
normalized.

No pretrained model or VecNormalize file is loaded or modified.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (SCRIPT_DIR, SRC_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)

from envs.docking3dof.closing_safety_exploration_env import (
    ClosingSafetyExplorationV18GymEnv,
)
from train_p3_reach_nonterminal_safety_continuation import (
    ExperimentCheckpointCallback,
    TrainingSafetyDiagnosticsCallback,
    checkpoint_rank,
    make_env,
    write_csv,
    write_json,
)


EXPERIMENT_FAMILY = "from_scratch_reward_progression"
EXPERIMENT_ENV_CLASS = ClosingSafetyExplorationV18GymEnv
EXPERIMENT_REWARD_SCHEME = EXPERIMENT_ENV_CLASS.reward_scheme
MODEL_ROOT = os.path.join(SCRIPT_DIR, "models", EXPERIMENT_FAMILY)
LOG_ROOT = os.path.join(SCRIPT_DIR, "logs", EXPERIMENT_FAMILY)
RESULT_ROOT = os.path.join(SCRIPT_DIR, "results", EXPERIMENT_FAMILY)
CHECKPOINT_RANK_FUNCTION = checkpoint_rank
CHECKPOINT_SELECTION_ORDER = [
    "highest success rate",
    "lowest mean final max target ratio",
    "lowest mean final position error",
    "lowest mean unsafe fraction",
    "highest mean return",
]
EXPERIMENT_REPLAY_BUFFER_CLASS = None
EXPERIMENT_REPLAY_BUFFER_KWARGS = {}
EXPERIMENT_NAME_EXTRA_TAG = ""
DEFAULT_ACTION_HOLD_SECONDS = 1

# V18 retains the stable vanilla-SAC settings used by V12 so this experiment
# isolates task-weight redistribution and bounded axial capture progress.
SAC_PARAMETERS = {
    "learning_rate": 1e-4,
    "buffer_size": 1_000_000,
    "learning_starts": 10_000,
    "batch_size": 512,
    "tau": 0.005,
    "gamma": 0.999,
    "train_freq": 16,
    "gradient_steps": 8,
    "ent_coef": "auto_0.05",
    "target_update_interval": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            f"Train a new closing SAC from scratch with "
            f"{EXPERIMENT_REWARD_SCHEME} and non-terminal safety "
            "exploration."
        )
    )
    parser.add_argument("--total-timesteps", type=int, default=600_000)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=list(range(25_000, 600_001, 25_000)),
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=list(range(100, 110)),
    )
    parser.add_argument("--max-time", type=float, default=12_000.0)
    parser.add_argument(
        "--action-hold-seconds",
        type=int,
        default=DEFAULT_ACTION_HOLD_SECONDS,
        help=(
            "Seconds each selected action remains constant. The simulator "
            "continues to integrate and check termination every second."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-regression-early-stop", action="store_true")
    parser.add_argument("--early-stop-start-step", type=int, default=250_000)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--regression-target-ratio-factor", type=float, default=2.0)
    parser.add_argument(
        "--regression-unsafe-fraction-increase",
        type=float,
        default=0.25,
    )
    return parser.parse_args()


def validate_args(args):
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive.")
    if args.max_time <= 0.0:
        raise ValueError("--max-time must be positive.")
    if args.action_hold_seconds <= 0:
        raise ValueError("--action-hold-seconds must be a positive integer.")
    if not args.validation_seeds:
        raise ValueError("At least one validation seed is required.")
    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        raise ValueError("Validation seeds must be unique.")
    if args.training_seed in set(args.validation_seeds):
        raise ValueError("Training and validation seeds must be disjoint.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.early_stop_start_step < 0:
        raise ValueError("--early-stop-start-step cannot be negative.")
    if args.early_stop_patience <= 0:
        raise ValueError("--early-stop-patience must be positive.")
    if args.regression_target_ratio_factor <= 1.0:
        raise ValueError("--regression-target-ratio-factor must exceed 1.")
    if args.regression_unsafe_fraction_increase <= 0.0:
        raise ValueError(
            "--regression-unsafe-fraction-increase must be positive."
        )


def resolve_checkpoint_steps(args):
    checkpoints = sorted(
        {
            int(step)
            for step in [*args.checkpoint_steps, args.total_timesteps]
            if 0 < int(step) <= args.total_timesteps
        }
    )
    if not checkpoints or checkpoints[-1] != args.total_timesteps:
        raise RuntimeError("The final training step must be a checkpoint.")
    return checkpoints


class RegressionEarlyStoppingCheckpointCallback(ExperimentCheckpointCallback):
    """Stop only after several consecutive severe validation regressions."""

    def __init__(
        self,
        *args,
        enabled,
        start_step,
        patience,
        target_ratio_factor,
        unsafe_fraction_increase,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.regression_stop_enabled = bool(enabled)
        self.regression_start_step = int(start_step)
        self.regression_patience = int(patience)
        self.regression_target_ratio_factor = float(target_ratio_factor)
        self.regression_unsafe_fraction_increase = float(
            unsafe_fraction_increase
        )
        self.regression_streak = 0
        self.regression_stop_requested = False
        self.replay_statistics_rows = []

    def _on_step(self):
        continue_training = super()._on_step()
        return bool(continue_training and not self.regression_stop_requested)

    def _evaluate_and_save(self, step):
        super()._evaluate_and_save(step)
        statistics = getattr(
            getattr(self.model, "replay_buffer", None),
            "replay_statistics",
            None,
        )
        if callable(statistics):
            self.replay_statistics_rows.append(
                {"additional_timestep": int(step), **statistics()}
            )
            write_csv(
                os.path.join(
                    self.result_dir,
                    "replay_buffer_checkpoint_metrics.csv",
                ),
                self.replay_statistics_rows,
            )
        if (
            not self.regression_stop_enabled
            or step < self.regression_start_step
            or self.best_step == step
        ):
            self.regression_streak = 0
            return

        current = self.summary_rows[-1]
        best = next(
            row
            for row in self.summary_rows
            if row["additional_timestep"] == self.best_step
        )
        lost_success = current["success_rate"] < best["success_rate"]
        target_regressed = (
            current["mean_final_max_target_ratio"]
            >= self.regression_target_ratio_factor
            * max(best["mean_final_max_target_ratio"], 1e-12)
        )
        safety_regressed = (
            current["mean_unsafe_fraction"]
            >= best["mean_unsafe_fraction"]
            + self.regression_unsafe_fraction_increase
        )
        clear_regression = bool(
            lost_success or (target_regressed and safety_regressed)
        )
        self.regression_streak = (
            self.regression_streak + 1 if clear_regression else 0
        )
        if self.regression_streak < self.regression_patience:
            return

        self.regression_stop_requested = True
        write_json(
            os.path.join(self.result_dir, "regression_early_stop.json"),
            {
                "stopped_at_timestep": int(step),
                "best_timestep": int(self.best_step),
                "regression_streak": int(self.regression_streak),
                "criteria": {
                    "start_step": self.regression_start_step,
                    "patience": self.regression_patience,
                    "target_ratio_factor": (
                        self.regression_target_ratio_factor
                    ),
                    "unsafe_fraction_increase": (
                        self.regression_unsafe_fraction_increase
                    ),
                },
                "best_summary": best,
                "stopping_summary": current,
            },
        )
        print(
            "Stopping after consecutive severe validation regressions: "
            f"step={step:,}, best_step={self.best_step:,}.",
            flush=True,
        )


def build_config(args, checkpoints, experiment_name):
    axial_capture_enabled = hasattr(
        EXPERIMENT_ENV_CLASS,
        "axial_capture_progress_clip",
    )
    replay_buffer_config = {
        "class": (
            EXPERIMENT_REPLAY_BUFFER_CLASS.__name__
            if EXPERIMENT_REPLAY_BUFFER_CLASS is not None
            else "ReplayBuffer"
        ),
        "kwargs": dict(EXPERIMENT_REPLAY_BUFFER_KWARGS),
        "experience_source": "online agent experience only",
        "offline_demonstrations_used": False,
        "lqr_transitions_used": False,
    }
    sac_parameters = dict(SAC_PARAMETERS)
    base_one_second_gamma = float(sac_parameters["gamma"])
    sac_parameters["gamma"] = base_one_second_gamma ** args.action_hold_seconds
    return {
        "experiment_name": experiment_name,
        "purpose": "closing SAC P3-reaching training from scratch",
        "phase": "closing",
        "initial_distribution": "normal P2 training distribution",
        "initialization": {
            "actor": "new random initialization",
            "critics": "new random initialization",
            "replay_buffer": "new empty buffer",
            "vecnormalize": "new statistics",
            "pretrained_model_loaded": False,
            "pretrained_vecnormalize_loaded": False,
        },
        "experiment_reward_scheme": EXPERIMENT_REWARD_SCHEME,
        "shared_gym_reward_version_used_for_experiment": False,
        "shared_reward_files_modified": False,
        "experiment_reward_weights": EXPERIMENT_ENV_CLASS.reward_weights,
        "goal_metric": {
            "components": list(EXPERIMENT_ENV_CLASS.goal_component_names),
            "normalization": "absolute P3 error divided by P3 tolerance",
            "error_transform": "log1p of each tolerance ratio",
            "aggregation": (
                "grouped equal blends of component mean and smooth worst"
            ),
            "aggregate_mean_weight": (
                EXPERIMENT_ENV_CLASS.aggregate_mean_weight
            ),
            "aggregate_smooth_worst_weight": (
                EXPERIMENT_ENV_CLASS.aggregate_smooth_worst_weight
            ),
            "smooth_max_beta": EXPERIMENT_ENV_CLASS.smooth_max_beta,
            "active_from_p2": True,
            "distance_gate": False,
            "logarithmic_error_compression": True,
            "group_weights": {
                "position": EXPERIMENT_ENV_CLASS.position_group_weight,
                "attitude_rate": EXPERIMENT_ENV_CLASS.attitude_group_weight,
                "terminal_velocity": EXPERIMENT_ENV_CLASS.velocity_group_weight,
            },
        },
        "terminal_velocity_shaping": {
            "target": "sampled exact P3 velocity",
            "gate": "sigmoid in Euclidean P3 position error",
            "gate_midpoint_m": (
                EXPERIMENT_ENV_CLASS.terminal_velocity_gate_distance_m
            ),
            "gate_width_m": (
                EXPERIMENT_ENV_CLASS.terminal_velocity_gate_width_m
            ),
            "gate_applied_after_velocity_progress_difference": True,
            "active_from_p2": False,
        },
        "reward_shaping": {
            "progress_clip": EXPERIMENT_ENV_CLASS.progress_clip,
            "state_cost_clip": None,
            "action_cost": "squared normalized action norm",
            "unsafe_penalty": (
                "bounded terminal failure penalty"
                if getattr(
                    EXPERIMENT_ENV_CLASS,
                    "safety_violation_termination_enabled",
                    False,
                )
                else "per hard-unsafe state; non-terminal"
            ),
        },
        "near_target_capture_shaping": {
            "type": "V18 bounded position/velocity-focused capture progress",
            "gate": "sigmoid in Euclidean P3 position error",
            "gate_midpoint_m": (
                EXPERIMENT_ENV_CLASS.near_target_gate_distance_m
            ),
            "gate_width_m": EXPERIMENT_ENV_CLASS.near_target_gate_width_m,
            "progress_metric": (
                "weighted changes in position, attitude/rate, and velocity "
                "group costs"
            ),
            "group_weights": {
                "position": EXPERIMENT_ENV_CLASS.position_group_weight,
                "attitude_rate": EXPERIMENT_ENV_CLASS.attitude_group_weight,
                "terminal_velocity": EXPERIMENT_ENV_CLASS.velocity_group_weight,
            },
            "progress_clip": EXPERIMENT_ENV_CLASS.capture_progress_clip,
            "persistent_occupancy_penalty": False,
        },
        "axial_capture_progress": {
            "enabled": axial_capture_enabled,
            "target": "episode-specific P3 axial position component",
            "cost": "log1p(abs(axial error) / axial P3 tolerance)",
            "type": "bounded symmetric temporal-difference progress",
            "gate": "same sigmoid near-target gate as V12 capture",
            "progress_clip": getattr(
                EXPERIMENT_ENV_CLASS,
                "axial_capture_progress_clip",
                None,
            ),
            "weight": EXPERIMENT_ENV_CLASS.reward_weights.get(
                "axial_capture_progress",
                0.0,
            ),
            "persistent_occupancy_penalty": False,
            "overshoot_penalty": False,
        },
        "soft_constraints": {
            "docking_cone": {
                "enabled": False,
                "training_filter_enabled": False,
                "deployment_filter": "frozen closing cone/speed CBF-QP",
            }
        },
        "departure_shaping": {
            "axial_overshoot_penalty": False,
            "axial_overshoot_metric_logged": True,
            "flyaway": (
                "bounded near-target log1p positive P3-relative radial speed"
            ),
            "flyaway_speed_scale_mps": (
                EXPERIMENT_ENV_CLASS.flyaway_speed_scale_mps
            ),
            "flyaway_penalty_clip": (
                EXPERIMENT_ENV_CLASS.flyaway_penalty_clip
            ),
            "uses_episode_history": False,
        },
        "failed_approach_reset": {
            "enabled": True,
            "entry_distance_m": (
                EXPERIMENT_ENV_CLASS.failed_approach_entry_distance_m
            ),
            "distance_growth_from_minimum_m": (
                EXPERIMENT_ENV_CLASS.failed_approach_distance_growth_m
            ),
            "minimum_radial_speed_mps": (
                EXPERIMENT_ENV_CLASS.failed_approach_radial_speed_mps
            ),
            "patience_steps": (
                EXPERIMENT_ENV_CLASS.failed_approach_patience_steps
            ),
            "added_reward_penalty": 0.0,
            "gym_semantics": "truncated",
        },
        "prescribed_velocity_profile": bool(
            getattr(
                EXPERIMENT_ENV_CLASS,
                "prescribed_velocity_profile_enabled",
                False,
            )
        ),
        "safety_classification_enabled": True,
        "safety_penalties_enabled": True,
        "safety_violation_termination_enabled": bool(
            getattr(
                EXPERIMENT_ENV_CLASS,
                "safety_violation_termination_enabled",
                False,
            )
        ),
        "terminal_conditions": [
            "P3 success",
            "failed-approach reset truncation",
            "time limit",
            "non-finite state",
        ],
        "training_seed": args.training_seed,
        "validation_seeds": args.validation_seeds,
        "total_timesteps": args.total_timesteps,
        "checkpoint_steps": checkpoints,
        "max_time": args.max_time,
        "time_discretization": {
            "integration_step_seconds": 1.0,
            "action_hold_seconds": args.action_hold_seconds,
            "control_interval_seconds": float(args.action_hold_seconds),
            "control_frequency_hz": 1.0 / args.action_hold_seconds,
            "policy_timestep_unit": "one control decision",
            "maximum_simulated_seconds_from_training_decisions": (
                args.total_timesteps * args.action_hold_seconds
            ),
            "reward_aggregation_within_hold": "undiscounted sum",
            "terminal_check_interval_seconds": 1.0,
        },
        "device": args.device,
        "sac_parameters": sac_parameters,
        "replay_buffer": replay_buffer_config,
        "base_one_second_gamma": base_one_second_gamma,
        "gamma": sac_parameters["gamma"],
        "vecnormalize": {
            "norm_obs": True,
            "norm_reward": False,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
            "gamma": sac_parameters["gamma"],
        },
        "checkpoint_selection": CHECKPOINT_SELECTION_ORDER,
        "regression_early_stop": {
            "enabled": not args.disable_regression_early_stop,
            "start_step": args.early_stop_start_step,
            "patience": args.early_stop_patience,
            "target_ratio_factor": args.regression_target_ratio_factor,
            "unsafe_fraction_increase": (
                args.regression_unsafe_fraction_increase
            ),
            "requires_target_and_safety_regression": True,
        },
        "training_learning_curve": {
            "file": "episode_return_learning_curve.png",
            "raw_episode_return": True,
            "moving_average_window_episodes": 10,
            "refresh_interval_episodes": 5,
            "evaluation_checkpoints_marked": True,
        },
    }


def main():
    args = parse_args()
    validate_args(args)
    checkpoints = resolve_checkpoint_steps(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reward_tag = EXPERIMENT_REWARD_SCHEME.replace("_", "")
    hold_tag = (
        f"action_hold_{args.action_hold_seconds}s_"
        if args.action_hold_seconds > 1
        else ""
    )
    termination_tag = (
        "safety_termination"
        if getattr(
            EXPERIMENT_ENV_CLASS,
            "safety_violation_termination_enabled",
            False,
        )
        else "nonterminal_safety"
    )
    experiment_name = args.experiment_name or (
        f"{reward_tag}_gamma0p999_from_scratch_"
        f"{termination_tag}_"
        f"{EXPERIMENT_NAME_EXTRA_TAG}"
        f"{hold_tag}"
        f"{args.total_timesteps // 1000}k_seed{args.training_seed}_{timestamp}"
    )
    config = build_config(args, checkpoints, experiment_name)

    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    model_dir = os.path.join(MODEL_ROOT, experiment_name)
    result_dir = os.path.join(RESULT_ROOT, experiment_name)
    log_dir = os.path.join(LOG_ROOT, experiment_name)
    for path in (model_dir, result_dir, log_dir):
        if os.path.exists(path):
            raise FileExistsError(
                f"Refusing to reuse experiment output directory: {path}"
            )
    for path in (model_dir, result_dir, log_dir):
        os.makedirs(path)
    write_json(os.path.join(result_dir, "run_config.json"), config)

    training_env = DummyVecEnv(
        [
            lambda: Monitor(
                make_env(
                    seed=args.training_seed,
                    mode="training",
                    max_time=args.max_time,
                    env_class=EXPERIMENT_ENV_CLASS,
                    action_hold_steps=args.action_hold_seconds,
                )
            )
        ]
    )
    training_env = VecNormalize(
        training_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=config["gamma"],
    )

    replay_buffer_options = {}
    if EXPERIMENT_REPLAY_BUFFER_CLASS is not None:
        replay_buffer_options = {
            "replay_buffer_class": EXPERIMENT_REPLAY_BUFFER_CLASS,
            "replay_buffer_kwargs": dict(EXPERIMENT_REPLAY_BUFFER_KWARGS),
        }

    model = SAC(
        policy="MlpPolicy",
        env=training_env,
        learning_rate=SAC_PARAMETERS["learning_rate"],
        buffer_size=SAC_PARAMETERS["buffer_size"],
        learning_starts=SAC_PARAMETERS["learning_starts"],
        batch_size=SAC_PARAMETERS["batch_size"],
        tau=SAC_PARAMETERS["tau"],
        gamma=config["gamma"],
        train_freq=SAC_PARAMETERS["train_freq"],
        gradient_steps=SAC_PARAMETERS["gradient_steps"],
        ent_coef=SAC_PARAMETERS["ent_coef"],
        target_update_interval=SAC_PARAMETERS["target_update_interval"],
        verbose=1,
        seed=args.training_seed,
        tensorboard_log=log_dir,
        device=args.device,
        **replay_buffer_options,
    )

    training_callback = TrainingSafetyDiagnosticsCallback(
        output_path=os.path.join(
            result_dir,
            "training_episode_safety_metrics.csv",
        ),
        plot_path=os.path.join(
            result_dir,
            "episode_return_learning_curve.png",
        ),
        checkpoint_steps=checkpoints,
        plot_subtitle=(
            "P3-reaching reward with terminal safety violations"
            if getattr(
                EXPERIMENT_ENV_CLASS,
                "safety_violation_termination_enabled",
                False,
            )
            else "P3-reaching reward with non-terminal safety exploration"
        ),
    )
    checkpoint_callback = RegressionEarlyStoppingCheckpointCallback(
        checkpoint_steps=checkpoints,
        validation_seeds=args.validation_seeds,
        max_time=args.max_time,
        model_dir=model_dir,
        result_dir=result_dir,
        evaluate_initial=False,
        env_class=EXPERIMENT_ENV_CLASS,
        action_hold_steps=args.action_hold_seconds,
        rank_function=CHECKPOINT_RANK_FUNCTION,
        selection_order=CHECKPOINT_SELECTION_ORDER,
        enabled=not args.disable_regression_early_stop,
        start_step=args.early_stop_start_step,
        patience=args.early_stop_patience,
        target_ratio_factor=args.regression_target_ratio_factor,
        unsafe_fraction_increase=args.regression_unsafe_fraction_increase,
    )

    print("=" * 80)
    print("Closing SAC from scratch: P3-reaching reward")
    print("=" * 80)
    print(f"Experiment:          {experiment_name}")
    print("Initialization:      new actor, critics, replay buffer, normalization")
    print("Initial states:      normal P2 training distribution")
    print(f"Reward:              isolated {EXPERIMENT_REWARD_SCHEME}")
    if EXPERIMENT_REPLAY_BUFFER_CLASS is not None:
        print(
            "Replay buffer:       "
            f"{EXPERIMENT_REPLAY_BUFFER_CLASS.__name__}"
        )
    if getattr(
        EXPERIMENT_ENV_CLASS,
        "safety_violation_termination_enabled",
        False,
    ):
        print("Safety violations:   terminate the episode; no training filter")
    else:
        print("Safety violations:   occupancy-penalized, but non-terminal")
    print(f"Training steps:      {args.total_timesteps:,}")
    print(
        "Control interval:    "
        f"{args.action_hold_seconds} s "
        "(1 s RK4 integration and terminal checks)"
    )
    print(f"SAC gamma/decision:  {config['gamma']:.9f}")
    print(f"Checkpoints:         {checkpoints}")
    print(f"Validation seeds:    {args.validation_seeds}")
    print(
        "Regression stop:     "
        + (
            f"enabled after {args.early_stop_start_step:,} steps"
            if not args.disable_regression_early_stop
            else "disabled"
        )
    )
    print("=" * 80)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            reset_num_timesteps=True,
            tb_log_name=experiment_name,
            progress_bar=args.progress_bar,
            callback=[training_callback, checkpoint_callback],
        )
        model.save(os.path.join(model_dir, "final_model"))
        training_env.save(os.path.join(model_dir, "final_vecnormalize.pkl"))
        replay_statistics = getattr(
            model.replay_buffer,
            "replay_statistics",
            None,
        )
        if callable(replay_statistics):
            write_json(
                os.path.join(result_dir, "replay_buffer_final.json"),
                replay_statistics(),
            )
    finally:
        training_env.close()

    print(f"Best checkpoint: {checkpoint_callback.best_step}")
    print(f"Results: {result_dir}")
    print(f"Models:  {model_dir}")


if __name__ == "__main__":
    main()
