"""Train closing SAC from scratch with the isolated P3-reaching reward.

This experiment starts every rollout from the normal P2 training distribution.
It creates a new actor, new critics, an empty replay buffer, and new observation
normalization statistics. Safety classifications and occupancy penalties are
retained, but a safety violation does not end an exploration episode. Raw
rewards are deliberately not normalized.

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
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.closing_safety_exploration_env import (
    ClosingSafetyExplorationV15GymEnv,
)
from train_sac_closing_no_safety_termination import (
    ExperimentCheckpointCallback,
    TrainingSafetyDiagnosticsCallback,
    make_env,
    write_json,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_FAMILY = "p3reach_from_scratch"
EXPERIMENT_ENV_CLASS = ClosingSafetyExplorationV15GymEnv
EXPERIMENT_REWARD_SCHEME = EXPERIMENT_ENV_CLASS.reward_scheme
MODEL_ROOT = os.path.join(SCRIPT_DIR, "models", EXPERIMENT_FAMILY)
LOG_ROOT = os.path.join(SCRIPT_DIR, "logs", EXPERIMENT_FAMILY)
RESULT_ROOT = os.path.join(SCRIPT_DIR, "results", EXPERIMENT_FAMILY)

# V15 retains the stable vanilla-SAC settings used by V12 so this experiment
# isolates only the effect of the lateral cone-containment soft constraint.
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
            "Train a new closing SAC from scratch with p3_reach_v15 and "
            "non-terminal safety exploration."
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

    def _on_step(self):
        continue_training = super()._on_step()
        return bool(continue_training and not self.regression_stop_requested)

    def _evaluate_and_save(self, step):
        super()._evaluate_and_save(step)
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
            "unsafe_penalty": "per hard-unsafe state; non-terminal",
        },
        "near_target_capture_shaping": {
            "type": "V12 bounded symmetric balanced capture progress",
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
        "soft_constraints": {
            "docking_cone": {
                "enabled": True,
                "type": "bounded non-terminal lateral-exit occupancy penalty",
                "constraint": "rho <= cone_slope * clipped_s",
                "normalization": "lateral excess / cone base radius",
                "clipping": [0.0, 1.0],
                "lateral_only": True,
                "axial_exit_penalty": False,
                "maximum_penalty_per_step": (
                    EXPERIMENT_ENV_CLASS.reward_weights["cone_constraint"]
                ),
                "zero_inside_cone": True,
                "terminates_episode": False,
                "restricts_initial_distribution": False,
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
        "prescribed_velocity_profile": False,
        "safety_classification_enabled": True,
        "safety_penalties_enabled": True,
        "safety_violation_termination_enabled": False,
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
        "device": args.device,
        "sac_parameters": SAC_PARAMETERS,
        "gamma": SAC_PARAMETERS["gamma"],
        "vecnormalize": {
            "norm_obs": True,
            "norm_reward": False,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
            "gamma": SAC_PARAMETERS["gamma"],
        },
        "checkpoint_selection": (
            "success, target ratio, position error, unsafe fraction, return"
        ),
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
    experiment_name = args.experiment_name or (
        "sac_closing_p3reachv15_gamma0999_from_scratch_"
        "no_safety_termination_"
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
        gamma=SAC_PARAMETERS["gamma"],
    )

    model = SAC(
        policy="MlpPolicy",
        env=training_env,
        learning_rate=SAC_PARAMETERS["learning_rate"],
        buffer_size=SAC_PARAMETERS["buffer_size"],
        learning_starts=SAC_PARAMETERS["learning_starts"],
        batch_size=SAC_PARAMETERS["batch_size"],
        tau=SAC_PARAMETERS["tau"],
        gamma=SAC_PARAMETERS["gamma"],
        train_freq=SAC_PARAMETERS["train_freq"],
        gradient_steps=SAC_PARAMETERS["gradient_steps"],
        ent_coef=SAC_PARAMETERS["ent_coef"],
        target_update_interval=SAC_PARAMETERS["target_update_interval"],
        verbose=1,
        seed=args.training_seed,
        tensorboard_log=log_dir,
        device=args.device,
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
    )
    checkpoint_callback = RegressionEarlyStoppingCheckpointCallback(
        checkpoint_steps=checkpoints,
        validation_seeds=args.validation_seeds,
        max_time=args.max_time,
        model_dir=model_dir,
        result_dir=result_dir,
        evaluate_initial=False,
        env_class=EXPERIMENT_ENV_CLASS,
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
    print("Safety violations:   occupancy-penalized, but non-terminal")
    print(f"Training steps:      {args.total_timesteps:,}")
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
    finally:
        training_env.close()

    print(f"Best checkpoint: {checkpoint_callback.best_step}")
    print(f"Results: {result_dir}")
    print(f"Models:  {model_dir}")


if __name__ == "__main__":
    main()
