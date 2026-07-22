"""Warm-start curriculum training for the 3-DoF SAC closing controller.

The default run keeps reward v15 and the Optuna trial-16 SAC configuration,
but uses a smaller learning rate for stable fine-tuning. It starts from seed
0's 400k checkpoint, first teaches a reachable P3 terminal capture, and
expands the reset distribution only after the current stage meets fixed
held-out success and safety thresholds.

Examples:
    python train_sac_closing_curriculum.py --device cuda --progress-bar
    python train_sac_closing_curriculum.py --dry-run
    python train_sac_closing_curriculum.py --from-scratch --seed 1
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
SRC_PATH = REPO_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from envs.docking3dof.gym_env import Docking3DoFGymEnv  # noqa: E402
from train_sac_closing import LearningCurveCallback  # noqa: E402
from train_sac_closing_multiseed import TRIAL_16_PARAMS  # noqa: E402
from tune_sac_closing_optuna import (  # noqa: E402
    evaluate_model,
    normalize_observation,
    score_summary,
    summarize_results,
    target_error,
    target_error_metrics,
)


MODEL_ROOT = SCRIPT_DIR / "models" / "curriculum"
LOG_ROOT = SCRIPT_DIR / "logs" / "curriculum"
RESULTS_ROOT = SCRIPT_DIR / "results" / "curriculum"

SOURCE_EXPERIMENT = "sac_closing_rewardv15_trial16_multiseed_600ksteps"
DEFAULT_SOURCE_ROOT = (
    SCRIPT_DIR / "models" / "multiseed" / SOURCE_EXPERIMENT / "seed_0"
)
DEFAULT_SOURCE_MODEL = DEFAULT_SOURCE_ROOT / "model_0400000_steps.zip"
DEFAULT_SOURCE_VECNORM = (
    DEFAULT_SOURCE_ROOT / "model_0400000_steps_vecnormalize.pkl"
)

# Stage 0 is a short success-acquisition task. Only axial position begins just
# outside its success tolerance; all other errors begin near the target.
EASY_AXIAL_TOLERANCE_RANGE = (1.001, 1.05)
EASY_OTHER_TOLERANCE_FRACTION = 0.10
DEFAULT_FINETUNE_LEARNING_RATE = 1.2e-4
DEFAULT_CURRICULUM_LEARNING_STARTS = 5_000


@dataclass(frozen=True)
class ClosingCurriculumStage:
    """One expansion level between an easy P3 reset and the full P2 set."""

    name: str
    max_difficulty: float
    default_budget: int
    full_reset_probability: float = 0.0
    max_time: float = 12_000.0
    min_completed_episodes: int = 0

    def __post_init__(self):
        if not 0.0 <= self.max_difficulty <= 1.0:
            raise ValueError("max_difficulty must be in [0, 1].")
        if not 0.0 <= self.full_reset_probability <= 1.0:
            raise ValueError("full_reset_probability must be in [0, 1].")
        if self.default_budget <= 0:
            raise ValueError("default_budget must be positive.")
        if self.max_time <= 0.0:
            raise ValueError("max_time must be positive.")
        if self.min_completed_episodes < 0:
            raise ValueError("min_completed_episodes must be non-negative.")


DEFAULT_CLOSING_CURRICULUM_STAGES = (
    ClosingCurriculumStage(
        "success_acquisition",
        0.00,
        500_000,
        max_time=2_000.0,
        min_completed_episodes=200,
    ),
    ClosingCurriculumStage("very_near_p3", 0.05, 100_000),
    ClosingCurriculumStage("near_field", 0.15, 150_000),
    ClosingCurriculumStage("mid_field", 0.35, 150_000),
    ClosingCurriculumStage("far_field", 0.65, 200_000),
    ClosingCurriculumStage(
        "full_p2",
        0.65,
        400_000,
        full_reset_probability=0.80,
    ),
)


def stage_for_evaluation(stage):
    """Evaluate the final stage on pure full-P2 resets."""
    if stage.name == "full_p2":
        return replace(stage, full_reset_probability=1.0)
    return stage


def sample_easy_target_error(core_env):
    """Sample a reset where only axial position starts outside tolerance."""
    tolerance = np.asarray(core_env.p3_target_tolerance, dtype=np.float64)
    rng = core_env.rng
    axial_low, axial_high = EASY_AXIAL_TOLERANCE_RANGE
    other_fraction = EASY_OTHER_TOLERANCE_FRACTION
    return np.array(
        [
            rng.uniform(axial_low * tolerance[0], axial_high * tolerance[0]),
            rng.uniform(
                -other_fraction * tolerance[1],
                other_fraction * tolerance[1],
            ),
            rng.uniform(
                -other_fraction * tolerance[2],
                other_fraction * tolerance[2],
            ),
            rng.uniform(
                -other_fraction * tolerance[3],
                other_fraction * tolerance[3],
            ),
            rng.uniform(
                -other_fraction * tolerance[4],
                other_fraction * tolerance[4],
            ),
            rng.uniform(
                -other_fraction * tolerance[5],
                other_fraction * tolerance[5],
            ),
        ],
        dtype=np.float64,
    )


def sample_curriculum_reset(core_env, stage, mode="training"):
    """Return a correlated initial state, P3 target, and diagnostics."""
    rng = core_env.rng
    target = core_env.sample_X_P3(mode)
    full_state = core_env.sample_X_P2(mode)
    full_reset = rng.random() < stage.full_reset_probability

    if full_reset:
        difficulty = 1.0
        initial_state = full_state
    else:
        difficulty = (
            0.0
            if stage.max_difficulty == 0.0
            else rng.uniform(0.0, stage.max_difficulty)
        )
        easy_error = sample_easy_target_error(core_env)
        full_error = np.asarray(full_state - target, dtype=np.float64)
        full_error[4] = core_env.wrap_to_pi(full_error[4])
        target_error_value = (
            (1.0 - difficulty) * easy_error
            + difficulty * full_error
        )
        initial_state = target + target_error_value
        initial_state[4] = core_env.wrap_to_pi(initial_state[4])

    diagnostics = {
        "curriculum_stage": stage.name,
        "curriculum_difficulty": float(difficulty),
        "curriculum_full_reset": bool(full_reset),
    }
    return initial_state, target, diagnostics


class CurriculumDocking3DoFGymEnv(Docking3DoFGymEnv):
    """Experiment-local closing environment with curriculum-controlled resets."""

    def __init__(self, curriculum_stage, **kwargs):
        kwargs.setdefault("phase", "closing")
        kwargs.setdefault("max_time", curriculum_stage.max_time)
        if kwargs["phase"] != "closing":
            raise ValueError("The closing curriculum requires phase='closing'.")
        super().__init__(**kwargs)
        self.curriculum_stage = curriculum_stage

    def set_curriculum_stage(self, curriculum_stage):
        self.curriculum_stage = curriculum_stage
        self.max_time = float(curriculum_stage.max_time)
        self.core_env.max_time = float(curriculum_stage.max_time)

    def reset(self, seed=None, options=None):
        options = {} if options is None else dict(options)
        if "initial_state" in options:
            return super().reset(seed=seed, options=options)

        if seed is not None:
            self.core_env.rng = np.random.default_rng(seed)

        mode = options.get("mode", self.mode)
        initial_state, p3_target, diagnostics = sample_curriculum_reset(
            self.core_env,
            self.curriculum_stage,
            mode=mode,
        )

        # core.reset() already supports a supplied initial state but normally
        # samples its P3 target internally. Override that sampler only for this
        # reset call so stable core/gym files remain completely unchanged.
        original_p3_sampler = self.core_env.sample_X_P3
        self.core_env.sample_X_P3 = lambda reset_mode: p3_target.copy()
        try:
            observation, info = super().reset(
                seed=None,
                options={"initial_state": initial_state, "mode": mode},
            )
        finally:
            self.core_env.sample_X_P3 = original_p3_sampler

        info.update(diagnostics)
        return observation, info


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train trial-16 SAC and expand the closing reset "
            "distribution through a success-gated curriculum."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gamma",
        type=float,
        default=TRIAL_16_PARAMS["gamma"],
        help=(
            "SAC discount factor. The same value is used by VecNormalize "
            "(default: trial-16 gamma)."
        ),
    )
    parser.add_argument(
        "--reward-version",
        choices=["v15", "v19", "v20"],
        default="v15",
        help=(
            "Closing reward used throughout the curriculum. v15 reproduces "
            "the baseline; v19 uses one persistent tolerance-normalized P3 "
            "state cost; v20 adds only the velocity warning margins needed "
            "after the v19 safety diagnosis (default: v15)."
        ),
    )
    parser.add_argument(
        "--stage-budgets",
        type=int,
        nargs=len(DEFAULT_CLOSING_CURRICULUM_STAGES),
        default=[stage.default_budget for stage in DEFAULT_CLOSING_CURRICULUM_STAGES],
        metavar=("S0", "S1", "S2", "S3", "S4", "S5"),
        help="Maximum training steps for stages 0 through 5.",
    )
    parser.add_argument(
        "--max-stages",
        type=int,
        default=len(DEFAULT_CLOSING_CURRICULUM_STAGES),
        help="Run only the first N curriculum stages (default: all).",
    )
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=list(range(1_000, 1_020)),
        help="Fixed seeds used for stage promotion (default: 1000--1019).",
    )
    parser.add_argument("--max-eval-steps", type=int, default=12_000)
    parser.add_argument("--promotion-success", type=float, default=0.80)
    parser.add_argument("--promotion-max-failure", type=float, default=0.02)
    parser.add_argument("--promotion-confirmations", type=int, default=2)
    parser.add_argument(
        "--retention-drop",
        type=float,
        default=0.10,
        help="Allowed success-rate drop on earlier stages during promotion.",
    )
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=DEFAULT_CURRICULUM_LEARNING_STARTS,
        help=(
            "Transitions collected before SAC updates begin "
            "(default: 5000)."
        ),
    )
    parser.add_argument(
        "--finetune-learning-rate",
        type=float,
        default=DEFAULT_FINETUNE_LEARNING_RATE,
        help=(
            "Learning rate used during curriculum training. The default is "
            "lower than trial 16 to reduce warm-start policy regression."
        ),
    )
    parser.add_argument(
        "--source-model",
        type=Path,
        default=DEFAULT_SOURCE_MODEL,
    )
    parser.add_argument(
        "--source-vecnormalize",
        type=Path,
        default=DEFAULT_SOURCE_VECNORM,
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Initialize a new SAC policy instead of the seed-0 400k policy.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample every reset stage and validate arguments without training.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if not 0.0 < args.gamma < 1.0:
        raise ValueError("--gamma must be strictly between 0 and 1.")
    if not 1 <= args.max_stages <= len(DEFAULT_CLOSING_CURRICULUM_STAGES):
        raise ValueError("--max-stages is outside the available stage range.")
    if any(budget <= 0 for budget in args.stage_budgets):
        raise ValueError("Every stage budget must be positive.")
    if args.eval_every <= 0 or args.max_eval_steps <= 0:
        raise ValueError("Evaluation intervals and rollout length must be positive.")
    if not args.validation_seeds:
        raise ValueError("At least one validation seed is required.")
    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        raise ValueError("Validation seeds must be unique.")
    if not 0.0 <= args.promotion_success <= 1.0:
        raise ValueError("--promotion-success must be in [0, 1].")
    if not 0.0 <= args.promotion_max_failure <= 1.0:
        raise ValueError("--promotion-max-failure must be in [0, 1].")
    if args.promotion_confirmations <= 0:
        raise ValueError("--promotion-confirmations must be positive.")
    if not 0.0 <= args.retention_drop <= 1.0:
        raise ValueError("--retention-drop must be in [0, 1].")
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be positive.")
    if args.learning_starts < 0:
        raise ValueError("--learning-starts must be non-negative.")
    if args.finetune_learning_rate <= 0.0:
        raise ValueError("--finetune-learning-rate must be positive.")
    if args.reward_version in {"v19", "v20"} and not args.from_scratch:
        raise ValueError(
            f"Reward {args.reward_version} must be trained with --from-scratch "
            "so the controlled experiment starts with fresh model and "
            "normalization statistics."
        )
    if not args.from_scratch:
        if not args.source_model.is_file():
            raise FileNotFoundError(f"Source model not found: {args.source_model}")
        if not args.source_vecnormalize.is_file():
            raise FileNotFoundError(
                f"Source VecNormalize state not found: {args.source_vecnormalize}"
            )


def stages_from_args(args):
    return tuple(
        replace(stage, default_budget=budget)
        for stage, budget in zip(
            DEFAULT_CLOSING_CURRICULUM_STAGES,
            args.stage_budgets,
        )
    )[: args.max_stages]


def make_curriculum_env(stage, seed, reward_version="v15"):
    return CurriculumDocking3DoFGymEnv(
        curriculum_stage=stage,
        phase="closing",
        mode="training",
        dt=1.0,
        max_time=stage.max_time,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
        closing_reward_version=reward_version,
    )


def build_training_stack(args, initial_stage, log_dir):
    dummy_env = DummyVecEnv(
        [
            lambda: Monitor(
                make_curriculum_env(
                    initial_stage,
                    seed=args.seed,
                    reward_version=args.reward_version,
                )
            )
        ]
    )

    if args.from_scratch:
        vecnorm = VecNormalize(
            dummy_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=args.gamma,
        )
        model = SAC(
            policy="MlpPolicy",
            env=vecnorm,
            learning_rate=args.finetune_learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=TRIAL_16_PARAMS["batch_size"],
            tau=TRIAL_16_PARAMS["tau"],
            gamma=args.gamma,
            train_freq=TRIAL_16_PARAMS["train_freq"],
            gradient_steps=TRIAL_16_PARAMS["gradient_steps"],
            ent_coef=TRIAL_16_PARAMS["ent_coef"],
            target_update_interval=1,
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(log_dir),
            device=args.device,
        )
    else:
        vecnorm = VecNormalize.load(args.source_vecnormalize, dummy_env)
        vecnorm.gamma = args.gamma
        vecnorm.training = True
        vecnorm.norm_reward = True
        model = SAC.load(
            args.source_model,
            env=vecnorm,
            device=args.device,
            tensorboard_log=str(log_dir),
        )
        if int(model.buffer_size) != args.buffer_size:
            raise ValueError(
                "The warm-start model buffer size does not match --buffer-size: "
                f"{model.buffer_size} != {args.buffer_size}."
            )
        model.learning_starts = args.learning_starts
        model.gamma = args.gamma
        model.learning_rate = args.finetune_learning_rate
        model.lr_schedule = get_schedule_fn(args.finetune_learning_rate)
        # Replay buffers are not part of SB3 model.zip files. Reset explicitly
        # and restart timestep accounting so the configured warm-up is applied.
        model.replay_buffer.reset()

    return model, vecnorm


def mean_and_std_return(results):
    values = np.asarray([result["episode_return"] for result in results])
    return float(np.mean(values)), float(np.std(values))


def evaluate_curriculum_model(model, vecnorm, stage, args):
    """Evaluate a curriculum stage without changing the Optuna utilities."""
    results = []
    evaluation_stage = stage_for_evaluation(stage)
    old_training = vecnorm.training
    old_norm_reward = vecnorm.norm_reward
    vecnorm.training = False
    vecnorm.norm_reward = False

    try:
        for seed in args.validation_seeds:
            env = make_curriculum_env(
                evaluation_stage,
                seed=seed,
                reward_version=args.reward_version,
            )
            try:
                obs, info = env.reset(seed=seed)
                final_info = info
                tolerance = np.asarray(
                    env.core_env.p3_target_tolerance,
                    dtype=np.float64,
                )
                initial_error = target_error(env)
                initial_metrics = target_error_metrics(
                    initial_error,
                    tolerance,
                )
                min_position_error = initial_metrics["position_error"]
                port_distances = []
                rho_values = []
                port_speeds = []
                abs_s_dot_values = []
                abs_psi_values = []
                abs_omega_values = []
                warning_steps = 0
                episode_return = 0.0

                for step in range(args.max_eval_steps):
                    norm_obs = normalize_observation(vecnorm, obs)
                    action, _ = model.predict(norm_obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(
                        action[0]
                    )
                    final_info = info
                    episode_return += float(reward)

                    current_error = target_error(env)
                    min_position_error = min(
                        min_position_error,
                        float(np.linalg.norm(current_error[0:2])),
                    )
                    port_distances.append(info.get("port_distance", np.nan))
                    rho_values.append(info.get("rho", np.nan))
                    port_speeds.append(info.get("port_speed", np.nan))
                    abs_s_dot_values.append(abs(info.get("s_dot", np.nan)))
                    abs_psi_values.append(abs(info.get("psi_e", np.nan)))
                    abs_omega_values.append(abs(info.get("omega", np.nan)))
                    warning_steps += int(bool(info.get("warning", False)))

                    if terminated or truncated:
                        break

                final_error = target_error(env)
                final_metrics = target_error_metrics(final_error, tolerance)
                initial_position_error = initial_metrics["position_error"]
                progress_fraction = (
                    initial_position_error - final_metrics["position_error"]
                ) / max(initial_position_error, 1e-12)

                results.append(
                    {
                        "seed": seed,
                        "steps": step + 1,
                        "episode_return": episode_return,
                        "success": bool(final_info.get("success", False)),
                        "failure": bool(final_info.get("failure", False)),
                        "time_limit": bool(final_info.get("time_limit", False)),
                        "termination_reason": final_info.get(
                            "termination_reason"
                        ),
                        "violated_constraints": final_info.get(
                            "violated_constraints",
                            [],
                        ),
                        "initial_position_error": initial_position_error,
                        "final_position_error": final_metrics[
                            "position_error"
                        ],
                        "min_position_error": min_position_error,
                        "progress_fraction": float(
                            np.clip(progress_fraction, -1.0, 1.0)
                        ),
                        "final_lateral_error": final_metrics[
                            "lateral_error"
                        ],
                        "final_velocity_error": final_metrics[
                            "velocity_error"
                        ],
                        "final_attitude_error_deg": final_metrics[
                            "attitude_error_deg"
                        ],
                        "final_rate_error_degs": final_metrics[
                            "rate_error_degs"
                        ],
                        "final_position_ratio": final_metrics[
                            "position_ratio"
                        ],
                        "final_velocity_ratio": final_metrics[
                            "velocity_ratio"
                        ],
                        "final_alignment_ratio": final_metrics[
                            "alignment_ratio"
                        ],
                        "final_max_target_ratio": final_metrics[
                            "max_target_ratio"
                        ],
                        "final_port_distance": float(port_distances[-1]),
                        "min_port_distance": float(
                            np.nanmin(port_distances)
                        ),
                        "final_rho": float(rho_values[-1]),
                        "max_rho": float(np.nanmax(rho_values)),
                        "max_port_speed": float(np.nanmax(port_speeds)),
                        "max_abs_s_dot": float(
                            np.nanmax(abs_s_dot_values)
                        ),
                        "max_abs_psi_deg": float(
                            np.rad2deg(np.nanmax(abs_psi_values))
                        ),
                        "max_abs_omega_degs": float(
                            np.rad2deg(np.nanmax(abs_omega_values))
                        ),
                        "warning_fraction": warning_steps / (step + 1),
                    }
                )
            finally:
                env.close()
    finally:
        vecnorm.training = old_training
        vecnorm.norm_reward = old_norm_reward

    return results


def evaluate_stage(model, vecnorm, stage, args):
    return evaluate_curriculum_model(model, vecnorm, stage, args)


def evaluate_full_test_distribution(model, vecnorm, args):
    return evaluate_model(
        model=model,
        vecnorm=vecnorm,
        seeds=args.validation_seeds,
        max_steps=args.max_eval_steps,
        closing_reward_version=args.reward_version,
    )


def append_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = list(rows[0])
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def serializable_rollout(result):
    row = dict(result)
    row["violated_constraints"] = json.dumps(
        row.get("violated_constraints", []),
        sort_keys=True,
    )
    return row


def record_evaluation(
    result_dir,
    results,
    stage_index,
    stage,
    stage_step,
    curriculum_step,
    completed_training_episodes,
    completed_stage_training_episodes,
    eval_kind,
):
    summary = summarize_results(results)
    mean_return, std_return = mean_and_std_return(results)
    score = score_summary(summary)
    summary_row = {
        "eval_kind": eval_kind,
        "curriculum_step": curriculum_step,
        "completed_training_episodes": completed_training_episodes,
        "completed_stage_training_episodes": completed_stage_training_episodes,
        "stage_index": stage_index,
        "stage_name": stage.name,
        "stage_step": stage_step,
        "max_difficulty": stage.max_difficulty,
        "full_reset_probability": stage.full_reset_probability,
        "n_episodes": len(results),
        "mean_return": mean_return,
        "std_return": std_return,
        "score": score,
        **summary,
    }
    append_csv(result_dir / "curriculum_evaluations.csv", [summary_row])

    if eval_kind == "current_stage":
        learning_curve_row = {
            "episode": completed_training_episodes,
            "timestep": curriculum_step,
            "stage_index": stage_index,
            "stage_name": stage.name,
            "stage_step": stage_step,
            "mean_return": mean_return,
            "std_return": std_return,
            "mean_length": float(
                np.mean([result["steps"] for result in results])
            ),
        }
        for result in results:
            seed = result["seed"]
            learning_curve_row[f"seed_{seed}_return"] = result[
                "episode_return"
            ]
            learning_curve_row[f"seed_{seed}_length"] = result["steps"]
        append_csv(
            result_dir / "evaluation_learning_curve.csv",
            [learning_curve_row],
        )

    rollout_rows = []
    for result in results:
        rollout_rows.append(
            {
                "eval_kind": eval_kind,
                "curriculum_step": curriculum_step,
                "stage_index": stage_index,
                "stage_name": stage.name,
                "stage_step": stage_step,
                **serializable_rollout(result),
            }
        )
    append_csv(result_dir / "curriculum_validation_rollouts.csv", rollout_rows)
    return summary_row


def save_bundle(model, vecnorm, model_dir, stem):
    model.save(model_dir / f"{stem}_model")
    vecnorm.save(model_dir / f"{stem}_vecnormalize.pkl")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def checkpoint_rank(summary_row):
    """Prefer success, then safety, then balanced terminal precision."""
    return (
        float(summary_row["success_rate"]),
        -float(summary_row["failure_rate"]),
        -float(summary_row["mean_final_max_target_ratio"]),
        -float(summary_row["mean_final_position_error"]),
        float(summary_row["mean_return"]),
    )


def meets_promotion(summary_row, args, success_floor=None):
    if success_floor is None:
        success_floor = args.promotion_success
    return (
        float(summary_row["success_rate"]) >= success_floor
        and float(summary_row["failure_rate"])
        <= args.promotion_max_failure
    )


def retention_is_satisfied(
    model,
    vecnorm,
    completed_stages,
    current_stage_index,
    stage_step,
    curriculum_step,
    completed_training_episodes,
    completed_stage_training_episodes,
    args,
    result_dir,
):
    retention_floor = max(0.0, args.promotion_success - args.retention_drop)
    for previous_index, previous_stage in enumerate(completed_stages):
        results = evaluate_stage(model, vecnorm, previous_stage, args)
        summary = record_evaluation(
            result_dir=result_dir,
            results=results,
            stage_index=previous_index,
            stage=previous_stage,
            stage_step=stage_step,
            curriculum_step=curriculum_step,
            completed_training_episodes=completed_training_episodes,
            completed_stage_training_episodes=completed_stage_training_episodes,
            eval_kind=f"retention_for_stage_{current_stage_index}",
        )
        if not meets_promotion(summary, args, success_floor=retention_floor):
            return False
    return True


def write_evaluation_learning_curve(result_dir, experiment_name):
    """Plot deterministic evaluation return against training episodes."""
    csv_path = result_dir / "evaluation_learning_curve.csv"
    if not csv_path.exists():
        return
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        return

    episodes = np.asarray([int(row["episode"]) for row in rows])
    mean_returns = np.asarray([float(row["mean_return"]) for row in rows])
    std_returns = np.asarray([float(row["std_return"]) for row in rows])

    plt.figure(figsize=(10, 6))
    plt.plot(
        episodes,
        mean_returns,
        color="tab:green",
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="Mean evaluation return",
    )
    plt.fill_between(
        episodes,
        mean_returns - std_returns,
        mean_returns + std_returns,
        color="tab:green",
        alpha=0.2,
        label="1 std across seeds",
    )
    plt.xlabel("Completed training episodes")
    plt.ylabel("Mean total evaluation reward")
    plt.title(f"Deterministic Evaluation Learning Curve\n{experiment_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "evaluation_learning_curve.png", dpi=200)
    plt.close()


def dry_run(stages, args):
    print("Curriculum reset samples")
    for stage_index, stage in enumerate(stages):
        env = make_curriculum_env(stage, seed=args.seed)
        try:
            _, info = env.reset(seed=args.seed)
            error = env.core_env.state - env.core_env.p3_target
            error[4] = env.core_env.wrap_to_pi(error[4])
            print(
                f"stage={stage_index} {stage.name:20s} "
                f"difficulty={info['curriculum_difficulty']:.3f} "
                f"full={info['curriculum_full_reset']} "
                f"position_error={np.linalg.norm(error[:2]):.3f} m "
                f"velocity_error={np.linalg.norm(error[2:4]):.5f} m/s "
                f"attitude_error={np.rad2deg(abs(error[4])):.3f} deg"
            )
        finally:
            env.close()


def main():
    args = parse_args()
    validate_args(args)
    stages = stages_from_args(args)

    if args.dry_run:
        dry_run(stages, args)
        return

    experiment_name = args.experiment_name or (
        f"sac_closing_reward{args.reward_version}_trial16_"
        f"curriculum_easy_s0_seed{args.seed}"
    )
    model_dir = MODEL_ROOT / experiment_name
    result_dir = RESULTS_ROOT / experiment_name
    log_dir = LOG_ROOT / experiment_name

    populated_paths = [
        path
        for path in (model_dir, result_dir, log_dir)
        if path.exists() and any(path.iterdir())
    ]
    if populated_paths and not args.overwrite:
        joined = "\n".join(str(path) for path in populated_paths)
        raise FileExistsError(
            "Output directories are not empty. Use a new --experiment-name "
            f"or pass --overwrite:\n{joined}"
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment_name": experiment_name,
        "seed": args.seed,
        "gamma": args.gamma,
        "reward_version": args.reward_version,
        "warm_start": not args.from_scratch,
        "source_model": None if args.from_scratch else str(args.source_model.resolve()),
        "source_vecnormalize": (
            None if args.from_scratch else str(args.source_vecnormalize.resolve())
        ),
        "trial_16_params": TRIAL_16_PARAMS,
        "finetune_learning_rate": args.finetune_learning_rate,
        "easy_axial_tolerance_range": EASY_AXIAL_TOLERANCE_RANGE,
        "easy_other_tolerance_fraction": EASY_OTHER_TOLERANCE_FRACTION,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "validation_seeds": args.validation_seeds,
        "max_eval_steps": args.max_eval_steps,
        "eval_every": args.eval_every,
        "promotion_success": args.promotion_success,
        "promotion_max_failure": args.promotion_max_failure,
        "promotion_confirmations": args.promotion_confirmations,
        "retention_drop": args.retention_drop,
        "stages": [asdict(stage) for stage in stages],
    }
    write_json(result_dir / "experiment_config.json", config)

    model = None
    vecnorm = None
    try:
        model, vecnorm = build_training_stack(args, stages[0], log_dir)
        training_callback = LearningCurveCallback(
            csv_path=result_dir / "learning_curve.csv",
            plot_path=result_dir / "episode_return_learning_curve.png",
            model_name=experiment_name,
        )
        first_learn_call = True
        curriculum_step = 0
        completed_stages = []
        stopped_before_full_curriculum = False

        print("=" * 80)
        print(f"Reward-{args.reward_version} / trial-16 closing curriculum")
        print(f"Experiment: {experiment_name}")
        print(f"Warm start: {not args.from_scratch}")
        print(f"Gamma:       {args.gamma}")
        print(f"Learning rate: {args.finetune_learning_rate}")
        print(f"Learning starts: {args.learning_starts}")
        print(f"Stages:     {[stage.name for stage in stages]}")
        print(f"Device:     {model.device}")
        print("=" * 80)

        for stage_index, stage in enumerate(stages):
            vecnorm.env_method("set_curriculum_stage", stage)
            model._last_obs = None
            stage_step = 0
            confirmations = 0
            best_rank = None
            promoted = False
            stage_episode_start = len(training_callback.episodes)

            print("=" * 80)
            print(
                f"Stage {stage_index}: {stage.name} | "
                f"max difficulty={stage.max_difficulty} | "
                f"budget={stage.default_budget} | "
                f"max time={stage.max_time:.0f} s | "
                f"minimum episodes={stage.min_completed_episodes}"
            )
            print("=" * 80)

            while stage_step < stage.default_budget:
                chunk = min(args.eval_every, stage.default_budget - stage_step)
                model.learn(
                    total_timesteps=chunk,
                    reset_num_timesteps=first_learn_call,
                    tb_log_name=experiment_name,
                    progress_bar=args.progress_bar,
                    callback=training_callback,
                )
                first_learn_call = False
                stage_step += chunk
                curriculum_step += chunk
                completed_training_episodes = len(training_callback.episodes)
                completed_stage_training_episodes = (
                    completed_training_episodes - stage_episode_start
                )

                results = evaluate_stage(model, vecnorm, stage, args)
                summary = record_evaluation(
                    result_dir=result_dir,
                    results=results,
                    stage_index=stage_index,
                    stage=stage,
                    stage_step=stage_step,
                    curriculum_step=curriculum_step,
                    completed_training_episodes=completed_training_episodes,
                    completed_stage_training_episodes=(
                        completed_stage_training_episodes
                    ),
                    eval_kind="current_stage",
                )
                write_evaluation_learning_curve(result_dir, experiment_name)
                save_bundle(model, vecnorm, model_dir, "latest")

                rank = checkpoint_rank(summary)
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    save_bundle(
                        model,
                        vecnorm,
                        model_dir,
                        f"stage_{stage_index:02d}_best",
                    )
                    write_json(
                        result_dir / f"stage_{stage_index:02d}_best.json",
                        summary,
                    )

                episode_requirement_met = (
                    completed_stage_training_episodes
                    >= stage.min_completed_episodes
                )
                eligible = (
                    episode_requirement_met
                    and meets_promotion(summary, args)
                )
                confirmations = confirmations + 1 if eligible else 0
                print(
                    f"step={curriculum_step} stage_step={stage_step} | "
                    f"success={summary['success_rate']:.3f} "
                    f"failure={summary['failure_rate']:.3f} "
                    f"timeout={summary['time_limit_rate']:.3f} | "
                    f"P3 error={summary['mean_final_position_error']:.3f} m | "
                    f"episodes={completed_stage_training_episodes}/"
                    f"{stage.min_completed_episodes} | "
                    f"confirmations={confirmations}/"
                    f"{args.promotion_confirmations}"
                )

                if confirmations >= args.promotion_confirmations:
                    retention_ok = retention_is_satisfied(
                        model=model,
                        vecnorm=vecnorm,
                        completed_stages=completed_stages,
                        current_stage_index=stage_index,
                        stage_step=stage_step,
                        curriculum_step=curriculum_step,
                        completed_training_episodes=completed_training_episodes,
                        completed_stage_training_episodes=(
                            completed_stage_training_episodes
                        ),
                        args=args,
                        result_dir=result_dir,
                    )
                    if retention_ok:
                        promoted = True
                        break
                    confirmations = 0
                    print("Promotion delayed: an earlier stage lost retention.")

            save_bundle(
                model,
                vecnorm,
                model_dir,
                f"stage_{stage_index:02d}_final",
            )
            full_results = evaluate_full_test_distribution(model, vecnorm, args)
            record_evaluation(
                result_dir=result_dir,
                results=full_results,
                stage_index=stage_index,
                stage=stage,
                stage_step=stage_step,
                curriculum_step=curriculum_step,
                completed_training_episodes=len(training_callback.episodes),
                completed_stage_training_episodes=(
                    len(training_callback.episodes) - stage_episode_start
                ),
                eval_kind="full_test_probe",
            )
            write_evaluation_learning_curve(result_dir, experiment_name)

            state = {
                "curriculum_step": curriculum_step,
                "last_stage_index": stage_index,
                "last_stage_name": stage.name,
                "last_stage_step": stage_step,
                "last_stage_completed_training_episodes": (
                    len(training_callback.episodes) - stage_episode_start
                ),
                "stage_promoted": promoted,
                "completed_stages": [
                    previous.name for previous in completed_stages
                ],
            }
            write_json(result_dir / "curriculum_state.json", state)

            is_last_stage = stage_index == len(stages) - 1
            if promoted:
                completed_stages.append(stage)
            elif not is_last_stage:
                print(
                    f"Stage {stage_index} exhausted its budget without meeting "
                    "the promotion criteria. Training stops here."
                )
                stopped_before_full_curriculum = True
                break

        save_bundle(model, vecnorm, model_dir, "final")
        write_evaluation_learning_curve(result_dir, experiment_name)
        print("=" * 80)
        if stopped_before_full_curriculum:
            print("Curriculum stopped at an unsolved stage.")
        else:
            print("Curriculum run complete.")
        print(f"Models:  {model_dir}")
        print(f"Results: {result_dir}")
        print("=" * 80)
    finally:
        if vecnorm is not None:
            vecnorm.close()


if __name__ == "__main__":
    main()
