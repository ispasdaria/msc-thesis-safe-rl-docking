"""Train terminal V15 SAC with guarded simultaneous-target replay V2."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import json
import os
import sys

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLOSING_DIR = os.path.dirname(SCRIPT_DIR)
for path in (SCRIPT_DIR, CLOSING_DIR, SRC_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)

from envs.docking3dof.gym_env import Docking3DoFGymEnv  # noqa: E402
from envs.docking3dof.closing_cbf_shield_env import (  # noqa: E402
    CBF_TRAINING_VARIANTS,
    ClosingCBFShieldedGymEnv,
    build_closing_cbf_config,
)
from near_p3_reset_wrapper import FixedNearP3ResetWrapper  # noqa: E402
from phase_balanced_replay_buffer_v2 import (  # noqa: E402
    SimultaneousTargetReplayBuffer,
)
from train_sac_closing_multiseed import (  # noqa: E402
    RESULTS_ROOT,
    TRIAL_16_PARAMS,
    TrainingReturnCallback,
    seed_output_paths,
    write_aggregate_outputs,
    write_json,
)
from train_sac_closing_v15_phase_balanced_replay import (  # noqa: E402
    DEFAULT_SOURCE_MODEL,
    DEFAULT_SOURCE_VECNORMALIZE,
    PhaseBalancedCheckpointCallback,
    file_sha256,
    resolve_checkpoint_steps,
)


MODEL_ROOT = os.path.join(SCRIPT_DIR, "models", "multiseed")
RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results", "multiseed")


def seed_output_paths(experiment_name, seed):
    return (
        os.path.join(MODEL_ROOT, experiment_name, f"seed_{seed}"),
        os.path.join(RESULTS_ROOT, experiment_name, f"seed_{seed}"),
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Continue terminal V15 SAC with immutable source anchors, "
            "simultaneous-target replay, and fixed near-P3 reset augmentation."
        )
    )
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument(
        "--source-vecnormalize",
        default=DEFAULT_SOURCE_VECNORMALIZE,
    )
    parser.add_argument("--training-timesteps", type=int, default=150_000)
    parser.add_argument(
        "--max-replay-warmup-steps",
        type=int,
        default=200_000,
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[25_000, 50_000, 75_000, 100_000, 125_000, 150_000],
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=list(range(100, 110)),
    )
    parser.add_argument("--max-eval-steps", type=int, default=12_000)
    parser.add_argument("--global-buffer-size", type=int, default=300_000)
    parser.add_argument("--anchor-buffer-size", type=int, default=100_000)
    parser.add_argument("--capture-buffer-size", type=int, default=100_000)
    parser.add_argument("--priority-buffer-size", type=int, default=25_000)
    parser.add_argument("--target-ratio-threshold", type=float, default=10.0)
    parser.add_argument("--anchor-batch-fraction", type=float, default=0.25)
    parser.add_argument("--capture-batch-fraction", type=float, default=0.25)
    parser.add_argument(
        "--priority-capture-fraction",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--minimum-anchor-transitions",
        type=int,
        default=20_000,
    )
    parser.add_argument(
        "--minimum-anchor-episodes",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--minimum-capture-transitions",
        type=int,
        default=5_000,
    )
    parser.add_argument(
        "--minimum-capture-episodes",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--warmup-near-p3-reset-fraction",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--near-p3-reset-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument("--near-p3-min-ratio", type=float, default=1.20)
    parser.add_argument("--near-p3-max-ratio", type=float, default=4.00)
    parser.add_argument(
        "--near-p3-other-max-ratio",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--deterministic-reset-mixture",
        action="store_true",
    )
    parser.add_argument("--continuation-learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--continuation-gradient-steps", type=int, default=1)
    parser.add_argument(
        "--training-cbf-variant",
        choices=["none", *CBF_TRAINING_VARIANTS],
        default="none",
        help=(
            "CBF included in both training and checkpoint validation. "
            "'all' protects cone and speed; 'full' also protects alignment."
        ),
    )
    parser.add_argument(
        "--cbf-intervention-penalty-weight",
        type=float,
        default=0.1,
        help=(
            "Weight on the squared action correction; used only when a "
            "training CBF is enabled (default: 0.1)."
        ),
    )
    parser.add_argument(
        "--cbf-solver",
        choices=["osqp", "scipy"],
        default="osqp",
    )
    parser.add_argument("--early-stop-after-steps", type=int, default=50_000)
    parser.add_argument("--early-stop-failure-evals", type=int, default=2)
    parser.add_argument("--early-stop-regression-evals", type=int, default=2)
    parser.add_argument(
        "--early-stop-regression-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--experiment-name",
        default="sac_closing_v15_trial16_terminal_phase_balanced_v2_150k_seed0",
    )
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(method_version="v2")
    return parser


def parse_args():
    return build_parser().parse_args()


def validate_args(args):
    for name in (
        "training_timesteps",
        "max_replay_warmup_steps",
        "max_eval_steps",
        "global_buffer_size",
        "anchor_buffer_size",
        "capture_buffer_size",
        "priority_buffer_size",
        "minimum_anchor_transitions",
        "minimum_capture_transitions",
        "minimum_capture_episodes",
        "continuation_gradient_steps",
        "early_stop_failure_evals",
        "early_stop_regression_evals",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.early_stop_after_steps < 0:
        raise ValueError("--early-stop-after-steps cannot be negative.")
    if args.minimum_anchor_episodes < 0:
        raise ValueError("--minimum-anchor-episodes cannot be negative.")
    if args.target_ratio_threshold <= 1.0:
        raise ValueError("--target-ratio-threshold must exceed one.")
    if args.near_p3_min_ratio <= 1.0:
        raise ValueError("--near-p3-min-ratio must exceed one.")
    if args.near_p3_max_ratio <= args.near_p3_min_ratio:
        raise ValueError("--near-p3-max-ratio must exceed the minimum.")
    if (
        args.near_p3_other_max_ratio is not None
        and not 0.0 <= args.near_p3_other_max_ratio < 1.0
    ):
        raise ValueError("--near-p3-other-max-ratio must lie in [0, 1).")
    for name in (
        "anchor_batch_fraction",
        "capture_batch_fraction",
        "priority_capture_fraction",
        "warmup_near_p3_reset_fraction",
        "near_p3_reset_fraction",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must lie in [0, 1].")
    if args.anchor_batch_fraction + args.capture_batch_fraction > 1.0:
        raise ValueError("Anchor and capture batch fractions cannot exceed one.")
    if args.early_stop_regression_fraction < 0.0:
        raise ValueError("--early-stop-regression-fraction cannot be negative.")
    if args.continuation_learning_rate <= 0.0:
        raise ValueError("--continuation-learning-rate must be positive.")
    if (
        not np.isfinite(args.cbf_intervention_penalty_weight)
        or args.cbf_intervention_penalty_weight < 0.0
    ):
        raise ValueError(
            "--cbf-intervention-penalty-weight must be finite and nonnegative."
        )
    if not args.validation_seeds:
        raise ValueError("At least one validation seed is required.")
    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        raise ValueError("Validation seeds must be unique.")
    if args.training_seed in set(args.validation_seeds):
        raise ValueError("Training and validation seeds must be disjoint.")
    if not os.path.isfile(args.source_model):
        raise FileNotFoundError(args.source_model)
    if not os.path.isfile(args.source_vecnormalize):
        raise FileNotFoundError(args.source_vecnormalize)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")


def simultaneous_capture_checkpoint_rank(summary, mean_return, score):
    """Rank true simultaneous convergence before positional fly-by distance."""
    return (
        float(summary["success_rate"]),
        -float(summary["failure_rate"]),
        -float(summary["mean_best_max_target_ratio"]),
        float(summary["mean_target_2x_fraction"]),
        float(summary["mean_longest_target_2x_dwell_steps"]),
        -float(summary["mean_final_max_target_ratio"]),
        float(score),
        float(mean_return),
    )


class ReplayV2ReadinessCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ready = False

    def _on_step(self):
        if self.model.replay_buffer.replay_ready:
            self.ready = True
            return False
        return True


class GuardedV2CheckpointCallback(PhaseBalancedCheckpointCallback):
    """Stop after persistent validation safety loss or target regression."""

    def __init__(
        self,
        *args,
        early_stop_after_steps,
        failure_evals,
        regression_evals,
        regression_fraction,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.early_stop_after_steps = int(early_stop_after_steps)
        self.failure_evals = int(failure_evals)
        self.regression_evals = int(regression_evals)
        self.regression_fraction = float(regression_fraction)
        self.failure_streak = 0
        self.regression_streak = 0
        self.best_safe_simultaneous_ratio = None
        self.stop_requested = False
        self.stop_reason = None

    def _evaluate_and_save(self, checkpoint):
        super()._evaluate_and_save(checkpoint)
        row = self.checkpoint_rows[-1]
        if checkpoint < self.early_stop_after_steps:
            if float(row["failure_rate"]) == 0.0:
                ratio = float(row["mean_best_max_target_ratio"])
                if (
                    self.best_safe_simultaneous_ratio is None
                    or ratio < self.best_safe_simultaneous_ratio
                ):
                    self.best_safe_simultaneous_ratio = ratio
            return

        if float(row["failure_rate"]) > 0.0:
            self.failure_streak += 1
        else:
            self.failure_streak = 0

        ratio = float(row["mean_best_max_target_ratio"])
        if float(row["failure_rate"]) == 0.0:
            if self.best_safe_simultaneous_ratio is None:
                self.best_safe_simultaneous_ratio = ratio
            regression_limit = self.best_safe_simultaneous_ratio * (
                1.0 + self.regression_fraction
            )
            if ratio > regression_limit:
                self.regression_streak += 1
            else:
                self.regression_streak = 0
                self.best_safe_simultaneous_ratio = min(
                    self.best_safe_simultaneous_ratio,
                    ratio,
                )

        if self.failure_streak >= self.failure_evals:
            self.stop_requested = True
            self.stop_reason = (
                f"validation failure persisted for {self.failure_streak} evaluations"
            )
        elif self.regression_streak >= self.regression_evals:
            self.stop_requested = True
            self.stop_reason = (
                "simultaneous target error regressed for "
                f"{self.regression_streak} evaluations"
            )

    def _on_step(self):
        keep_training = super()._on_step()
        return bool(keep_training and not self.stop_requested)


def make_augmented_training_env(args):
    env_class = (
        Docking3DoFGymEnv
        if args.training_cbf_variant == "none"
        else ClosingCBFShieldedGymEnv
    )
    cbf_kwargs = {}
    if args.training_cbf_variant != "none":
        cbf_kwargs = {
            "cbf_variant": args.training_cbf_variant,
            "cbf_config": build_closing_cbf_config(
                args.training_cbf_variant,
                solver_backend=args.cbf_solver,
            ),
            "intervention_penalty_weight": (
                args.cbf_intervention_penalty_weight
            ),
        }
    env = env_class(
        phase="closing",
        mode="training",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=args.training_seed,
        closing_reward_version="v15",
        **cbf_kwargs,
    )
    env = FixedNearP3ResetWrapper(
        env,
        near_p3_fraction=args.warmup_near_p3_reset_fraction,
        minimum_target_ratio=args.near_p3_min_ratio,
        maximum_target_ratio=args.near_p3_max_ratio,
        other_component_max_ratio=args.near_p3_other_max_ratio,
        deterministic_mixture=args.deterministic_reset_mixture,
    )
    return Monitor(env)


def make_checkpoint_evaluation_env(
    *,
    args,
    seed,
    closing_reward_version,
    action_hold_steps,
):
    """Create a validation environment matching the training shield."""
    if args.training_cbf_variant == "none":
        return Docking3DoFGymEnv(
            phase="closing",
            mode="testing",
            dt=1.0,
            max_time=None,
            integrator="rk4",
            normalize_action=True,
            action_hold_steps=action_hold_steps,
            seed=seed,
            closing_reward_version=closing_reward_version,
        )
    return ClosingCBFShieldedGymEnv(
        phase="closing",
        mode="testing",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        action_hold_steps=action_hold_steps,
        seed=seed,
        closing_reward_version=closing_reward_version,
        cbf_variant=args.training_cbf_variant,
        cbf_config=build_closing_cbf_config(
            args.training_cbf_variant,
            solver_backend=args.cbf_solver,
        ),
        intervention_penalty_weight=args.cbf_intervention_penalty_weight,
    )


def find_reset_wrapper(vecnorm):
    env = vecnorm.venv.envs[0]
    while env is not None:
        if isinstance(env, FixedNearP3ResetWrapper):
            return env
        env = getattr(env, "env", None)
    raise RuntimeError("FixedNearP3ResetWrapper is missing from training env.")


def make_replay_buffer(model, args):
    kwargs = {
        "anchor_buffer_size": args.anchor_buffer_size,
        "capture_buffer_size": args.capture_buffer_size,
        "priority_buffer_size": args.priority_buffer_size,
        "target_ratio_threshold": args.target_ratio_threshold,
        "anchor_batch_fraction": args.anchor_batch_fraction,
        "capture_batch_fraction": args.capture_batch_fraction,
        "priority_capture_fraction": args.priority_capture_fraction,
        "minimum_anchor_transitions": args.minimum_anchor_transitions,
        "minimum_anchor_episodes": args.minimum_anchor_episodes,
        "minimum_capture_transitions": args.minimum_capture_transitions,
        "minimum_capture_episodes": args.minimum_capture_episodes,
    }
    replay_buffer = SimultaneousTargetReplayBuffer(
        buffer_size=args.global_buffer_size,
        observation_space=model.observation_space,
        action_space=model.action_space,
        device=model.device,
        n_envs=model.n_envs,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
        **kwargs,
    )
    model.buffer_size = args.global_buffer_size
    model.replay_buffer_class = SimultaneousTargetReplayBuffer
    model.replay_buffer_kwargs = kwargs
    model.replay_buffer = replay_buffer
    return replay_buffer


def build_config(args, checkpoints, source_num_timesteps):
    cbf_enabled = args.training_cbf_variant != "none"
    cbf_config = (
        build_closing_cbf_config(
            args.training_cbf_variant,
            solver_backend=args.cbf_solver,
        )
        if cbf_enabled
        else None
    )
    return {
        "experiment_name": args.experiment_name,
        "method_version": args.method_version,
        "purpose": (
            "terminal V15 continuation with immutable anchors, simultaneous-"
            "target replay, and fixed near-P3 reset augmentation"
        ),
        "training_seed": args.training_seed,
        "training_seeds": [args.training_seed],
        "validation_seeds": args.validation_seeds,
        "reward_version": "v15",
        "source_model": os.path.abspath(args.source_model),
        "source_vecnormalize": os.path.abspath(args.source_vecnormalize),
        "source_model_sha256": file_sha256(args.source_model),
        "source_vecnormalize_sha256": file_sha256(args.source_vecnormalize),
        "source_model_num_timesteps": int(source_num_timesteps),
        "gradient_training_timesteps": args.training_timesteps,
        "total_timesteps": args.training_timesteps,
        "maximum_replay_warmup_interaction_steps": args.max_replay_warmup_steps,
        "checkpoint_steps": [0, *checkpoints],
        "max_eval_steps": args.max_eval_steps,
        "device": args.device,
        "reward_changed": False,
        "curriculum_used": False,
        "reset_augmentation": {
            "type": "fixed mixture, not scheduled curriculum",
            "warmup_near_p3_fraction": args.warmup_near_p3_reset_fraction,
            "training_near_p3_fraction": args.near_p3_reset_fraction,
            "minimum_initial_max_target_ratio": args.near_p3_min_ratio,
            "maximum_initial_max_target_ratio": args.near_p3_max_ratio,
            "other_component_max_target_ratio": (
                args.near_p3_other_max_ratio
            ),
            "deterministic_mixture": args.deterministic_reset_mixture,
            "standard_p2_fraction_during_training": (
                1.0 - args.near_p3_reset_fraction
            ),
            "validation_uses_original_p2_distribution": True,
        },
        "replay_v2": {
            "global_buffer_size": args.global_buffer_size,
            "anchor_buffer_size": args.anchor_buffer_size,
            "capture_buffer_size": args.capture_buffer_size,
            "event_buffer_size": args.priority_buffer_size,
            "target_ratio_threshold": args.target_ratio_threshold,
            "anchor_batch_fraction": args.anchor_batch_fraction,
            "global_batch_fraction": (
                1.0
                - args.anchor_batch_fraction
                - args.capture_batch_fraction
            ),
            "capture_batch_fraction": args.capture_batch_fraction,
            "priority_fraction_within_capture": (
                args.priority_capture_fraction
            ),
            "minimum_anchor_transitions": args.minimum_anchor_transitions,
            "minimum_anchor_episodes": args.minimum_anchor_episodes,
            "minimum_capture_transitions": args.minimum_capture_transitions,
            "minimum_capture_episodes": args.minimum_capture_episodes,
            "anchor_source": "standard-P2 frozen-policy warmup only",
            "anchor_overwrite_enabled": False,
            "capture_definition": (
                "max(abs(P3 error) / exact P3 tolerance) <= threshold"
            ),
            "event_sampling": (
                "disabled"
                if args.priority_capture_fraction == 0.0
                else "uniform across non-empty event categories"
            ),
            "braking_limit": "at most one event per episode",
            "success_storage": "non-overwriting",
        },
        "continuation_optimizer": {
            "learning_rate": args.continuation_learning_rate,
            "gradient_steps": args.continuation_gradient_steps,
            "train_freq": TRIAL_16_PARAMS["train_freq"],
        },
        "training_environment": (
            "ClosingCBFShieldedGymEnv V15 with hard safety termination"
            if cbf_enabled
            else "Docking3DoFGymEnv V15 with hard safety termination"
        ),
        "safety_violation_termination_enabled": True,
        "cbf_filter_used_during_training": cbf_enabled,
        "checkpoint_validation_matches_training_filter": True,
        "cbf_training": {
            "enabled": cbf_enabled,
            "variant": args.training_cbf_variant,
            "intervention_penalty_weight": (
                args.cbf_intervention_penalty_weight if cbf_enabled else 0.0
            ),
            "intervention_penalty_definition": (
                "-weight * ||(u_filtered-u_nominal)/action_limits||_2^2"
            ),
            "replay_action": "nominal SAC action",
            "simulator_action": (
                "CBF-filtered action" if cbf_enabled else "nominal SAC action"
            ),
            "deployment_filter_required": cbf_enabled,
            "filter_config": asdict(cbf_config) if cbf_config else None,
        },
        "trial_16_params": TRIAL_16_PARAMS,
        "vecnormalize_statistics_frozen": True,
        "checkpoint_selection": [
            "highest success rate",
            "lowest safety-failure rate",
            "lowest best simultaneous six-component target ratio",
            "highest occupancy within two times all tolerances",
            "longest dwell within two times all tolerances",
            "lowest final simultaneous target ratio",
            "highest safety-first score",
            "highest return",
        ],
        "early_stopping": {
            "active_after_steps": args.early_stop_after_steps,
            "failure_evaluations": args.early_stop_failure_evals,
            "regression_evaluations": args.early_stop_regression_evals,
            "regression_fraction": args.early_stop_regression_fraction,
        },
    }


def main(args=None):
    if args is None:
        args = parse_args()
    validate_args(args)
    checkpoints = resolve_checkpoint_steps(args)

    source_probe = SAC.load(args.source_model, device="cpu")
    source_num_timesteps = int(source_probe.num_timesteps)
    del source_probe
    config = build_config(args, checkpoints, source_num_timesteps)
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    model_dir, result_dir = seed_output_paths(
        args.experiment_name,
        args.training_seed,
    )
    log_dir = os.path.join(
        SCRIPT_DIR,
        "logs",
        "multiseed",
        args.experiment_name,
    )
    for path in (model_dir, result_dir, log_dir):
        if os.path.exists(path):
            raise FileExistsError(
                f"Refusing to reuse experiment output directory: {path}"
            )
    for path in (model_dir, result_dir, log_dir):
        os.makedirs(path)
    config_path = os.path.join(
        RESULTS_ROOT,
        args.experiment_name,
        "experiment_config.json",
    )
    write_json(config_path, config)

    training_env = DummyVecEnv([lambda: make_augmented_training_env(args)])
    training_env = VecNormalize.load(args.source_vecnormalize, training_env)
    training_env.training = False
    training_env.norm_reward = True
    reset_wrapper = find_reset_wrapper(training_env)

    model = SAC.load(
        args.source_model,
        env=training_env,
        device=args.device,
        seed=args.training_seed,
        tensorboard_log=log_dir,
    )
    replay_buffer = make_replay_buffer(model, args)

    checkpoint_callback = GuardedV2CheckpointCallback(
        training_seed=args.training_seed,
        checkpoint_steps=checkpoints,
        validation_seeds=args.validation_seeds,
        max_eval_steps=args.max_eval_steps,
        closing_reward_version="v15",
        model_dir=model_dir,
        result_dir=result_dir,
        model_name=f"{args.experiment_name} - seed {args.training_seed}",
        rank_function=simultaneous_capture_checkpoint_rank,
        selection_metric=(
            "success, safety, best simultaneous target ratio, 2x-target "
            "occupancy/dwell, final target ratio, score, return"
        ),
        early_stop_after_steps=args.early_stop_after_steps,
        failure_evals=args.early_stop_failure_evals,
        regression_evals=args.early_stop_regression_evals,
        regression_fraction=args.early_stop_regression_fraction,
        evaluation_env_factory=partial(
            make_checkpoint_evaluation_env,
            args=args,
        ),
    )
    training_return_callback = TrainingReturnCallback(
        training_seed=args.training_seed,
        result_dir=result_dir,
        model_name=f"{args.experiment_name} - seed {args.training_seed}",
    )

    print("=" * 80)
    print(
        "Terminal V15 SAC with guarded simultaneous-target replay "
        f"{args.method_version.upper()}"
    )
    print("=" * 80)
    print(f"Experiment:       {args.experiment_name}")
    print(f"Gradient steps:   {args.training_timesteps:,}")
    print(
        "Batch mixture:    "
        f"{1.0 - args.anchor_batch_fraction - args.capture_batch_fraction:.0%} "
        f"recent / {args.anchor_batch_fraction:.0%} immutable anchor / "
        f"{args.capture_batch_fraction:.0%} simultaneous capture"
    )
    print(
        "Reset mixture:    "
        f"warmup {args.warmup_near_p3_reset_fraction:.0%} near-P3, "
        f"training {args.near_p3_reset_fraction:.0%} near-P3"
    )
    print(f"Target threshold: max component ratio <= {args.target_ratio_threshold:g}")
    if args.training_cbf_variant == "none":
        print("Safety:           hard violations terminate; no CBF during training")
    else:
        print(
            "Safety:           hard closing CBF "
            f"'{args.training_cbf_variant}' in training and validation"
        )
        print(
            "CBF penalty:      "
            f"{args.cbf_intervention_penalty_weight:g} * squared scaled "
            "correction"
        )
    print(f"Checkpoints:      {[0, *checkpoints]}")
    print("=" * 80)

    try:
        readiness_callback = ReplayV2ReadinessCallback()
        model.learning_starts = 0
        model.gradient_steps = 0
        model.learn(
            total_timesteps=args.max_replay_warmup_steps,
            reset_num_timesteps=True,
            tb_log_name=f"{args.experiment_name}_frozen_replay_warmup",
            progress_bar=args.progress_bar,
            callback=readiness_callback,
        )
        warmup_steps = int(model.num_timesteps)
        warmup_record = {
            "ready_for_gradient_training": bool(readiness_callback.ready),
            "warmup_interaction_steps": warmup_steps,
            "reset_statistics": reset_wrapper.reset_statistics(),
            **replay_buffer.replay_statistics(),
        }
        write_json(
            os.path.join(result_dir, "replay_buffer_after_warmup.json"),
            warmup_record,
        )
        config["actual_replay_warmup_interaction_steps"] = warmup_steps
        config["replay_ready_after_warmup"] = bool(readiness_callback.ready)
        if not readiness_callback.ready:
            write_json(config_path, config)
            raise RuntimeError(
                "Replay warmup reached its maximum before anchor/capture "
                "readiness. No gradient update was performed."
            )

        replay_buffer.freeze_anchor()
        reset_wrapper.set_near_p3_fraction(args.near_p3_reset_fraction)
        config["anchor_frozen_at_size"] = replay_buffer.anchor_size()
        write_json(config_path, config)

        model.learning_starts = 0
        model.gradient_steps = args.continuation_gradient_steps
        model.learning_rate = args.continuation_learning_rate
        model.lr_schedule = FloatSchedule(args.continuation_learning_rate)
        model.learn(
            total_timesteps=args.training_timesteps,
            reset_num_timesteps=True,
            tb_log_name=args.experiment_name,
            progress_bar=args.progress_bar,
            callback=[training_return_callback, checkpoint_callback],
        )
        model.save(os.path.join(model_dir, "final_model"))
        training_env.save(os.path.join(model_dir, "final_vecnormalize.pkl"))
        write_json(
            os.path.join(result_dir, "replay_buffer_final.json"),
            replay_buffer.replay_statistics(),
        )
        write_json(
            os.path.join(result_dir, "near_p3_reset_statistics.json"),
            reset_wrapper.reset_statistics(),
        )
        config["actual_gradient_training_steps"] = int(model.num_timesteps)
        config["early_stop_triggered"] = bool(
            checkpoint_callback.stop_requested
        )
        config["early_stop_reason"] = checkpoint_callback.stop_reason
        write_json(config_path, config)
    finally:
        training_env.close()

    best_checkpoint, best_mean_return, best_episode = (
        checkpoint_callback.get_best_checkpoint()
    )
    training_return_callback.set_best_checkpoint(best_episode, best_checkpoint)
    training_return_callback.save_outputs()
    write_aggregate_outputs(
        args.experiment_name,
        [args.training_seed],
        best_checkpoint_by_seed={args.training_seed: best_checkpoint},
    )
    print(f"Best additional checkpoint: {best_checkpoint}")
    print(f"Best validation return:      {best_mean_return:.3f}")
    if checkpoint_callback.stop_requested:
        print(f"Early stop:                  {checkpoint_callback.stop_reason}")
    print(f"Results: {result_dir}")
    print(f"Models:  {model_dir}")


if __name__ == "__main__":
    main()
