"""Adaptive-cadence SAC continuation at 0.1 s integration.

The experiment is isolated from the shared environment and existing model
artifacts.  It starts from the audited guarded 1 Hz checkpoint, integrates at
0.1 s everywhere, holds actions for one physical second far from P3, and
switches to true 10 Hz control near P3 or in a safety warning.

Training mixes ordinary P2 resets with bounded near-P3 resets, protects
standard-P2 and simultaneous-target replay partitions, freezes the source
observation/reward normalization, and fixes entropy at the source checkpoint's
already-learned low value.  Full-P2 validation never uses augmented resets.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import re
import sys
from types import MethodType

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
SRC_PATH = REPO_ROOT / "src"
CLOSING_DIR = SCRIPT_DIR.parent
PHASE_BALANCED_DIR = CLOSING_DIR / "phase_balanced_replay"
for path in (SCRIPT_DIR, CLOSING_DIR, PHASE_BALANCED_DIR, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from envs.docking3dof.gym_env import Docking3DoFGymEnv
from near_p3_reset_wrapper import FixedNearP3ResetWrapper
from train_sac_closing_multiseed import TrainingReturnCallback, write_json
from train_sac_closing_v15_phase_balanced_replay import (
    file_sha256,
    resolve_checkpoint_steps,
)
from train_sac_closing_v15_phase_balanced_replay_v2 import (
    GuardedV2CheckpointCallback,
    ReplayV2ReadinessCallback,
    find_reset_wrapper,
    make_replay_buffer,
    simultaneous_capture_checkpoint_rank,
)


DT_SECONDS = 0.1
REFERENCE_DT_SECONDS = 1.0
FAR_ACTION_HOLD_STEPS = 10
DEFAULT_SWITCH_DISTANCE_M = 200.0
BASE_GAMMA_PER_SECOND = 0.99
GAMMA_PER_NEAR_DECISION = BASE_GAMMA_PER_SECOND ** DT_SECONDS
REWARD_VERSION = "v15_dt"

SOURCE_EXPERIMENT = (
    "sac_closing_v15_trial16_terminal_phase_balanced_guarded600k_seed0"
)
SOURCE_DIR = (
    CLOSING_DIR / "models" / "multiseed" / SOURCE_EXPERIMENT / "seed_0"
)
DEFAULT_SOURCE_MODEL = SOURCE_DIR / "model_0100000_steps.zip"
DEFAULT_SOURCE_VECNORMALIZE = (
    SOURCE_DIR / "model_0100000_steps_vecnormalize.pkl"
)

EXPERIMENT_FAMILY = "adaptive_dt01_terminal"
MODEL_ROOT = SCRIPT_DIR / "models" / EXPERIMENT_FAMILY
RESULT_ROOT = SCRIPT_DIR / "results" / EXPERIMENT_FAMILY
LOG_ROOT = SCRIPT_DIR / "logs" / EXPERIMENT_FAMILY
DEFAULT_EXPERIMENT_NAME = (
    "sac_closing_adaptive_dt01_terminal_last_200k_seed0"
)


class ResetSpecificTimeLimit(gym.Wrapper):
    """Use a short horizon only for augmented near-P3 training episodes."""

    def __init__(self, env, near_p3_max_time_seconds=600.0):
        super().__init__(env)
        self.near_p3_max_time_seconds = float(near_p3_max_time_seconds)
        if self.near_p3_max_time_seconds <= 0.0:
            raise ValueError("near_p3_max_time_seconds must be positive.")
        self._reset_distribution = FixedNearP3ResetWrapper.RESET_STANDARD
        self._episode_start_time = 0.0

    @property
    def core_env(self):
        return self.env.unwrapped.core_env

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._reset_distribution = info.get(
            "reset_distribution",
            FixedNearP3ResetWrapper.RESET_STANDARD,
        )
        self._episode_start_time = float(self.core_env.time)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        elapsed = float(self.core_env.time - self._episode_start_time)
        near_p3_timeout = bool(
            self._reset_distribution == FixedNearP3ResetWrapper.RESET_NEAR_P3
            and not terminated
            and not truncated
            and elapsed + 1e-9 >= self.near_p3_max_time_seconds
        )
        if near_p3_timeout:
            truncated = True
            info = dict(info)
            info["time_limit"] = True
            info["termination_reason"] = "near_p3_training_timeout"
            info["near_p3_training_timeout"] = True
            timeout_penalty = -200.0
            reward = float(reward) + timeout_penalty
            components = dict(info.get("reward_components", {}))
            components["reward_timeout"] = float(
                components.get("reward_timeout", 0.0) + timeout_penalty
            )
            components["reward_total"] = float(reward)
            info["reward_components"] = components
        else:
            info = dict(info)
            info["near_p3_training_timeout"] = False
        info["training_episode_time_limit_seconds"] = float(
            self.near_p3_max_time_seconds
            if self._reset_distribution == FixedNearP3ResetWrapper.RESET_NEAR_P3
            else self.env.unwrapped.max_time
        )
        return observation, float(reward), terminated, truncated, info


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Continue the audited closing SAC at 0.1 s with one-second far "
            "action holds and true 10 Hz terminal control."
        )
    )
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument(
        "--source-vecnormalize",
        type=Path,
        default=DEFAULT_SOURCE_VECNORMALIZE,
    )
    parser.add_argument("--training-timesteps", type=int, default=200_000)
    parser.add_argument("--max-replay-warmup-steps", type=int, default=100_000)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[25_000, 50_000, 100_000, 150_000, 200_000],
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=[100, 101, 102],
    )
    parser.add_argument("--max-eval-steps", type=int, default=120_000)
    parser.add_argument("--max-time", type=float, default=12_000.0)
    parser.add_argument(
        "--adaptive-switch-distance-m",
        type=float,
        default=DEFAULT_SWITCH_DISTANCE_M,
    )
    parser.add_argument(
        "--near-p3-max-time",
        type=float,
        default=600.0,
    )

    parser.add_argument("--global-buffer-size", type=int, default=300_000)
    parser.add_argument("--anchor-buffer-size", type=int, default=80_000)
    parser.add_argument("--capture-buffer-size", type=int, default=120_000)
    parser.add_argument("--priority-buffer-size", type=int, default=30_000)
    parser.add_argument("--target-ratio-threshold", type=float, default=10.0)
    parser.add_argument("--anchor-batch-fraction", type=float, default=0.25)
    parser.add_argument("--capture-batch-fraction", type=float, default=0.50)
    parser.add_argument("--priority-capture-fraction", type=float, default=0.0)
    parser.add_argument("--minimum-anchor-transitions", type=int, default=24_000)
    parser.add_argument("--minimum-anchor-episodes", type=int, default=2)
    parser.add_argument("--minimum-capture-transitions", type=int, default=8_000)
    parser.add_argument("--minimum-capture-episodes", type=int, default=3)

    parser.add_argument(
        "--warmup-near-p3-reset-fraction",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--near-p3-reset-fraction",
        type=float,
        default=0.35,
    )
    parser.add_argument("--near-p3-min-ratio", type=float, default=1.05)
    parser.add_argument("--near-p3-max-ratio", type=float, default=6.0)
    parser.add_argument("--near-p3-other-max-ratio", type=float, default=0.75)

    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=GAMMA_PER_NEAR_DECISION)
    parser.add_argument(
        "--fixed-ent-coef",
        type=float,
        default=1.7e-4,
        help="Fixed low entropy coefficient; disables automatic alpha updates.",
    )
    parser.add_argument("--early-stop-after-steps", type=int, default=50_000)
    parser.add_argument("--early-stop-failure-evals", type=int, default=1)
    parser.add_argument("--early-stop-regression-evals", type=int, default=2)
    parser.add_argument(
        "--early-stop-regression-fraction",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def validate_args(args):
    positive_integer_fields = (
        "training_timesteps",
        "max_replay_warmup_steps",
        "max_eval_steps",
        "global_buffer_size",
        "anchor_buffer_size",
        "capture_buffer_size",
        "priority_buffer_size",
        "minimum_anchor_transitions",
        "minimum_anchor_episodes",
        "minimum_capture_transitions",
        "minimum_capture_episodes",
        "batch_size",
        "train_freq",
        "gradient_steps",
        "early_stop_failure_evals",
        "early_stop_regression_evals",
    )
    for field in positive_integer_fields:
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")
    if args.early_stop_after_steps < 0:
        raise ValueError("--early-stop-after-steps cannot be negative.")
    for field in (
        "max_time",
        "adaptive_switch_distance_m",
        "near_p3_max_time",
        "learning_rate",
        "fixed_ent_coef",
    ):
        if not np.isfinite(getattr(args, field)) or getattr(args, field) <= 0.0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")
    if not 0.0 < args.gamma < 1.0:
        raise ValueError("--gamma must lie in (0, 1).")
    if not 0.0 < args.tau <= 1.0:
        raise ValueError("--tau must lie in (0, 1].")
    for field in (
        "anchor_batch_fraction",
        "capture_batch_fraction",
        "priority_capture_fraction",
        "warmup_near_p3_reset_fraction",
        "near_p3_reset_fraction",
    ):
        if not 0.0 <= float(getattr(args, field)) <= 1.0:
            raise ValueError(f"--{field.replace('_', '-')} must lie in [0, 1].")
    if args.anchor_batch_fraction + args.capture_batch_fraction > 1.0:
        raise ValueError("Anchor and capture batch fractions cannot exceed one.")
    if args.target_ratio_threshold <= 1.0:
        raise ValueError("--target-ratio-threshold must exceed one.")
    if args.near_p3_min_ratio <= 1.0:
        raise ValueError("--near-p3-min-ratio must exceed one.")
    if args.near_p3_max_ratio <= args.near_p3_min_ratio:
        raise ValueError("--near-p3-max-ratio must exceed its minimum.")
    if not 0.0 <= args.near_p3_other_max_ratio < 1.0:
        raise ValueError("--near-p3-other-max-ratio must lie in [0, 1).")
    if args.near_p3_max_time >= args.max_time:
        raise ValueError("--near-p3-max-time must be smaller than --max-time.")
    if not args.validation_seeds:
        raise ValueError("At least one validation seed is required.")
    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        raise ValueError("Validation seeds must be unique.")
    if args.training_seed in set(args.validation_seeds):
        raise ValueError("Training and validation seeds must be disjoint.")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*",
        args.experiment_name,
    ):
        raise ValueError("--experiment-name contains unsafe characters.")
    args.source_model = args.source_model.expanduser().resolve()
    args.source_vecnormalize = args.source_vecnormalize.expanduser().resolve()
    if not args.source_model.is_file():
        raise FileNotFoundError(args.source_model)
    if not args.source_vecnormalize.is_file():
        raise FileNotFoundError(args.source_vecnormalize)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    resolve_checkpoint_steps(args)


def make_training_env(args):
    env = Docking3DoFGymEnv(
        phase="closing",
        mode="training",
        dt=DT_SECONDS,
        max_time=args.max_time,
        integrator="rk4",
        normalize_action=True,
        seed=args.training_seed,
        closing_reward_version=REWARD_VERSION,
        action_hold_steps=FAR_ACTION_HOLD_STEPS,
        adaptive_action_hold_distance_m=args.adaptive_switch_distance_m,
    )
    env = FixedNearP3ResetWrapper(
        env,
        near_p3_fraction=args.warmup_near_p3_reset_fraction,
        minimum_target_ratio=args.near_p3_min_ratio,
        maximum_target_ratio=args.near_p3_max_ratio,
        other_component_max_ratio=args.near_p3_other_max_ratio,
        deterministic_mixture=True,
    )
    env = ResetSpecificTimeLimit(
        env,
        near_p3_max_time_seconds=args.near_p3_max_time,
    )
    return Monitor(env)


def make_validation_env(
    *,
    args,
    seed,
    closing_reward_version=None,
    action_hold_steps=None,
    adaptive_action_hold_distance_m=None,
):
    """Create an unaugmented full-P2 environment matching deployment."""
    del closing_reward_version, action_hold_steps, adaptive_action_hold_distance_m
    return Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        dt=DT_SECONDS,
        max_time=args.max_time,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
        closing_reward_version=REWARD_VERSION,
        action_hold_steps=FAR_ACTION_HOLD_STEPS,
        adaptive_action_hold_distance_m=args.adaptive_switch_distance_m,
    )


def freeze_entropy_coefficient(model, coefficient):
    """Replace automatic entropy tuning with a fixed source-scale alpha."""
    coefficient = float(coefficient)
    model.ent_coef = coefficient
    model.ent_coef_optimizer = None
    model.log_ent_coef = None
    model.ent_coef_tensor = torch.tensor(coefficient, device=model.device)


def configure_model(model, args):
    model.learning_starts = 0
    model.learning_rate = args.learning_rate
    model.lr_schedule = FloatSchedule(args.learning_rate)
    model.batch_size = args.batch_size
    model.tau = args.tau
    model.gamma = args.gamma
    model.train_freq = TrainFreq(args.train_freq, TrainFrequencyUnit.STEP)
    model.gradient_steps = args.gradient_steps
    model.target_update_interval = 1
    freeze_entropy_coefficient(model, args.fixed_ent_coef)


def install_deterministic_warmup_sampler(model):
    """Temporarily collect immutable replay anchors without action noise."""
    def sample_action(self, learning_starts, action_noise=None, n_envs=1):
        del learning_starts, action_noise, n_envs
        if self._last_obs is None:
            raise RuntimeError("The warmup observation is not initialized.")
        unscaled_action, _ = self.predict(self._last_obs, deterministic=True)
        scaled_action = self.policy.scale_action(unscaled_action)
        action = self.policy.unscale_action(scaled_action)
        return action, scaled_action

    model._sample_action = MethodType(sample_action, model)


def remove_deterministic_warmup_sampler(model):
    """Restore the class implementation without storing a bound self-reference."""
    if "_sample_action" in model.__dict__:
        del model.__dict__["_sample_action"]


def output_paths(args):
    model_dir = MODEL_ROOT / args.experiment_name
    result_dir = RESULT_ROOT / args.experiment_name
    log_dir = LOG_ROOT / args.experiment_name
    return model_dir, result_dir, log_dir


def build_config(args, checkpoints, source_num_timesteps, source_alpha):
    return {
        "experiment_name": args.experiment_name,
        "purpose": (
            "warm-started SAC with 0.1 s integration, adaptive "
            "far/terminal cadence, and simultaneous-target training coverage"
        ),
        "method": "warm-started adaptive-cadence SAC",
        "strict_from_scratch_vanilla_sac": False,
        "cbf_used": False,
        "lqr_actions_or_transitions_used": False,
        "source_model": str(args.source_model),
        "source_vecnormalize": str(args.source_vecnormalize),
        "source_model_sha256": file_sha256(args.source_model),
        "source_vecnormalize_sha256": file_sha256(args.source_vecnormalize),
        "source_model_num_timesteps": int(source_num_timesteps),
        "source_entropy_coefficient": float(source_alpha),
        "training_seed": args.training_seed,
        "validation_seeds": list(args.validation_seeds),
        "training_timesteps": args.training_timesteps,
        "maximum_replay_warmup_steps": args.max_replay_warmup_steps,
        "checkpoint_steps": [0, *checkpoints],
        "dynamics_and_cadence": {
            "integration_dt_seconds": DT_SECONDS,
            "far_action_hold_steps": FAR_ACTION_HOLD_STEPS,
            "far_decision_interval_seconds": (
                DT_SECONDS * FAR_ACTION_HOLD_STEPS
            ),
            "near_decision_interval_seconds": DT_SECONDS,
            "adaptive_switch_distance_m": args.adaptive_switch_distance_m,
            "warning_forces_ten_hz_control": True,
            "maximum_physical_episode_time_seconds": args.max_time,
        },
        "reward": {
            "version": REWARD_VERSION,
            "persistent_costs_scaled_by_dt": True,
            "held_step_rewards_aggregated_by_environment": True,
        },
        "discount": {
            "gamma": args.gamma,
            "base_gamma_per_second": BASE_GAMMA_PER_SECOND,
            "near_ten_hz_conversion": "base_gamma_per_second ** 0.1",
            "adaptive_macro_step_caveat": (
                "SB3 uses one fixed gamma; the selected value is physically "
                "matched in the terminal 10 Hz region and deliberately gives "
                "the far one-second macro-MDP a longer planning horizon."
            ),
        },
        "optimizer": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "update_to_data_ratio": args.gradient_steps / args.train_freq,
            "tau": args.tau,
            "entropy_coefficient": args.fixed_ent_coef,
            "automatic_entropy_tuning": False,
        },
        "reset_augmentation": {
            "warmup_near_p3_fraction": args.warmup_near_p3_reset_fraction,
            "training_near_p3_fraction": args.near_p3_reset_fraction,
            "minimum_target_ratio": args.near_p3_min_ratio,
            "maximum_target_ratio": args.near_p3_max_ratio,
            "other_component_max_ratio": args.near_p3_other_max_ratio,
            "deterministic_mixture": True,
            "near_p3_max_time_seconds": args.near_p3_max_time,
            "standard_p2_max_time_seconds": args.max_time,
            "validation_uses_augmented_resets": False,
        },
        "replay": {
            "type": "SimultaneousTargetReplayBuffer",
            "global_buffer_size": args.global_buffer_size,
            "anchor_buffer_size": args.anchor_buffer_size,
            "capture_buffer_size": args.capture_buffer_size,
            "priority_buffer_size": args.priority_buffer_size,
            "target_ratio_threshold": args.target_ratio_threshold,
            "global_batch_fraction": (
                1.0 - args.anchor_batch_fraction - args.capture_batch_fraction
            ),
            "anchor_batch_fraction": args.anchor_batch_fraction,
            "capture_batch_fraction": args.capture_batch_fraction,
            "priority_capture_fraction": args.priority_capture_fraction,
            "minimum_anchor_transitions": args.minimum_anchor_transitions,
            "minimum_anchor_episodes": args.minimum_anchor_episodes,
            "minimum_capture_transitions": args.minimum_capture_transitions,
            "minimum_capture_episodes": args.minimum_capture_episodes,
            "warmup_policy_deterministic": True,
        },
        "normalization": {
            "source_statistics_loaded": True,
            "statistics_frozen": True,
            "observation_normalization": True,
            "reward_normalization": True,
        },
        "checkpoint_selection": (
            "success, safety, best simultaneous target ratio, 2x-target "
            "occupancy/dwell, final target ratio, safety score, return"
        ),
        "early_stopping": {
            "active_after_steps": args.early_stop_after_steps,
            "failure_evaluations": args.early_stop_failure_evals,
            "regression_evaluations": args.early_stop_regression_evals,
            "regression_fraction": args.early_stop_regression_fraction,
        },
        "device": args.device,
    }


def source_model_metadata(path):
    model = SAC.load(str(path), device="cpu")
    source_num_timesteps = int(model.num_timesteps)
    source_alpha = (
        float(torch.exp(model.log_ent_coef.detach()).cpu())
        if model.log_ent_coef is not None
        else float(model.ent_coef_tensor.detach().cpu())
    )
    del model
    return source_num_timesteps, source_alpha


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    checkpoints = resolve_checkpoint_steps(args)
    source_num_timesteps, source_alpha = source_model_metadata(args.source_model)
    config = build_config(
        args,
        checkpoints=checkpoints,
        source_num_timesteps=source_num_timesteps,
        source_alpha=source_alpha,
    )
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return config

    model_dir, result_dir, log_dir = output_paths(args)
    for path in (model_dir, result_dir, log_dir):
        if path.exists():
            raise FileExistsError(
                f"Refusing to reuse experiment output directory: {path}"
            )
    for path in (model_dir, result_dir, log_dir):
        path.mkdir(parents=True)
    config_path = result_dir / "experiment_config.json"
    write_json(str(config_path), config)

    training_env = DummyVecEnv([lambda: make_training_env(args)])
    training_env = VecNormalize.load(
        str(args.source_vecnormalize),
        training_env,
    )
    training_env.training = False
    training_env.norm_reward = True
    reset_wrapper = find_reset_wrapper(training_env)

    model = SAC.load(
        str(args.source_model),
        env=training_env,
        device=args.device,
        seed=args.training_seed,
        tensorboard_log=str(log_dir),
    )
    configure_model(model, args)
    replay_buffer = make_replay_buffer(model, args)

    checkpoint_callback = GuardedV2CheckpointCallback(
        training_seed=args.training_seed,
        checkpoint_steps=checkpoints,
        validation_seeds=args.validation_seeds,
        max_eval_steps=args.max_eval_steps,
        closing_reward_version=REWARD_VERSION,
        model_dir=str(model_dir),
        result_dir=str(result_dir),
        model_name=args.experiment_name,
        rank_function=simultaneous_capture_checkpoint_rank,
        selection_metric=config["checkpoint_selection"],
        early_stop_after_steps=args.early_stop_after_steps,
        failure_evals=args.early_stop_failure_evals,
        regression_evals=args.early_stop_regression_evals,
        regression_fraction=args.early_stop_regression_fraction,
        action_hold_steps=FAR_ACTION_HOLD_STEPS,
        adaptive_action_hold_distance_m=args.adaptive_switch_distance_m,
        evaluation_env_factory=partial(make_validation_env, args=args),
    )
    return_callback = TrainingReturnCallback(
        training_seed=args.training_seed,
        result_dir=str(result_dir),
        model_name=args.experiment_name,
    )

    print("=" * 80)
    print("Last-chance adaptive-cadence closing SAC at dt=0.1 s")
    print("=" * 80)
    print(f"Experiment:          {args.experiment_name}")
    print(f"Source alpha:        {source_alpha:.8g}")
    print(f"Fixed training alpha:{args.fixed_ent_coef:.8g}")
    print(f"Far control:         1 Hz above {args.adaptive_switch_distance_m:g} m")
    print(f"Terminal control:    10 Hz at/below {args.adaptive_switch_distance_m:g} m")
    print(f"Gamma:               {args.gamma:.12f}")
    print(f"Gradient interactions: {args.training_timesteps:,}")
    print(f"Checkpoints:         {[0, *checkpoints]}")
    print("=" * 80)

    try:
        readiness_callback = ReplayV2ReadinessCallback()
        install_deterministic_warmup_sampler(model)
        model.gradient_steps = 0
        try:
            model.learn(
                total_timesteps=args.max_replay_warmup_steps,
                reset_num_timesteps=True,
                tb_log_name=f"{args.experiment_name}_deterministic_warmup",
                progress_bar=args.progress_bar,
                callback=readiness_callback,
            )
        finally:
            remove_deterministic_warmup_sampler(model)

        warmup_steps = int(model.num_timesteps)
        warmup_record = {
            "ready_for_gradient_training": bool(readiness_callback.ready),
            "warmup_interaction_steps": warmup_steps,
            "reset_statistics": reset_wrapper.reset_statistics(),
            **replay_buffer.replay_statistics(),
        }
        write_json(
            str(result_dir / "replay_buffer_after_warmup.json"),
            warmup_record,
        )
        config["actual_replay_warmup_steps"] = warmup_steps
        config["replay_ready_after_warmup"] = bool(readiness_callback.ready)
        if not readiness_callback.ready:
            write_json(str(config_path), config)
            raise RuntimeError(
                "Replay warmup reached its limit before anchor/capture "
                "readiness; no gradient update was performed."
            )

        replay_buffer.freeze_anchor()
        reset_wrapper.set_near_p3_fraction(args.near_p3_reset_fraction)
        config["anchor_frozen_at_size"] = replay_buffer.anchor_size()
        write_json(str(config_path), config)

        configure_model(model, args)
        model.learn(
            total_timesteps=args.training_timesteps,
            reset_num_timesteps=True,
            tb_log_name=args.experiment_name,
            progress_bar=args.progress_bar,
            callback=[return_callback, checkpoint_callback],
        )
        model.save(str(model_dir / "final_model"))
        training_env.save(str(model_dir / "final_vecnormalize.pkl"))
        write_json(
            str(result_dir / "replay_buffer_final.json"),
            replay_buffer.replay_statistics(),
        )
        write_json(
            str(result_dir / "near_p3_reset_statistics.json"),
            reset_wrapper.reset_statistics(),
        )
        config["actual_gradient_training_steps"] = int(model.num_timesteps)
        config["early_stop_triggered"] = bool(checkpoint_callback.stop_requested)
        config["early_stop_reason"] = checkpoint_callback.stop_reason
        write_json(str(config_path), config)
    finally:
        training_env.close()

    best_checkpoint, best_mean_return, best_episode = (
        checkpoint_callback.get_best_checkpoint()
    )
    return_callback.set_best_checkpoint(best_episode, best_checkpoint)
    return_callback.save_outputs()
    print(f"Best additional checkpoint: {best_checkpoint}")
    print(f"Best validation return:      {best_mean_return:.3f}")
    if checkpoint_callback.stop_requested:
        print(f"Early stop: {checkpoint_callback.stop_reason}")
    print(f"Results: {result_dir}")
    print(f"Models:  {model_dir}")
    return config


if __name__ == "__main__":
    main()
