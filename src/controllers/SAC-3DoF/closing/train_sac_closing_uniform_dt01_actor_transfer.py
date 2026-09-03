"""Uniform-0.1 s closing SAC transfer with actor protection.

Only the actor weights are copied from the strongest audited non-LQR 1 Hz
checkpoint.  Twin critics, target critics, optimizers, replay, and reward
normalization are new.  Replay warmup is deterministic and episode-gated;
critics are then calibrated while the actor is exactly frozen before cautious
online SAC updates begin.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from functools import partial
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
SRC_PATH = REPO_ROOT / "src"
for path in (SCRIPT_DIR, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from envs.docking3dof.gym_env import Docking3DoFGymEnv
from episode_stratified_replay_buffer import EpisodeStratifiedReplayBuffer
from near_p3_reset_wrapper import FixedNearP3ResetWrapper
from train_sac_closing_adaptive_dt01_terminal import (
    DEFAULT_SOURCE_MODEL,
    DEFAULT_SOURCE_VECNORMALIZE,
    ResetSpecificTimeLimit,
    install_deterministic_warmup_sampler,
    remove_deterministic_warmup_sampler,
)
from train_sac_closing_multiseed import TrainingReturnCallback, write_json
from train_sac_closing_v15_phase_balanced_replay import (
    file_sha256,
    resolve_checkpoint_steps,
    write_csv,
)
from train_sac_closing_v15_phase_balanced_replay_v2 import (
    GuardedV2CheckpointCallback,
    find_reset_wrapper,
    simultaneous_capture_checkpoint_rank,
)


DT_SECONDS = 0.1
BASE_GAMMA_PER_SECOND = 0.99
GAMMA_PER_STEP = BASE_GAMMA_PER_SECOND**DT_SECONDS
REWARD_VERSION = "v15_dt"
ACTION_HOLD_STEPS = 1

EXPERIMENT_FAMILY = "uniform_dt01_actor_transfer"
MODEL_ROOT = SCRIPT_DIR / "models" / EXPERIMENT_FAMILY
RESULT_ROOT = SCRIPT_DIR / "results" / EXPERIMENT_FAMILY
LOG_ROOT = SCRIPT_DIR / "logs" / EXPERIMENT_FAMILY
DEFAULT_EXPERIMENT_NAME = "sac_closing_uniform_dt01_actor_only_seed0"


class FrozenObservationFreshRewardVecNormalize(VecNormalize):
    """Preserve source actor inputs while fitting new reward statistics."""

    def reset(self):
        obs = self.venv.reset()
        if not isinstance(obs, (np.ndarray, dict)):
            raise TypeError("VecNormalize reset returned an unsupported observation.")
        self.old_obs = obs
        self.returns = np.zeros(self.num_envs)
        return self.normalize_obs(obs)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        if not isinstance(obs, (np.ndarray, dict)):
            raise TypeError("VecNormalize step returned an unsupported observation.")
        self.old_obs = obs
        self.old_reward = rewards
        obs = self.normalize_obs(obs)
        if self.training:
            self._update_reward(rewards)
        rewards = self.normalize_reward(rewards)
        for index, done in enumerate(dones):
            if done and "terminal_observation" in infos[index]:
                infos[index]["terminal_observation"] = self.normalize_obs(
                    infos[index]["terminal_observation"]
                )
        self.returns[dones] = 0
        return obs, rewards, dones, infos


class ProtectedActorSAC(SAC):
    """SAC with separate learning rates and an explicit actor update gate."""

    def __init__(
        self,
        *args,
        actor_learning_rate=5.0e-7,
        critic_learning_rate=1.0e-5,
        actor_update_interval=20,
        **kwargs,
    ):
        self.actor_learning_rate = float(actor_learning_rate)
        self.critic_learning_rate = float(critic_learning_rate)
        self.actor_update_interval = int(actor_update_interval)
        self.actor_updates_enabled = False
        self.actor_gradient_updates = 0
        self.critic_gradient_updates = 0
        self.last_train_metrics = {}
        if self.actor_learning_rate <= 0.0:
            raise ValueError("actor_learning_rate must be positive.")
        if self.critic_learning_rate <= 0.0:
            raise ValueError("critic_learning_rate must be positive.")
        if self.actor_update_interval <= 0:
            raise ValueError("actor_update_interval must be positive.")
        super().__init__(*args, **kwargs)

    def set_actor_updates_enabled(self, enabled):
        self.actor_updates_enabled = bool(enabled)

    @staticmethod
    def _set_optimizer_learning_rate(optimizer, learning_rate):
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = float(learning_rate)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self._set_optimizer_learning_rate(
            self.critic.optimizer, self.critic_learning_rate
        )
        self._set_optimizer_learning_rate(
            self.actor.optimizer, self.actor_learning_rate
        )

        ent_coefs = []
        actor_losses = []
        critic_losses = []
        actor_updates_this_call = 0

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )
            if self.use_sde:
                self.actor.reset_noise()
            if self.ent_coef_optimizer is not None or self.log_ent_coef is not None:
                raise RuntimeError("ProtectedActorSAC requires a fixed entropy coefficient.")
            ent_coef = self.ent_coef_tensor
            ent_coefs.append(float(ent_coef.item()))

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = torch.cat(
                    self.critic_target(
                        replay_data.next_observations, next_actions
                    ),
                    dim=1,
                )
                next_q_values, _ = torch.min(
                    next_q_values, dim=1, keepdim=True
                )
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(
                    -1, 1
                )
                target_q_values = replay_data.rewards + (
                    (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            critic_losses.append(float(critic_loss.item()))
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            self.critic_gradient_updates += 1

            absolute_update = self._n_updates + gradient_step + 1
            update_actor = bool(
                self.actor_updates_enabled
                and absolute_update % self.actor_update_interval == 0
            )
            if update_actor:
                actions_pi, log_prob = self.actor.action_log_prob(
                    replay_data.observations
                )
                log_prob = log_prob.reshape(-1, 1)
                q_values_pi = torch.cat(
                    self.critic(replay_data.observations, actions_pi), dim=1
                )
                min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(float(actor_loss.item()))
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                self.actor_gradient_updates += 1
                actor_updates_this_call += 1

            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(
                    self.batch_norm_stats, self.batch_norm_stats_target, 1.0
                )

        self._n_updates += gradient_steps
        self.last_train_metrics = {
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else np.nan,
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else np.nan,
            "critic_updates": int(gradient_steps),
            "actor_updates": int(actor_updates_this_call),
        }
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", float(np.mean(ent_coefs)))
        self.logger.record("train/critic_loss", self.last_train_metrics["critic_loss"])
        self.logger.record(
            "train/actor_update_fraction",
            actor_updates_this_call / max(gradient_steps, 1),
        )
        if actor_losses:
            self.logger.record("train/actor_loss", self.last_train_metrics["actor_loss"])


class EpisodeReplayReadinessCallback(BaseCallback):
    """Stop deterministic collection only after episode-level gates pass."""

    def __init__(self):
        super().__init__()
        self.ready = False
        self.ready_at_timestep = None

    def _on_step(self):
        if self.model.replay_buffer.replay_ready:
            self.ready = True
            self.ready_at_timestep = int(self.num_timesteps)
            return False
        return True


def tensor_state_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Transfer only the audited 1 Hz actor into uniform 0.1 s SAC, "
            "calibrate fresh critics while the actor is frozen, then train "
            "with episode-aware P3-ratio replay."
        )
    )
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument(
        "--source-vecnormalize",
        type=Path,
        default=DEFAULT_SOURCE_VECNORMALIZE,
    )
    parser.add_argument("--training-timesteps", type=int, default=200_000)
    parser.add_argument("--max-replay-warmup-steps", type=int, default=800_000)
    parser.add_argument("--critic-pretrain-updates", type=int, default=20_000)
    parser.add_argument("--critic-pretrain-log-interval", type=int, default=2_000)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[10_000, 25_000, 50_000, 100_000, 150_000, 200_000],
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument(
        "--validation-seeds", type=int, nargs="+", default=[100, 101, 102]
    )
    parser.add_argument("--max-eval-steps", type=int, default=120_000)
    parser.add_argument("--max-time", type=float, default=12_000.0)
    parser.add_argument("--near-p3-max-time", type=float, default=600.0)
    parser.add_argument("--near-p3-reset-fraction", type=float, default=0.80)
    parser.add_argument("--near-p3-min-ratio", type=float, default=1.05)
    parser.add_argument("--near-p3-max-ratio", type=float, default=6.0)
    parser.add_argument("--near-p3-other-max-ratio", type=float, default=0.75)

    parser.add_argument("--global-buffer-size", type=int, default=300_000)
    parser.add_argument("--target-le-2-capacity", type=int, default=25_000)
    parser.add_argument("--target-2-to-5-capacity", type=int, default=75_000)
    parser.add_argument("--target-5-to-10-capacity", type=int, default=75_000)
    parser.add_argument("--approach-capacity", type=int, default=400_000)
    parser.add_argument("--target-le-2-fraction", type=float, default=0.10)
    parser.add_argument("--target-2-to-5-fraction", type=float, default=0.30)
    parser.add_argument("--target-5-to-10-fraction", type=float, default=0.25)
    parser.add_argument("--approach-fraction", type=float, default=0.35)
    parser.add_argument("--minimum-completed-episodes", type=int, default=20)
    parser.add_argument("--minimum-standard-episodes", type=int, default=4)
    parser.add_argument("--minimum-near-p3-episodes", type=int, default=16)
    parser.add_argument("--minimum-target-le-2-episodes", type=int, default=0)
    parser.add_argument("--minimum-target-2-to-5-episodes", type=int, default=8)
    parser.add_argument("--minimum-target-5-to-10-episodes", type=int, default=4)
    parser.add_argument("--minimum-approach-episodes", type=int, default=4)

    parser.add_argument("--actor-learning-rate", type=float, default=5.0e-7)
    parser.add_argument("--critic-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--actor-update-interval", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=GAMMA_PER_STEP)
    parser.add_argument("--fixed-ent-coef", type=float, default=1.7e-4)
    parser.add_argument("--early-stop-after-steps", type=int, default=10_000)
    parser.add_argument("--early-stop-failure-evals", type=int, default=1)
    parser.add_argument("--early-stop-regression-evals", type=int, default=2)
    parser.add_argument(
        "--early-stop-regression-fraction", type=float, default=0.15
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete only this experiment's existing model, result, and log "
            "directories before starting a clean run."
        ),
    )
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def validate_args(args):
    positive_integers = (
        "training_timesteps",
        "max_replay_warmup_steps",
        "critic_pretrain_updates",
        "critic_pretrain_log_interval",
        "global_buffer_size",
        "target_le_2_capacity",
        "target_2_to_5_capacity",
        "target_5_to_10_capacity",
        "approach_capacity",
        "minimum_completed_episodes",
        "batch_size",
        "train_freq",
        "gradient_steps",
        "actor_update_interval",
        "max_eval_steps",
        "early_stop_failure_evals",
        "early_stop_regression_evals",
    )
    for field in positive_integers:
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")
    episode_minimum_fields = (
        "minimum_standard_episodes",
        "minimum_near_p3_episodes",
        "minimum_target_le_2_episodes",
        "minimum_target_2_to_5_episodes",
        "minimum_target_5_to_10_episodes",
        "minimum_approach_episodes",
    )
    for field in episode_minimum_fields:
        if int(getattr(args, field)) < 0:
            raise ValueError(f"--{field.replace('_', '-')} cannot be negative.")
    if args.early_stop_after_steps < 0:
        raise ValueError("--early-stop-after-steps cannot be negative.")
    for field in (
        "max_time",
        "near_p3_max_time",
        "actor_learning_rate",
        "critic_learning_rate",
        "fixed_ent_coef",
    ):
        value = float(getattr(args, field))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")
    if args.near_p3_max_time >= args.max_time:
        raise ValueError("--near-p3-max-time must be smaller than --max-time.")
    if not 0.0 < args.gamma < 1.0:
        raise ValueError("--gamma must lie in (0, 1).")
    if not 0.0 < args.tau <= 1.0:
        raise ValueError("--tau must lie in (0, 1].")
    if not 0.0 <= args.near_p3_reset_fraction <= 1.0:
        raise ValueError("--near-p3-reset-fraction must lie in [0, 1].")
    if args.near_p3_min_ratio <= 1.0:
        raise ValueError("--near-p3-min-ratio must exceed one.")
    if args.near_p3_max_ratio <= args.near_p3_min_ratio:
        raise ValueError("--near-p3-max-ratio must exceed its minimum.")
    if not 0.0 <= args.near_p3_other_max_ratio < 1.0:
        raise ValueError("--near-p3-other-max-ratio must lie in [0, 1).")
    fractions = (
        args.target_le_2_fraction,
        args.target_2_to_5_fraction,
        args.target_5_to_10_fraction,
        args.approach_fraction,
    )
    if any(value < 0.0 for value in fractions) or not np.isclose(
        sum(fractions), 1.0
    ):
        raise ValueError("Replay stratum fractions must be nonnegative and sum to one.")
    if not args.validation_seeds or len(set(args.validation_seeds)) != len(
        args.validation_seeds
    ):
        raise ValueError("Validation seeds must be nonempty and unique.")
    if args.training_seed in set(args.validation_seeds):
        raise ValueError("Training and validation seeds must be disjoint.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.experiment_name):
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


def make_base_env(args, *, mode, seed):
    return Docking3DoFGymEnv(
        phase="closing",
        mode=mode,
        dt=DT_SECONDS,
        max_time=args.max_time,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
        closing_reward_version=REWARD_VERSION,
        action_hold_steps=ACTION_HOLD_STEPS,
        adaptive_action_hold_distance_m=None,
    )


def make_training_env(args):
    env = make_base_env(args, mode="training", seed=args.training_seed)
    env = FixedNearP3ResetWrapper(
        env,
        near_p3_fraction=args.near_p3_reset_fraction,
        minimum_target_ratio=args.near_p3_min_ratio,
        maximum_target_ratio=args.near_p3_max_ratio,
        other_component_max_ratio=args.near_p3_other_max_ratio,
        deterministic_mixture=True,
    )
    env = ResetSpecificTimeLimit(
        env, near_p3_max_time_seconds=args.near_p3_max_time
    )
    return Monitor(env)


def make_validation_env(
    *, args, seed, closing_reward_version=None, action_hold_steps=None
):
    del closing_reward_version, action_hold_steps
    return make_base_env(args, mode="testing", seed=seed)


def make_vecnormalize(args, raw_vec_env):
    source_raw = DummyVecEnv([lambda: make_training_env(args)])
    source_vecnorm = VecNormalize.load(str(args.source_vecnormalize), source_raw)
    try:
        source_obs_rms = deepcopy(source_vecnorm.obs_rms)
        clip_obs = float(source_vecnorm.clip_obs)
        clip_reward = float(source_vecnorm.clip_reward)
        epsilon = float(source_vecnorm.epsilon)
    finally:
        source_vecnorm.close()

    vecnorm = FrozenObservationFreshRewardVecNormalize(
        raw_vec_env,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=clip_obs,
        clip_reward=clip_reward,
        gamma=args.gamma,
        epsilon=epsilon,
    )
    vecnorm.obs_rms = source_obs_rms
    return vecnorm


def replay_kwargs(args):
    return {
        "target_le_2_capacity": args.target_le_2_capacity,
        "target_2_to_5_capacity": args.target_2_to_5_capacity,
        "target_5_to_10_capacity": args.target_5_to_10_capacity,
        "approach_capacity": args.approach_capacity,
        "target_le_2_fraction": args.target_le_2_fraction,
        "target_2_to_5_fraction": args.target_2_to_5_fraction,
        "target_5_to_10_fraction": args.target_5_to_10_fraction,
        "approach_fraction": args.approach_fraction,
        "minimum_completed_episodes": args.minimum_completed_episodes,
        "minimum_standard_episodes": args.minimum_standard_episodes,
        "minimum_near_p3_episodes": args.minimum_near_p3_episodes,
        "minimum_target_le_2_episodes": args.minimum_target_le_2_episodes,
        "minimum_target_2_to_5_episodes": args.minimum_target_2_to_5_episodes,
        "minimum_target_5_to_10_episodes": args.minimum_target_5_to_10_episodes,
        "minimum_approach_episodes": args.minimum_approach_episodes,
    }


def build_model(args, training_env, source_model):
    model = ProtectedActorSAC(
        policy=source_model.policy_class,
        env=training_env,
        learning_rate=args.critic_learning_rate,
        buffer_size=args.global_buffer_size,
        learning_starts=0,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.fixed_ent_coef,
        policy_kwargs=deepcopy(source_model.policy_kwargs),
        replay_buffer_class=EpisodeStratifiedReplayBuffer,
        replay_buffer_kwargs=replay_kwargs(args),
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        actor_update_interval=args.actor_update_interval,
        seed=args.training_seed,
        device=args.device,
        tensorboard_log=str(LOG_ROOT / args.experiment_name),
        verbose=0,
    )
    model.actor.load_state_dict(source_model.actor.state_dict(), strict=True)
    model.set_actor_updates_enabled(False)
    return model


def output_paths(args):
    return (
        MODEL_ROOT / args.experiment_name,
        RESULT_ROOT / args.experiment_name,
        LOG_ROOT / args.experiment_name,
    )


def prepare_output_directories(args):
    """Create clean, exactly scoped experiment directories."""
    paths = output_paths(args)
    roots = (MODEL_ROOT, RESULT_ROOT, LOG_ROOT)
    for path, root in zip(paths, roots):
        resolved_path = path.resolve()
        resolved_root = root.resolve()
        if (
            resolved_path.parent != resolved_root
            or resolved_path.name != args.experiment_name
        ):
            raise RuntimeError(
                f"Refusing unsafe experiment output path: {resolved_path}"
            )

    existing = [path for path in paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to reuse experiment output directories without "
            f"--overwrite: {', '.join(str(path) for path in existing)}"
        )
    if args.overwrite:
        for path in existing:
            if not path.is_dir() or path.is_symlink():
                raise RuntimeError(
                    f"Refusing to overwrite non-directory or symlink: {path}"
                )
        for path in existing:
            print(f"Overwrite enabled; removing prior experiment directory: {path}")
            shutil.rmtree(path)
    for path in paths:
        path.mkdir(parents=True)
    return paths


def build_config(args, checkpoints, source_model):
    return {
        "experiment_name": args.experiment_name,
        "method": "uniform-0.1-s actor-only transfer with protected SAC",
        "source_model": str(args.source_model),
        "source_model_sha256": file_sha256(args.source_model),
        "source_vecnormalize": str(args.source_vecnormalize),
        "source_vecnormalize_sha256": file_sha256(args.source_vecnormalize),
        "source_model_num_timesteps": int(source_model.num_timesteps),
        "source_role": (
            "actor weights and frozen observation statistics only; selected as "
            "the strongest audited non-LQR closing checkpoint"
        ),
        "actor_initialization": "exact source actor weight copy",
        "critic_initialization": "fresh random twin critics and target critics",
        "optimizer_initialization": "fresh actor and critic optimizers",
        "source_critic_loaded": False,
        "source_optimizer_loaded": False,
        "replay_initialization": "fresh episode-aware ratio-stratified replay",
        "training_seed": args.training_seed,
        "validation_seeds": list(args.validation_seeds),
        "training_timesteps": args.training_timesteps,
        "checkpoint_steps": [0, *checkpoints],
        "dynamics": {
            "integration_dt_seconds": DT_SECONDS,
            "action_hold_steps": ACTION_HOLD_STEPS,
            "decision_interval_seconds": DT_SECONDS,
            "mixed_duration_transitions": False,
            "gamma": args.gamma,
            "gamma_definition": "0.99 ** 0.1",
        },
        "reward": {
            "version": REWARD_VERSION,
            "observation_statistics": "copied from source and frozen",
            "reward_statistics": "freshly fitted during replay warmup then frozen",
        },
        "actor_protection": {
            "frozen_during_replay_warmup": True,
            "frozen_during_critic_pretraining": True,
            "critic_pretrain_updates": args.critic_pretrain_updates,
            "actor_learning_rate": args.actor_learning_rate,
            "critic_learning_rate": args.critic_learning_rate,
            "actor_update_interval": args.actor_update_interval,
        },
        "episode_gates": {
            "minimum_completed_episodes": args.minimum_completed_episodes,
            "minimum_standard_episodes": args.minimum_standard_episodes,
            "minimum_near_p3_episodes": args.minimum_near_p3_episodes,
            "minimum_target_le_2_episodes": args.minimum_target_le_2_episodes,
            "minimum_target_2_to_5_episodes": (
                args.minimum_target_2_to_5_episodes
            ),
            "minimum_target_5_to_10_episodes": (
                args.minimum_target_5_to_10_episodes
            ),
            "minimum_approach_episodes": args.minimum_approach_episodes,
            "maximum_warmup_interactions": args.max_replay_warmup_steps,
        },
        "replay": {
            "sampling_unit": "episode first, transition second",
            "strata": ["ratio <= 2", "2 < ratio <= 5", "5 < ratio <= 10", "ratio > 10"],
            "fractions": {
                "ratio_le_2": args.target_le_2_fraction,
                "ratio_2_to_5": args.target_2_to_5_fraction,
                "ratio_5_to_10": args.target_5_to_10_fraction,
                "approach_gt_10": args.approach_fraction,
            },
            "unavailable_stratum_policy": (
                "redistribute its batch share proportionally across available strata"
            ),
        },
        "reset_augmentation": {
            "near_p3_fraction": args.near_p3_reset_fraction,
            "minimum_target_ratio": args.near_p3_min_ratio,
            "maximum_target_ratio": args.near_p3_max_ratio,
            "other_component_max_ratio": args.near_p3_other_max_ratio,
            "near_p3_time_limit_seconds": args.near_p3_max_time,
            "validation_uses_augmented_resets": False,
        },
        "checkpoint_selection": (
            "success, safety, best simultaneous target ratio, 2x-target "
            "occupancy/dwell, final target ratio, safety score, return"
        ),
        "device": args.device,
        "overwrite_requested": bool(args.overwrite),
    }


def critic_pretrain(model, updates, log_interval, result_dir):
    rows = []
    completed = 0
    while completed < updates:
        chunk = min(log_interval, updates - completed)
        model.train(gradient_steps=chunk, batch_size=model.batch_size)
        completed += chunk
        row = {
            "critic_pretrain_update": completed,
            "critic_loss": model.last_train_metrics["critic_loss"],
            "actor_updates": model.last_train_metrics["actor_updates"],
        }
        rows.append(row)
        write_csv(str(result_dir / "critic_pretraining.csv"), rows)
        print(
            f"Critic pretraining {completed:,}/{updates:,} | "
            f"loss={row['critic_loss']:.6g} | actor updates=0"
        )
    return rows


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    checkpoints = resolve_checkpoint_steps(args)
    source_model = SAC.load(str(args.source_model), device="cpu")
    config = build_config(args, checkpoints, source_model)
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return config

    model_dir, result_dir, log_dir = prepare_output_directories(args)
    config_path = result_dir / "experiment_config.json"
    write_json(str(config_path), config)

    raw_vec_env = DummyVecEnv([lambda: make_training_env(args)])
    training_env = make_vecnormalize(args, raw_vec_env)
    reset_wrapper = find_reset_wrapper(training_env)
    model = build_model(args, training_env, source_model)

    source_actor_hash = tensor_state_sha256(source_model.actor)
    source_critic_hash = tensor_state_sha256(source_model.critic)
    initialization_audit = {
        "source_actor_sha256": source_actor_hash,
        "new_actor_sha256": tensor_state_sha256(model.actor),
        "actor_weights_exactly_equal": bool(
            source_actor_hash == tensor_state_sha256(model.actor)
        ),
        "source_critic_sha256": source_critic_hash,
        "fresh_critic_sha256": tensor_state_sha256(model.critic),
        "critic_weights_equal_to_source": bool(
            source_critic_hash == tensor_state_sha256(model.critic)
        ),
    }
    write_json(str(result_dir / "initialization_audit.json"), initialization_audit)
    if not initialization_audit["actor_weights_exactly_equal"]:
        raise RuntimeError("Actor-only transfer did not reproduce the source actor.")
    if initialization_audit["critic_weights_equal_to_source"]:
        raise RuntimeError("Fresh critic unexpectedly equals the source critic.")
    del source_model

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
        action_hold_steps=ACTION_HOLD_STEPS,
        adaptive_action_hold_distance_m=None,
        evaluation_env_factory=partial(make_validation_env, args=args),
    )
    return_callback = TrainingReturnCallback(
        training_seed=args.training_seed,
        result_dir=str(result_dir),
        model_name=args.experiment_name,
    )

    print("=" * 80)
    print("Uniform 0.1 s closing SAC: actor-only transfer and fresh critics")
    print("=" * 80)
    print(f"Experiment:             {args.experiment_name}")
    print(f"Source actor:           {args.source_model}")
    print("Source critics:         NOT LOADED")
    print("Transition duration:    0.1 s everywhere")
    print(f"Replay episode gate:    {args.minimum_completed_episodes} completed")
    print(f"Critic-only updates:    {args.critic_pretrain_updates:,}")
    print(f"Actor learning rate:    {args.actor_learning_rate:.3g}")
    print(f"Actor update interval:  1/{args.actor_update_interval} critic updates")
    print("=" * 80)

    try:
        readiness_callback = EpisodeReplayReadinessCallback()
        install_deterministic_warmup_sampler(model)
        model.gradient_steps = 0
        try:
            model.learn(
                total_timesteps=args.max_replay_warmup_steps,
                reset_num_timesteps=True,
                tb_log_name=f"{args.experiment_name}_episode_warmup",
                progress_bar=args.progress_bar,
                callback=readiness_callback,
            )
        finally:
            remove_deterministic_warmup_sampler(model)

        warmup_steps = int(model.num_timesteps)
        warmup_statistics = {
            "ready_for_critic_pretraining": bool(readiness_callback.ready),
            "warmup_interaction_steps": warmup_steps,
            "reset_statistics": reset_wrapper.reset_statistics(),
            "reward_rms_mean": float(training_env.ret_rms.mean),
            "reward_rms_variance": float(training_env.ret_rms.var),
            **model.replay_buffer.replay_statistics(),
        }
        write_json(
            str(result_dir / "replay_buffer_after_warmup.json"),
            warmup_statistics,
        )
        config["actual_replay_warmup_steps"] = warmup_steps
        config["replay_ready_after_warmup"] = bool(readiness_callback.ready)
        if not readiness_callback.ready:
            write_json(str(config_path), config)
            raise RuntimeError(
                "Episode-level replay gates were not satisfied before the "
                "warmup interaction limit; no gradient update was performed."
            )

        # Reward statistics are now matched to the new 0.1 s reward stream and
        # frozen so the critic target scale cannot drift during optimization.
        training_env.training = False
        actor_hash_before_pretraining = tensor_state_sha256(model.actor)
        model.set_actor_updates_enabled(False)
        critic_pretrain(
            model,
            updates=args.critic_pretrain_updates,
            log_interval=args.critic_pretrain_log_interval,
            result_dir=result_dir,
        )
        actor_hash_after_pretraining = tensor_state_sha256(model.actor)
        critic_audit = {
            "actor_sha256_before": actor_hash_before_pretraining,
            "actor_sha256_after": actor_hash_after_pretraining,
            "actor_unchanged": bool(
                actor_hash_before_pretraining == actor_hash_after_pretraining
            ),
            "critic_updates": int(model.critic_gradient_updates),
            "actor_updates": int(model.actor_gradient_updates),
        }
        write_json(str(result_dir / "critic_pretraining_audit.json"), critic_audit)
        if not critic_audit["actor_unchanged"] or model.actor_gradient_updates != 0:
            raise RuntimeError("Actor changed during critic-only pretraining.")
        model.save(str(model_dir / "critic_pretrained_actor_frozen"))
        training_env.save(str(model_dir / "critic_pretrained_vecnormalize.pkl"))

        model.gradient_steps = args.gradient_steps
        model.set_actor_updates_enabled(True)
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
            model.replay_buffer.replay_statistics(),
        )
        write_json(
            str(result_dir / "near_p3_reset_statistics.json"),
            reset_wrapper.reset_statistics(),
        )
        config["actual_gradient_training_steps"] = int(model.num_timesteps)
        config["critic_gradient_updates"] = int(model.critic_gradient_updates)
        config["actor_gradient_updates"] = int(model.actor_gradient_updates)
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
