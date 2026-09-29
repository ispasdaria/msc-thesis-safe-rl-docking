"""Train residual SAC+HER around the exact frozen V15 closing controller."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from her_closing_utils import (
    checkpoint_rank,
    evaluate_model,
    make_her_env,
    summarize_rollouts,
    write_csv,
    write_json,
)
from her_residual_env import (
    DEFAULT_SOURCE_MODEL,
    DEFAULT_SOURCE_VECNORMALIZE,
    ResidualV15GoalConditionedClosingEnv,
)
from her_source_warmstart import prefill_her_replay_buffer
from train_sac_closing_her import HerEvaluationCheckpointCallback


CLOSING_DIR = Path(__file__).resolve().parent
MODEL_ROOT = CLOSING_DIR / "models" / "her"
RESULT_ROOT = CLOSING_DIR / "results" / "her"
LOG_ROOT = CLOSING_DIR / "logs" / "her"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_observation(observation):
    return {
        key: np.asarray(value, dtype=np.float32).copy()
        for key, value in observation.items()
    }


def _compact_info(info, truncated):
    keys = (
        "failure",
        "time_limit",
        "success",
        "port_speed",
        "s_dot",
        "psi_e",
        "omega",
        "state",
        "violated_constraints",
        "termination_reason",
    )
    compact = {key: copy.deepcopy(info[key]) for key in keys if key in info}
    compact["TimeLimit.truncated"] = bool(truncated)
    return compact


def collect_residual_source_trajectories(
    *,
    minimum_transitions,
    seed_start,
    max_episodes,
    max_steps,
    residual_scale,
    residual_std=0.0,
    residual_clip=1.0,
    residual_hold_steps=20,
    zero_episode_fraction=1.0,
    random_seed=5_000,
    retain_failure_episodes=False,
):
    """Collect complete frozen-source episodes with optional residual probes.

    Failed exploratory episodes are useful boundary data and may be retained;
    the core closing environment still terminates immediately at a hard safety
    violation.  Setting ``residual_std=0`` and not retaining failures exactly
    recovers the V4 zero-residual collection behavior.
    """
    if residual_std < 0.0:
        raise ValueError("residual_std must be non-negative.")
    if not 0.0 < residual_clip <= 1.0:
        raise ValueError("residual_clip must lie in (0, 1].")
    if residual_hold_steps <= 0:
        raise ValueError("residual_hold_steps must be positive.")
    if not 0.0 <= zero_episode_fraction <= 1.0:
        raise ValueError("zero_episode_fraction must lie in [0, 1].")
    rng = np.random.default_rng(random_seed)
    env = make_her_env(
        seed=seed_start,
        mode="training",
        max_time=max_steps,
        reward_version="v4",
        residual_scale=residual_scale,
    )
    transitions = []
    summaries = []
    accepted_episode_id = 0
    try:
        for attempt in range(max_episodes):
            if len(transitions) >= minimum_transitions:
                break
            seed = int(seed_start + attempt)
            observation, _ = env.reset(seed=seed)
            episode = []
            final_info = {}
            zero_episode = bool(
                residual_std == 0.0 or rng.random() < zero_episode_fraction
            )
            residual_action = np.zeros(3, dtype=np.float32)
            action_square_sum = np.zeros(3, dtype=np.float64)
            action_abs_max = np.zeros(3, dtype=np.float64)
            for step in range(int(max_steps)):
                if not zero_episode and step % residual_hold_steps == 0:
                    residual_action = np.clip(
                        rng.normal(0.0, residual_std, size=3),
                        -residual_clip,
                        residual_clip,
                    ).astype(np.float32)
                next_observation, reward, terminated, truncated, final_info = (
                    env.step(residual_action)
                )
                done = bool(terminated or truncated)
                episode.append(
                    {
                        "observation": _copy_observation(observation),
                        "next_observation": _copy_observation(next_observation),
                        "action": residual_action.copy(),
                        "reward": float(reward),
                        "done": done,
                        "info": _compact_info(final_info, truncated),
                        "episode_id": accepted_episode_id,
                    }
                )
                action_square_sum += np.square(residual_action)
                action_abs_max = np.maximum(
                    action_abs_max,
                    np.abs(residual_action),
                )
                observation = next_observation
                if done:
                    break
            failure = bool(final_info.get("failure", False))
            retain_episode = bool(not failure or retain_failure_episodes)
            action_rms = np.sqrt(action_square_sum / max(len(episode), 1))
            summaries.append(
                {
                    "attempt": attempt + 1,
                    "seed": seed,
                    "steps": len(episode),
                    "accepted": int(retain_episode),
                    "zero_residual_episode": int(zero_episode),
                    "failure": int(failure),
                    "residual_rms_ax": float(action_rms[0]),
                    "residual_rms_ay": float(action_rms[1]),
                    "residual_rms_alpha": float(action_rms[2]),
                    "residual_max_ax": float(action_abs_max[0]),
                    "residual_max_ay": float(action_abs_max[1]),
                    "residual_max_alpha": float(action_abs_max[2]),
                    "termination_reason": final_info.get(
                        "termination_reason",
                        "",
                    ),
                    "violated_constraints": str(
                        final_info.get("violated_constraints", [])
                    ),
                }
            )
            if retain_episode:
                transitions.extend(episode)
                accepted_episode_id += 1
    finally:
        env.close()
    if len(transitions) < minimum_transitions:
        raise RuntimeError(
            f"Collected only {len(transitions):,} retained residual "
            f"transitions; requested {minimum_transitions:,}."
        )
    return transitions, summaries


def collect_zero_residual_source_trajectories(**kwargs):
    """Backward-compatible V4 collector."""
    return collect_residual_source_trajectories(**kwargs)


def initialize_zero_residual_actor(model, log_std=-4.0):
    with torch.no_grad():
        model.actor.mu.weight.zero_()
        model.actor.mu.bias.zero_()
        model.actor.log_std.weight.zero_()
        model.actor.log_std.bias.fill_(float(log_std))


class ResidualSafetyGuardCallback(HerEvaluationCheckpointCallback):
    def __init__(self, *, maximum_failure_rate, **kwargs):
        super().__init__(**kwargs)
        self.maximum_failure_rate = float(maximum_failure_rate)
        self.guard_triggered = False
        self.guard_reason = ""

    def _evaluate_checkpoint(self, checkpoint):
        super()._evaluate_checkpoint(checkpoint)
        current = self.checkpoint_summaries[-1]
        if float(current["failure_rate"]) > self.maximum_failure_rate:
            self.guard_triggered = True
            self.guard_reason = (
                f"validation failure rate {current['failure_rate']:.3f} "
                f"exceeded {self.maximum_failure_rate:.3f} at "
                f"{checkpoint:,} steps"
            )
            print(f"Residual safety guard: {self.guard_reason}", flush=True)

    def _on_step(self):
        keep_training = super()._on_step()
        return bool(keep_training and not self.guard_triggered)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Residual SAC+HER with exact frozen V15 nominal actions."
    )
    parser.add_argument(
        "--experiment-name",
        default="sac_closing_goal_her_v4_residual_v15_guarded50k_seed0",
    )
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[10_000, 25_000, 50_000],
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=list(range(100, 110)),
    )
    parser.add_argument("--source-transition-count", type=int, default=60_000)
    parser.add_argument("--source-seed-start", type=int, default=0)
    parser.add_argument("--source-max-episodes", type=int, default=12)
    parser.add_argument("--max-time", type=float, default=12_000.0)
    parser.add_argument("--max-eval-steps", type=int, default=12_000)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--initial-log-std", type=float, default=-4.0)
    parser.add_argument("--prefill-residual-std", type=float, default=0.0)
    parser.add_argument("--prefill-residual-clip", type=float, default=1.0)
    parser.add_argument(
        "--prefill-residual-hold-steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--prefill-zero-episode-fraction",
        type=float,
        default=1.0,
    )
    parser.add_argument("--prefill-random-seed", type=int, default=5_000)
    parser.add_argument("--retain-failure-prefill", action="store_true")
    parser.add_argument("--maximum-gate-failure-rate", type=float, default=0.0)
    parser.add_argument(
        "--maximum-online-validation-failure-rate",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--maximum-gate-min-position-error",
        type=float,
        default=100.0,
    )
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--train-freq", type=int, default=16)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--ent-coef", default="auto_0.01")
    parser.add_argument("--n-sampled-goal", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not DEFAULT_SOURCE_MODEL.is_file() or not DEFAULT_SOURCE_VECNORMALIZE.is_file():
        parser.error("Frozen V15 source artifacts are missing.")
    if not 0.0 < args.residual_scale <= 1.0:
        parser.error("--residual-scale must lie in (0, 1].")
    if args.prefill_residual_std < 0.0:
        parser.error("--prefill-residual-std must be non-negative.")
    if not 0.0 < args.prefill_residual_clip <= 1.0:
        parser.error("--prefill-residual-clip must lie in (0, 1].")
    if args.prefill_residual_hold_steps <= 0:
        parser.error("--prefill-residual-hold-steps must be positive.")
    if not 0.0 <= args.prefill_zero_episode_fraction <= 1.0:
        parser.error("--prefill-zero-episode-fraction must lie in [0, 1].")
    checkpoints = sorted(
        {step for step in args.checkpoint_steps if 0 < step <= args.total_timesteps}
    )
    if args.total_timesteps not in checkpoints:
        checkpoints.append(args.total_timesteps)

    reward_class = ResidualV15GoalConditionedClosingEnv
    config = {
        "experiment_name": args.experiment_name,
        "method": (
            "exploratory residual goal-conditioned SAC+HER around frozen V15"
            if args.prefill_residual_std > 0.0
            else "residual goal-conditioned SAC+HER around frozen V15"
        ),
        "reward_version": "v4",
        "reward_scheme": reward_class.reward_scheme,
        "source_model": str(DEFAULT_SOURCE_MODEL.resolve()),
        "source_model_sha256": file_sha256(DEFAULT_SOURCE_MODEL),
        "source_vecnormalize": str(DEFAULT_SOURCE_VECNORMALIZE.resolve()),
        "source_vecnormalize_sha256": file_sha256(DEFAULT_SOURCE_VECNORMALIZE),
        "residual_scale": args.residual_scale,
        "initial_residual_policy": "zero deterministic residual",
        "source_transition_count_requested": args.source_transition_count,
        "source_seed_start": args.source_seed_start,
        "source_max_episodes": args.source_max_episodes,
        "prefill_exploration": {
            "residual_std": args.prefill_residual_std,
            "residual_clip": args.prefill_residual_clip,
            "residual_hold_steps": args.prefill_residual_hold_steps,
            "zero_episode_fraction": args.prefill_zero_episode_fraction,
            "random_seed": args.prefill_random_seed,
            "retain_failure_episodes": args.retain_failure_prefill,
            "hard_safety_termination": True,
        },
        "training_seed": args.training_seed,
        "validation_seeds": args.validation_seeds,
        "total_timesteps": args.total_timesteps,
        "checkpoint_steps": [0, *checkpoints],
        "max_time": args.max_time,
        "max_eval_steps": args.max_eval_steps,
        "initial_gate": {
            "maximum_failure_rate": args.maximum_gate_failure_rate,
            "maximum_min_position_error": args.maximum_gate_min_position_error,
        },
        "online_guard_maximum_failure_rate": (
            args.maximum_online_validation_failure_rate
        ),
        "cbf_filter_used_during_training": False,
        "near_p3_resets_used": False,
        "curriculum_used": False,
        "sac": {
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "learning_starts": 0,
            "batch_size": args.batch_size,
            "tau": args.tau,
            "gamma": args.gamma,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "ent_coef": args.ent_coef,
        },
        "her": {
            "n_sampled_goal": args.n_sampled_goal,
            "goal_selection_strategy": "future",
            "copy_info_dict": True,
        },
    }
    if args.dry_run:
        import json

        print(json.dumps(config, indent=2, sort_keys=True))
        return

    model_dir = MODEL_ROOT / args.experiment_name
    result_dir = RESULT_ROOT / args.experiment_name
    log_dir = LOG_ROOT / args.experiment_name
    for path in (model_dir, result_dir, log_dir):
        if path.exists():
            raise FileExistsError(f"Refusing to reuse output directory: {path}")
    for path in (model_dir, result_dir, log_dir):
        path.mkdir(parents=True)
    write_json(result_dir / "experiment_config.json", config)

    transitions, episode_rows = collect_residual_source_trajectories(
        minimum_transitions=args.source_transition_count,
        seed_start=args.source_seed_start,
        max_episodes=args.source_max_episodes,
        max_steps=args.max_time,
        residual_scale=args.residual_scale,
        residual_std=args.prefill_residual_std,
        residual_clip=args.prefill_residual_clip,
        residual_hold_steps=args.prefill_residual_hold_steps,
        zero_episode_fraction=args.prefill_zero_episode_fraction,
        random_seed=args.prefill_random_seed,
        retain_failure_episodes=args.retain_failure_prefill,
    )
    write_csv(result_dir / "source_episode_collection.csv", episode_rows)

    training_env = DummyVecEnv(
        [
            lambda: Monitor(
                make_her_env(
                    seed=args.training_seed,
                    mode="training",
                    max_time=args.max_time,
                    reward_version="v4",
                    residual_scale=args.residual_scale,
                )
            )
        ]
    )
    model = SAC(
        "MultiInputPolicy",
        training_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": args.n_sampled_goal,
            "goal_selection_strategy": "future",
            "copy_info_dict": True,
        },
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=0,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        verbose=1,
        seed=args.training_seed,
        tensorboard_log=str(log_dir),
        device=args.device,
    )
    initialize_zero_residual_actor(model, args.initial_log_std)
    prefill_metrics = prefill_her_replay_buffer(model, transitions)

    initial_rows = evaluate_model(
        model,
        args.validation_seeds,
        max_steps=args.max_eval_steps,
        reward_version="v4",
        env_kwargs={"residual_scale": args.residual_scale},
    )
    initial_summary = summarize_rollouts(initial_rows)
    write_csv(result_dir / "zero_residual_validation_rollouts.csv", initial_rows)
    write_json(
        result_dir / "source_initialization.json",
        {**prefill_metrics, "zero_residual_validation": initial_summary},
    )
    if initial_summary["failure_rate"] > args.maximum_gate_failure_rate:
        raise RuntimeError("Exact zero-residual source failed the safety gate.")
    if (
        initial_summary["mean_minimum_position_error"]
        > args.maximum_gate_min_position_error
    ):
        raise RuntimeError(
            "Exact zero-residual source failed the approach gate: "
            f"{initial_summary['mean_minimum_position_error']:.2f} m."
        )

    initial_rank = checkpoint_rank(initial_summary)
    model.save(model_dir / "checkpoint_0_model")
    model.save(model_dir / "best_model")
    write_json(
        result_dir / "best_checkpoint.json",
        {"checkpoint": 0, "summary": initial_summary},
    )
    callback = ResidualSafetyGuardCallback(
        maximum_failure_rate=args.maximum_online_validation_failure_rate,
        checkpoint_steps=checkpoints,
        validation_seeds=args.validation_seeds,
        max_eval_steps=args.max_eval_steps,
        model_dir=model_dir,
        result_dir=result_dir,
        initial_rank=initial_rank,
        initial_summary=initial_summary,
        initial_rows=initial_rows,
        reward_version="v4",
        env_kwargs={"residual_scale": args.residual_scale},
    )
    callback._save_tables()
    print(f"Experiment:       {args.experiment_name}")
    print(
        f"Exact V15 gate:   failure={initial_summary['failure_rate']:.3f}, "
        f"min P3 error={initial_summary['mean_minimum_position_error']:.2f} m"
    )
    print(f"Residual scale:   {args.residual_scale:.3f}")
    print(f"Online HER steps: {args.total_timesteps:,}")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        tb_log_name=args.experiment_name,
        progress_bar=args.progress_bar,
    )
    model.save(model_dir / "final_model")
    config.update(
        {
            "actual_timesteps": int(model.num_timesteps),
            "best_checkpoint": int(callback.best_checkpoint),
            "guard_triggered": callback.guard_triggered,
            "guard_reason": callback.guard_reason,
            "source_prefill": prefill_metrics,
        }
    )
    write_json(result_dir / "experiment_config.json", config)
    training_env.close()


if __name__ == "__main__":
    main()
