"""Pilot C: fixed-goal SAC reference residuals around LQR, without HER/CBF."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

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
from lqr_reference_residual_env import (
    ReferenceResidualLQRGoalConditionedClosingEnv,
)
from train_sac_closing_her import HerEvaluationCheckpointCallback
from train_sac_closing_her_reference_residual_lqr import (
    collect_reference_residual_trajectories,
)
from train_sac_closing_her_v4_residual_v15 import (
    initialize_zero_residual_actor,
)
from zero_residual_constrained_sac import ZeroResidualConstrainedSAC


CLOSING_DIR = Path(__file__).resolve().parent
MODEL_ROOT = CLOSING_DIR / "models" / "lqr_reference_residual_no_her"
RESULT_ROOT = CLOSING_DIR / "results" / "lqr_reference_residual_no_her"
LOG_ROOT = CLOSING_DIR / "logs" / "lqr_reference_residual_no_her"
REWARD_VERSION = "lqr_reference_residual"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fixed-P3 SAC reference residual around closing LQR."
    )
    parser.add_argument(
        "--experiment-name",
        default=(
            "sac_closing_lqr_reference_residual_"
            "pilot_c_no_her_1hz_50k_seed0"
        ),
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
    parser.add_argument("--vx-reference-bound", type=float, default=0.02)
    parser.add_argument("--vy-reference-bound", type=float, default=0.01)
    parser.add_argument(
        "--attitude-reference-bound-deg", type=float, default=1.0
    )
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument(
        "--reference-magnitude-weight", type=float, default=0.01
    )
    parser.add_argument(
        "--reference-smoothness-weight", type=float, default=0.01
    )
    parser.add_argument("--success-bonus", type=float, default=1.0)
    parser.add_argument("--failure-penalty", type=float, default=10.0)
    parser.add_argument("--timeout-penalty", type=float, default=1.0)
    parser.add_argument("--initial-log-std", type=float, default=-3.0)
    parser.add_argument("--prefill-residual-std", type=float, default=0.25)
    parser.add_argument("--prefill-residual-clip", type=float, default=0.75)
    parser.add_argument(
        "--prefill-zero-episode-fraction", type=float, default=0.5
    )
    parser.add_argument("--prefill-random-seed", type=int, default=8_000)
    parser.add_argument("--retain-failure-prefill", action="store_true")
    parser.add_argument("--maximum-gate-failure-rate", type=float, default=0.0)
    parser.add_argument("--maximum-gate-min-position-error", type=float, default=100.0)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.9999)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--ent-coef", default="auto_0.01")
    parser.add_argument("--critic-warmup-steps", type=int, default=10_000)
    parser.add_argument("--zero-residual-actor-weight", type=float, default=1.0)
    parser.add_argument("--actor-learning-rate-scale", type=float, default=0.25)
    parser.add_argument("--target-entropy", type=float, default=-3.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(parser, args):
    positive = (
        "total_timesteps",
        "source_transition_count",
        "source_max_episodes",
        "max_time",
        "max_eval_steps",
        "vx_reference_bound",
        "vy_reference_bound",
        "attitude_reference_bound_deg",
        "buffer_size",
        "batch_size",
        "learning_rate",
        "train_freq",
        "gradient_steps",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if not 0 <= args.critic_warmup_steps < args.total_timesteps:
        parser.error("--critic-warmup-steps must lie in [0, total-timesteps).")
    if not 0.0 < args.gamma <= 1.0:
        parser.error("--gamma must lie in (0, 1].")
    if args.prefill_residual_std < 0.0:
        parser.error("--prefill-residual-std must be non-negative.")
    if not 0.0 < args.prefill_residual_clip <= 1.0:
        parser.error("--prefill-residual-clip must lie in (0, 1].")
    if not 0.0 <= args.prefill_zero_episode_fraction <= 1.0:
        parser.error("--prefill-zero-episode-fraction must lie in [0, 1].")
    nonnegative = (
        "progress_weight",
        "reference_magnitude_weight",
        "reference_smoothness_weight",
        "success_bonus",
        "failure_penalty",
        "timeout_penalty",
        "zero_residual_actor_weight",
        "actor_learning_rate_scale",
    )
    for name in nonnegative:
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative.")


def environment_kwargs(args):
    return {
        "reference_residual_bounds": (
            args.vx_reference_bound,
            args.vy_reference_bound,
            np.deg2rad(args.attitude_reference_bound_deg),
        ),
        "progress_gamma": args.gamma,
        "progress_weight": args.progress_weight,
        "reference_magnitude_weight": args.reference_magnitude_weight,
        "reference_smoothness_weight": args.reference_smoothness_weight,
        "success_bonus": args.success_bonus,
        "safety_failure_penalty": args.failure_penalty,
        "timeout_penalty": args.timeout_penalty,
        "terminate_on_success": True,
    }


def prefill_standard_replay_buffer(model, transitions):
    """Load fixed-goal transitions without HER relabeling."""
    replay_buffer = model.replay_buffer
    for transition in transitions:
        observation = {
            key: value.reshape(1, -1)
            for key, value in transition["observation"].items()
        }
        next_observation = {
            key: value.reshape(1, -1)
            for key, value in transition["next_observation"].items()
        }
        replay_buffer.add(
            observation,
            next_observation,
            transition["action"].reshape(1, -1),
            np.asarray([transition["reward"]], dtype=np.float32),
            np.asarray([transition["done"]], dtype=np.float32),
            [transition["info"]],
        )
    return {
        "prefill_transition_count": len(transitions),
        "prefill_episode_count": len(
            {transition["episode_id"] for transition in transitions}
        ),
        "replay_buffer_class": type(replay_buffer).__name__,
        "replay_buffer_position_after_prefill": int(replay_buffer.pos),
        "replay_buffer_full_after_prefill": bool(replay_buffer.full),
    }


def write_model_readme(
    *, model_dir, config, checkpoint_summaries, best_checkpoint, complete
):
    rows = []
    for summary in checkpoint_summaries:
        rows.append(
            "| {checkpoint:,} | {success:.1%} | {failure:.1%} | "
            "{timeout:.1%} | {position:.3f} | {ratio:.3f} | {steps:.1f} |".format(
                checkpoint=int(summary["checkpoint"]),
                success=float(summary["success_rate"]),
                failure=float(summary["failure_rate"]),
                timeout=float(summary["time_limit_rate"]),
                position=float(summary["mean_minimum_position_error"]),
                ratio=float(summary["mean_best_max_target_ratio"]),
                steps=float(summary["mean_steps"]),
            )
        )
    baseline = checkpoint_summaries[0]
    final = checkpoint_summaries[-1]
    if not complete:
        interpretation = "The run is in progress; selection is provisional."
    elif int(best_checkpoint) == 0:
        interpretation = (
            "No learned checkpoint outranked the zero-reference baseline. "
            f"Success changed from {float(baseline['success_rate']):.1%} to "
            f"{float(final['success_rate']):.1%}; `best_model.zip` therefore "
            "remains checkpoint 0."
        )
    else:
        interpretation = (
            f"Checkpoint {int(best_checkpoint):,} outranked the baseline and "
            "is stored as `best_model.zip`."
        )
    reward = config["reward"]
    sac = config["sac"]
    bounds = config["reference_residual_bounds"]
    status = "complete" if complete else "in progress"
    content = f"""# Pilot C fixed-goal reference-residual SAC model card

Run: `{config['experiment_name']}`  
Status: **{status}**  
Selected best checkpoint: **{int(best_checkpoint):,} SAC decisions**

## What changed from Pilot B

Pilot C retains the same 1 Hz SAC reference residual and unchanged closing
LQR, but removes HER completely. Every replay transition keeps the real fixed
P3 target. Exact P3 success terminates an episode. The task reward was divided
by 100 so critic values and the actor residual prior operate at comparable
scales.

No CBF is instantiated, called, observed, or used in the reward.

## Controller

```text
delta_reference = [
    {bounds['vx_mps']:.3f} * a[0] m/s,
    {bounds['vy_mps']:.3f} * a[1] m/s,
    {bounds['attitude_deg']:.3f} * a[2] deg,
]
u_nominal = LQR(state, scheduled_P3_reference + delta_reference)
```

SAC and LQR both update every second. There is no action hold or distance
gate, and zero SAC action exactly reproduces the established LQR.

## Reward

```text
reward = {reward['progress_weight']:.3f} * (J(previous) - gamma * J(next))
         + {reward['success_bonus']:.3f} * exact_P3_success
         - {reward['failure_penalty']:.3f} * hard_failure
         - {reward['timeout_penalty']:.3f} * unsuccessful_timeout
         - {reward['magnitude_weight']:.3f} * ||a||^2
         - {reward['smoothness_weight']:.3f} * ||a - a_previous||^2
```

`J` equally combines the mean and worst log-transformed P3 tolerance ratios.

## SAC configuration

| Parameter | Value |
|---|---:|
| Online interactions | {config['total_timesteps']:,} |
| Replay prefill | {config['source_transition_count_requested']:,} |
| Replay method | Fixed-goal `DictReplayBuffer`; no HER |
| Critic-only warm-up | {config['critic_warmup_steps']:,} interactions |
| Learning rate | {sac['learning_rate']:.2g} |
| Effective actor learning rate | {sac['learning_rate'] * config['actor_learning_rate_scale']:.2g} |
| Actor zero-reference weight | {config['zero_residual_actor_weight']:.3g} |
| Train frequency | every {sac['train_freq']} interactions |
| Gradient steps | {sac['gradient_steps']} |
| Update-to-data ratio | {sac['update_to_data_ratio']:.3f} |
| Batch size | {sac['batch_size']} |
| Discount factor | {sac['gamma']:.4f} |
| Target smoothing | {sac['tau']:.4f} |
| Entropy coefficient | `{sac['ent_coef']}` |
| Target entropy | {sac['target_entropy']:.1f} |

## Validation results

Validation uses seeds {min(config['validation_seeds'])}--{max(config['validation_seeds'])}.

| Checkpoint | Success | Hard failure | Timeout | Mean minimum position error [m] | Mean best worst ratio | Mean seconds |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Interpretation

{interpretation}

## Artifacts

- `checkpoint_0_model.zip`: exact zero-reference LQR baseline;
- `checkpoint_<N>_model.zip`: evaluated SAC checkpoint;
- `best_model.zip`: validation-ranked checkpoint;
- `final_model.zip`: end-of-training policy.

Metrics and configuration are in
`../../../results/lqr_reference_residual_no_her/{config['experiment_name']}/`.
"""
    (Path(model_dir) / "README.md").write_text(content, encoding="utf-8")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    checkpoints = sorted(
        {step for step in args.checkpoint_steps if 0 < step <= args.total_timesteps}
    )
    if args.total_timesteps not in checkpoints:
        checkpoints.append(args.total_timesteps)
    env_kwargs = environment_kwargs(args)
    config = {
        "experiment_name": args.experiment_name,
        "method": "fixed-P3 1 Hz SAC reference residual around closing LQR",
        "reward_version": REWARD_VERSION,
        "reward_scheme": (
            ReferenceResidualLQRGoalConditionedClosingEnv.reward_scheme
        ),
        "reference_residual_components": ["vx", "vy", "attitude"],
        "reference_residual_bounds": {
            "vx_mps": args.vx_reference_bound,
            "vy_mps": args.vy_reference_bound,
            "attitude_deg": args.attitude_reference_bound_deg,
        },
        "sac_decision_interval_seconds": 1.0,
        "lqr_update_interval_seconds": 1.0,
        "residual_hold_steps": 1,
        "distance_gate_used": False,
        "her_used": False,
        "actual_p3_success_terminal": True,
        "cbf_filter_used_during_training": False,
        "cbf_signal_in_observation_or_reward": False,
        "reward": {
            "progress_weight": args.progress_weight,
            "progress_form": "J(previous) - gamma * J(next)",
            "success_bonus": args.success_bonus,
            "failure_penalty": args.failure_penalty,
            "timeout_penalty": args.timeout_penalty,
            "magnitude_weight": args.reference_magnitude_weight,
            "smoothness_weight": args.reference_smoothness_weight,
            "scale_relative_to_pilot_b": 0.01,
        },
        "training_seed": args.training_seed,
        "validation_seeds": args.validation_seeds,
        "total_timesteps": args.total_timesteps,
        "checkpoint_steps": [0, *checkpoints],
        "max_time": args.max_time,
        "max_eval_steps": args.max_eval_steps,
        "source_transition_count_requested": args.source_transition_count,
        "source_seed_start": args.source_seed_start,
        "source_max_episodes": args.source_max_episodes,
        "prefill_exploration": {
            "residual_std": args.prefill_residual_std,
            "residual_clip": args.prefill_residual_clip,
            "zero_episode_fraction": args.prefill_zero_episode_fraction,
            "decision_interval_seconds": 1.0,
            "random_seed": args.prefill_random_seed,
            "retain_failure_episodes": args.retain_failure_prefill,
        },
        "critic_warmup_steps": args.critic_warmup_steps,
        "zero_residual_actor_weight": args.zero_residual_actor_weight,
        "actor_learning_rate_scale": args.actor_learning_rate_scale,
        "sac": {
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "replay_buffer": "DictReplayBuffer",
            "batch_size": args.batch_size,
            "tau": args.tau,
            "gamma": args.gamma,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "update_to_data_ratio": args.gradient_steps / args.train_freq,
            "ent_coef": args.ent_coef,
            "target_entropy": args.target_entropy,
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

    transitions, source_rows = collect_reference_residual_trajectories(
        minimum_transitions=args.source_transition_count,
        seed_start=args.source_seed_start,
        max_episodes=args.source_max_episodes,
        max_steps=args.max_time,
        env_kwargs=env_kwargs,
        residual_std=args.prefill_residual_std,
        residual_clip=args.prefill_residual_clip,
        zero_episode_fraction=args.prefill_zero_episode_fraction,
        random_seed=args.prefill_random_seed,
        retain_failure_episodes=args.retain_failure_prefill,
    )
    write_csv(result_dir / "source_episode_collection.csv", source_rows)

    training_env = DummyVecEnv(
        [
            lambda: Monitor(
                make_her_env(
                    seed=args.training_seed,
                    mode="training",
                    max_time=args.max_time,
                    reward_version=REWARD_VERSION,
                    **env_kwargs,
                )
            )
        ]
    )
    model = ZeroResidualConstrainedSAC(
        "MultiInputPolicy",
        training_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=0,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        target_entropy=args.target_entropy,
        critic_warmup_steps=args.critic_warmup_steps,
        zero_residual_actor_weight=args.zero_residual_actor_weight,
        actor_learning_rate_scale=args.actor_learning_rate_scale,
        verbose=1,
        seed=args.training_seed,
        tensorboard_log=str(log_dir),
        device=args.device,
    )
    initialize_zero_residual_actor(model, args.initial_log_std)
    prefill_metrics = prefill_standard_replay_buffer(model, transitions)

    initial_rows = evaluate_model(
        model,
        args.validation_seeds,
        max_steps=args.max_eval_steps,
        reward_version=REWARD_VERSION,
        env_kwargs=env_kwargs,
    )
    initial_summary = summarize_rollouts(initial_rows)
    write_csv(result_dir / "zero_residual_validation_rollouts.csv", initial_rows)
    write_json(
        result_dir / "source_initialization.json",
        {**prefill_metrics, "zero_residual_validation": initial_summary},
    )
    if initial_summary["failure_rate"] > args.maximum_gate_failure_rate:
        raise RuntimeError("Exact zero-reference LQR failed the initial gate.")
    if (
        initial_summary["mean_minimum_position_error"]
        > args.maximum_gate_min_position_error
    ):
        raise RuntimeError("Exact zero-reference LQR failed the approach gate.")

    model.save(model_dir / "checkpoint_0_model")
    model.save(model_dir / "best_model")
    write_json(
        result_dir / "best_checkpoint.json",
        {"checkpoint": 0, "summary": initial_summary},
    )
    write_model_readme(
        model_dir=model_dir,
        config=config,
        checkpoint_summaries=[{"checkpoint": 0, **initial_summary}],
        best_checkpoint=0,
        complete=False,
    )
    callback = HerEvaluationCheckpointCallback(
        checkpoint_steps=checkpoints,
        validation_seeds=args.validation_seeds,
        max_eval_steps=args.max_eval_steps,
        model_dir=model_dir,
        result_dir=result_dir,
        initial_rank=checkpoint_rank(initial_summary),
        initial_summary=initial_summary,
        initial_rows=initial_rows,
        reward_version=REWARD_VERSION,
        env_kwargs=env_kwargs,
    )
    callback._save_tables()
    print(f"Experiment:       {args.experiment_name}")
    print("HER in training:  no")
    print("CBF in training:  no")
    print("SAC/LQR cadence:  1 Hz / 1 Hz")
    print(f"Online steps:     {args.total_timesteps:,}")
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
            "source_prefill": prefill_metrics,
        }
    )
    write_json(result_dir / "experiment_config.json", config)
    write_model_readme(
        model_dir=model_dir,
        config=config,
        checkpoint_summaries=callback.checkpoint_summaries,
        best_checkpoint=callback.best_checkpoint,
        complete=True,
    )
    training_env.close()


if __name__ == "__main__":
    main()
