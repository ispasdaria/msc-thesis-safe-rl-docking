"""Train 10 Hz SAC reference residuals around the dt=0.1 s closing LQR."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from stable_baselines3 import HerReplayBuffer
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
from her_source_warmstart import prefill_her_replay_buffer
from lqr_reference_residual_env import (
    ReferenceResidualLQRGoalConditionedClosingEnv,
)
from residual_lqr_config import (
    CLOSING_LQR_CONTROL_HZ,
    CLOSING_LQR_DT_SECONDS,
    discount_for_dt,
    steps_for_duration,
)
from train_sac_closing_her import HerEvaluationCheckpointCallback
from train_sac_closing_her_v4_residual_v15 import (
    initialize_zero_residual_actor,
)
from zero_residual_constrained_sac import ZeroResidualConstrainedSAC


CLOSING_DIR = Path(__file__).resolve().parent
MODEL_ROOT = CLOSING_DIR / "models" / "her_lqr_reference_residual"
RESULT_ROOT = CLOSING_DIR / "results" / "her_lqr_reference_residual"
LOG_ROOT = CLOSING_DIR / "logs" / "her_lqr_reference_residual"
REWARD_VERSION = "lqr_reference_residual"


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
        "p3_max_target_ratio",
        "state",
        "previous_achieved_goal",
        "reference_residual_normalized_action",
        "previous_reference_residual_normalized_action",
        "violated_constraints",
        "termination_reason",
    )
    compact = {key: copy.deepcopy(info[key]) for key in keys if key in info}
    compact["TimeLimit.truncated"] = bool(truncated)
    return compact


def reference_env_kwargs(args):
    return {
        "dt": CLOSING_LQR_DT_SECONDS,
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
    }


def collect_reference_residual_trajectories(
    *,
    minimum_transitions,
    seed_start,
    max_episodes,
    max_steps,
    env_kwargs,
    residual_std,
    residual_clip,
    zero_episode_fraction,
    random_seed,
    retain_failure_episodes,
    residual_hold_steps=1,
    env_factory=None,
):
    """Collect complete LQR episodes with reference-residual probes."""
    if minimum_transitions <= 0:
        raise ValueError("minimum_transitions must be positive.")
    if residual_hold_steps <= 0:
        raise ValueError("residual_hold_steps must be positive.")
    rng = np.random.default_rng(random_seed)
    env = (
        env_factory(seed_start)
        if env_factory is not None
        else make_her_env(
            seed=seed_start,
            mode="training",
            max_time=max_steps,
            reward_version=REWARD_VERSION,
            **env_kwargs,
        )
    )
    transitions = []
    summaries = []
    accepted_episode_id = 0
    # Read the resolved period from the environment. Residual-LQR factories
    # default to the canonical 0.1 s period even when callers omit ``dt``.
    environment_dt = float(env.core_env.dt)
    maximum_control_steps = int(np.ceil(float(max_steps) / environment_dt))
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
            action_square_sum = np.zeros(3, dtype=np.float64)
            action_abs_max = np.zeros(3, dtype=np.float64)
            lqr_saturated_steps = 0
            action = np.zeros(3, dtype=np.float32)
            for step in range(maximum_control_steps):
                if zero_episode:
                    action = np.zeros(3, dtype=np.float32)
                elif step % int(residual_hold_steps) == 0:
                    action = np.clip(
                        rng.normal(0.0, residual_std, size=3),
                        -residual_clip,
                        residual_clip,
                    ).astype(np.float32)
                next_observation, reward, terminated, truncated, final_info = (
                    env.step(action)
                )
                done = bool(terminated or truncated)
                episode.append(
                    {
                        "observation": _copy_observation(observation),
                        "next_observation": _copy_observation(next_observation),
                        "action": action.copy(),
                        "reward": float(reward),
                        "done": done,
                        "info": _compact_info(final_info, truncated),
                        "episode_id": accepted_episode_id,
                    }
                )
                action_square_sum += np.square(action)
                action_abs_max = np.maximum(action_abs_max, np.abs(action))
                lqr_saturated_steps += int(final_info.get("lqr_saturated", False))
                observation = next_observation
                if done:
                    break
            failure = bool(final_info.get("failure", False))
            retained = bool(not failure or retain_failure_episodes)
            action_rms = np.sqrt(action_square_sum / max(len(episode), 1))
            summaries.append(
                {
                    "attempt": attempt + 1,
                    "seed": seed,
                    "steps": len(episode),
                    "accepted": int(retained),
                    "zero_reference_residual_episode": int(zero_episode),
                    "residual_hold_steps": int(residual_hold_steps),
                    "success": int(bool(final_info.get("success", False))),
                    "failure": int(failure),
                    "time_limit": int(bool(final_info.get("time_limit", False))),
                    "lqr_saturation_fraction": float(
                        lqr_saturated_steps / max(len(episode), 1)
                    ),
                    "residual_rms_vx": float(action_rms[0]),
                    "residual_rms_vy": float(action_rms[1]),
                    "residual_rms_attitude": float(action_rms[2]),
                    "residual_max_vx": float(action_abs_max[0]),
                    "residual_max_vy": float(action_abs_max[1]),
                    "residual_max_attitude": float(action_abs_max[2]),
                    "termination_reason": final_info.get("termination_reason", ""),
                    "violated_constraints": str(
                        final_info.get("violated_constraints", [])
                    ),
                }
            )
            if retained:
                transitions.extend(episode)
                accepted_episode_id += 1
    finally:
        env.close()
    if len(transitions) < minimum_transitions:
        raise RuntimeError(
            f"Collected only {len(transitions):,} retained transitions; "
            f"requested {minimum_transitions:,}."
        )
    return transitions, summaries


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train SAC reference residuals around closing LQR."
    )
    parser.add_argument(
        "--experiment-name",
        default=(
            "sac_closing_goal_her_lqr_reference_residual_"
            "dt0p1_her_50k_seed0"
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
    parser.add_argument(
        "--max-eval-steps",
        type=int,
        default=steps_for_duration(12_000.0),
    )
    parser.add_argument("--vx-reference-bound", type=float, default=0.02)
    parser.add_argument("--vy-reference-bound", type=float, default=0.01)
    parser.add_argument(
        "--attitude-reference-bound-deg", type=float, default=1.0
    )
    parser.add_argument("--progress-weight", type=float, default=100.0)
    parser.add_argument(
        "--reference-magnitude-weight", type=float, default=0.01
    )
    parser.add_argument(
        "--reference-smoothness-weight", type=float, default=0.01
    )
    parser.add_argument("--success-bonus", type=float, default=100.0)
    parser.add_argument("--failure-penalty", type=float, default=1000.0)
    parser.add_argument("--timeout-penalty", type=float, default=100.0)
    parser.add_argument("--initial-log-std", type=float, default=-3.0)
    parser.add_argument("--prefill-residual-std", type=float, default=0.25)
    parser.add_argument("--prefill-residual-clip", type=float, default=0.75)
    parser.add_argument(
        "--prefill-zero-episode-fraction", type=float, default=0.5
    )
    parser.add_argument("--prefill-random-seed", type=int, default=7_000)
    parser.add_argument("--retain-failure-prefill", action="store_true")
    parser.add_argument("--maximum-gate-failure-rate", type=float, default=0.0)
    parser.add_argument("--maximum-gate-min-position-error", type=float, default=100.0)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--gamma",
        type=float,
        default=discount_for_dt(0.9999),
        help="Per-step discount preserving gamma=0.9999 per second.",
    )
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--ent-coef", default="auto_0.01")
    parser.add_argument("--critic-warmup-steps", type=int, default=10_000)
    parser.add_argument("--zero-residual-actor-weight", type=float, default=1.0)
    parser.add_argument("--actor-learning-rate-scale", type=float, default=0.25)
    parser.add_argument("--target-entropy", type=float, default=-3.0)
    parser.add_argument("--n-sampled-goal", type=int, default=4)
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
        "n_sampled_goal",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if not 0 <= args.critic_warmup_steps < args.total_timesteps:
        parser.error("--critic-warmup-steps must lie in [0, total-timesteps).")
    if not 0.0 < args.gamma <= 1.0:
        parser.error("--gamma must lie in (0, 1].")
    if not 0.0 < args.prefill_residual_clip <= 1.0:
        parser.error("--prefill-residual-clip must lie in (0, 1].")
    if args.prefill_residual_std < 0.0:
        parser.error("--prefill-residual-std must be non-negative.")
    if not 0.0 <= args.prefill_zero_episode_fraction <= 1.0:
        parser.error("--prefill-zero-episode-fraction must lie in [0, 1].")
    for name in (
        "progress_weight",
        "reference_magnitude_weight",
        "reference_smoothness_weight",
        "success_bonus",
        "failure_penalty",
        "timeout_penalty",
        "zero_residual_actor_weight",
        "actor_learning_rate_scale",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative.")


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
    reward = config["reward"]
    sac = config["sac"]
    bounds = config["reference_residual_bounds"]
    status = "complete" if complete else "in progress"
    baseline = checkpoint_summaries[0]
    final = checkpoint_summaries[-1]
    if complete and int(best_checkpoint) == 0:
        interpretation = (
            "No learned checkpoint outranked the zero-reference LQR. "
            f"Success changed from {float(baseline['success_rate']):.1%} "
            f"at checkpoint 0 to {float(final['success_rate']):.1%} at the "
            "final checkpoint, so `best_model.zip` intentionally remains "
            "the exact zero-reference baseline. `final_model.zip` is kept "
            "for diagnosis and is not the recommended controller."
        )
    elif complete:
        interpretation = (
            f"Checkpoint {int(best_checkpoint):,} produced the highest "
            "validation rank. `best_model.zip` contains that checkpoint; "
            "`final_model.zip` separately contains the end of training."
        )
    else:
        interpretation = (
            "The experiment is still running; checkpoint selection and the "
            "final interpretation are provisional."
        )
    content = f"""# Pilot B LQR-reference-residual SAC model card

Run: `{config['experiment_name']}`  
Status: **{status}**  
Selected best checkpoint: **{int(best_checkpoint):,} SAC decisions**

## Controller architecture

SAC produces a fresh normalized reference correction every simulator second.
There is no action hold and no distance gate. The correction changes only the
reference tracked by the existing closing LQR:

```text
delta_reference = [
    {bounds['vx_mps']:.3f} * a[0] m/s,
    {bounds['vy_mps']:.3f} * a[1] m/s,
    {bounds['attitude_deg']:.3f} * a[2] deg,
]
u_nominal = LQR(state, scheduled_P3_reference + delta_reference)
```

The zero action exactly reproduces the established closing LQR. CBF is not
instantiated, called, observed, or used in the reward during this experiment.
Hard failure termination remains part of the unfiltered simulator.

## Reward

Let `J(g, x)` be an equally weighted mean/worst log transform of the six
absolute P3 tolerance ratios. For the SAC discount `{sac['gamma']:.4f}`:

```text
reward = {reward['progress_weight']:.1f} * (J(previous) - gamma * J(next))
         + {reward['success_bonus']:.1f} * exact_P3_success
         - {reward['failure_penalty']:.1f} * hard_failure
         - {reward['timeout_penalty']:.1f} * unsuccessful_timeout
         - {reward['magnitude_weight']:.3f} * ||a||^2
         - {reward['smoothness_weight']:.3f} * ||a - a_previous||^2
```

The previous achieved goal and both actions are stored in every replay info
dictionary, allowing HER to recompute all terms after goal relabeling.

## SAC configuration

| Parameter | Value |
|---|---:|
| Online interactions / decisions | {config['total_timesteps']:,} |
| SAC and LQR decision interval | 1 s |
| Replay prefill transitions | {config['source_transition_count_requested']:,} |
| Critic-only warm-up | {config['critic_warmup_steps']:,} interactions |
| Learning rate | {sac['learning_rate']:.2g} |
| Effective actor learning rate | {sac['learning_rate'] * config['actor_learning_rate_scale']:.2g} |
| Actor zero-reference weight | {config['zero_residual_actor_weight']:.3g} |
| Train frequency | every {sac['train_freq']} interactions |
| Gradient steps per update | {sac['gradient_steps']} |
| Update-to-data ratio | {sac['update_to_data_ratio']:.3f} |
| Batch size | {sac['batch_size']} |
| Discount factor | {sac['gamma']:.4f} |
| Target smoothing coefficient | {sac['tau']:.4f} |
| Entropy coefficient | `{sac['ent_coef']}` |
| Target entropy | {sac['target_entropy']:.1f} |
| HER future goals per transition | {config['her']['n_sampled_goal']} |

## Validation results

Validation uses seeds {min(config['validation_seeds'])}--{max(config['validation_seeds'])}.

| Checkpoint | Success | Hard failure | Timeout | Mean minimum position error [m] | Mean best worst ratio | Mean seconds |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Interpretation

{interpretation}

## Artifacts

- `checkpoint_0_model.zip`: zero-reference-residual LQR baseline;
- `checkpoint_<N>_model.zip`: evaluated SAC checkpoint;
- `best_model.zip`: validation-ranked checkpoint;
- `final_model.zip`: final policy, whether or not it ranked best.

Metrics and the machine-readable configuration are in
`../../../results/her_lqr_reference_residual/{config['experiment_name']}/`.
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
    env_kwargs = reference_env_kwargs(args)
    config = {
        "experiment_name": args.experiment_name,
        "method": "10 Hz SAC reference residual around closing LQR",
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
        "dt_seconds": CLOSING_LQR_DT_SECONDS,
        "control_frequency_hz": CLOSING_LQR_CONTROL_HZ,
        "sac_decision_interval_seconds": CLOSING_LQR_DT_SECONDS,
        "lqr_update_interval_seconds": CLOSING_LQR_DT_SECONDS,
        "residual_hold_steps": 1,
        "distance_gate_used": False,
        "cbf_filter_used_during_training": False,
        "cbf_signal_in_observation_or_reward": False,
        "unfiltered_simulator_hard_failure_termination": True,
        "reward": {
            "progress_weight": args.progress_weight,
            "progress_form": "J(previous) - gamma * J(next)",
            "success_bonus": args.success_bonus,
            "failure_penalty": args.failure_penalty,
            "timeout_penalty": args.timeout_penalty,
            "magnitude_weight": args.reference_magnitude_weight,
            "smoothness_weight": args.reference_smoothness_weight,
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
            "decision_interval_seconds": CLOSING_LQR_DT_SECONDS,
            "random_seed": args.prefill_random_seed,
            "retain_failure_episodes": args.retain_failure_prefill,
        },
        "critic_warmup_steps": args.critic_warmup_steps,
        "zero_residual_actor_weight": args.zero_residual_actor_weight,
        "actor_learning_rate_scale": args.actor_learning_rate_scale,
        "sac": {
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "learning_starts": 0,
            "batch_size": args.batch_size,
            "tau": args.tau,
            "gamma": args.gamma,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "update_to_data_ratio": args.gradient_steps / args.train_freq,
            "ent_coef": args.ent_coef,
            "target_entropy": args.target_entropy,
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
    prefill_metrics = prefill_her_replay_buffer(model, transitions)

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
    print("CBF in training:  no")
    print(
        f"SAC/LQR cadence:  {CLOSING_LQR_CONTROL_HZ:g} Hz / "
        f"{CLOSING_LQR_CONTROL_HZ:g} Hz"
    )
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
