"""Train goal-conditioned residual SAC+HER around the closing LQR."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
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
from lqr_residual_env import ResidualLQRGoalConditionedClosingEnv
from train_sac_closing_her_v4_residual_v15 import (
    ResidualSafetyGuardCallback,
    initialize_zero_residual_actor,
)
from zero_residual_constrained_sac import ZeroResidualConstrainedSAC


CLOSING_DIR = Path(__file__).resolve().parent
MODEL_ROOT = CLOSING_DIR / "models" / "her_lqr_residual"
RESULT_ROOT = CLOSING_DIR / "results" / "her_lqr_residual"
LOG_ROOT = CLOSING_DIR / "logs" / "her_lqr_residual"


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


def collect_lqr_residual_trajectories(
    *,
    minimum_transitions,
    seed_start,
    max_episodes,
    max_steps,
    residual_scale,
    residual_zero_distance=None,
    residual_full_distance=None,
    environment_residual_hold_steps=1,
    residual_std=0.25,
    residual_clip=0.75,
    residual_hold_steps=20,
    zero_episode_fraction=0.5,
    random_seed=6_000,
    retain_failure_episodes=False,
):
    """Collect complete LQR episodes with zero or bounded residual probes."""
    if minimum_transitions <= 0:
        raise ValueError("minimum_transitions must be positive.")
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
        reward_version="lqr_residual",
        residual_scale=residual_scale,
        residual_zero_distance=residual_zero_distance,
        residual_full_distance=residual_full_distance,
        residual_hold_steps=environment_residual_hold_steps,
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
            lqr_saturated_steps = 0

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
                action_abs_max = np.maximum(action_abs_max, np.abs(residual_action))
                lqr_saturated_steps += int(final_info.get("lqr_saturated", False))
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
                    "success": int(bool(final_info.get("success", False))),
                    "failure": int(failure),
                    "time_limit": int(bool(final_info.get("time_limit", False))),
                    "lqr_saturation_fraction": float(
                        lqr_saturated_steps / max(len(episode), 1)
                    ),
                    "residual_rms_ax": float(action_rms[0]),
                    "residual_rms_ay": float(action_rms[1]),
                    "residual_rms_alpha": float(action_rms[2]),
                    "residual_max_ax": float(action_abs_max[0]),
                    "residual_max_ay": float(action_abs_max[1]),
                    "residual_max_alpha": float(action_abs_max[2]),
                    "termination_reason": final_info.get("termination_reason", ""),
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
            f"Collected only {len(transitions):,} retained LQR-residual "
            f"transitions; requested {minimum_transitions:,}."
        )
    return transitions, summaries


def build_parser():
    parser = argparse.ArgumentParser(
        description="Residual SAC+HER around the closing LQR controller."
    )
    parser.add_argument(
        "--experiment-name",
        default="sac_closing_goal_her_residual_lqr_pilot_a_1hz_20k_seed0",
    )
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[5_000, 10_000, 20_000],
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
    parser.add_argument("--residual-zero-distance", type=float, default=100.0)
    parser.add_argument("--residual-full-distance", type=float, default=200.0)
    parser.add_argument("--residual-hold-steps", type=int, default=1)
    parser.add_argument("--initial-log-std", type=float, default=-4.0)
    parser.add_argument("--prefill-residual-std", type=float, default=0.25)
    parser.add_argument("--prefill-residual-clip", type=float, default=0.75)
    parser.add_argument("--prefill-residual-hold-steps", type=int, default=1)
    parser.add_argument("--prefill-zero-episode-fraction", type=float, default=0.5)
    parser.add_argument("--prefill-random-seed", type=int, default=6_000)
    parser.add_argument("--retain-failure-prefill", action="store_true")
    parser.add_argument("--maximum-gate-failure-rate", type=float, default=0.0)
    parser.add_argument(
        "--maximum-online-validation-failure-rate",
        type=float,
        default=0.1,
    )
    parser.add_argument("--maximum-gate-min-position-error", type=float, default=100.0)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--ent-coef", default="auto_0.001")
    parser.add_argument("--critic-warmup-steps", type=int, default=10_000)
    parser.add_argument("--zero-residual-actor-weight", type=float, default=10.0)
    parser.add_argument("--actor-learning-rate-scale", type=float, default=0.1)
    parser.add_argument("--target-entropy", type=float, default=-6.0)
    parser.add_argument("--n-sampled-goal", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(parser, args):
    if args.total_timesteps <= 0:
        parser.error("--total-timesteps must be positive.")
    if not 0 <= args.critic_warmup_steps < args.total_timesteps:
        parser.error(
            "--critic-warmup-steps must lie in [0, total-timesteps)."
        )
    if args.zero_residual_actor_weight < 0.0:
        parser.error("--zero-residual-actor-weight must be non-negative.")
    if args.actor_learning_rate_scale < 0.0:
        parser.error("--actor-learning-rate-scale must be non-negative.")
    if not 0.0 < args.residual_scale <= 1.0:
        parser.error("--residual-scale must lie in (0, 1].")
    if args.residual_zero_distance < 0.0:
        parser.error("--residual-zero-distance must be non-negative.")
    if args.residual_full_distance <= args.residual_zero_distance:
        parser.error(
            "--residual-full-distance must exceed --residual-zero-distance."
        )
    if args.residual_hold_steps <= 0:
        parser.error("--residual-hold-steps must be positive.")
    if args.prefill_residual_std < 0.0:
        parser.error("--prefill-residual-std must be non-negative.")
    if not 0.0 < args.prefill_residual_clip <= 1.0:
        parser.error("--prefill-residual-clip must lie in (0, 1].")
    if args.prefill_residual_hold_steps <= 0:
        parser.error("--prefill-residual-hold-steps must be positive.")
    if not 0.0 <= args.prefill_zero_episode_fraction <= 1.0:
        parser.error("--prefill-zero-episode-fraction must lie in [0, 1].")


def write_pilot_a_model_readme(
    model_dir,
    config,
    checkpoint_summaries,
    best_checkpoint,
    run_complete,
):
    """Write a self-contained model card beside the Pilot A artifacts."""
    rows = []
    hold_steps = int(config["residual_hold_steps"])
    for summary in checkpoint_summaries:
        mean_decisions = float(summary["mean_steps"])
        rows.append(
            "| {checkpoint:,} | {success:.1%} | {failure:.1%} | "
            "{timeout:.1%} | {position:.3f} | {ratio:.3f} | "
            "{decisions:.1f} | {seconds:.1f} |".format(
                checkpoint=int(summary["checkpoint"]),
                success=float(summary["success_rate"]),
                failure=float(summary["failure_rate"]),
                timeout=float(summary["time_limit_rate"]),
                position=float(summary["mean_minimum_position_error"]),
                ratio=float(summary["mean_best_max_target_ratio"]),
                decisions=mean_decisions,
                seconds=mean_decisions * hold_steps,
            )
        )
    status = "complete" if run_complete else "in progress"
    content = f"""# Pilot A residual SAC model card

Run: `{config['experiment_name']}`  
Status: **{status}**  
Selected best checkpoint: **{int(best_checkpoint):,} residual decisions**

## Purpose

Pilot A tests stronger critic training while leaving the closing LQR law
unchanged. SAC selects one normalized residual action every {hold_steps}
simulator second(s), and the LQR action is recomputed from the same current
physical state before the two commands are combined. In the configured Pilot
A run, both feedback decisions therefore operate at 1 Hz.

```text
u_executed = clip(u_LQR_clipped + 0.05 * gate(distance) * a_residual, -1, 1)
```

The residual gate is zero at or below
{config['residual_gate']['zero_authority_at_or_below_m']:.0f} m, uses a
smoothstep ramp, and reaches full authority at or above
{config['residual_gate']['full_authority_at_or_above_m']:.0f} m.

## SAC configuration

| Parameter | Value |
|---|---:|
| Online residual decisions | {int(config['total_timesteps']):,} |
| Residual decision interval | {config['residual_decision_interval_seconds']:.0f} s |
| LQR update interval | {config['lqr_update_interval_seconds']:.0f} s |
| Replay prefill transitions | {int(config['source_transition_count_requested']):,} |
| Critic warm-up | {int(config['critic_warmup_steps']):,} residual decisions/updates |
| Critic learning rate | {config['sac']['learning_rate']:.2g} |
| Actor learning-rate scale | {config['actor_learning_rate_scale']:.3g} |
| Effective actor learning rate | {config['sac']['learning_rate'] * config['actor_learning_rate_scale']:.2g} |
| Train frequency | {int(config['sac']['train_freq'])} |
| Gradient steps | {int(config['sac']['gradient_steps'])} |
| Update-to-data ratio | {config['sac']['update_to_data_ratio']:.3g} |
| Discount factor | {config['sac']['gamma']:.6f} per residual decision |
| Target smoothing coefficient | {config['sac']['tau']:.4f} |
| Entropy coefficient | `{config['sac']['ent_coef']}` |
| Target entropy | {config['sac']['target_entropy']:.1f} |
| Zero-residual actor weight | {config['zero_residual_actor_weight']:.1f} |
| Batch size | {int(config['sac']['batch_size'])} |
| HER future goals per transition | {int(config['her']['n_sampled_goal'])} |

The first {int(config['critic_warmup_steps']):,} online residual decisions use
the deterministic zero actor and update only the critic. With `train_freq=1`
and `gradient_steps=1`, this produces one critic update per collected residual
transition during warm-up.

## Reward

The environment reward remains `goal_conditioned_p3_residual_lqr_v1`, whose
numerical calculation is inherited from `GoalConditionedClosingV3Env`:

```text
reward = -0.01 * goal_cost
         +100 * exact_P3_success
         -1000 * hard_failure
         -velocity_warning_depth
         -geometry_warning_depth
         -alignment_warning_depth
         -100 * unsuccessful_timeout
```

The constrained actor additionally optimizes
`{config['zero_residual_actor_weight']:.1f} * ||a_residual||^2`. This is an
actor-only regularizer and is not part of the environment reward or critic
target. The reward returned for a held residual decision is the final
one-second substep reward so HER can recompute it consistently; the sum across
substeps is retained only as an `info` diagnostic.

## Validation results

Validation uses seeds {min(config['validation_seeds'])}--{max(config['validation_seeds'])}.
`Mean decisions` counts SAC residual decisions; `Approx. seconds` multiplies
that count by {hold_steps} and may overestimate an episode whose final held
action ended early on success or termination.

| Checkpoint | Success | Hard failure | Timeout | Mean minimum position error [m] | Mean best worst ratio | Mean decisions | Approx. seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Artifacts

- `checkpoint_0_model.zip`: exact zero-residual LQR baseline;
- `checkpoint_<N>_model.zip`: evaluated model at residual decision `N`;
- `best_model.zip`: best validation-ranked checkpoint;
- `final_model.zip`: policy at the end of training.

The matching metrics and exact configuration are under
`../../../results/her_lqr_residual/{config['experiment_name']}/`.
"""
    (Path(model_dir) / "README.md").write_text(content, encoding="utf-8")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    checkpoints = sorted(
        {step for step in args.checkpoint_steps if 0 < step <= args.total_timesteps}
    )
    if args.total_timesteps not in checkpoints:
        checkpoints.append(args.total_timesteps)

    config = {
        "experiment_name": args.experiment_name,
        "method": "goal-conditioned residual SAC+HER around closing LQR",
        "reward_version": "lqr_residual",
        "reward_scheme": ResidualLQRGoalConditionedClosingEnv.reward_scheme,
        "baseline_controller": "LQRClosingController",
        "baseline_action_semantics": (
            "clip physical LQR command, normalize, then add bounded residual"
        ),
        "residual_scale": args.residual_scale,
        "residual_gate": {
            "zero_authority_at_or_below_m": args.residual_zero_distance,
            "full_authority_at_or_above_m": args.residual_full_distance,
            "interpolation": "smoothstep",
        },
        "initial_residual_policy": "zero deterministic residual",
        "pilot_design": "Pilot A: 1 Hz residual and stronger critic training",
        "critic_warmup_steps": args.critic_warmup_steps,
        "critic_warmup_interpretation": (
            "one critic update per residual transition during warm-up"
        ),
        "zero_residual_actor_weight": args.zero_residual_actor_weight,
        "actor_learning_rate_scale": args.actor_learning_rate_scale,
        "residual_hold_steps": args.residual_hold_steps,
        "residual_decision_interval_seconds": float(
            args.residual_hold_steps
        ),
        "lqr_update_interval_seconds": 1.0,
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
        "action_hold_steps": 1,
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
            "target_entropy": args.target_entropy,
            "update_to_data_ratio": float(
                args.gradient_steps / args.train_freq
            ),
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

    transitions, episode_rows = collect_lqr_residual_trajectories(
        minimum_transitions=args.source_transition_count,
        seed_start=args.source_seed_start,
        max_episodes=args.source_max_episodes,
        max_steps=args.max_time,
        residual_scale=args.residual_scale,
        residual_zero_distance=args.residual_zero_distance,
        residual_full_distance=args.residual_full_distance,
        environment_residual_hold_steps=args.residual_hold_steps,
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
                    reward_version="lqr_residual",
                    residual_scale=args.residual_scale,
                    residual_zero_distance=args.residual_zero_distance,
                    residual_full_distance=args.residual_full_distance,
                    residual_hold_steps=args.residual_hold_steps,
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

    env_kwargs = {
        "residual_scale": args.residual_scale,
        "residual_zero_distance": args.residual_zero_distance,
        "residual_full_distance": args.residual_full_distance,
        "residual_hold_steps": args.residual_hold_steps,
    }
    initial_rows = evaluate_model(
        model,
        args.validation_seeds,
        max_steps=args.max_eval_steps,
        reward_version="lqr_residual",
        env_kwargs=env_kwargs,
    )
    initial_summary = summarize_rollouts(initial_rows)
    write_csv(result_dir / "zero_residual_validation_rollouts.csv", initial_rows)
    write_json(
        result_dir / "source_initialization.json",
        {**prefill_metrics, "zero_residual_validation": initial_summary},
    )
    if initial_summary["failure_rate"] > args.maximum_gate_failure_rate:
        raise RuntimeError("Exact zero-residual LQR failed the safety gate.")
    if (
        initial_summary["mean_minimum_position_error"]
        > args.maximum_gate_min_position_error
    ):
        raise RuntimeError(
            "Exact zero-residual LQR failed the approach gate: "
            f"{initial_summary['mean_minimum_position_error']:.2f} m."
        )

    initial_rank = checkpoint_rank(initial_summary)
    model.save(model_dir / "checkpoint_0_model")
    model.save(model_dir / "best_model")
    write_json(
        result_dir / "best_checkpoint.json",
        {"checkpoint": 0, "summary": initial_summary},
    )
    write_pilot_a_model_readme(
        model_dir=model_dir,
        config=config,
        checkpoint_summaries=[{"checkpoint": 0, **initial_summary}],
        best_checkpoint=0,
        run_complete=False,
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
        reward_version="lqr_residual",
        env_kwargs=env_kwargs,
    )
    callback._save_tables()
    print(f"Experiment:       {args.experiment_name}")
    print(
        f"Exact LQR gate:    failure={initial_summary['failure_rate']:.3f}, "
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
    write_pilot_a_model_readme(
        model_dir=model_dir,
        config=config,
        checkpoint_summaries=callback.checkpoint_summaries,
        best_checkpoint=callback.best_checkpoint,
        run_complete=True,
    )
    training_env.close()


if __name__ == "__main__":
    main()
