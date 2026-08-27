"""Pilot D: Pilot C with fixed 50/25/25 P3-stratified replay."""

from __future__ import annotations

from pathlib import Path
import sys


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
from p3_stratified_dict_replay_buffer import P3StratifiedDictReplayBuffer
from train_sac_closing_her import HerEvaluationCheckpointCallback
from train_sac_closing_her_reference_residual_lqr import (
    collect_reference_residual_trajectories,
)
from train_sac_closing_her_v4_residual_v15 import (
    initialize_zero_residual_actor,
)
from train_sac_closing_reference_residual_lqr_no_her import (
    build_parser as build_pilot_c_parser,
    environment_kwargs,
    prefill_standard_replay_buffer,
    validate_args as validate_pilot_c_args,
)
from zero_residual_constrained_sac import ZeroResidualConstrainedSAC


CLOSING_DIR = Path(__file__).resolve().parent
MODEL_ROOT = CLOSING_DIR / "models" / "lqr_reference_residual_stratified"
RESULT_ROOT = CLOSING_DIR / "results" / "lqr_reference_residual_stratified"
LOG_ROOT = CLOSING_DIR / "logs" / "lqr_reference_residual_stratified"
REWARD_VERSION = "lqr_reference_residual"


def build_parser():
    parser = build_pilot_c_parser()
    parser.description = (
        "Pilot D: fixed-P3 SAC reference residual with 50/25/25 replay."
    )
    parser.set_defaults(
        experiment_name=(
            "sac_closing_lqr_reference_residual_"
            "pilot_d_stratified_1hz_50k_seed0"
        )
    )
    parser.add_argument("--outer-replay-size", type=int, default=100_000)
    parser.add_argument("--inner-replay-size", type=int, default=100_000)
    parser.add_argument("--global-batch-fraction", type=float, default=0.50)
    parser.add_argument("--outer-batch-fraction", type=float, default=0.25)
    parser.add_argument("--inner-batch-fraction", type=float, default=0.25)
    parser.add_argument("--outer-ratio-limit", type=float, default=5.0)
    parser.add_argument("--inner-ratio-limit", type=float, default=2.0)
    parser.add_argument("--pilot-label", choices=("D", "E"), default="D")
    parser.add_argument(
        "--zero-residual-actor-final-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--zero-residual-actor-anneal-end-step",
        type=int,
        default=None,
    )
    return parser


def validate_args(parser, args):
    validate_pilot_c_args(parser, args)
    if args.outer_replay_size <= 0 or args.inner_replay_size <= 0:
        parser.error("Near-P3 replay sizes must be positive.")
    fractions = (
        args.global_batch_fraction,
        args.outer_batch_fraction,
        args.inner_batch_fraction,
    )
    if any(value < 0.0 for value in fractions):
        parser.error("Replay fractions must be non-negative.")
    if abs(sum(fractions) - 1.0) > 1e-9:
        parser.error("Replay fractions must sum to 1.")
    if not 0.0 < args.inner_ratio_limit < args.outer_ratio_limit:
        parser.error(
            "Ratio limits must satisfy 0 < inner-ratio-limit < "
            "outer-ratio-limit."
        )
    schedule_values = (
        args.zero_residual_actor_final_weight,
        args.zero_residual_actor_anneal_end_step,
    )
    if (schedule_values[0] is None) != (schedule_values[1] is None):
        parser.error(
            "Actor-prior final weight and anneal end step must be provided "
            "together."
        )
    if args.zero_residual_actor_final_weight is not None:
        if args.zero_residual_actor_final_weight < 0.0:
            parser.error("Actor-prior final weight must be non-negative.")
        if not (
            args.critic_warmup_steps
            < args.zero_residual_actor_anneal_end_step
            <= args.total_timesteps
        ):
            parser.error(
                "Actor-prior anneal end must be after critic warm-up and no "
                "later than total timesteps."
            )


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
    replay = config["replay_distribution"]
    actor_constraint = config["actor_constraint"]
    reward = config["reward"]
    sac = config["sac"]
    bounds = config["reference_residual_bounds"]
    status = "complete" if complete else "in progress"
    if not complete:
        interpretation = "Training is in progress; model selection is provisional."
    elif int(best_checkpoint) == 0:
        interpretation = (
            "No trained checkpoint outranked the exact zero-residual LQR "
            "baseline on the validation seeds."
        )
    else:
        interpretation = (
            f"Checkpoint {int(best_checkpoint):,} ranked best on the "
            "validation seeds and is stored as `best_model.zip`."
        )
    statistics = config.get("final_replay_statistics", {})
    replay_summary = (
        "Not available yet."
        if not statistics
        else (
            f"Global/outer/inner stored: "
            f"{statistics['global_buffer_size']:,}/"
            f"{statistics['outer_buffer_size']:,}/"
            f"{statistics['inner_buffer_size']:,}. Realized sampled fractions: "
            f"{statistics['realized_global_sample_fraction']:.1%}/"
            f"{statistics['realized_outer_sample_fraction']:.1%}/"
            f"{statistics['realized_inner_sample_fraction']:.1%}."
        )
    )
    if actor_constraint["schedule_used"]:
        change_description = (
            f"Pilot {config['pilot']} changes exactly one Pilot D setting: "
            "the zero-residual actor penalty is annealed linearly from "
            f"{actor_constraint['initial_weight']:.3g} at interaction "
            f"{actor_constraint['anneal_start_step']:,} to "
            f"{actor_constraint['final_weight']:.3g} at interaction "
            f"{actor_constraint['anneal_end_step']:,}, then held constant. "
            "The controller, reward, replay distribution, seeds, "
            "initialization, and all other SAC settings are unchanged."
        )
        actor_weight_description = (
            f"{actor_constraint['initial_weight']:.3g} at "
            f"{actor_constraint['anneal_start_step']:,}; linear to "
            f"{actor_constraint['final_weight']:.3g} at "
            f"{actor_constraint['anneal_end_step']:,}"
        )
    else:
        change_description = """Pilot D changes only Pilot C's replay sampling
distribution. The controller, fixed P3 goal, reward, one-second SAC/LQR
cadence, zero-residual actor prior, and no-HER setup are unchanged."""
        actor_weight_description = f"{actor_constraint['initial_weight']:.3g}"
    content = f"""# Pilot {config['pilot']} P3-stratified reference-residual SAC model card

Run: `{config['experiment_name']}`  
Status: **{status}**  
Selected best checkpoint: **{int(best_checkpoint):,} SAC decisions**

## What this experiment changes

{change_description}

No CBF is used during training.

Each batch is constructed from:

- **{replay['global_fraction']:.0%} uniform:** all stored closing transitions;
- **{replay['outer_fraction']:.0%} near-P3 pool:**
  `worst ratio <= {replay['outer_ratio_limit']:g}`;
- **{replay['inner_fraction']:.0%} near-P3 inner band:**
  `worst ratio <= {replay['inner_ratio_limit']:g}`.

The targeted pools are deliberately nested: every `ratio <= 2` transition is
eligible for both targeted quarters, giving the closest states extra replay
weight exactly as implied by the two requested thresholds. Uniform replay also
samples from the complete buffer and may naturally include near-P3 transitions.
If a targeted ring is empty early in collection, its missing batch slots fall
back to uniform replay.

Final replay audit: {replay_summary}

## Controller

```text
delta_reference = [
    {bounds['vx_mps']:.3f} * a[0] m/s,
    {bounds['vy_mps']:.3f} * a[1] m/s,
    {bounds['attitude_deg']:.3f} * a[2] deg,
]
u_nominal = LQR(state, scheduled_P3_reference + delta_reference)
```

Zero SAC action exactly reproduces the original LQR. There is no action hold,
distance gate, CBF signal, HER relabeling, or safety filtering in training.

## Reward

```text
reward = {reward['progress_weight']:.3f} * (J(previous) - gamma * J(next))
         + {reward['success_bonus']:.3f} * exact_P3_success
         - {reward['failure_penalty']:.3f} * hard_failure
         - {reward['timeout_penalty']:.3f} * unsuccessful_timeout
         - {reward['magnitude_weight']:.3f} * ||a||^2
         - {reward['smoothness_weight']:.3f} * ||a - a_previous||^2
```

`J` equally combines the mean and maximum of the six log-transformed,
normalized P3 tolerance ratios. The replay strata use the maximum untransformed
ratio from the actual next state and actual fixed P3 target.

## SAC configuration

| Parameter | Value |
|---|---:|
| Online interactions | {config['total_timesteps']:,} |
| Replay prefill | {config['source_transition_count_requested']:,} |
| Global/outer/inner capacities | {sac['buffer_size']:,}/{replay['outer_buffer_size']:,}/{replay['inner_buffer_size']:,} |
| Critic-only warm-up | {config['critic_warmup_steps']:,} interactions |
| Learning rate | {sac['learning_rate']:.2g} |
| Effective actor learning rate | {sac['learning_rate'] * config['actor_learning_rate_scale']:.2g} |
| Actor zero-reference weight | {actor_weight_description} |
| Train frequency / gradient steps | {sac['train_freq']} / {sac['gradient_steps']} |
| Batch size | {sac['batch_size']} |
| Discount factor | {sac['gamma']:.4f} |
| Target smoothing | {sac['tau']:.4f} |
| Entropy coefficient | `{sac['ent_coef']}` |
| Target entropy | {sac['target_entropy']:.1f} |

## Validation results

Validation seeds: {min(config['validation_seeds'])}--{max(config['validation_seeds'])}.

| Checkpoint | Success | Hard failure | Timeout | Mean minimum position error [m] | Mean best worst ratio | Mean seconds |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Interpretation

{interpretation}

## Artifacts

- `checkpoint_0_model.zip`: exact zero-residual LQR baseline;
- `checkpoint_<N>_model.zip`: evaluated SAC checkpoints;
- `best_model.zip`: validation-ranked checkpoint;
- `final_model.zip`: end-of-training policy.

Configuration, rollout tables, and `replay_statistics.json` are in
`../../../results/lqr_reference_residual_stratified/{config['experiment_name']}/`.
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
    replay_kwargs = {
        "outer_buffer_size": args.outer_replay_size,
        "inner_buffer_size": args.inner_replay_size,
        "global_batch_fraction": args.global_batch_fraction,
        "outer_batch_fraction": args.outer_batch_fraction,
        "inner_batch_fraction": args.inner_batch_fraction,
        "outer_ratio_limit": args.outer_ratio_limit,
        "inner_ratio_limit": args.inner_ratio_limit,
    }
    config = {
        "experiment_name": args.experiment_name,
        "pilot": args.pilot_label,
        "method": (
            "Pilot D with annealed zero-residual actor constraint"
            if args.zero_residual_actor_final_weight is not None
            else "Pilot C with P3-stratified fixed-goal replay"
        ),
        "reward_version": REWARD_VERSION,
        "reward_scheme": ReferenceResidualLQRGoalConditionedClosingEnv.reward_scheme,
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
        "replay_distribution": {
            "semantics": (
                "global includes all transitions; ratio<=2 is nested inside "
                "the ratio<=5 targeted partition"
            ),
            "global_fraction": args.global_batch_fraction,
            "outer_fraction": args.outer_batch_fraction,
            "inner_fraction": args.inner_batch_fraction,
            "outer_ratio_band": f"ratio <= {args.outer_ratio_limit:g}",
            "inner_ratio_band": f"ratio <= {args.inner_ratio_limit:g}",
            "target_partitions_are_nested": True,
            "outer_ratio_limit": args.outer_ratio_limit,
            "inner_ratio_limit": args.inner_ratio_limit,
            "outer_buffer_size": args.outer_replay_size,
            "inner_buffer_size": args.inner_replay_size,
            "missing_partition_fallback": "uniform global replay",
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
        "actor_constraint": {
            "schedule_used": (
                args.zero_residual_actor_final_weight is not None
            ),
            "initial_weight": args.zero_residual_actor_weight,
            "final_weight": (
                args.zero_residual_actor_weight
                if args.zero_residual_actor_final_weight is None
                else args.zero_residual_actor_final_weight
            ),
            "anneal_start_step": args.critic_warmup_steps,
            "anneal_end_step": (
                None
                if args.zero_residual_actor_anneal_end_step is None
                else args.zero_residual_actor_anneal_end_step
            ),
            "form": (
                "constant"
                if args.zero_residual_actor_final_weight is None
                else "linear in absolute online interactions, then constant"
            ),
        },
        "actor_learning_rate_scale": args.actor_learning_rate_scale,
        "sac": {
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "replay_buffer": "P3StratifiedDictReplayBuffer",
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
        replay_buffer_class=P3StratifiedDictReplayBuffer,
        replay_buffer_kwargs=replay_kwargs,
        critic_warmup_steps=args.critic_warmup_steps,
        zero_residual_actor_weight=args.zero_residual_actor_weight,
        zero_residual_actor_final_weight=(
            args.zero_residual_actor_final_weight
        ),
        zero_residual_actor_anneal_end_step=(
            args.zero_residual_actor_anneal_end_step
        ),
        actor_learning_rate_scale=args.actor_learning_rate_scale,
        verbose=1,
        seed=args.training_seed,
        tensorboard_log=str(log_dir),
        device=args.device,
    )
    initialize_zero_residual_actor(model, args.initial_log_std)
    prefill_metrics = prefill_standard_replay_buffer(model, transitions)
    prefill_replay_statistics = model.replay_buffer.replay_statistics()
    write_json(
        result_dir / "replay_statistics_after_prefill.json",
        prefill_replay_statistics,
    )

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
        {
            **prefill_metrics,
            "replay_statistics_after_prefill": prefill_replay_statistics,
            "zero_residual_validation": initial_summary,
        },
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
    print("Replay batches:   50% global / 25% ratio <=5 / 25% ratio <=2")
    if args.zero_residual_actor_final_weight is not None:
        print(
            "Actor prior:      "
            f"{args.zero_residual_actor_weight:g} at "
            f"{args.critic_warmup_steps:,} -> "
            f"{args.zero_residual_actor_final_weight:g} at "
            f"{args.zero_residual_actor_anneal_end_step:,}"
        )
    print(f"Online steps:     {args.total_timesteps:,}")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        tb_log_name=args.experiment_name,
        progress_bar=args.progress_bar,
    )
    model.save(model_dir / "final_model")
    replay_statistics = model.replay_buffer.replay_statistics()
    write_json(result_dir / "replay_statistics.json", replay_statistics)
    config.update(
        {
            "actual_timesteps": int(model.num_timesteps),
            "best_checkpoint": int(callback.best_checkpoint),
            "source_prefill": prefill_metrics,
            "prefill_replay_statistics": prefill_replay_statistics,
            "final_replay_statistics": replay_statistics,
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
