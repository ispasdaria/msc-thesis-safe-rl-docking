"""Pilot F: empirical terminal-reset SAC reference residual around closing LQR."""

from __future__ import annotations

import json
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
from terminal_reference_residual_env import (
    EmpiricalTerminalReferenceResidualWrapper,
    load_terminal_reset_dataset,
)
from train_sac_closing_her import HerEvaluationCheckpointCallback
from train_sac_closing_her_reference_residual_lqr import (
    collect_reference_residual_trajectories,
)
from train_sac_closing_her_v4_residual_v15 import (
    initialize_zero_residual_actor,
)
from train_sac_closing_reference_residual_lqr_no_her import (
    environment_kwargs,
    prefill_standard_replay_buffer,
)
from train_sac_closing_reference_residual_lqr_stratified import (
    build_parser as build_pilot_d_parser,
    validate_args as validate_pilot_d_args,
)
from zero_residual_constrained_sac import ZeroResidualConstrainedSAC


CLOSING_DIR = Path(__file__).resolve().parent
MODEL_ROOT = CLOSING_DIR / "models" / "lqr_reference_residual_terminal"
RESULT_ROOT = CLOSING_DIR / "results" / "lqr_reference_residual_terminal"
LOG_ROOT = CLOSING_DIR / "logs" / "lqr_reference_residual_terminal"
DEFAULT_DATASET = (
    CLOSING_DIR / "demonstrations" / "terminal_reference_residual_training_bank.npz"
)
DEFAULT_ORACLE_DIR = RESULT_ROOT / "oracle_study_n20_h1"
REWARD_VERSION = "lqr_reference_residual"


def build_parser():
    parser = build_pilot_d_parser()
    parser.description = (
        "Pilot F: fixed empirical terminal resets for SAC reference residuals."
    )
    parser.set_defaults(
        experiment_name=(
            "sac_closing_lqr_reference_residual_"
            "pilot_f_terminal_r2_h600_1hz_50k_seed0"
        ),
        pilot_label="D",
    )
    parser.add_argument("--terminal-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--terminal-minimum-ratio", type=float, default=1.05)
    parser.add_argument("--terminal-maximum-ratio", type=float, default=2.0)
    parser.add_argument("--terminal-max-time", type=float, default=600.0)
    parser.add_argument("--oracle-study-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    return parser


def validate_args(parser, args):
    validate_pilot_d_args(parser, args)
    if args.zero_residual_actor_final_weight is not None:
        parser.error(
            "Pilot F keeps Pilot D's constant actor constraint; do not pass "
            "the Pilot E annealing arguments."
        )
    if not 1.0 < args.terminal_minimum_ratio < args.terminal_maximum_ratio:
        parser.error("terminal ratios must satisfy 1 < minimum < maximum.")
    if args.terminal_max_time <= 0.0:
        parser.error("--terminal-max-time must be positive.")
    if args.terminal_max_time >= args.max_time:
        parser.error("terminal max time must be shorter than full closing max time.")


def _dataset_audit(dataset, args):
    ratios = dataset["max_target_ratios"]
    outcomes = dataset["source_outcomes"].astype(str)
    eligible = (
        (ratios >= args.terminal_minimum_ratio)
        & (ratios <= args.terminal_maximum_ratio)
    )
    eligible_outcomes = outcomes[eligible]
    eligible_seeds = dataset["source_seeds"][eligible]
    return {
        "path": str(args.terminal_dataset.resolve()),
        "total_snapshots": int(len(ratios)),
        "eligible_snapshots": int(eligible.sum()),
        "eligible_source_episodes": int(len(set(eligible_seeds.tolist()))),
        "eligible_source_success_snapshots": int(
            (eligible_outcomes == "success").sum()
        ),
        "eligible_source_failure_snapshots": int(
            (eligible_outcomes == "failure").sum()
        ),
        "eligible_source_timeout_snapshots": int(
            (eligible_outcomes == "time_limit").sum()
        ),
    }


def _load_oracle_audit(path):
    summary_path = Path(path) / "summary.json"
    config_path = Path(path) / "config.json"
    if not summary_path.is_file() or not config_path.is_file():
        return {"available": False, "path": str(Path(path).resolve())}
    return {
        "available": True,
        "path": str(Path(path).resolve()),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "config": json.loads(config_path.read_text(encoding="utf-8")),
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
    status = "complete" if complete else "in progress"
    if complete and int(best_checkpoint) == 0:
        interpretation = (
            "No trained checkpoint outranked exact zero-residual LQR on the "
            "full-closing validation seeds; `best_model.zip` remains baseline."
        )
    elif complete:
        interpretation = (
            f"Checkpoint {int(best_checkpoint):,} ranked best on full-closing "
            "validation and is stored as `best_model.zip`."
        )
    else:
        interpretation = "Training is in progress; selection is provisional."
    terminal = config["terminal_training_distribution"]
    audit = terminal["dataset_audit"]
    oracle = config["oracle_controllability_study"]
    if oracle["available"]:
        oracle_summary = oracle["summary"]
        oracle_text = (
            f"The pre-training paired diagnostic used "
            f"{oracle_summary['state_count']} states. Zero residual succeeded "
            f"on {oracle_summary['baseline_successes']}; the finite-grid "
            f"one-step search succeeded on {oracle_summary['oracle_successes']}, "
            f"with {oracle_summary['oracle_rescues']} rescues and "
            f"{oracle_summary['oracle_losses']} losses. This demonstrates some "
            "action authority but does not claim that the myopic oracle is a "
            "better controller."
        )
    else:
        oracle_text = "No oracle summary was available when the run began."
    reward = config["reward"]
    sac = config["sac"]
    bounds = config["reference_residual_bounds"]
    content = f"""# Pilot F terminal-reset reference-residual SAC model card

Run: `{config['experiment_name']}`  
Status: **{status}**  
Selected best checkpoint: **{int(best_checkpoint):,} SAC decisions**

## Experimental change

Pilot F changes Pilot D's **online episode start distribution**. Training
episodes begin at empirical states recorded from ordinary zero-residual LQR
closing trajectories with worst P3 tolerance ratio in
`[{terminal['minimum_ratio']:g}, {terminal['maximum_ratio']:g}]`. Their horizon
is {terminal['max_time_seconds']:g} seconds. This is a fixed terminal
distribution, not a curriculum.

The bank contains {audit['eligible_snapshots']:,} eligible snapshots from
{audit['eligible_source_episodes']:,} source episodes. Snapshot provenance is
success/failure/timeout = {audit['eligible_source_success_snapshots']:,}/
{audit['eligible_source_failure_snapshots']:,}/
{audit['eligible_source_timeout_snapshots']:,}.

Full LQR trajectories still provide the {config['source_transition_count_requested']:,}
transition replay prefill. Replay remains 50% global, 25% ratio <= 5, and 25%
ratio <= 2. Model selection and all reported checkpoint results below use the
original full closing reset and {config['max_eval_steps']:,}-second horizon.

## Oracle controllability diagnostic

{oracle_text}

## Controller and exclusions

```text
delta_reference = [
    {bounds['vx_mps']:.3f} * a[0] m/s,
    {bounds['vy_mps']:.3f} * a[1] m/s,
    {bounds['attitude_deg']:.3f} * a[2] deg,
]
u = LQR(state, scheduled_P3_reference + delta_reference)
```

SAC and LQR update every second. Zero SAC action exactly reproduces LQR. There
is no action hold, distance gate, HER, CBF, safety-filter signal, or curriculum.
The actor is initialized at zero and retains Pilot D's constant zero-residual
penalty.

## Reward (unchanged)

```text
reward = {reward['progress_weight']:.3f} * (J(previous) - gamma * J(next))
         + {reward['success_bonus']:.3f} * exact_P3_success
         - {reward['failure_penalty']:.3f} * hard_failure
         - {reward['timeout_penalty']:.3f} * unsuccessful_timeout
         - {reward['magnitude_weight']:.3f} * ||a||^2
         - {reward['smoothness_weight']:.3f} * ||a-a_previous||^2
```

`J` equally combines the mean and maximum of the six log-transformed P3
tolerance ratios.

## SAC configuration

| Parameter | Value |
|---|---:|
| Online interactions | {config['total_timesteps']:,} |
| Full-closing replay prefill | {config['source_transition_count_requested']:,} |
| Critic-only warm-up | {config['critic_warmup_steps']:,} |
| Learning rate | {sac['learning_rate']:.2g} |
| Effective actor learning rate | {sac['learning_rate'] * config['actor_learning_rate_scale']:.2g} |
| Zero-residual actor weight | {config['zero_residual_actor_weight']:.3g} |
| Batch size | {sac['batch_size']} |
| Train frequency / gradient steps | {sac['train_freq']} / {sac['gradient_steps']} |
| Discount factor | {sac['gamma']:.4f} |
| Target smoothing | {sac['tau']:.4f} |
| Entropy coefficient | `{sac['ent_coef']}` |
| Target entropy | {sac['target_entropy']:.1f} |

## Full-closing validation

Validation seeds: {min(config['validation_seeds'])}--{max(config['validation_seeds'])}.

| Checkpoint | Success | Hard failure | Timeout | Mean minimum position error [m] | Mean best worst ratio | Mean seconds |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Interpretation

{interpretation}

## Artifacts

- `checkpoint_0_model.zip`: exact zero-residual LQR baseline;
- `checkpoint_<N>_model.zip`: trained checkpoints;
- `best_model.zip`: validation-selected controller;
- `final_model.zip`: final trained controller.

Machine-readable configuration and rollout tables are in
`../../../results/lqr_reference_residual_terminal/{config['experiment_name']}/`.
"""
    (Path(model_dir) / "README.md").write_text(content, encoding="utf-8")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    dataset = load_terminal_reset_dataset(args.terminal_dataset)
    dataset_audit = _dataset_audit(dataset, args)
    if dataset_audit["eligible_snapshots"] == 0:
        parser.error("terminal dataset has no eligible snapshots.")
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
        "pilot": "F",
        "method": "fixed empirical terminal resets with Pilot D SAC/replay",
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
        "curriculum_used": False,
        "terminal_training_distribution": {
            "type": "fixed empirical LQR trajectory snapshots",
            "minimum_ratio": args.terminal_minimum_ratio,
            "maximum_ratio": args.terminal_maximum_ratio,
            "max_time_seconds": args.terminal_max_time,
            "dataset_audit": dataset_audit,
        },
        "oracle_controllability_study": _load_oracle_audit(args.oracle_study_dir),
        "reward": {
            "progress_weight": args.progress_weight,
            "progress_form": "J(previous) - gamma * J(next)",
            "success_bonus": args.success_bonus,
            "failure_penalty": args.failure_penalty,
            "timeout_penalty": args.timeout_penalty,
            "magnitude_weight": args.reference_magnitude_weight,
            "smoothness_weight": args.reference_smoothness_weight,
        },
        "replay_distribution": {
            "global_fraction": args.global_batch_fraction,
            "outer_fraction": args.outer_batch_fraction,
            "inner_fraction": args.inner_batch_fraction,
            "outer_ratio_limit": args.outer_ratio_limit,
            "inner_ratio_limit": args.inner_ratio_limit,
            "outer_buffer_size": args.outer_replay_size,
            "inner_buffer_size": args.inner_replay_size,
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
        "critic_warmup_steps": args.critic_warmup_steps,
        "zero_residual_actor_weight": args.zero_residual_actor_weight,
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

    def make_training_env():
        base = make_her_env(
            seed=args.training_seed,
            mode="training",
            max_time=args.terminal_max_time,
            reward_version=REWARD_VERSION,
            **env_kwargs,
        )
        return Monitor(
            EmpiricalTerminalReferenceResidualWrapper(
                base,
                args.terminal_dataset,
                minimum_max_target_ratio=args.terminal_minimum_ratio,
                maximum_max_target_ratio=args.terminal_maximum_ratio,
            )
        )

    training_env = DummyVecEnv([make_training_env])
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
        actor_learning_rate_scale=args.actor_learning_rate_scale,
        verbose=1,
        seed=args.training_seed,
        tensorboard_log=str(log_dir),
        device=args.device,
    )
    initialize_zero_residual_actor(model, args.initial_log_std)
    prefill_metrics = prefill_standard_replay_buffer(model, transitions)
    prefill_statistics = model.replay_buffer.replay_statistics()
    write_json(result_dir / "replay_statistics_after_prefill.json", prefill_statistics)

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
            "replay_statistics_after_prefill": prefill_statistics,
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
    print(f"Experiment:          {args.experiment_name}")
    print("Training reset:      empirical terminal LQR states")
    print(
        f"Training ratio band: {args.terminal_minimum_ratio:g}--"
        f"{args.terminal_maximum_ratio:g}"
    )
    print(f"Training horizon:    {args.terminal_max_time:g} s")
    print("Validation reset:    ordinary full closing")
    print("HER / CBF:           no / no")
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
            "prefill_replay_statistics": prefill_statistics,
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
