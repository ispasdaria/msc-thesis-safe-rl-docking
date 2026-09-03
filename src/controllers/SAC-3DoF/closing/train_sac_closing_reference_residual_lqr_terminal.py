"""Pilot F: empirical terminal-reset SAC reference residual around closing LQR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
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
from p3_stratified_dict_replay_buffer import P3StratifiedDictReplayBuffer
from terminal_reference_residual_env import (
    EmpiricalTerminalReferenceResidualWrapper,
    LateralTerminalCurriculumWrapper,
    load_terminal_reset_dataset,
    validate_terminal_reset_provenance,
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
    prepare_output_directories,
    validate_args as validate_pilot_d_args,
)
from zero_residual_constrained_sac import ZeroResidualConstrainedSAC
from validate_residual_lqr_dt01_baseline import (
    DEFAULT_OUTPUT_DIR as DEFAULT_BASELINE_DIR,
    validate_saved_baseline_manifest,
)


CLOSING_DIR = Path(__file__).resolve().parent
MODEL_ROOT = CLOSING_DIR / "models" / "lqr_reference_residual_terminal"
RESULT_ROOT = CLOSING_DIR / "results" / "lqr_reference_residual_terminal"
LOG_ROOT = CLOSING_DIR / "logs" / "lqr_reference_residual_terminal"
DEFAULT_DATASET = (
    CLOSING_DIR
    / "demonstrations"
    / "terminal_reference_residual_final_lqr_dt0p1_bank.npz"
)
DEFAULT_BASELINE_MANIFEST = DEFAULT_BASELINE_DIR / "baseline_manifest.json"
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
            "dt0p1_terminal_r2_h600_50k_seed0"
        ),
        pilot_label="D",
    )
    parser.add_argument("--terminal-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--baseline-manifest", type=Path, default=DEFAULT_BASELINE_MANIFEST
    )
    parser.add_argument("--terminal-minimum-ratio", type=float, default=1.05)
    parser.add_argument("--terminal-maximum-ratio", type=float, default=2.0)
    parser.add_argument("--terminal-max-time", type=float, default=600.0)
    parser.add_argument("--oracle-study-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument(
        "--lateral-terminal-curriculum",
        action="store_true",
        help=(
            "Restrict resets to lateral-limited states and expand the "
            "maximum lateral ratio through an easy-to-hard curriculum."
        ),
    )
    parser.add_argument(
        "--curriculum-stage-maximum-ratios",
        type=float,
        nargs="+",
        default=(1.25, 1.5, 2.0),
    )
    parser.add_argument(
        "--curriculum-stage-boundaries",
        type=float,
        nargs="*",
        default=(0.25, 0.60),
    )
    parser.add_argument(
        "--terminal-prefill",
        action="store_true",
        help="Collect replay prefill from the terminal reset distribution.",
    )
    parser.add_argument(
        "--balance-terminal-source-outcomes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample success- and timeout-derived terminal resets equally.",
    )
    parser.add_argument(
        "--acceptance-seeds",
        type=int,
        nargs="+",
        default=list(range(300, 310)),
        help=(
            "Independent full-closing seeds used only for the final "
            "candidate-versus-LQR acceptance gate."
        ),
    )
    parser.add_argument(
        "--terminal-residual-fade-full-ratio",
        type=float,
        default=None,
        help=(
            "Apply full SAC residual authority at or above this worst P3 "
            "ratio and fade linearly below it. Disabled when omitted."
        ),
    )
    parser.add_argument(
        "--terminal-residual-fade-zero-ratio",
        type=float,
        default=1.0,
        help="Force zero SAC residual authority at or below this P3 ratio.",
    )
    return parser


def validate_args(parser, args):
    validate_pilot_d_args(parser, args)
    if not 1.0 < args.terminal_minimum_ratio < args.terminal_maximum_ratio:
        parser.error("terminal ratios must satisfy 1 < minimum < maximum.")
    if args.terminal_max_time <= 0.0:
        parser.error("--terminal-max-time must be positive.")
    if not args.acceptance_seeds:
        parser.error("--acceptance-seeds must not be empty.")
    if set(args.validation_seeds) & set(args.acceptance_seeds):
        parser.error("validation and acceptance seeds must be disjoint.")
    if args.terminal_max_time >= args.max_time:
        parser.error("terminal max time must be shorter than full closing max time.")
    if args.terminal_residual_fade_zero_ratio < 0.0:
        parser.error("terminal residual fade zero ratio must be non-negative.")
    if (
        args.terminal_residual_fade_full_ratio is not None
        and args.terminal_residual_fade_full_ratio
        <= args.terminal_residual_fade_zero_ratio
    ):
        parser.error(
            "terminal residual fade full ratio must be above its zero ratio."
        )
    if args.lateral_terminal_curriculum:
        maxima = args.curriculum_stage_maximum_ratios
        boundaries = args.curriculum_stage_boundaries
        if len(boundaries) != len(maxima) - 1:
            parser.error(
                "curriculum boundaries must contain one fewer value than "
                "stage maximum ratios."
            )
        if not maxima or not (
            1.0 < args.terminal_minimum_ratio < maxima[0]
        ):
            parser.error(
                "terminal minimum ratio must be above one and below the "
                "first curriculum maximum."
            )
        if any(b <= a for a, b in zip(maxima, maxima[1:])):
            parser.error("curriculum stage maximum ratios must increase.")
        if abs(maxima[-1] - args.terminal_maximum_ratio) > 1e-12:
            parser.error(
                "the final curriculum maximum must equal terminal maximum ratio."
            )
        if any(not 0.0 < value < 1.0 for value in boundaries) or any(
            b <= a for a, b in zip(boundaries, boundaries[1:])
        ):
            parser.error("curriculum boundaries must increase within (0, 1).")


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


def make_terminal_training_env(args, env_kwargs, *, seed, total_steps):
    """Build the fixed terminal reset or lateral curriculum environment."""
    base = make_her_env(
        seed=seed,
        mode="training",
        max_time=args.terminal_max_time,
        reward_version=REWARD_VERSION,
        **env_kwargs,
    )
    if args.lateral_terminal_curriculum:
        return LateralTerminalCurriculumWrapper(
            base,
            args.terminal_dataset,
            minimum_lateral_ratio=args.terminal_minimum_ratio,
            stage_maximum_ratios=args.curriculum_stage_maximum_ratios,
            stage_boundaries=args.curriculum_stage_boundaries,
            total_curriculum_steps=total_steps,
            balance_source_outcomes=args.balance_terminal_source_outcomes,
        )
    return EmpiricalTerminalReferenceResidualWrapper(
        base,
        args.terminal_dataset,
        minimum_max_target_ratio=args.terminal_minimum_ratio,
        maximum_max_target_ratio=args.terminal_maximum_ratio,
        include_source_outcomes=("success", "time_limit"),
        balance_source_outcomes=args.balance_terminal_source_outcomes,
    )


def _acceptance_gate(
    *,
    model_dir,
    result_dir,
    training_env,
    candidate_checkpoint,
    acceptance_seeds,
    max_eval_steps,
    env_kwargs,
    device,
):
    """Select a trained policy only after an independent safety-aware gate."""
    baseline = ZeroResidualConstrainedSAC.load(
        model_dir / "checkpoint_0_model.zip", env=training_env, device=device
    )
    candidate = ZeroResidualConstrainedSAC.load(
        model_dir / "best_model.zip", env=training_env, device=device
    )
    baseline_rows = evaluate_model(
        baseline,
        acceptance_seeds,
        max_steps=max_eval_steps,
        reward_version=REWARD_VERSION,
        env_kwargs=env_kwargs,
    )
    candidate_rows = evaluate_model(
        candidate,
        acceptance_seeds,
        max_steps=max_eval_steps,
        reward_version=REWARD_VERSION,
        env_kwargs=env_kwargs,
    )
    baseline_summary = summarize_rollouts(baseline_rows)
    candidate_summary = summarize_rollouts(candidate_rows)
    accepted = trained_checkpoint_passes_acceptance(
        candidate_checkpoint=candidate_checkpoint,
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
    )
    selected_checkpoint = int(candidate_checkpoint) if accepted else 0
    selected_source = (
        model_dir / "best_model.zip"
        if accepted
        else model_dir / "checkpoint_0_model.zip"
    )
    shutil.copy2(selected_source, model_dir / "accepted_model.zip")
    write_csv(
        result_dir / "acceptance_rollout_metrics.csv",
        [
            {"controller": "lqr_zero_residual", **row}
            for row in baseline_rows
        ]
        + [
            {"controller": "trained_candidate", **row}
            for row in candidate_rows
        ],
    )
    decision = {
        "accepted_trained_checkpoint": accepted,
        "selected_checkpoint": selected_checkpoint,
        "candidate_checkpoint": int(candidate_checkpoint),
        "acceptance_seeds": list(map(int, acceptance_seeds)),
        "rule": (
            "strictly higher success rate and no higher hard-failure rate "
            "than zero-residual final LQR"
        ),
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "selected_model": str((model_dir / "accepted_model.zip").resolve()),
    }
    write_json(result_dir / "acceptance_decision.json", decision)
    return decision


def trained_checkpoint_passes_acceptance(
    *, candidate_checkpoint, baseline_summary, candidate_summary
):
    """Return the deployment decision for an independently tested candidate."""
    return bool(
        int(candidate_checkpoint) > 0
        and float(candidate_summary["success_rate"])
        > float(baseline_summary["success_rate"])
        and float(candidate_summary["failure_rate"])
        <= float(baseline_summary["failure_rate"])
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
    acceptance = config.get("acceptance")
    if acceptance is None:
        acceptance_text = "Independent acceptance has not run yet."
    elif acceptance["accepted_trained_checkpoint"]:
        acceptance_text = (
            f"The trained checkpoint {acceptance['candidate_checkpoint']:,} "
            "strictly beat the zero-residual LQR success rate without a "
            "higher hard-failure rate. `accepted_model.zip` is the trained "
            "candidate."
        )
    else:
        acceptance_text = (
            "The trained candidate did not pass the independent success and "
            "safety gate. `accepted_model.zip` is therefore checkpoint 0, the "
            "exact final LQR baseline."
        )
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
    residual_fade = config.get("terminal_residual_fade", {"enabled": False})
    if residual_fade.get("enabled", False):
        experiment_text = f"""Pilot {config['pilot']} uses a terminal
residual-authority fade based on the worst simultaneous P3 tolerance ratio:
full authority at ratio >=
{residual_fade['full_authority_at_or_above_ratio']:g}, linear fade through the
terminal band, and exact zero residual at ratio <=
{residual_fade['zero_authority_at_or_below_ratio']:g}. The established LQR
therefore performs the final tolerance crossing."""
        exclusion_text = (
            "There is no action hold, distance gate, HER, CBF, or safety-filter "
            "signal. Full-closing validation uses the same residual fade."
        )
    elif config.get("curriculum_used", False):
        experiment_text = f"""Pilot {config['pilot']} uses an empirical replay
prefill and online start-state distribution. Every reset is a state
where lateral position is the worst P3 component. The maximum reset ratio
expands through {terminal['stage_maximum_ratios']} at training fractions
{terminal['stage_boundaries']}. Episodes last at most
{terminal['max_time_seconds']:g} physical seconds."""
        exclusion_text = (
            "There is no action hold, distance gate, HER, CBF, or safety-filter "
            "signal. Full-closing validation remains unchanged."
        )
    else:
        if config["pilot"] == "J":
            experiment_text = f"""Pilot J is the provenance-locked terminal
follow-up to Pilot G. Training episodes begin at balanced success/timeout
states recorded only from the fingerprinted final dt=0.1 LQR, with worst P3
tolerance ratio in `[{terminal['minimum_ratio']:g},
{terminal['maximum_ratio']:g}]`. Their horizon is
{terminal['max_time_seconds']:g} seconds. This is a fixed terminal
distribution, not a curriculum."""
        else:
            experiment_text = f"""Pilot {config['pilot']} uses a fixed empirical
terminal start distribution. Training episodes begin at states recorded from
ordinary zero-residual LQR closing trajectories with worst P3 tolerance ratio
in `[{terminal['minimum_ratio']:g}, {terminal['maximum_ratio']:g}]`. Their
horizon is {terminal['max_time_seconds']:g} seconds."""
        exclusion_text = (
            "There is no action hold, distance gate, HER, CBF, safety-filter "
            "signal, or curriculum."
        )
    if reward.get("design") == "dt01_lateral_v2":
        reward_text = f"""reward = {reward['progress_weight']:.3f} * weighted_P3_progress
         + {reward['lateral_constraint_progress_weight']:.3f} * lateral_violation_progress
         + {reward['success_bonus']:.3f} * exact_P3_success
         - {reward['failure_penalty']:.3f} * hard_failure
         - {reward['timeout_penalty']:.3f} * severity_scaled_timeout
         - dt * {reward['magnitude_weight']:.3f} * ||a||^2
         - (1 / dt) * {reward['smoothness_weight']:.3f} * ||a-a_previous||^2"""
    else:
        reward_text = f"""reward = {reward['progress_weight']:.3f} * (J(previous) - gamma * J(next))
         + {reward['success_bonus']:.3f} * exact_P3_success
         - {reward['failure_penalty']:.3f} * hard_failure
         - {reward['timeout_penalty']:.3f} * unsuccessful_timeout
         - {reward['magnitude_weight']:.3f} * ||a||^2
         - {reward['smoothness_weight']:.3f} * ||a-a_previous||^2"""
    fade = config.get("terminal_residual_fade", {"enabled": False})
    if fade.get("enabled", False):
        residual_equation = f"""g(r) = clip((r - {fade['zero_authority_at_or_below_ratio']:.3f}) /
            ({fade['full_authority_at_or_above_ratio']:.3f} - {fade['zero_authority_at_or_below_ratio']:.3f}), 0, 1)
delta_reference = g(max_P3_ratio) * [
    {bounds['vx_mps']:.3f} * a[0] m/s,
    {bounds['vy_mps']:.3f} * a[1] m/s,
    {bounds['attitude_deg']:.3f} * a[2] deg,
]"""
    else:
        residual_equation = f"""delta_reference = [
    {bounds['vx_mps']:.3f} * a[0] m/s,
    {bounds['vy_mps']:.3f} * a[1] m/s,
    {bounds['attitude_deg']:.3f} * a[2] deg,
]"""
    content = f"""# Pilot {config['pilot']} terminal-reset reference-residual SAC model card

Run: `{config['experiment_name']}`  
Status: **{status}**  
Selected best checkpoint: **{int(best_checkpoint):,} SAC decisions**

## Experimental change

{experiment_text}

The bank contains {audit['eligible_snapshots']:,} eligible snapshots from
{audit['eligible_source_episodes']:,} source episodes. Snapshot provenance is
success/failure/timeout = {audit['eligible_source_success_snapshots']:,}/
{audit['eligible_source_failure_snapshots']:,}/
{audit['eligible_source_timeout_snapshots']:,}.

The {config['source_transition_count_requested']:,}-transition replay prefill
uses `{config['source_prefill_distribution']}`. Replay remains 50% global,
25% ratio <= 5, and 25% ratio <= 2. Model selection and all reported
checkpoint results below use the original full-closing reset.

## Oracle controllability diagnostic

{oracle_text}

## Controller and exclusions

```text
{residual_equation}
u = LQR(state, scheduled_P3_reference + delta_reference)
```

SAC and LQR update every {config['dt_seconds']:g} seconds. Zero SAC action
exactly reproduces LQR. {exclusion_text}

## Reward

```text
{reward_text}
```

`J` equally combines the mean and maximum of the six log-transformed P3
tolerance ratios.

## SAC configuration

| Parameter | Value |
|---|---:|
| Online interactions | {config['total_timesteps']:,} |
| Replay prefill transitions | {config['source_transition_count_requested']:,} |
| Critic-only warm-up | {config['critic_warmup_steps']:,} |
| Learning rate | {sac['learning_rate']:.2g} |
| Effective actor learning rate | {sac['learning_rate'] * config['actor_learning_rate_scale']:.2g} |
| Zero-residual actor weight | {config['zero_residual_actor_weight']:.3g} |
| Batch size | {sac['batch_size']} |
| Train frequency / gradient steps | {sac['train_freq']} / {sac['gradient_steps']} |
| Discount factor | {sac['gamma']:.8f} |
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

## Independent acceptance

{acceptance_text}

## Artifacts

- `checkpoint_0_model.zip`: exact zero-residual LQR baseline;
- `checkpoint_<N>_model.zip`: trained checkpoints;
- `best_model.zip`: validation-selected controller;
- `accepted_model.zip`: controller selected by the independent acceptance gate;
- `final_model.zip`: final trained controller.

Machine-readable configuration and rollout tables are in
`../../../results/{config['artifact_family']}/{config['experiment_name']}/`.
"""
    (Path(model_dir) / "README.md").write_text(content, encoding="utf-8")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    baseline_manifest = validate_saved_baseline_manifest(args.baseline_manifest)
    dataset = load_terminal_reset_dataset(args.terminal_dataset)
    provenance_audit = validate_terminal_reset_provenance(
        dataset,
        dataset_path=args.terminal_dataset,
        baseline_manifest=baseline_manifest,
        baseline_manifest_path=args.baseline_manifest,
    )
    source_seeds = set(map(int, np.asarray(dataset["source_seeds"])))
    evaluation_seeds = set(map(int, args.validation_seeds)) | set(
        map(int, args.acceptance_seeds)
    )
    overlap = sorted(source_seeds & evaluation_seeds)
    if overlap:
        parser.error(
            "terminal-bank source seeds overlap validation/acceptance seeds: "
            f"{overlap}"
        )
    dataset_audit = _dataset_audit(dataset, args)
    dataset_audit["provenance"] = provenance_audit
    if dataset_audit["eligible_snapshots"] == 0:
        parser.error("terminal dataset has no eligible snapshots.")
    checkpoints = sorted(
        {step for step in args.checkpoint_steps if 0 < step <= args.total_timesteps}
    )
    if args.total_timesteps not in checkpoints:
        checkpoints.append(args.total_timesteps)
    env_kwargs = environment_kwargs(args)
    env_kwargs.update(
        {
            "terminal_residual_fade_full_ratio": (
                args.terminal_residual_fade_full_ratio
            ),
            "terminal_residual_fade_zero_ratio": (
                args.terminal_residual_fade_zero_ratio
            ),
        }
    )
    if args.lateral_terminal_curriculum:
        audit_env = make_terminal_training_env(
            args,
            env_kwargs,
            seed=args.training_seed,
            total_steps=args.total_timesteps,
        )
        try:
            dataset_audit.update(
                {
                    "lateral_limited_stage_snapshot_counts": [
                        int(len(indices))
                        for indices in audit_env.stage_eligible_indices
                    ],
                    "lateral_limited_source_episodes": int(
                        len(
                            np.unique(
                                audit_env.dataset["source_seeds"][
                                    audit_env.eligible_indices
                                ]
                            )
                        )
                    ),
                }
            )
        finally:
            audit_env.close()
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
        "artifact_family": MODEL_ROOT.name,
        "pilot": args.pilot_label,
        "method": (
            "10 Hz lateral-limited near-P3 reset curriculum"
            if args.lateral_terminal_curriculum
            else (
                "provenance-locked final-LQR terminal resets with Pilot G SAC/replay"
                if args.pilot_label == "J"
                else "fixed empirical terminal resets"
            )
        ),
        "reward_version": REWARD_VERSION,
        "reward_scheme": (
            ReferenceResidualLQRGoalConditionedClosingEnv.reward_scheme
            if args.reward_design == "v1"
            else (
                "goal_conditioned_p3_reference_residual_lqr_"
                "dt01_lateral_v2"
            )
        ),
        "reference_residual_components": ["vx", "vy", "attitude"],
        "reference_residual_bounds": {
            "vx_mps": args.vx_reference_bound,
            "vy_mps": args.vy_reference_bound,
            "attitude_deg": args.attitude_reference_bound_deg,
        },
        "dt_seconds": args.dt,
        "control_frequency_hz": 1.0 / args.dt,
        "sac_decision_interval_seconds": args.dt,
        "lqr_update_interval_seconds": args.dt,
        "residual_hold_steps": 1,
        "distance_gate_used": False,
        "terminal_residual_fade": {
            "enabled": args.terminal_residual_fade_full_ratio is not None,
            "ratio_definition": "maximum of six physical P3 tolerance ratios",
            "interpolation": "linear",
            "full_authority_at_or_above_ratio": (
                args.terminal_residual_fade_full_ratio
            ),
            "zero_authority_at_or_below_ratio": (
                args.terminal_residual_fade_zero_ratio
            ),
        },
        "her_used": False,
        "actual_p3_success_terminal": True,
        "cbf_filter_used_during_training": False,
        "cbf_signal_in_observation_or_reward": False,
        "curriculum_used": bool(args.lateral_terminal_curriculum),
        "terminal_training_distribution": {
            "type": (
                "easy-to-hard lateral-limited empirical LQR snapshots"
                if args.lateral_terminal_curriculum
                else "fixed empirical LQR trajectory snapshots"
            ),
            "minimum_ratio": args.terminal_minimum_ratio,
            "maximum_ratio": args.terminal_maximum_ratio,
            "max_time_seconds": args.terminal_max_time,
            "stage_maximum_ratios": (
                list(args.curriculum_stage_maximum_ratios)
                if args.lateral_terminal_curriculum
                else None
            ),
            "stage_boundaries": (
                list(args.curriculum_stage_boundaries)
                if args.lateral_terminal_curriculum
                else None
            ),
            "limiting_component": (
                "lateral" if args.lateral_terminal_curriculum else "any"
            ),
            "dataset_audit": dataset_audit,
        },
        "oracle_controllability_study": _load_oracle_audit(args.oracle_study_dir),
        "reward": {
            "design": args.reward_design,
            "progress_weight": args.progress_weight,
            "progress_form": "J(previous) - gamma * J(next)",
            "goal_component_weights": list(args.goal_component_weights),
            "worst_component_weight": args.worst_component_weight,
            "lateral_constraint_progress_weight": (
                args.lateral_constraint_progress_weight
            ),
            "success_bonus": args.success_bonus,
            "failure_penalty": args.failure_penalty,
            "timeout_penalty": args.timeout_penalty,
            "timeout_violation_weight": args.timeout_violation_weight,
            "magnitude_weight": args.reference_magnitude_weight,
            "smoothness_weight": args.reference_smoothness_weight,
            "persistent_terms_time_scaled": bool(
                args.time_scale_persistent_reward
            ),
            "persistent_time_scale": (
                args.dt if args.time_scale_persistent_reward else 1.0
            ),
            "smoothness_is_action_rate_cost": bool(
                args.time_consistent_smoothness_reward
            ),
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
        "acceptance_seeds": args.acceptance_seeds,
        "controller_provenance": {
            "baseline_manifest": provenance_audit["baseline_manifest_path"],
            "baseline_manifest_sha256": provenance_audit[
                "baseline_manifest_sha256"
            ],
            "numeric_fingerprint_sha256": provenance_audit[
                "controller_numeric_fingerprint_sha256"
            ],
            "terminal_dataset_sha256": provenance_audit["dataset_sha256"],
            "lqr_actions_or_transitions_from_dt1_used": False,
        },
        "total_timesteps": args.total_timesteps,
        "checkpoint_steps": [0, *checkpoints],
        "max_time": args.max_time,
        "max_eval_steps": args.max_eval_steps,
        "source_transition_count_requested": args.source_transition_count,
        "source_seed_start": args.source_seed_start,
        "source_max_episodes": args.source_max_episodes,
        "source_prefill_distribution": (
            "terminal curriculum"
            if args.terminal_prefill
            else "ordinary full closing"
        ),
        "critic_warmup_steps": args.critic_warmup_steps,
        "zero_residual_actor_weight": args.zero_residual_actor_weight,
        "actor_constraint": {
            "schedule_used": args.zero_residual_actor_final_weight is not None,
            "initial_weight": args.zero_residual_actor_weight,
            "final_weight": (
                args.zero_residual_actor_weight
                if args.zero_residual_actor_final_weight is None
                else args.zero_residual_actor_final_weight
            ),
            "anneal_start_step": args.critic_warmup_steps,
            "anneal_end_step": args.zero_residual_actor_anneal_end_step,
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
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    model_dir = MODEL_ROOT / args.experiment_name
    result_dir = RESULT_ROOT / args.experiment_name
    log_dir = LOG_ROOT / args.experiment_name
    prepare_output_directories(
        (model_dir, result_dir, log_dir),
        (MODEL_ROOT, RESULT_ROOT, LOG_ROOT),
        overwrite=args.overwrite,
    )
    write_json(result_dir / "experiment_config.json", config)

    prefill_env_factory = None
    prefill_max_steps = args.max_time
    if args.terminal_prefill:
        prefill_max_steps = args.terminal_max_time
        prefill_env_factory = lambda seed: make_terminal_training_env(
            args,
            env_kwargs,
            seed=seed,
            total_steps=args.source_transition_count,
        )
    transitions, source_rows = collect_reference_residual_trajectories(
        minimum_transitions=args.source_transition_count,
        seed_start=args.source_seed_start,
        max_episodes=args.source_max_episodes,
        max_steps=prefill_max_steps,
        env_kwargs=env_kwargs,
        residual_std=args.prefill_residual_std,
        residual_clip=args.prefill_residual_clip,
        zero_episode_fraction=args.prefill_zero_episode_fraction,
        random_seed=args.prefill_random_seed,
        retain_failure_episodes=args.retain_failure_prefill,
        residual_hold_steps=args.prefill_residual_hold_steps,
        env_factory=prefill_env_factory,
    )
    write_csv(result_dir / "source_episode_collection.csv", source_rows)

    def make_training_env():
        return Monitor(
            make_terminal_training_env(
                args,
                env_kwargs,
                seed=args.training_seed,
                total_steps=args.total_timesteps,
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
    print(
        "Training reset:      "
        + (
            "lateral-limited terminal curriculum"
            if args.lateral_terminal_curriculum
            else "empirical terminal LQR states"
        )
    )
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
    acceptance = _acceptance_gate(
        model_dir=model_dir,
        result_dir=result_dir,
        training_env=training_env,
        candidate_checkpoint=callback.best_checkpoint,
        acceptance_seeds=args.acceptance_seeds,
        max_eval_steps=args.max_eval_steps,
        env_kwargs=env_kwargs,
        device=args.device,
    )
    config.update(
        {
            "actual_timesteps": int(model.num_timesteps),
            "best_checkpoint": int(callback.best_checkpoint),
            "source_prefill": prefill_metrics,
            "prefill_replay_statistics": prefill_statistics,
            "final_replay_statistics": replay_statistics,
            "acceptance": acceptance,
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
