"""Paired evaluation of a final-approach SAC policy with CBF-QP variants.

The same evaluation seeds are used for every requested variant, which makes
the baseline-versus-filter comparison paired rather than dependent on a new
set of randomly sampled initial states.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from controllers.cbf_3DoF.barriers import (
    FinalApproachCBFConfig,
    build_final_approach_constraints,
)
from controllers.cbf_3DoF.qp_filter import FinalApproachCBFFilter
from envs.docking3dof.gym_env import Docking3DoFGymEnv


SAC_FINAL_DIR = SRC_PATH / "controllers" / "SAC-3DoF" / "final_approach"
DEFAULT_MODEL_STEM = (
    "sac_final_approach_portobs_rewardv9_600ksteps_"
    "seed10_optuna12_buffer600k_best"
)
DEFAULT_MODEL_DIR = SAC_FINAL_DIR / "models" / "optuna12_600ksteps_buffer600k"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / f"{DEFAULT_MODEL_STEM}.zip"
DEFAULT_VECNORM_PATH = DEFAULT_MODEL_DIR / f"{DEFAULT_MODEL_STEM}_vecnormalize.pkl"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "final_approach"

VARIANT_FLAGS = {
    "baseline": (False, False),
    "cone": (True, False),
    "speed": (False, True),
    "both": (True, True),
}


def make_env(seed: int, mode: str) -> Docking3DoFGymEnv:
    return Docking3DoFGymEnv(
        phase="final",
        mode=mode,
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
    )


def load_vecnormalize(path: Path) -> tuple[VecNormalize, DummyVecEnv]:
    dummy = DummyVecEnv([lambda: Monitor(make_env(seed=999_999, mode="testing"))])
    vecnormalize = VecNormalize.load(str(path), dummy)
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    return vecnormalize, dummy


def normalize_observation(vecnormalize: VecNormalize, observation: np.ndarray):
    return vecnormalize.normalize_obs(observation.reshape(1, -1))


def _serialize_step_records(records: list[dict[str, object]], path: Path) -> None:
    """Store fixed-schema scalar/vector step records in a compact NPZ file."""
    if not records:
        np.savez_compressed(path, step=np.empty(0, dtype=np.int64))
        return

    keys = records[0].keys()
    arrays = {}
    for key in keys:
        values = [record[key] for record in records]
        first = values[0]
        if isinstance(first, str):
            arrays[key] = np.asarray(values, dtype=np.str_)
        elif isinstance(first, (bool, np.bool_)):
            arrays[key] = np.asarray(values, dtype=np.bool_)
        else:
            arrays[key] = np.asarray(values)
    np.savez_compressed(path, **arrays)


def _baseline_diagnostics(core_env, state, nominal, config):
    constraints = build_final_approach_constraints(core_env, state, config)
    residual = constraints.A @ nominal - constraints.b
    return {
        "action": nominal,
        "status": "not_run",
        "backend": "none",
        "intervened": False,
        "scaled_norm": 0.0,
        "values": constraints.barrier_values,
        "residuals": dict(zip(constraints.names, residual)),
        "slacks": {name: 0.0 for name in constraints.names},
        "failed": False,
        "kinematics": constraints.kinematics,
    }


def run_rollout(
    *,
    model,
    vecnormalize,
    eval_seed: int,
    mode: str,
    max_steps: int,
    variant: str,
    base_config: FinalApproachCBFConfig,
    output_dir: Path,
) -> dict[str, object]:
    env = make_env(eval_seed, mode)
    observation, info = env.reset(seed=eval_seed)
    total_reward = 0.0
    records: list[dict[str, object]] = []
    cone_enabled, speed_enabled = VARIANT_FLAGS[variant]

    # Baseline constraints are evaluated in shadow mode for diagnostics only.
    config = replace(
        base_config,
        enable_cone=True if variant == "baseline" else cone_enabled,
        enable_speed=True if variant == "baseline" else speed_enabled,
    )
    cbf_filter = (
        None if variant == "baseline" else FinalApproachCBFFilter(env.core_env, config)
    )

    final_info = info
    for step_index in range(max_steps):
        state = env.get_physical_state()
        normalized_observation = normalize_observation(vecnormalize, observation)
        action_normalized, _ = model.predict(
            normalized_observation,
            deterministic=True,
        )
        nominal_physical = env.normalized_action_to_physical(action_normalized[0])

        if cbf_filter is None:
            diagnostics = _baseline_diagnostics(
                env.core_env,
                state,
                nominal_physical,
                config,
            )
        else:
            result = cbf_filter.filter(state, nominal_physical)
            constraints = build_final_approach_constraints(env.core_env, state, config)
            diagnostics = {
                "action": result.action_filtered,
                "status": result.solver_status,
                "backend": result.solver_backend,
                "intervened": result.intervened,
                "scaled_norm": result.intervention_scaled_norm,
                "values": result.barrier_values,
                "residuals": result.cbf_residuals,
                "slacks": result.slack_values,
                "failed": result.qp_failed,
                "kinematics": constraints.kinematics,
            }

        filtered_physical = np.asarray(diagnostics["action"], dtype=np.float64)
        filtered_normalized = env.physical_action_to_normalized(filtered_physical)
        observation, reward, terminated, truncated, final_info = env.step(
            filtered_normalized
        )
        total_reward += reward

        values = diagnostics["values"]
        residuals = diagnostics["residuals"]
        slacks = diagnostics["slacks"]
        kinematics = diagnostics["kinematics"]
        records.append(
            {
                "step": step_index,
                "time": float(env.core_env.time),
                "state": state,
                "action_nominal": nominal_physical,
                "action_filtered": filtered_physical,
                "intervened": diagnostics["intervened"],
                "intervention_scaled_norm": diagnostics["scaled_norm"],
                "solver_status": str(diagnostics["status"]),
                "solver_backend": str(diagnostics["backend"]),
                "qp_failed": diagnostics["failed"],
                "s": float(kinematics.s),
                "y_p": float(kinematics.y_p),
                "s_dot": float(kinematics.s_dot),
                "y_p_dot": float(kinematics.y_p_dot),
                "port_speed": float(np.linalg.norm(kinematics.v_p)),
                "h_cone_upper": float(values.get("cone_upper", np.nan)),
                "h_cone_lower": float(values.get("cone_lower", np.nan)),
                "h_port_speed": float(values.get("port_speed", np.nan)),
                "residual_cone_upper": float(residuals.get("cone_upper", np.nan)),
                "residual_cone_lower": float(residuals.get("cone_lower", np.nan)),
                "residual_port_speed": float(residuals.get("port_speed", np.nan)),
                "slack_cone_upper": float(slacks.get("cone_upper", np.nan)),
                "slack_cone_lower": float(slacks.get("cone_lower", np.nan)),
                "slack_port_speed": float(slacks.get("port_speed", np.nan)),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        if terminated or truncated:
            break

    env.close()
    steps = len(records)
    rollout_path = output_dir / f"steps_{variant}_seed{eval_seed:04d}.npz"
    _serialize_step_records(records, rollout_path)

    def values(key):
        return np.asarray([record[key] for record in records], dtype=np.float64)

    def finite_min(key):
        array = values(key)
        finite = array[np.isfinite(array)]
        return float(np.min(finite)) if finite.size else np.nan

    intervention = values("intervention_scaled_norm")
    slacks = np.column_stack(
        [
            values("slack_cone_upper"),
            values("slack_cone_lower"),
            values("slack_port_speed"),
        ]
    )
    finite_slacks = slacks[np.isfinite(slacks)]
    ended = bool(final_info.get("success") or final_info.get("failure") or final_info.get("time_limit"))
    reason = final_info.get("termination_reason") if ended else "evaluation_step_limit"

    return {
        "variant": variant,
        "eval_seed": eval_seed,
        "steps": steps,
        "success": bool(final_info.get("success", False)),
        "failure": bool(final_info.get("failure", False)),
        "time_limit": bool(final_info.get("time_limit", False)) or not ended,
        "termination_reason": reason,
        "violated_constraints": ",".join(final_info.get("violated_constraints", [])),
        "total_reward": total_reward,
        "final_port_distance": float(final_info.get("port_distance", np.nan)),
        "final_port_speed": float(final_info.get("port_speed", np.nan)),
        "intervention_count": int(sum(record["intervened"] for record in records)),
        "intervention_fraction": float(np.mean([record["intervened"] for record in records])),
        "mean_intervention_scaled_norm": float(np.mean(intervention)),
        "max_intervention_scaled_norm": float(np.max(intervention)),
        "qp_failure_count": int(sum(record["qp_failed"] for record in records)),
        "max_slack": float(np.max(finite_slacks)) if finite_slacks.size else 0.0,
        "min_h_cone_upper": finite_min("h_cone_upper"),
        "min_h_cone_lower": finite_min("h_cone_lower"),
        "min_h_port_speed": finite_min("h_port_speed"),
        "step_log": rollout_path.name,
    }


def summarize_variant(variant: str, rows: list[dict[str, object]]):
    reasons = Counter(row["termination_reason"] for row in rows)
    violations = Counter()
    for row in rows:
        violations.update(item for item in str(row["violated_constraints"]).split(",") if item)
    return {
        "variant": variant,
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "failure_rate": float(np.mean([row["failure"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_intervention_fraction": float(
            np.mean([row["intervention_fraction"] for row in rows])
        ),
        "qp_failure_count": int(sum(row["qp_failure_count"] for row in rows)),
        "maximum_slack": float(max(row["max_slack"] for row in rows)),
        "termination_reasons": ";".join(f"{key}:{value}" for key, value in reasons.items()),
        "violated_constraints": ";".join(f"{key}:{value}" for key, value in violations.items()),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired final-approach SAC versus CBF-QP evaluation."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vecnormalize", type=Path, default=DEFAULT_VECNORM_PATH)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANT_FLAGS),
        default=["baseline", "both"],
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--mode", choices=("nominal", "testing", "training"), default="testing")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--solver", choices=("auto", "osqp", "scipy"), default="auto")
    parser.add_argument("--hard-constraints", action="store_true")
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--port-speed-max", type=float, default=0.15)
    parser.add_argument("--cone-k1", type=float, default=0.1)
    parser.add_argument("--cone-k2", type=float, default=0.1)
    parser.add_argument("--speed-alpha", type=float, default=0.5)
    parser.add_argument("--cone-slack-penalty", type=float, default=1.0e8)
    parser.add_argument("--speed-slack-penalty", type=float, default=1.0e8)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_seeds <= 0 or args.max_steps <= 0:
        raise ValueError("--num-seeds and --max-steps must be positive.")
    if not args.model.is_file():
        raise FileNotFoundError(f"SAC checkpoint not found: {args.model}")
    if not args.vecnormalize.is_file():
        raise FileNotFoundError(f"VecNormalize statistics not found: {args.vecnormalize}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (DEFAULT_RESULTS_DIR / timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)

    config = FinalApproachCBFConfig(
        kappa=args.kappa,
        port_speed_max=args.port_speed_max,
        cone_k1=args.cone_k1,
        cone_k2=args.cone_k2,
        speed_alpha=args.speed_alpha,
        cone_slack_penalty=args.cone_slack_penalty,
        speed_slack_penalty=args.speed_slack_penalty,
        allow_slack=not args.hard_constraints,
        solver_backend=args.solver,
    )
    model = SAC.load(str(args.model))
    vecnormalize, dummy_env = load_vecnormalize(args.vecnormalize)

    rows = []
    try:
        for variant in args.variants:
            for eval_seed in range(args.seed_start, args.seed_start + args.num_seeds):
                row = run_rollout(
                    model=model,
                    vecnormalize=vecnormalize,
                    eval_seed=eval_seed,
                    mode=args.mode,
                    max_steps=args.max_steps,
                    variant=variant,
                    base_config=config,
                    output_dir=output_dir,
                )
                rows.append(row)
                print(
                    f"{variant:8s} seed={eval_seed:4d} "
                    f"success={row['success']} reason={row['termination_reason']} "
                    f"steps={row['steps']} intervention={row['intervention_fraction']:.3f}"
                )
    finally:
        dummy_env.close()

    summaries = [
        summarize_variant(variant, [row for row in rows if row["variant"] == variant])
        for variant in args.variants
    ]
    write_csv(output_dir / "rollout_metrics.csv", rows)
    write_csv(output_dir / "variant_summary.csv", summaries)
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "model": str(args.model.resolve()),
                "vecnormalize": str(args.vecnormalize.resolve()),
                "variants": args.variants,
                "seed_start": args.seed_start,
                "num_seeds": args.num_seeds,
                "mode": args.mode,
                "max_steps": args.max_steps,
                "cbf": asdict(config),
            },
            stream,
            indent=2,
        )

    print(f"\nResults written to: {output_dir.resolve()}")
    for summary in summaries:
        print(
            f"{summary['variant']:8s}: success={summary['success_rate']:.3f}, "
            f"failure={summary['failure_rate']:.3f}, "
            f"intervention={summary['mean_intervention_fraction']:.3f}, "
            f"QP failures={summary['qp_failure_count']}"
        )


if __name__ == "__main__":
    main()
