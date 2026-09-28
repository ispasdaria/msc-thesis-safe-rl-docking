"""Robustness evaluation of the selected final-approach SAC policies.

This is evaluation-only: it loads fixed checkpoints and frozen VecNormalize
statistics, and supplies an explicit enlarged initial state to ``reset``.
The environment's standard testing and training distributions are untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from envs.docking3dof.gym_env import Docking3DoFGymEnv


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR / "models" / "retained_policies"
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR / "results" / "robustness_evaluations" / "robustness_expanded_test_bounds"
)
MAX_STEPS = 5_000
DT_SECONDS = 1.0

POLICIES = {
    "round1_seed12_final100k": {
        "round": 1,
        "training_seed": 12,
        "training_budget_steps": 100_000,
        "replay_buffer_size": 100_000,
        "output_directory": "round1_seed12_100k_steps",
        "group": "optuna12_100ksteps_buffer100k",
        "stem": "sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12",
    },
    "round5_seed11_final600k": {
        "round": 5,
        "training_seed": 11,
        "training_budget_steps": 600_000,
        "replay_buffer_size": 600_000,
        "output_directory": "round5_seed11_600k_steps_buffer600k",
        "group": "optuna12_600ksteps_buffer600k",
        "stem": "sac_final_approach_portobs_rewardv9_600ksteps_seed11_optuna12_buffer600k",
    },
}

INITIAL_STATE_COLUMNS = (
    "initial_x_O_m",
    "initial_y_O_m",
    "initial_xdot_O_mps",
    "initial_ydot_O_mps",
    "initial_psi_deg",
    "initial_omega_z_deg_s",
)

DEFAULT_INITIAL_RANGES = {
    "x_o": (260.0, 340.0),
    "y_o": (-4.0, 4.0),
    "xdot_o": (-0.055, -0.01),
    "ydot_o": (-0.015, 0.015),
    "psi_deg": (-1.5, 1.5),
    "omega_z_deg_s": (-0.075, 0.075),
}

RANGE_CLI_NAMES = {
    "x_o": ("x-o-min", "x-o-max"),
    "y_o": ("y-o-min", "y-o-max"),
    "xdot_o": ("xdot-o-min", "xdot-o-max"),
    "ydot_o": ("ydot-o-min", "ydot-o-max"),
    "psi_deg": ("psi-min-deg", "psi-max-deg"),
    "omega_z_deg_s": ("omega-z-min-deg-s", "omega-z-max-deg-s"),
}


def portable_path(path: Path) -> str:
    """Return repository paths in a platform-independent form."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def sample_initial_state(seed: int, ranges: dict | None = None) -> np.ndarray:
    """Sample the enlarged test range reproducibly in simulator state order."""
    ranges = DEFAULT_INITIAL_RANGES if ranges is None else ranges
    rng = np.random.default_rng(seed)
    return np.array(
        [
            rng.uniform(*ranges["x_o"]),
            rng.uniform(*ranges["y_o"]),
            rng.uniform(*ranges["xdot_o"]),
            rng.uniform(*ranges["ydot_o"]),
            np.deg2rad(rng.uniform(*ranges["psi_deg"])),
            np.deg2rad(rng.uniform(*ranges["omega_z_deg_s"])),
        ],
        dtype=np.float64,
    )


def make_env(seed: int) -> Docking3DoFGymEnv:
    return Docking3DoFGymEnv(
        phase="final",
        mode="testing",
        dt=DT_SECONDS,
        max_time=5_000.0,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
    )


def load_policy(
    spec: dict,
    model_override: Path | None = None,
    vecnorm_override: Path | None = None,
) -> tuple[SAC, VecNormalize, DummyVecEnv, Path, Path]:
    model_path = model_override or MODEL_ROOT / spec["group"] / f"{spec['stem']}.zip"
    vecnorm_path = (
        vecnorm_override
        or MODEL_ROOT / spec["group"] / f"{spec['stem']}_vecnormalize.pkl"
    )
    if not model_path.is_file() or not vecnorm_path.is_file():
        raise FileNotFoundError(f"Missing model pair: {model_path}; {vecnorm_path}")

    dummy_env = DummyVecEnv([lambda: Monitor(make_env(999))])
    vecnorm = VecNormalize.load(str(vecnorm_path), dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    model = SAC.load(str(model_path))
    return model, vecnorm, dummy_env, model_path, vecnorm_path


def evaluate_rollout(
    policy_id: str,
    model: SAC,
    vecnorm: VecNormalize,
    eval_seed: int,
    max_steps: int = MAX_STEPS,
    initial_ranges: dict | None = None,
) -> dict:
    initial_state = sample_initial_state(eval_seed, initial_ranges)
    env = make_env(eval_seed)
    obs, info = env.reset(
        seed=eval_seed,
        options={"initial_state": initial_state, "mode": "testing"},
    )
    final_info = info
    saturation_steps = 0
    cone_violations = 0

    for step_index in range(max_steps):
        normalized_obs = vecnorm.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(normalized_obs, deterministic=True)
        normalized_action = np.asarray(action[0], dtype=np.float64)
        saturation_steps += int(np.any(np.abs(normalized_action) >= 1.0 - 1e-6))
        obs, _, terminated, truncated, info = env.step(normalized_action)
        final_info = info
        geometry = info.get("geometry", {})
        cone_violations += int(geometry.get("in_C_dock") is False)
        if terminated or truncated:
            break
    else:
        step_index = max_steps - 1

    steps = step_index + 1
    reason = final_info.get("termination_reason") or "evaluation_step_limit"
    row = {
        "policy_identifier": policy_id,
        "evaluation_seed": eval_seed,
        "initial_x_O_m": float(initial_state[0]),
        "initial_y_O_m": float(initial_state[1]),
        "initial_xdot_O_mps": float(initial_state[2]),
        "initial_ydot_O_mps": float(initial_state[3]),
        "initial_psi_deg": float(np.rad2deg(initial_state[4])),
        "initial_omega_z_deg_s": float(np.rad2deg(initial_state[5])),
        "success": bool(final_info.get("success", False)),
        "termination_reason": reason,
        "duration_s": float(env.core_env.time),
        "terminal_port_distance_m": float(final_info["port_distance"]),
        "terminal_port_speed_mps": float(final_info["port_speed"]),
        "terminal_attitude_error_deg": abs(float(np.rad2deg(final_info["psi_e"]))),
        "terminal_angular_rate_error_deg_s": abs(
            float(np.rad2deg(final_info["omega"]))
        ),
        "action_saturation_fraction": saturation_steps / steps,
        "docking_cone_violation_count": cone_violations,
        "violated_constraints": ";".join(final_info.get("violated_constraints", [])),
    }
    env.close()
    return row


def summarize(policy_id: str, rows: list[dict]) -> dict:
    reasons = Counter(str(row["termination_reason"]) for row in rows)

    def values(field: str) -> np.ndarray:
        return np.asarray([float(row[field]) for row in rows], dtype=np.float64)

    return {
        "policy_identifier": policy_id,
        "evaluation_rollouts": len(rows),
        "successful_rollouts": sum(bool(row["success"]) for row in rows),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
        "timeouts": reasons.get("timeout", 0),
        "hard_safety_terminations": reasons.get("safety_violation", 0),
        "termination_reason_breakdown": dict(sorted(reasons.items())),
        "mean_duration_s": float(np.mean(values("duration_s"))),
        "std_duration_s": float(np.std(values("duration_s"))),
        "mean_terminal_port_distance_m": float(
            np.mean(values("terminal_port_distance_m"))
        ),
        "max_terminal_port_distance_m": float(
            np.max(values("terminal_port_distance_m"))
        ),
        "mean_terminal_port_speed_mps": float(
            np.mean(values("terminal_port_speed_mps"))
        ),
        "mean_terminal_attitude_error_deg": float(
            np.mean(values("terminal_attitude_error_deg"))
        ),
        "mean_terminal_angular_rate_error_deg_s": float(
            np.mean(values("terminal_angular_rate_error_deg_s"))
        ),
        "mean_action_saturation_fraction": float(
            np.mean(values("action_saturation_fraction"))
        ),
        "docking_cone_violation_count": int(
            np.sum(values("docking_cone_violation_count"))
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_comparison(summaries: list[dict]) -> None:
    print("\nEnlarged-range deterministic evaluation")
    print(
        f"{'policy':30} {'success':>9} {'timeout':>8} {'safety':>7} "
        f"{'duration mean+/-sd [s]':>23} {'port err mean/max [m]':>23} {'sat. mean':>10}"
    )
    for item in summaries:
        success = f"{item['successful_rollouts']}/{item['evaluation_rollouts']}"
        duration = f"{item['mean_duration_s']:.1f}+/-{item['std_duration_s']:.1f}"
        distance = (
            f"{item['mean_terminal_port_distance_m']:.4f}/"
            f"{item['max_terminal_port_distance_m']:.4f}"
        )
        print(
            f"{item['policy_identifier']:30} {success:>9} {item['timeouts']:>8} "
            f"{item['hard_safety_terminations']:>7} {duration:>23} "
            f"{distance:>23} {item['mean_action_saturation_fraction']:>10.3f}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=("both", *POLICIES),
        default="both",
        help="Evaluate both selected policies by default.",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed-end", type=int)
    seed_group.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--vecnormalize-path", type=Path)
    for name, default in DEFAULT_INITIAL_RANGES.items():
        min_name, max_name = RANGE_CLI_NAMES[name]
        parser.add_argument(
            f"--{min_name}", dest=f"{name}_min", type=float, default=default[0]
        )
        parser.add_argument(
            f"--{max_name}", dest=f"{name}_max", type=float, default=default[1]
        )
    args = parser.parse_args(argv)
    if (args.model_path is None) != (args.vecnormalize_path is None):
        parser.error("--model-path and --vecnormalize-path must be supplied together")
    if args.model_path is not None and args.policy == "both":
        parser.error("explicit checkpoint paths require a single --policy")
    if args.seed_end is None:
        if args.num_seeds <= 0:
            parser.error("--num-seeds must be positive")
        args.seed_end = args.seed_start + args.num_seeds - 1
    if args.seed_end < args.seed_start:
        parser.error("--seed-end must be greater than or equal to --seed-start")
    args.initial_ranges = {}
    for name in DEFAULT_INITIAL_RANGES:
        min_name, max_name = RANGE_CLI_NAMES[name]
        low = getattr(args, f"{name}_min")
        high = getattr(args, f"{name}_max")
        if high < low:
            parser.error(f"--{max_name} must be >= --{min_name}")
        args.initial_ranges[name] = (low, high)
    return args


def main() -> None:
    args = parse_args()
    selected = POLICIES if args.policy == "both" else {args.policy: POLICIES[args.policy]}
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []

    for policy_id, spec in selected.items():
        model, vecnorm, dummy_env, model_path, vecnorm_path = load_policy(
            spec, args.model_path, args.vecnormalize_path
        )
        print(f"Evaluating {policy_id}: seeds {args.seed_start}--{args.seed_end}", flush=True)
        rows = []
        for eval_seed in range(args.seed_start, args.seed_end + 1):
            row = evaluate_rollout(
                policy_id,
                model,
                vecnorm,
                eval_seed,
                max_steps=args.max_steps,
                initial_ranges=args.initial_ranges,
            )
            rows.append(row)
            print(
                f"  seed={eval_seed:02d} reason={row['termination_reason']} "
                f"duration={row['duration_s']:.1f}s",
                flush=True,
            )
        dummy_env.close()

        summary = summarize(policy_id, rows)
        summary.update(
            {
                "round": spec["round"],
                "training_seed": spec["training_seed"],
                "training_budget_steps": spec["training_budget_steps"],
                "replay_buffer_size": spec["replay_buffer_size"],
                "model_path": portable_path(model_path),
                "vecnormalize_path": portable_path(vecnorm_path),
                "vecnormalize_training": False,
                "vecnormalize_reward_normalization": False,
                "deterministic_inference": True,
                "evaluation_seed_start": args.seed_start,
                "evaluation_seed_end": args.seed_end,
            }
        )
        policy_dir = args.output_root / spec["output_directory"]
        write_csv(policy_dir / "rollout_results.csv", rows)
        with (policy_dir / "evaluation_summary.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(summary, stream, indent=2)
        summaries.append(summary)

    with (args.output_root / "policy_comparison_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summaries, stream, indent=2)
    print_comparison(summaries)
    print(f"\nSaved outputs to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
