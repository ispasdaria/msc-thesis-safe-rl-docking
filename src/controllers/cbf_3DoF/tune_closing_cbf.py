"""Safety-focused discrete gain search for the closing-phase hard CBF-QP.

The fixed phase-balanced replay SAC checkpoint is evaluated deterministically.
Task success is reported for context but is deliberately excluded from ranking.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from stable_baselines3 import SAC

from controllers.cbf_3DoF.closing_barriers import ClosingCBFConfig
from controllers.cbf_3DoF.evaluate_closing_cbf import (
    PROTECTED_BARRIER_TOLERANCE,
    load_vecnormalize,
    run_rollout,
)


CHECKPOINT_DIR = (
    SRC_PATH
    / "controllers"
    / "SAC-3DoF"
    / "closing"
    / "models"
    / "multiseed"
    / "sac_closing_v15_trial16_terminal_phase_balanced_guarded600k_seed0"
    / "seed_0"
)
DEFAULT_MODEL_PATH = CHECKPOINT_DIR / "model_0100000_steps.zip"
DEFAULT_VECNORM_PATH = CHECKPOINT_DIR / "model_0100000_steps_vecnormalize.pkl"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "closing" / "tuning"

DEFAULT_CONE_K = (0.05, 0.10, 0.20)
DEFAULT_PORT_ALPHA = (0.25, 0.50, 1.00)
DEFAULT_AXIAL_ALPHA = (0.25, 0.50, 1.00)
DEFAULT_ATTITUDE_K = (0.10, 0.30, 0.50)
DEFAULT_RATE_ALPHA = (0.25, 0.50, 1.00)

FAMILY_FIELDS = (
    "protected_cone_violation_count",
    "protected_total_speed_violation_count",
    "protected_axial_speed_violation_count",
    "protected_attitude_violation_count",
    "protected_angular_rate_violation_count",
)


def parse_float_values(text: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in text.split(",") if item.strip())
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("gain lists must contain positive finite values")
    return values


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_result_file(path: Path, filename: str) -> Path:
    candidate = path / filename if path.is_dir() else path
    if not candidate.is_file():
        raise FileNotFoundError(f"Required result file not found: {candidate}")
    return candidate


def aggregate_configuration(
    *, config_id: str, stage: str, variant: str, gains: dict[str, float],
    episodes: list[dict[str, object]], model_path: Path, seeds: list[int],
) -> dict[str, object]:
    steps = int(sum(int(row["steps"]) for row in episodes))

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in episodes]))

    def finite_mean(field: str) -> float:
        values = np.asarray([float(row[field]) for row in episodes], dtype=float)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else np.nan

    result: dict[str, object] = {
        "configuration_id": config_id,
        "stage": stage,
        "filter_variant": variant,
        "k_c": gains["k_c"],
        "alpha_v": gains["alpha_v"],
        "alpha_s": gains["alpha_s"],
        "k_psi": gains["k_psi"],
        "alpha_omega": gains["alpha_omega"],
        "number_of_evaluation_rollouts": len(episodes),
        "total_steps": steps,
    }
    for field in FAMILY_FIELDS:
        result[field] = int(sum(int(row[field]) for row in episodes))
    result.update(
        total_protected_violations=int(sum(int(result[field]) for field in FAMILY_FIELDS)),
        strict_crossing_count=int(
            sum(int(row["strict_protected_barrier_violation_count"]) for row in episodes)
        ),
        qp_failure_count=int(sum(int(row["qp_failure_count"]) for row in episodes)),
        mean_intervention_fraction=mean("intervention_fraction"),
        mean_action_correction_norm=mean("mean_action_correction_norm"),
        maximum_action_correction_norm=float(
            max(float(row["max_action_correction_norm"]) for row in episodes)
        ),
        mean_normalized_action_correction_norm=mean("mean_intervention_scaled_norm"),
        maximum_normalized_action_correction_norm=float(
            max(float(row["max_intervention_scaled_norm"]) for row in episodes)
        ),
        actuator_saturation_fraction=mean("actuator_saturation_fraction"),
        minimum_cbf_residual=float(min(float(row["minimum_cbf_residual"]) for row in episodes)),
        success_count=int(sum(bool(row["success"]) for row in episodes)),
        failure_count=int(sum(bool(row["failure"]) for row in episodes)),
        timeout_count=int(sum(bool(row["time_limit"]) for row in episodes)),
        success_rate=mean("success"),
        failure_rate=mean("failure"),
        timeout_rate=mean("time_limit"),
        mean_terminal_p3_position_error=finite_mean("final_p3_position_error"),
        mean_terminal_p3_velocity_error=finite_mean("final_p3_velocity_error"),
        mean_terminal_p3_attitude_error=finite_mean("final_p3_attitude_error"),
        mean_terminal_p3_rate_error=finite_mean("final_p3_rate_error"),
        checkpoint_path=str(model_path.resolve()),
        seed_list=",".join(str(seed) for seed in seeds),
        numerical_tolerance=PROTECTED_BARRIER_TOLERANCE,
    )
    result["safety_feasible"] = bool(
        all(int(result[field]) == 0 for field in FAMILY_FIELDS)
        and int(result["qp_failure_count"]) == 0
    )
    return result


def ranking_key(row: dict[str, object]) -> tuple[object, ...]:
    residual_violation = max(0.0, -float(row["minimum_cbf_residual"]))
    return (
        int(row["protected_cone_violation_count"]),
        int(row["protected_total_speed_violation_count"]),
        int(row["protected_axial_speed_violation_count"]),
        int(row["protected_attitude_violation_count"]),
        int(row["protected_angular_rate_violation_count"]),
        int(row["qp_failure_count"]),
        float(row["mean_intervention_fraction"]),
        float(row["mean_action_correction_norm"]),
        float(row["maximum_action_correction_norm"]),
        float(row["actuator_saturation_fraction"]),
        residual_violation,
        str(row["configuration_id"]),
    )


def rank_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked = sorted(rows, key=ranking_key)
    result = []
    for index, row in enumerate(ranked, start=1):
        item = dict(row)
        item["rank"] = index
        item["selection_status"] = "selected" if index == 1 else "not_selected"
        result.append(item)
    return result


def plot_rows(rows: list[dict[str, object]], output_dir: Path, label: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [str(row["configuration_id"]) for row in rows]
    metrics = (
        ("total_protected_violations", "Protected violations", "protected_violations.png"),
        ("mean_intervention_fraction", "Mean intervention fraction", "intervention_fraction.png"),
        ("mean_action_correction_norm", "Mean physical correction norm", "action_correction.png"),
    )
    for field, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(max(7.0, 0.32 * len(rows)), 4.2))
        ax.bar(labels, [float(row[field]) for row in rows])
        ax.set_xlabel(f"{label} gain configuration")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=90)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def evaluate_grid(
    *, stage: str, variant: str, grid: list[dict[str, float]], seeds: list[int],
    model, vecnormalize, model_path: Path, output_dir: Path, mode: str,
    max_steps: int,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    episode_rows = []
    for index, gains in enumerate(grid, start=1):
        config_id = f"{stage}_{index:03d}"
        config = ClosingCBFConfig(
            cone_k1=gains["k_c"], cone_k2=gains["k_c"],
            port_speed_alpha=gains["alpha_v"],
            axial_port_speed_alpha=gains["alpha_s"],
            attitude_k1=gains["k_psi"], attitude_k2=gains["k_psi"],
            angular_rate_alpha=gains["alpha_omega"], allow_slack=False,
            solver_backend="osqp",
        )
        current = []
        for seed in seeds:
            episode = run_rollout(
                model=model, vecnormalize=vecnormalize, eval_seed=seed, mode=mode,
                max_steps=max_steps, variant=variant, base_config=config,
                output_dir=output_dir, save_timeseries=False,
            )
            current.append(episode)
            episode_rows.append({"configuration_id": config_id, **gains, **episode})
        row = aggregate_configuration(
            config_id=config_id, stage=stage, variant=variant, gains=gains,
            episodes=current, model_path=model_path, seeds=seeds,
        )
        rows.append(row)
        print(
            f"{config_id}: protected={row['total_protected_violations']}, "
            f"QP failures={row['qp_failure_count']}, "
            f"intervention={row['mean_intervention_fraction']:.6f}",
            flush=True,
        )
        write_csv(output_dir / "gain_configuration_results.csv", rows)
        write_csv(output_dir / "per_episode_metrics.csv", episode_rows)
    ranked = rank_rows(rows)
    write_csv(output_dir / "ranked_gain_configurations.csv", ranked)
    selected = ranked[0]
    safe_rows = [row for row in ranked if row["safety_feasible"]]
    summary = {
        "selected_configuration": selected,
        "safe_configuration_ids": [row["configuration_id"] for row in safe_rows],
        "safe_configuration_count": len(safe_rows),
        "selection_reason": (
            "All protected families and QP failures are zero; selected by minimum "
            "intervention and the declared tie-breakers."
            if selected["safety_feasible"] else
            "No tested configuration met every safety criterion; this is the "
            "lexicographically least-violating provisional configuration."
        ),
        "ranking_rule": [
            "protected cone violations", "protected total-speed violations",
            "protected axial-speed violations", "protected attitude violations",
            "protected angular-rate violations", "QP failures",
            "mean intervention fraction", "mean physical correction norm",
            "maximum physical correction norm", "actuator saturation fraction",
            "negative minimum CBF residual magnitude", "configuration ID",
        ],
    }
    write_json(output_dir / "selected_configuration.json", summary)
    plot_rows(ranked, output_dir, stage)
    return ranked


def gains_from_row(row: dict[str, object]) -> dict[str, float]:
    return {name: float(row[name]) for name in ("k_c", "alpha_v", "alpha_s", "k_psi", "alpha_omega")}


def selected_gains(path: Path) -> dict[str, float]:
    data = read_json(resolve_result_file(path, "selected_configuration.json"))
    return gains_from_row(data["selected_configuration"])


def make_stage1_grid(args) -> list[dict[str, float]]:
    return [
        {"k_c": kc, "alpha_v": av, "alpha_s": axial, "k_psi": 0.3, "alpha_omega": 0.5}
        for kc, av, axial in product(args.cone_k_values, args.port_alpha_values, args.axial_alpha_values)
    ]


def make_stage2_grid(args, translation: dict[str, float]) -> list[dict[str, float]]:
    return [
        {**translation, "k_psi": kp, "alpha_omega": rate}
        for kp, rate in product(args.attitude_k_values, args.rate_alpha_values)
    ]


def load_shortlist(path: Path, count: int) -> list[dict[str, float]]:
    ranked_path = resolve_result_file(path, "ranked_gain_configurations.csv")
    with ranked_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    safe = [row for row in rows if str(row["safety_feasible"]).lower() == "true"]
    source = safe if safe else rows
    if not source:
        raise ValueError(f"No configurations found in {ranked_path}")
    return [gains_from_row(row) for row in source[:count]]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("1", "2", "validate", "all"), default="all")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vecnormalize", type=Path, default=DEFAULT_VECNORM_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stage-1-results", type=Path, default=None)
    parser.add_argument("--stage-2-results", type=Path, default=None)
    parser.add_argument("--screen-seed-start", type=int, default=200)
    parser.add_argument("--screen-num-seeds", type=int, default=10)
    parser.add_argument("--validation-seed-start", type=int, default=210)
    parser.add_argument("--validation-num-seeds", type=int, default=50)
    parser.add_argument("--shortlist-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=12_000)
    parser.add_argument("--mode", choices=("nominal", "testing", "training"), default="testing")
    parser.add_argument("--cone-k-values", type=parse_float_values, default=DEFAULT_CONE_K)
    parser.add_argument("--port-alpha-values", type=parse_float_values, default=DEFAULT_PORT_ALPHA)
    parser.add_argument("--axial-alpha-values", type=parse_float_values, default=DEFAULT_AXIAL_ALPHA)
    parser.add_argument("--attitude-k-values", type=parse_float_values, default=DEFAULT_ATTITUDE_K)
    parser.add_argument("--rate-alpha-values", type=parse_float_values, default=DEFAULT_RATE_ALPHA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in ((args.model, "model"), (args.vecnormalize, "VecNormalize")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if min(args.screen_num_seeds, args.validation_num_seeds, args.shortlist_size, args.max_steps) <= 0:
        raise ValueError("seed counts, shortlist size, and max steps must be positive")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.output_dir or DEFAULT_RESULTS_DIR / f"closing_gain_search_{timestamp}"
    root.mkdir(parents=True, exist_ok=False)
    screen_seeds = list(range(args.screen_seed_start, args.screen_seed_start + args.screen_num_seeds))
    validation_seeds = list(range(args.validation_seed_start, args.validation_seed_start + args.validation_num_seeds))
    if set(screen_seeds) & set(validation_seeds):
        raise ValueError("screening and validation seed sets must be disjoint")
    plan = {
        "model": str(args.model.resolve()), "vecnormalize": str(args.vecnormalize.resolve()),
        "stage": args.stage, "screening_seeds": screen_seeds,
        "validation_seeds": validation_seeds, "max_steps": args.max_steps,
        "dt_seconds": 1.0, "integrator": "rk4", "hard_constraints": True,
        "protected_barrier_tolerance": PROTECTED_BARRIER_TOLERANCE,
        "physical_limits_are_not_tuned": True,
        "candidate_values": {
            "k_c": args.cone_k_values, "alpha_v": args.port_alpha_values,
            "alpha_s": args.axial_alpha_values, "k_psi": args.attitude_k_values,
            "alpha_omega": args.rate_alpha_values,
        },
        "task_performance_used_for_ranking": False,
    }
    write_json(root / "gain_search_plan.json", plan)
    model = SAC.load(str(args.model))
    vecnormalize, dummy_env = load_vecnormalize(args.vecnormalize)
    all_rows = []
    try:
        stage1_dir = root / "stage1_translational"
        if args.stage in ("1", "all"):
            stage1 = evaluate_grid(
                stage="stage1", variant="all", grid=make_stage1_grid(args),
                seeds=screen_seeds, model=model, vecnormalize=vecnormalize,
                model_path=args.model, output_dir=stage1_dir, mode=args.mode,
                max_steps=args.max_steps,
            )
            all_rows.extend(stage1)
            translation = gains_from_row(stage1[0])
        else:
            if args.stage_1_results is None and args.stage == "2":
                raise ValueError("--stage-1-results is required for --stage 2")
            translation = selected_gains(args.stage_1_results) if args.stage == "2" else {}

        stage2_dir = root / "stage2_alignment"
        if args.stage in ("2", "all"):
            stage2 = evaluate_grid(
                stage="stage2", variant="full", grid=make_stage2_grid(args, translation),
                seeds=screen_seeds, model=model, vecnormalize=vecnormalize,
                model_path=args.model, output_dir=stage2_dir, mode=args.mode,
                max_steps=args.max_steps,
            )
            all_rows.extend(stage2)

        if args.stage in ("validate", "all"):
            if args.stage == "all":
                shortlist = [gains_from_row(row) for row in ([row for row in stage2 if row["safety_feasible"]] or stage2)[:args.shortlist_size]]
            else:
                if args.stage_2_results is None:
                    raise ValueError("--stage-2-results is required for --stage validate")
                shortlist = load_shortlist(args.stage_2_results, args.shortlist_size)
            validation = evaluate_grid(
                stage="validation", variant="full", grid=shortlist,
                seeds=validation_seeds, model=model, vecnormalize=vecnormalize,
                model_path=args.model, output_dir=root / "validation", mode=args.mode,
                max_steps=args.max_steps,
            )
            all_rows.extend(validation)
        if all_rows:
            write_csv(root / "all_tested_gain_configurations.csv", all_rows)
    finally:
        dummy_env.close()
    print(f"Results saved to: {root.resolve()}")


if __name__ == "__main__":
    main()
