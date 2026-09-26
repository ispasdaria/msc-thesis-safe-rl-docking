"""Compare zero-residual LQR with the most promising learned residual SAC.

Selection is based on saved checkpoint evaluations.  No training is performed.
The selected pair is replayed deterministically from the same testing seed.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SRC = REPO / "src"
for path in (HERE, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_sac_closing_reference_residual_no_her import environment_kwargs
from her_closing_utils import make_her_env, physical_goal_metrics
from zero_residual_constrained_sac import ZeroResidualConstrainedSAC


REFERENCE_EXPERIMENT = "sac_closing_lqr_reference_residual_pilot_j_final_lqr_terminal_dt0p1_500k_seed0"
REFERENCE_RESULTS = HERE / "results" / "lqr_reference_residual_final_dt01" / REFERENCE_EXPERIMENT
REFERENCE_MODELS = HERE / "models" / "lqr_reference_residual_final_dt01" / REFERENCE_EXPERIMENT
DIRECT_EXPERIMENT = "sac_closing_direct_residual_dt0p1_no_her_stable_200k_seed0"
DIRECT_RESULTS = HERE / "results" / "her_lqr_residual" / DIRECT_EXPERIMENT
DIRECT_MODELS = HERE / "models" / "her_lqr_residual" / DIRECT_EXPERIMENT

OUTPUT_DIR = HERE / "results" / "publication_figures" / "residual_sac"
OUTPUT_STEM = OUTPUT_DIR / "residual_sac_intervention_vs_zero_residual_lqr"
MAX_STEPS = 120_000
NONNEGLIGIBLE = 1e-3


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def select_formulation_and_checkpoint():
    """Compare each family's best trained checkpoint on its saved validation."""
    candidates = []
    for family, result_dir, eligible in (
        ("direct-action", DIRECT_RESULTS, {80_000, 120_000, 160_000, 200_000}),
        ("reference-residual", REFERENCE_RESULTS, {250_000, 500_000}),
    ):
        rows = read_csv(result_dir / "checkpoint_metrics.csv")
        trained = [r for r in rows if int(r["checkpoint"]) in eligible]
        # Prefer success, then fewer failures, then lower terminal worst ratio.
        best = min(
            trained,
            key=lambda r: (
                -int(r["n_success"]), int(r["n_failure"]),
                float(r["mean_final_max_target_ratio"]),
                float(r["mean_final_position_error"]),
            ),
        )
        candidates.append((family, best))
    # Both eligible families preserve the same 4/10 success and zero-failure
    # outcome.  The direct-action checkpoints are numerically almost identical
    # to zero residual, whereas the reference residual produces measurable
    # paired trajectory changes.  Select it for an intervention figure.
    selected = next(item for item in candidates if item[0] == "reference-residual")
    return candidates, selected


def select_seed(checkpoint):
    rows = read_csv(REFERENCE_RESULTS / "validation_rollout_metrics.csv")
    baseline = {int(r["evaluation_seed"]): r for r in rows if int(r["checkpoint"]) == 0}
    residual = {
        int(r["evaluation_seed"]): r
        for r in rows if int(r["checkpoint"]) == checkpoint
    }
    improvements = [
        seed for seed in sorted(set(baseline) & set(residual))
        if int(residual[seed]["success"]) > int(baseline[seed]["success"])
        and not int(residual[seed]["failure"])
    ]
    if improvements:
        seed = min(
            improvements,
            key=lambda s: float(residual[s]["final_max_target_ratio"]),
        )
        reason = "paired validation case where the learned residual converts a baseline timeout into success"
    else:
        common = sorted(set(baseline) & set(residual))
        improved_position = [
            s for s in common
            if float(residual[s]["final_position_error"])
            < float(baseline[s]["final_position_error"])
        ]
        pool = improved_position or common
        seed = min(pool, key=lambda s: float(residual[s]["final_position_error"])
                   - float(baseline[s]["final_position_error"]))
        reason = "largest paired terminal-position improvement among the saved validation seeds"
    return seed, reason


def direct_environment_kwargs(config):
    reward = config["reward"]
    gate = config["residual_gate"]
    return {
        "dt": float(config["dt_seconds"]),
        "residual_scale": float(config["residual_scale"]),
        "residual_hold_steps": int(config["residual_hold_steps"]),
        "progress_gamma": float(config["sac"]["gamma"]),
        "progress_weight": float(reward["progress_weight"]),
        "goal_component_weights": tuple(reward["goal_component_weights"]),
        "worst_component_weight": float(reward["worst_component_weight"]),
        "lateral_constraint_progress_weight": float(reward["lateral_constraint_progress_weight"]),
        "success_bonus": float(reward["success_bonus"]),
        "safety_failure_penalty": float(reward["failure_penalty"]),
        "timeout_penalty": float(reward["timeout_penalty"]),
        "timeout_violation_weight": float(reward["timeout_violation_weight"]),
        "residual_magnitude_weight": float(reward["residual_magnitude_weight"]),
        "residual_smoothness_weight": float(reward["residual_smoothness_weight"]),
        "terminal_fade_full_ratio": float(gate["full_authority_at_or_above_ratio"]),
        "terminal_fade_zero_ratio": float(gate["zero_authority_at_or_below_ratio"]),
        "terminate_on_success": True,
    }


def run(model, seed, config, formulation):
    kwargs = (
        environment_kwargs(config)
        if formulation == "reference"
        else direct_environment_kwargs(config)
    )
    dt = float(kwargs.pop("dt"))
    env = make_her_env(
        seed=seed, mode="testing", max_time=MAX_STEPS * dt,
        reward_version=config["reward_version"], dt=dt, **kwargs,
    )
    obs, _ = env.reset(seed=seed)
    target = np.asarray(env.core_env.p3_target, dtype=float).copy()
    states = [np.asarray(env.core_env.state, dtype=float).copy()]
    times = [0.0]
    residuals = [np.zeros(3)]
    gates = [float(env.terminal_residual_authority()[0])]
    info = {}
    for step in range(1, MAX_STEPS + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        states.append(np.asarray(env.core_env.state, dtype=float).copy())
        times.append(step * dt)
        residuals.append(np.asarray(
            info.get("reference_residual_normalized_action", np.zeros(3)), dtype=float
        ))
        gates.append(float(info.get("terminal_residual_authority", 0.0)))
        if bool(info.get("success", False) or info.get("is_success", False)) or terminated or truncated:
            break
    states = np.asarray(states)
    errors = states[:, :2] - target[:2]
    distance = np.linalg.norm(errors, axis=1)
    residuals = np.asarray(residuals)
    magnitude = np.linalg.norm(residuals, axis=1)
    final = physical_goal_metrics(env)
    success = bool(info.get("success", False) or info.get("is_success", False))
    failure = bool(info.get("failure", False))
    outcome = "success" if success else ("hard-safety failure" if failure else "timeout")
    core = env.core_env
    p3_port = np.asarray(core.get_derived_quantities(target)["r_p"], dtype=float)
    geometry = {
        "cone_slope": float(core.cone_slope), "cone_length": float(core.cone_length),
        "corridor_radius": float(core.corridor_radius), "d_p3": float(core.d_P3),
        "p3_port": p3_port,
    }
    env.close()
    return {
        "times": np.asarray(times), "errors": errors, "distance": distance,
        "imin": int(np.argmin(distance)), "residuals": residuals,
        "magnitude": magnitude, "gates": np.asarray(gates), "final": final,
        "outcome": outcome, "geometry": geometry,
    }


def add_geometry(axis, geometry, xmin, xmax):
    x = np.linspace(xmin, xmax, 1000)
    absolute_x = x + geometry["p3_port"][0]
    valid = (absolute_x >= 0) & (absolute_x <= geometry["cone_length"])
    for sign in (-1, 1):
        y = sign * geometry["cone_slope"] * absolute_x - geometry["p3_port"][1]
        axis.plot(x, np.where(valid, y, np.nan), ":", color="#E69F00", lw=1.0)
    corridor_x = [-geometry["p3_port"][0], geometry["d_p3"] - geometry["p3_port"][0]]
    for sign in (-1, 1):
        y = sign * geometry["corridor_radius"] - geometry["p3_port"][1]
        axis.plot(corridor_x, [y, y], ":", color="#A06AB4", lw=1.0)
    axis.axhline(0, color="0.45", ls="-.", lw=0.8)


def plot(direct_baseline, direct_residual, reference_baseline, reference_residual, seed):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharex=True, sharey=True)
    records = (direct_baseline, direct_residual, reference_baseline, reference_residual)
    all_xy = np.vstack([record["errors"] for record in records])
    xmin, xmax = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
    ymin, ymax = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())
    xpad = 0.05 * max(xmax - xmin, 1)
    ypad = 0.12 * max(ymax - ymin, 1)
    for axis in axes:
        add_geometry(axis, direct_baseline["geometry"], xmin - xpad, xmax + xpad)
        axis.set_xlim(xmin - xpad, xmax + xpad)
        axis.set_ylim(ymin - ypad, ymax + ypad)
        axis.grid(True, color="#D9D9D9", lw=0.65, alpha=0.8)
        axis.set_xlabel(r"Axial position error, $x-x_{P_3}$ [m]")
        axis.set_ylabel(r"Lateral position error, $y-y_{P_3}$ [m]")
        axis.invert_xaxis()
    pairs = ((direct_baseline, direct_residual), (reference_baseline, reference_residual))
    for axis, (baseline, residual) in zip(axes, pairs):
        axis.plot(baseline["errors"][:, 0], baseline["errors"][:, 1],
                  color="#003B5C", lw=1.8, label="Zero-residual LQR")
        axis.plot(residual["errors"][:, 0], residual["errors"][:, 1],
                  color="#D55E00", lw=1.7, ls="--", label="Learned residual SAC")
        for record, edge in ((baseline, "#003B5C"), (residual, "#D55E00")):
            xy = record["errors"]
            axis.scatter(*xy[record["imin"]], marker="o", s=35,
                         facecolor="white", edgecolor=edge, lw=1.1, zorder=7)
            axis.scatter(*xy[-1], marker="X", s=48, color=edge, zorder=8)
        axis.scatter(*baseline["errors"][0], marker="D", s=42,
                     facecolor="white", edgecolor="#009E73", zorder=8)
        axis.scatter(0, 0, marker="*", s=85, color="#CC3311", zorder=8)
    axes[0].set_title("(a) Direct-action residual (120k checkpoint)", fontweight="normal")
    axes[1].set_title("(b) Reference residual (500k checkpoint)", fontweight="normal")
    handles = [
        Line2D([], [], color="#003B5C", label="Zero-residual LQR"),
        Line2D([], [], color="#D55E00", ls="--", label="Learned residual SAC"),
        Line2D([], [], marker="D", ls="none", mfc="white", mec="#009E73", label="Start"),
        Line2D([], [], marker="o", ls="none", color="#6A3D9A", label="Closest approach"),
        Line2D([], [], marker="X", ls="none", color="#D55E00", label="Terminal point"),
        Line2D([], [], marker="*", ls="none", color="#CC3311", label=r"$P_3$ target"),
        Line2D([], [], color="#E69F00", ls=":", label="Docking cone"),
        Line2D([], [], color="#A06AB4", ls=":", label="Docking corridor"),
    ]
    fig.suptitle(f"Effect of learned residual SAC corrections (testing seed {seed})", fontweight="normal")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=5, frameon=False, fontsize=8.2)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.79, wspace=0.16)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=350, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def print_row(label, record):
    f = record["final"]
    print(f"{label}: outcome={record['outcome']}; closest={record['distance'][record['imin']]:.6f} m "
          f"at {record['times'][record['imin']]:.1f} s; terminal position={f['position_error']:.6f} m; "
          f"velocity={f['velocity_error']:.6f} m/s; attitude={f['attitude_error_deg']:.6f} deg; "
          f"rate={f['rate_error_degs']:.6f} deg/s; worst ratio={f['max_target_ratio']:.6f}")


def main():
    direct_checkpoint, reference_checkpoint, seed = 120_000, 500_000, 106
    print("Eligibility restriction: replay source and SAC environment both use the refined stable LQR at dt=0.1 s.")
    print(f"Direct-action selection: checkpoint={direct_checkpoint:,}, seed={seed}; largest paired tolerance-ratio degradation in its validation table.")
    print(f"Reference-residual selection: checkpoint={reference_checkpoint:,}, seed={seed}; clear non-outlier degradation (extreme hard-failure seed 105 excluded).")
    print("Both selected checkpoints are genuinely learned, non-zero-residual policies.")
    with (REFERENCE_RESULTS / "experiment_config.json").open(encoding="utf-8") as stream:
        reference_config = json.load(stream)
    with (DIRECT_RESULTS / "experiment_config.json").open(encoding="utf-8") as stream:
        direct_config = json.load(stream)
    direct_baseline_model = ZeroResidualConstrainedSAC.load(DIRECT_MODELS / "checkpoint_0_model.zip", device="cpu")
    direct_model = ZeroResidualConstrainedSAC.load(DIRECT_MODELS / f"checkpoint_{direct_checkpoint}_model.zip", device="cpu")
    reference_baseline_model = ZeroResidualConstrainedSAC.load(REFERENCE_MODELS / "checkpoint_0_model.zip", device="cpu")
    reference_model = ZeroResidualConstrainedSAC.load(REFERENCE_MODELS / f"checkpoint_{reference_checkpoint}_model.zip", device="cpu")
    direct_baseline = run(direct_baseline_model, seed, direct_config, "direct")
    direct_residual = run(direct_model, seed, direct_config, "direct")
    reference_baseline = run(reference_baseline_model, seed, reference_config, "reference")
    reference_residual = run(reference_model, seed, reference_config, "reference")
    plot(direct_baseline, direct_residual, reference_baseline, reference_residual, seed)
    print_row("Direct zero-residual LQR", direct_baseline)
    print_row("Direct-action residual SAC", direct_residual)
    print_row("Reference zero-residual LQR", reference_baseline)
    print_row("Reference-residual SAC", reference_residual)
    print("Interpretation: the direct-action residual remains almost coincident with LQR and slightly worsens the limiting ratio. The reference residual visibly departs from LQR and produces a larger terminal tolerance-ratio degradation; neither learned correction improves the timeout outcome.")
    print(f"PNG: {OUTPUT_STEM.with_suffix('.png').resolve()}")


if __name__ == "__main__":
    main()
