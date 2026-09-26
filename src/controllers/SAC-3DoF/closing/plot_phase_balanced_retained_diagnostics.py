"""Publication diagnostics for the retained 100k phase-balanced closing SAC.

The aggregate figures use the saved deterministic seeds 0--49 CSV files.
Full histories are regenerated only for selected seeds because the aggregate
evaluation intentionally stores metrics rather than time series. No training
is performed and no inset/embedded axes are created.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2]
for path in (HERE, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_sac_closing as evaluator


EXPERIMENT = "sac_closing_v15_trial16_terminal_phase_balanced_guarded600k_seed0"
MODEL_DIR = HERE / "models" / "multiseed" / EXPERIMENT / "seed_0"
MODEL_PATH = MODEL_DIR / "model_0100000_steps.zip"
VECNORM_PATH = MODEL_DIR / "model_0100000_steps_vecnormalize.pkl"
RESULT_ROOT = HERE / "results" / "timestep_transfer"
HORIZONS = (12_000, 13_000, 20_000)
EVAL_DIRS = {
    h: RESULT_ROOT / f"phase_balanced_guarded100k_seeds0_49_horizon{h}"
    for h in HORIZONS
}
DEFAULT_OUTPUT = (
    HERE / "results" / "publication_figures" / "phase_balanced_retained100k_diagnostics"
)

# Match the trajectory blue used by the vanilla V15 state-convergence figure
# so the two controller figures are directly comparable.
BLUE = "#003B5C"
LIGHT_BLUE = "#9ECAE1"
ORANGE = "#E69F00"
PURPLE = "#A06AB4"
GREEN = "#009E73"
RED = "#D55E00"


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row, *names):
    for name in names:
        if row.get(name) not in (None, ""):
            return float(row[name])
    raise KeyError(f"Missing all alternative columns: {names}")


def truth(row, *names):
    return bool(round(number(row, *names)))


def load_metrics():
    result = {}
    paths = {}
    for horizon, directory in EVAL_DIRS.items():
        path = directory / "native_1hz" / "rollout_metrics.csv"
        rows = read_csv(path)
        result[horizon] = {int(row["seed"]): row for row in rows}
        paths[horizon] = path
    common = set.intersection(*(set(rows) for rows in result.values()))
    if common != set(range(50)):
        raise ValueError("Expected identical evaluation seeds 0--49 at every horizon.")
    return result, paths


def robust_representative(rows, seeds):
    """Return the seed closest to the bivariate group median in robust units."""
    values = np.asarray(
        [[number(rows[s], "minimum_position_error_m"),
          number(rows[s], "final_position_error_m")] for s in seeds]
    )
    center = np.median(values, axis=0)
    scale = np.median(np.abs(values - center), axis=0)
    scale[scale == 0.0] = 1.0
    index = int(np.argmin(np.linalg.norm((values - center) / scale, axis=1)))
    return int(seeds[index])


def select_seeds(metrics):
    rows = metrics[20_000]
    timeout_seeds = [s for s, r in rows.items() if truth(r, "timeout")]
    failure_seeds = [s for s, r in rows.items() if truth(r, "safety_failure")]
    typical = robust_representative(rows, timeout_seeds)
    closest = min(
        timeout_seeds,
        key=lambda s: number(rows[s], "minimum_position_error_m"),
    )
    geometry = robust_representative(rows, failure_seeds)
    return typical, closest, geometry, failure_seeds


def rollout(model, vecnorm, seed, horizon=20_000):
    env = evaluator.Docking3DoFGymEnv(
        phase="closing", mode="testing", dt=1.0, max_time=float(horizon),
        integrator="rk4", normalize_action=True, seed=seed,
        closing_reward_version="v15",
    )
    obs, _ = env.reset(seed=seed)
    target = np.asarray(env.core_env.p3_target, dtype=float).copy()
    target_port = np.asarray(
        env.core_env.get_derived_quantities(target)["r_p"], dtype=float
    )
    tolerance = np.asarray(env.core_env.p3_target_tolerance, dtype=float).copy()
    limits = np.asarray(env.core_env.action_high, dtype=float).copy()
    info = {}
    for _ in range(horizon):
        norm_obs = evaluator.normalize_observation(vecnorm, obs)
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action[0])
        if terminated or truncated:
            break
    times, states, raw_actions, clipped_actions, _ = env.get_history_arrays()
    times, states = np.asarray(times), np.asarray(states)
    errors = states - target
    errors[:, 4] = np.arctan2(np.sin(errors[:, 4]), np.cos(errors[:, 4]))
    distances = np.linalg.norm(errors[:, :2], axis=1)
    imin = int(np.argmin(distances))
    record = {
        "seed": seed, "times": times, "states": states, "errors": errors,
        "distances": distances, "imin": imin, "target": target,
        "target_port": target_port, "tolerance": tolerance,
        "normalized_actions": np.asarray(raw_actions) / limits,
        "termination_reason": str(info.get("termination_reason", "unknown")),
        "cone_slope": float(env.core_env.cone_slope),
        "cone_length": float(env.core_env.cone_length),
        "corridor_radius": float(env.core_env.corridor_radius),
        "d_p3": float(env.core_env.d_P3),
    }
    env.close()
    return record


def add_geometry(ax, record, xgrid):
    absolute_x = xgrid + record["target_port"][0]
    valid = (absolute_x >= 0) & (absolute_x <= record["cone_length"])
    for sign in (-1, 1):
        y = sign * record["cone_slope"] * absolute_x - record["target_port"][1]
        ax.plot(xgrid, np.where(valid, y, np.nan), ":", color=ORANGE, lw=1.2)
    corridor_x = [
        -record["target_port"][0],
        record["d_p3"] - record["target_port"][0],
    ]
    for sign in (-1, 1):
        y = sign * record["corridor_radius"] - record["target_port"][1]
        ax.plot(corridor_x, [y, y], ":", color=PURPLE, lw=1.2)
    ax.axhline(0, color="0.45", ls="-.", lw=0.9)


def save(fig, output, stem):
    png = output / f"{stem}.png"
    fig.savefig(png, dpi=350, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return (png,)


def figure_seed_trajectories(records, output):
    labels = ("Representative timeout", "Smallest closest approach", "Geometry failure")
    all_xy = np.vstack([r["errors"][:, :2] for r in records])
    xmin, xmax = np.min(all_xy[:, 0]), np.max(all_xy[:, 0])
    xgrid = np.linspace(min(xmin - 20, -20), xmax + 20, 1200)
    ylimit = max(185, 1.08 * np.max(np.abs(all_xy[:, 1])))
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharex=True, sharey=True)
    for ax, record, label in zip(axes, records, labels):
        xy, imin = record["errors"][:, :2], record["imin"]
        add_geometry(ax, record, xgrid)
        ax.plot(xy[:, 0], xy[:, 1], color=BLUE, lw=1.7, label="Trajectory")
        ax.scatter(*xy[0], marker="D", s=42, fc="white", ec=GREEN, lw=1.3, zorder=5)
        ax.scatter(*xy[imin], marker="o", s=48, color="#6A3D9A", ec="white", lw=0.6, zorder=6)
        for marker_time, marker, color in (
            (12_000, "s", "#4477AA"),
            (13_000, "^", "#EE7733"),
        ):
            marker_index = int(np.argmin(np.abs(record["times"] - marker_time)))
            ax.scatter(
                *xy[marker_index],
                marker=marker,
                s=52,
                facecolor=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=7,
            )
        ax.scatter(*xy[-1], marker="X", s=55, color=RED, ec="white", lw=0.6, zorder=7)
        ax.scatter(0, 0, marker="*", s=95, color="#CC3311", ec="white", lw=0.5, zorder=8)
        ax.set_title(f"{label} (seed {record['seed']})", fontweight="normal")
        ax.set_xlabel(r"Axial position error to $P_3$ [m]")
        ax.grid(True, color="#D9D9D9", lw=0.65, alpha=0.8)
        ax.set_xlim(xmax + 20, min(xmin - 20, -20))
        ax.set_ylim(-ylimit, ylimit)
    axes[0].set_ylabel(r"Lateral position error to $P_3$ [m]")
    handles = [
        Line2D([0], [0], color=BLUE, label="Trajectory"),
        Line2D([0], [0], marker="D", ls="none", mfc="white", mec=GREEN, label="Start"),
        Line2D([0], [0], marker="o", ls="none", color="#6A3D9A", label="Closest approach"),
        Line2D([0], [0], marker="s", ls="none", color="#4477AA", label="12,000 s"),
        Line2D([0], [0], marker="^", ls="none", color="#EE7733", label="13,000 s"),
        Line2D([0], [0], marker="X", ls="none", color=RED, label="Terminal point"),
        Line2D([0], [0], marker="*", ls="none", color="#CC3311", markersize=10, label="$P_3$ target"),
        Line2D([0], [0], color=ORANGE, ls=":", label="Docking cone"),
        Line2D([0], [0], color=PURPLE, ls=":", label="Docking corridor"),
    ]
    fig.suptitle("Representative retained-policy trajectories at the 20,000-s horizon", fontweight="normal", y=1.04)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=9, frameon=False)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.13, top=0.78, wspace=0.08)
    return save(fig, output, "phase_balanced_representative_seed_trajectories_20000s")


def figure_error_histories(record, output):
    seed = int(record["seed"])
    t, e, tol = record["times"], record["errors"], record["tolerance"]
    plot_values = [
        np.abs(e[:, 0]),
        np.abs(e[:, 1]),
        np.abs(e[:, 2]),
        np.abs(e[:, 3]),
        np.abs(np.rad2deg(e[:, 4])),
        np.abs(np.rad2deg(e[:, 5])),
    ]
    plot_tol = [tol[0], tol[1], tol[2], tol[3], np.rad2deg(tol[4]), np.rad2deg(tol[5])]
    labels = [
        r"Axial position error, $|e_x|$ [m]",
        r"Lateral position error, $|e_y|$ [m]",
        r"Axial velocity error, $|e_{v_x}|$ [m/s]",
        r"Lateral velocity error, $|e_{v_y}|$ [m/s]",
        r"Attitude error, $|e_{\psi}|$ [deg]",
        r"Angular-rate error, $|e_{\omega}|$ [deg/s]",
    ]
    tolerance_labels = (
        "Tolerance: 20 m",
        "Tolerance: 2 m",
        "Tolerance: 0.01 m/s",
        "Tolerance: 0.01 m/s",
        "Tolerance: 0.5 deg",
        "Tolerance: 0.05 deg/s",
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.2), sharex=True)
    closest_time = t[record["imin"]]
    for index, (ax, values, threshold, ylabel) in enumerate(zip(axes.flat, plot_values, plot_tol, labels)):
        ax.plot(t, values, color=BLUE, lw=1.35)
        ax.axhline(threshold, color=GREEN, ls="--", lw=1.0)
        ax.axvline(closest_time, color="#CC6677", ls=":", lw=1.0, alpha=0.75)
        ax.axvline(12_000, color="#4477AA", ls=":", lw=1.0, alpha=0.55)
        ax.axvline(13_000, color="#EE7733", ls=":", lw=1.0, alpha=0.55)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#D9D9D9", lw=0.65, alpha=0.8)
        if index in (3, 4, 5):
            ax.text(
                0.98,
                threshold,
                tolerance_labels[index],
                transform=ax.get_yaxis_transform(),
                color=GREEN,
                fontsize=8,
                ha="right",
                va="bottom",
            )
        else:
            ax.text(
                0.02,
                threshold,
                tolerance_labels[index],
                color=GREEN,
                fontsize=8,
                va="bottom",
            )
        ax.text(0.015, 0.94, f"({chr(97 + index)})", transform=ax.transAxes, va="top", fontweight="bold")
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#CC6677",
            ls=":",
            label=f"Closest approach ({closest_time:,.0f} s)".replace(",", "."),
        ),
        Line2D([0], [0], color="#4477AA", ls=":", label="12,000-s horizon"),
        Line2D([0], [0], color="#EE7733", ls=":", label="13,000-s horizon"),
    ]
    fig.suptitle(
        f"Closing-phase phase balanced replay reward SAC state convergence (testing seed {seed})",
        fontweight="normal",
    )
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return save(fig, output, f"phase_balanced_seed{seed}_p3_error_histories")


def paired_summary(metrics):
    minimum = np.asarray([[number(metrics[h][s], "minimum_position_error_m") for h in HORIZONS] for s in range(50)])
    terminal = np.asarray([[number(metrics[h][s], "final_position_error_m") for h in HORIZONS] for s in range(50)])
    return minimum, terminal


def figure_paired(metrics, output):
    minimum, terminal = paired_summary(metrics)
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))
    for panel, (ax, values, ylabel) in enumerate(zip(axes, (minimum, terminal), ("Minimum position error [m]", "Terminal position error [m]"))):
        for row in values:
            ax.plot(x, row, color=LIGHT_BLUE, alpha=0.48, lw=0.75)
        means = np.mean(values, axis=0)
        ax.plot(x, means, color=BLUE, marker="o", ms=6, lw=2.2, label="Mean")
        ax.set_xticks(x, ["12,000", "13,000", "20,000"])
        ax.set_xlabel("Evaluation horizon [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#D9D9D9", lw=0.65, alpha=0.8)
        ax.text(0.015, 0.96, f"({chr(97 + panel)})", transform=ax.transAxes, va="top", fontweight="bold")
        ax.legend(loc="best", frameon=False)
    fig.suptitle(r"Per-seed evolution of minimum and terminal $P_3$-position error across horizons", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    paths = save(fig, output, "phase_balanced_paired_horizon_comparison")
    summary = {
        "minimum_means": np.mean(minimum, axis=0),
        "terminal_means": np.mean(terminal, axis=0),
        "minimum_improved_12_to_13": int(np.sum(minimum[:, 1] < minimum[:, 0])),
        "minimum_improved_13_to_20": int(np.sum(minimum[:, 2] < minimum[:, 1])),
        "terminal_worsened_12_to_13": int(np.sum(terminal[:, 1] > terminal[:, 0])),
        "terminal_worsened_13_to_20": int(np.sum(terminal[:, 2] > terminal[:, 1])),
        "terminal_change_13_to_20": terminal[:, 2] - terminal[:, 1],
    }
    return paths, summary


def termination_counts(rows):
    timeout = sum(truth(row, "timeout") for row in rows.values())
    geometry = sum(
        truth(row, "safety_failure") and "geometry" in row.get("violated_constraint_counts", row.get("termination_reason", ""))
        for row in rows.values()
    )
    return timeout, geometry


def figure_termination(metrics, output):
    counts = termination_counts(metrics[20_000])
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    bars = ax.bar(["Timeout", "Geometry violation"], counts, color=[BLUE, RED], width=0.62)
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_ylabel("Number of rollouts [-]")
    ax.set_ylim(0, max(counts) * 1.16)
    ax.set_title("Termination modes at the 20,000-s horizon", fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", lw=0.65, alpha=0.8)
    fig.tight_layout()
    return save(fig, output, "phase_balanced_termination_modes_20000s"), counts


def residence_interval(record):
    distance, times, imin = record["distances"], record["times"], record["imin"]
    threshold = 1.1 * distance[imin]
    left = right = imin
    while left > 0 and distance[left - 1] <= threshold:
        left -= 1
    while right + 1 < len(distance) and distance[right + 1] <= threshold:
        right += 1
    return left, right, threshold, float(times[right] - times[left])


def figure_actuation_residence(record, output):
    times, actions = record["times"], record["normalized_actions"]
    left, right, threshold, duration = residence_interval(record)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    colors = (BLUE, ORANGE, GREEN)
    labels = ("Axial acceleration", "Lateral acceleration", "Angular acceleration")
    for index, (color, label) in enumerate(zip(colors, labels)):
        axes[0].plot(times, actions[:, index], color=color, lw=1.0, label=label)
    axes[0].axhline(1, color="0.35", ls="--", lw=1.0)
    axes[0].axhline(-1, color="0.35", ls="--", lw=1.0, label=r"Limits, $\pm1$")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Normalized command [-]")
    axes[0].set_title("(a) Normalized actuator commands", loc="left", fontweight="bold")
    axes[0].legend(loc="best", frameon=False, fontsize=8)

    axes[1].plot(times, record["distances"], color=BLUE, lw=1.4, label="Position error")
    axes[1].axhline(threshold, color=GREEN, ls="--", lw=1.0, label=r"$1.1r_{\min}$")
    axes[1].axvspan(times[left], times[right], color=GREEN, alpha=0.18, label="Close-proximity residence")
    imin = record["imin"]
    axes[1].scatter(times[imin], record["distances"][imin], color=RED, s=40, zorder=4, label="Minimum")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel(r"$P_3$-relative position error [m]")
    axes[1].set_title("(b) Residence near closest approach", loc="left", fontweight="bold")
    axes[1].legend(loc="best", frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(True, color="#D9D9D9", lw=0.65, alpha=0.8)
    fig.suptitle(f"Actuator usage and close-proximity residence before drift (seed {record['seed']})", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fractions = np.mean(np.abs(actions) >= 0.95, axis=0)
    return save(fig, output, "phase_balanced_actuation_and_proximity_residence"), fractions, duration, threshold


def save_histories(output, records):
    payload = {}
    for seed, record in records.items():
        for key in ("times", "states", "errors", "distances", "normalized_actions", "target", "tolerance"):
            payload[f"seed{seed}_{key}"] = record[key]
    path = output / "phase_balanced_selected_rollout_histories.npz"
    np.savez_compressed(path, **payload)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics, metric_paths = load_metrics()
    typical, closest, geometry, failure_seeds = select_seeds(metrics)

    evaluator.MODEL_PATH = str(MODEL_PATH)
    evaluator.VECNORM_PATH = str(VECNORM_PATH)
    with contextlib.redirect_stdout(io.StringIO()):
        vecnorm, dummy = evaluator.load_vecnormalize_stats()
        model = evaluator.SAC.load(str(MODEL_PATH))
    selected = list(dict.fromkeys([typical, closest, geometry, 3]))
    records = {seed: rollout(model, vecnorm, seed) for seed in selected}
    dummy.close()

    outputs = []
    outputs += figure_seed_trajectories([records[typical], records[closest], records[geometry]], output)
    outputs += figure_error_histories(records[3], output)
    paired_paths, paired = figure_paired(metrics, output)
    outputs += paired_paths
    termination_paths, counts = figure_termination(metrics, output)
    outputs += termination_paths
    actuation_paths, saturation, residence, residence_threshold = figure_actuation_residence(records[3], output)
    outputs += actuation_paths
    history_path = save_histories(output, records)

    print("Input evaluation files:")
    for horizon, path in metric_paths.items():
        print(f"  {horizon:,} s: {path}")
    print(f"Model: {MODEL_PATH}")
    print(f"VecNormalize: {VECNORM_PATH}")
    print(f"Representative timeout seed: {typical} (closest to timeout-group median minimum/terminal errors)")
    print(f"Smallest closest-approach timeout seed: {closest}")
    print(f"Representative geometry-failure seed: {geometry} (closest to failure-group median minimum/terminal errors)")
    print(f"Confirmed geometry-failure seeds: {failure_seeds}")
    for seed in (typical, closest, geometry):
        r = records[seed]
        print(f"  seed {seed}: minimum={r['distances'][r['imin']]:.6f} m, terminal={r['distances'][-1]:.6f} m, outcome={r['termination_reason']}")
    print(f"20,000-s termination counts: timeout={counts[0]}, geometry violation={counts[1]}")
    print("Paired minimum-error means [12k, 13k, 20k] m: " + np.array2string(paired["minimum_means"], precision=3))
    print("Paired terminal-error means [12k, 13k, 20k] m: " + np.array2string(paired["terminal_means"], precision=3))
    print(f"Minimum error improved: 12k->13k for {paired['minimum_improved_12_to_13']}/50; 13k->20k for {paired['minimum_improved_13_to_20']}/50")
    print(f"Terminal error worsened: 12k->13k for {paired['terminal_worsened_12_to_13']}/50; 13k->20k for {paired['terminal_worsened_13_to_20']}/50")
    delta = paired["terminal_change_13_to_20"]
    print(f"20k terminal deterioration: median change={np.median(delta):.3f} m; positive for {np.sum(delta > 0)}/50 seeds")
    r3 = records[3]
    print(f"Seed 3 closest approach: {r3['distances'][r3['imin']]:.6f} m at t={r3['times'][r3['imin']]:.1f} s; terminal={r3['distances'][-1]:.6f} m")
    print("Seed 3 saturation fractions at |u|>=0.95 [axial, lateral, angular]: " + np.array2string(saturation, precision=6))
    print(f"Seed 3 residence within 1.1*r_min (threshold {residence_threshold:.6f} m): {residence:.1f} s contiguous around closest approach")
    print("Assumption: logged raw body commands divided by physical command limits are the normalized commands supplied by the policy.")
    print("No figure contains an inset or embedded axes.")
    print(f"Saved histories: {history_path}")
    print("Saved figures:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
