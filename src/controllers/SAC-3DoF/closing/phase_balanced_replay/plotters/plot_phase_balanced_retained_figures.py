"""Create retained-policy figures for the phase-balanced closing SAC.

Outputs
-------
1. One inset-free axes comparing a representative seed at 12,000, 13,000,
   and 20,000 seconds. Each evaluation horizon uses a fixed curve color.
2. The continuation-training learning curve after replay warm-up, with a
   10-episode moving average computed only from post-warm-up episodes.

No training is performed. The representative rollout is regenerated only
because the aggregate evaluation directories contain per-rollout metrics but
not full state histories.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FAMILY_DIR = SCRIPT_DIR.parent
CLOSING_DIR = FAMILY_DIR.parent
VANILLA_DIR = CLOSING_DIR / "vanilla_sac"
SRC_PATH = SCRIPT_DIR.parents[4]
for path in (FAMILY_DIR, CLOSING_DIR, VANILLA_DIR, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_sac_closing as evaluator


EXPERIMENT = "sac_closing_v15_trial16_terminal_phase_balanced_guarded600k_seed0"
MODEL_DIR = FAMILY_DIR / "models" / "multiseed" / EXPERIMENT / "seed_0"
MODEL_PATH = MODEL_DIR / "model_0100000_steps.zip"
VECNORM_PATH = MODEL_DIR / "model_0100000_steps_vecnormalize.pkl"
TRAINING_DIR = FAMILY_DIR / "results" / "multiseed" / EXPERIMENT
LEARNING_CURVE_PATH = TRAINING_DIR / "seed_0" / "learning_curve.csv"
VANILLA_LEARNING_CURVE_PATH = (
    VANILLA_DIR
    / "results"
    / "reward_shaping_trials"
    / "episode_return_learning_curves"
    / "sac_closing_p3augobs_rewardv15_optuna_trial16_400ksteps_seed0_buffer300k"
    / "learning_curve.csv"
)
VANILLA_LEARNING_STARTS = 10_000
CONFIG_PATH = TRAINING_DIR / "experiment_config.json"
RETAINED_CHECKPOINT_STEPS = 100_000
TRANSFER_ROOT = CLOSING_DIR / "results" / "timestep_transfer"
HORIZONS = (12_000, 13_000, 20_000)
EVALUATION_DIRS = {
    horizon: TRANSFER_ROOT
    / f"phase_balanced_guarded100k_seeds0_49_horizon{horizon}"
    for horizon in HORIZONS
}
DEFAULT_OUTPUT_DIR = (
    FAMILY_DIR / "results" / "retained_figures" / "retained_checkpoint"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_available(row: dict[str, str], names: tuple[str, ...]) -> float:
    """Read a numeric field while tolerating minor historical name changes."""

    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"None of these columns exists: {names}")


def load_evaluation_rows() -> dict[int, dict[int, dict[str, str]]]:
    indexed: dict[int, dict[int, dict[str, str]]] = {}
    for horizon, directory in EVALUATION_DIRS.items():
        path = directory / "native_1hz" / "rollout_metrics.csv"
        rows = read_csv(path)
        indexed[horizon] = {int(row["seed"]): row for row in rows}
    return indexed


def select_representative_seed(rows: dict[int, dict[int, dict[str, str]]]):
    """Choose the clearest monotonic-approach, long-horizon-drift example."""

    common = set.intersection(*(set(horizon_rows) for horizon_rows in rows.values()))
    candidates = []
    for seed in sorted(common):
        minimum = {
            horizon: first_available(
                rows[horizon][seed],
                ("minimum_position_error_m", "min_position_error"),
            )
            for horizon in HORIZONS
        }
        terminal_20k = first_available(
            rows[20_000][seed],
            ("final_position_error_m", "final_position_error"),
        )
        if minimum[13_000] < minimum[12_000] and minimum[20_000] < minimum[13_000]:
            drift = terminal_20k - minimum[20_000]
            candidates.append((drift, seed, minimum, terminal_20k))

    if not candidates:
        raise RuntimeError(
            "No common seed showed decreasing minimum error at all three horizons."
        )
    drift, seed, minimum, terminal_20k = max(candidates)
    reason = (
        "monotonically decreasing closest-approach error "
        f"({minimum[12000]:.2f}, {minimum[13000]:.2f}, "
        f"{minimum[20000]:.2f} m) and the largest subsequent 20,000-s drift "
        f"among qualifying seeds ({drift:.2f} m, terminal {terminal_20k:.2f} m)"
    )
    return seed, reason


def run_rollout(model, vecnorm, seed: int, horizon: int) -> dict[str, object]:
    env = evaluator.Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        dt=1.0,
        max_time=float(horizon),
        integrator="rk4",
        normalize_action=True,
        seed=seed,
        closing_reward_version="v15",
    )
    obs, _ = env.reset(seed=seed)
    p3_target = np.asarray(env.core_env.p3_target, dtype=float).copy()
    p3_port = np.asarray(
        env.core_env.get_derived_quantities(p3_target)["r_p"], dtype=float
    )
    info: dict[str, object] = {}
    for _ in range(horizon):
        normalized_obs = evaluator.normalize_observation(vecnorm, obs)
        action, _ = model.predict(normalized_obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action[0])
        if terminated or truncated:
            break

    times, states, *_ = env.get_history_arrays()
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    errors = states[:, :2] - p3_target[:2]
    distances = np.linalg.norm(errors, axis=1)
    minimum_index = int(np.argmin(distances))
    record = {
        "times": times,
        "states": states,
        "errors": errors,
        "p3_target": p3_target,
        "p3_port": p3_port,
        "minimum_index": minimum_index,
        "minimum_position_error_m": float(distances[minimum_index]),
        "terminal_position_error_m": float(distances[-1]),
        "termination_reason": str(info.get("termination_reason", "horizon")),
        "cone_slope": float(env.core_env.cone_slope),
        "cone_length": float(env.core_env.cone_length),
        "corridor_radius": float(env.core_env.corridor_radius),
        "d_p3": float(env.core_env.d_P3),
    }
    env.close()
    return record


def plot_trajectory(records: dict[int, dict[str, object]], seed: int, output_dir: Path):
    fig, axis = plt.subplots(figsize=(9.3, 5.6))
    styles = {
        12_000: ("Blues", "#08519C"),
        13_000: ("Oranges", "#D95F02"),
        20_000: ("Greens", "#238B45"),
    }

    all_errors = np.vstack([records[h]["errors"] for h in HORIZONS])
    x_min, x_max = np.min(all_errors[:, 0]), np.max(all_errors[:, 0])
    x_pad = 0.035 * (x_max - x_min)
    geometry_x = np.linspace(min(x_min - x_pad, -10.0), x_max + x_pad, 1200)
    reference = records[12_000]
    absolute_axial = geometry_x + reference["p3_port"][0]
    valid = (absolute_axial >= 0.0) & (absolute_axial <= reference["cone_length"])
    for sign in (-1.0, 1.0):
        cone_y = (
            sign * reference["cone_slope"] * absolute_axial
            - reference["p3_port"][1]
        )
        axis.plot(
            geometry_x,
            np.where(valid, cone_y, np.nan),
            color="#E69F00",
            linestyle=":",
            linewidth=1.2,
            zorder=0,
        )
    corridor_x = np.asarray(
        [-reference["p3_port"][0], reference["d_p3"] - reference["p3_port"][0]]
    )
    for sign in (-1.0, 1.0):
        y = sign * reference["corridor_radius"] - reference["p3_port"][1]
        axis.plot(
            corridor_x,
            [y, y],
            color="#A06AB4",
            linestyle=":",
            linewidth=1.2,
            zorder=1,
        )
    axis.axhline(0.0, color="0.45", linestyle="-.", linewidth=0.9, zorder=0)

    # Plot the longest rollout first so shorter coincident prefixes remain visible.
    for zorder, horizon in enumerate(reversed(HORIZONS), start=2):
        record = records[horizon]
        errors = record["errors"]
        _, terminal_color = styles[horizon]
        axis.plot(
            errors[:, 0], errors[:, 1], color=terminal_color,
            linewidth=2.0, zorder=zorder
        )
        minimum = errors[record["minimum_index"]]
        axis.scatter(
            minimum[0], minimum[1], marker="o", s=47, facecolor=terminal_color,
            edgecolor="white", linewidth=0.8, zorder=8
        )
        axis.scatter(
            errors[-1, 0], errors[-1, 1], marker="X", s=62,
            facecolor=terminal_color, edgecolor="white", linewidth=0.7, zorder=9
        )

    start = records[12_000]["errors"][0]
    axis.scatter(
        start[0], start[1], marker="D", s=52, facecolor="white",
        edgecolor="0.15", linewidth=1.2, zorder=10
    )
    axis.scatter(
        0.0, 0.0, marker="*", s=120, facecolor="#CC3311",
        edgecolor="white", linewidth=0.6, zorder=11
    )

    horizon_handles = [
        Line2D([0], [0], color=styles[h][1], linewidth=2.2, label=f"{h:,} s")
        for h in HORIZONS
    ]
    marker_handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor="0.15", label="Start"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.35", markeredgecolor="white", label="Closest approach"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="0.35", markeredgecolor="white", label="Terminal point"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#CC3311", markeredgecolor="white", markersize=11, label="$P_3$ target"),
    ]
    first_legend = axis.legend(
        handles=horizon_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
    )
    axis.add_artist(first_legend)
    axis.legend(
        handles=marker_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.095),
        ncol=4,
        frameon=False,
    )
    axis.set_title(
        "Representative $P_3$-relative trajectory of the phase-balanced SAC policy "
        f"(testing seed {seed})",
        fontweight="bold",
        pad=57,
    )
    axis.set_xlabel("Axial position error to P3 [m]", labelpad=10)
    axis.set_ylabel("Lateral position error to P3 [m]", labelpad=12)
    axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.margins(x=0.035, y=0.08)
    axis.invert_xaxis()
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.73)

    stem = output_dir / "phase_balanced_representative_p3_trajectory_three_horizons"
    fig.savefig(stem.with_suffix(".png"), dpi=350, facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


def load_post_learning_episodes():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    warmup_steps = int(config["actual_replay_warmup_interaction_steps"])
    rows = read_csv(LEARNING_CURVE_PATH)
    episodes = np.asarray([int(row["episode"]) for row in rows])
    timesteps = np.asarray([int(row["timestep"]) for row in rows])
    returns = np.asarray([float(row["episode_return"]) for row in rows])

    # The warm-up ended at interaction 72,001. Episodes ending at or before
    # 72,000 saw no updates; episode 7 is the first completed episode that
    # contains post-warm-up learning interactions.
    mask = timesteps >= warmup_steps
    return episodes[mask], returns[mask], warmup_steps, config


def plot_learning_curve(output_dir: Path):
    episodes, returns, warmup_steps, config = load_post_learning_episodes()
    vanilla_rows = read_csv(VANILLA_LEARNING_CURVE_PATH)
    vanilla_episodes_all = np.asarray(
        [int(row["episode"]) for row in vanilla_rows]
    )
    vanilla_timesteps = np.asarray(
        [int(row["timestep"]) for row in vanilla_rows]
    )
    vanilla_returns_all = np.asarray(
        [float(row["episode_return"]) for row in vanilla_rows]
    )
    vanilla_mask = vanilla_timesteps >= VANILLA_LEARNING_STARTS
    vanilla_episodes = vanilla_episodes_all[vanilla_mask]
    vanilla_returns = vanilla_returns_all[vanilla_mask]
    all_rows = read_csv(LEARNING_CURVE_PATH)
    retained_episode = min(
        int(row["episode"])
        for row in all_rows
        if int(row["timestep"]) >= RETAINED_CHECKPOINT_STEPS
    )
    window = 10
    if returns.size < window:
        raise ValueError("Fewer than ten post-learning episodes are available.")
    kernel = np.ones(window, dtype=float) / window
    moving_average = np.convolve(returns, kernel, mode="valid")
    moving_episodes = episodes[window - 1 :]
    vanilla_moving_average = np.convolve(vanilla_returns, kernel, mode="valid")
    vanilla_moving_episodes = vanilla_episodes[window - 1 :]

    fig, axis = plt.subplots(figsize=(8.1, 4.8))
    axis.plot(
        vanilla_episodes,
        vanilla_returns,
        color="#E6B07A",
        linewidth=0.8,
        alpha=0.48,
        label="Vanilla individual episode",
    )
    axis.plot(
        vanilla_moving_episodes,
        vanilla_moving_average,
        color="#D55E00",
        linewidth=2.0,
        label="Vanilla 10-episode moving average",
    )
    axis.plot(
        episodes,
        returns,
        color="#8EC7E8",
        linewidth=0.8,
        alpha=0.62,
        label="Phase-balanced individual episode",
    )
    axis.plot(
        moving_episodes,
        moving_average,
        color="#005A8D",
        linewidth=2.0,
        label="Phase-balanced 10-episode moving average",
    )
    axis.axvline(
        retained_episode,
        color="#7A3E9D",
        linestyle=":",
        linewidth=1.5,
        label="Retained checkpoint (100k steps)",
    )
    axis.set_title(
        "Learning curves of the vanilla and phase-balanced SAC policies",
        fontweight="normal",
        pad=10,
    )
    axis.set_xlabel("Episode number [-]")
    axis.set_ylabel("Episode return [-]")
    axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False, fontsize=8.5)
    axis.margins(x=0.015)
    fig.tight_layout()

    stem = output_dir / "phase_balanced_learning_curve_from_learning_start"
    fig.savefig(stem.with_suffix(".png"), dpi=350, facecolor="white")
    plt.close(fig)
    return (
        stem.with_suffix(".png"),
        int(episodes[0]),
        int(episodes[9]),
        warmup_steps,
        config,
    )


def save_trajectory_data(output_dir: Path, seed: int, records):
    payload = {"seed": np.asarray(seed), "horizons_s": np.asarray(HORIZONS)}
    for horizon, record in records.items():
        prefix = f"h{horizon}"
        payload[f"{prefix}_time_s"] = record["times"]
        payload[f"{prefix}_state"] = record["states"]
        payload[f"{prefix}_p3_relative_xy_m"] = record["errors"]
        payload[f"{prefix}_p3_target"] = record["p3_target"]
    path = output_dir / "phase_balanced_representative_trajectory_data.npz"
    np.savez_compressed(path, **payload)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    required = [
        MODEL_PATH,
        VECNORM_PATH,
        LEARNING_CURVE_PATH,
        CONFIG_PATH,
    ]
    required.extend(
        directory / "native_1hz" / "rollout_metrics.csv"
        for directory in EVALUATION_DIRS.values()
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(map(str, missing)))
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_rows = load_evaluation_rows()
    seed, selection_reason = select_representative_seed(evaluation_rows)
    evaluator.MODEL_PATH = str(MODEL_PATH)
    evaluator.VECNORM_PATH = str(VECNORM_PATH)
    with contextlib.redirect_stdout(io.StringIO()):
        vecnorm, dummy_env = evaluator.load_vecnormalize_stats()
        model = evaluator.SAC.load(str(MODEL_PATH))
    records = {
        horizon: run_rollout(model, vecnorm, seed, horizon) for horizon in HORIZONS
    }
    dummy_env.close()

    trajectory_png = plot_trajectory(records, seed, output_dir)
    data_path = save_trajectory_data(output_dir, seed, records)
    learning_png, first_episode, first_average_episode, warmup, config = (
        plot_learning_curve(output_dir)
    )

    print("Detected evaluation files:")
    for horizon, directory in EVALUATION_DIRS.items():
        print(f"  {horizon:,} s: {directory / 'native_1hz' / 'rollout_metrics.csv'}")
    print(f"Detected learning-curve data: {LEARNING_CURVE_PATH}")
    print(f"Detected experiment configuration: {CONFIG_PATH}")
    print(f"Selected representative seed: {seed}")
    print(f"Selection reason: {selection_reason}")
    for horizon in HORIZONS:
        record = records[horizon]
        print(
            f"  {horizon:,} s: minimum={record['minimum_position_error_m']:.6f} m, "
            f"terminal={record['terminal_position_error_m']:.6f} m, "
            f"termination={record['termination_reason']}"
        )
    print(
        "Learning onset: dedicated replay warm-up ended at interaction step "
        f"{warmup:,}; first included completed episode={first_episode}."
    )
    print(
        "Source trial learning_starts="
        f"{config['trial_16_params']['learning_starts']:,}, but it does not define "
        "the continuation cutoff because continuation gradients were explicitly "
        "disabled during replay warm-up."
    )
    print(f"First 10-episode moving-average point: episode {first_average_episode}")
    print(f"Trajectory PNG: {trajectory_png}")
    print(f"Trajectory data: {data_path}")
    print(f"Learning-curve PNG: {learning_png}")
    print(f"Learning-curve PDF: {learning_pdf}")
    print("Assumption: CoM x/y state errors are plotted directly relative to the sampled P3 target.")
    print("No inset axes are created by this script.")


if __name__ == "__main__":
    main()
