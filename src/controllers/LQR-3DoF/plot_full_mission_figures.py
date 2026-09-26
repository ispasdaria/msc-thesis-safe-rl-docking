"""Create the two publication figures for the held-out full-mission campaign.

The campaign CSV contains all 600 mission outcomes, but the continuous state
histories were intentionally not retained during the large evaluation.  This
script therefore selects a deterministic, geometry-spanning subset of the 328
successful missions, reruns only those seeds with the frozen campaign
configuration, and caches compact state histories before making:

1. full-mission docking-port x-y trajectories; and
2. representative full-mission position and velocity time histories for two
   successful seeds.

Both PNG and PDF versions are written.  Subsequent runs reuse the cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


LQR_ROOT = Path(__file__).resolve().parent
SRC_ROOT = LQR_ROOT.parents[1]
if str(LQR_ROOT) not in sys.path:
    sys.path.insert(0, str(LQR_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from envs.docking3dof.core import Docking3DoFEnv
from run_lqr_full_mission import run_full_mission


CAMPAIGN_DIR = (
    LQR_ROOT
    / "results"
    / "lqr_mission"
    / "heldout_300_899_dtclose_0p1_dtfinal_1p0"
)
DEFAULT_CSV = CAMPAIGN_DIR / "full_mission_rollouts.csv"
DEFAULT_CACHE_DIR = CAMPAIGN_DIR / "representative_histories"
DEFAULT_OUTPUT_DIR = CAMPAIGN_DIR / "figures"
DEFAULT_ADDITIONAL_TIME_HISTORY_SEED = 306

CHASER_PORT_BODY = np.array([4.1, 0.0])
TARGET_PORT_ORBITAL = np.array([20.0, 0.0])
PHASE_COLORS = {"closing": "#0072B2", "final": "#D55E00"}
TRAJECTORY_COLORS = ("#0072B2", "#D55E00", "#CC79A7")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=3,
        help="Number of successful missions in the geometry-spanning subset.",
    )
    parser.add_argument(
        "--additional-time-history-seed",
        type=int,
        default=DEFAULT_ADDITIONAL_TIME_HISTORY_SEED,
        help=(
            "Successful campaign seed for the additional time-history figure "
            f"(default: {DEFAULT_ADDITIONAL_TIME_HISTORY_SEED})."
        ),
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Regenerate cached histories even when they already exist.",
    )
    return parser.parse_args()


def read_success_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        records = [
            record
            for record in csv.DictReader(handle)
            if record.get("mission_success", "").strip().lower() == "true"
        ]
    if not records:
        raise ValueError(f"No successful missions found in {path}")
    return records


def port_position(states: np.ndarray) -> np.ndarray:
    """Return vectorized chaser-port position relative to the target port."""
    states = np.asarray(states, dtype=float)
    psi = states[:, 4]
    chaser_x, chaser_y = CHASER_PORT_BODY
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    # Match Docking3DoFEnv.rotation_body_to_orbital_2d. At psi = 0,
    # the chaser body +x axis points opposite to the orbital +x axis.
    chaser_offset = np.column_stack(
        (
            -cos_psi * chaser_x + sin_psi * chaser_y,
            -sin_psi * chaser_x - cos_psi * chaser_y,
        )
    )
    return states[:, :2] + chaser_offset - TARGET_PORT_ORBITAL


def port_velocity(states: np.ndarray) -> np.ndarray:
    """Return vectorized chaser-port velocity relative to the target port."""
    states = np.asarray(states, dtype=float)
    psi = states[:, 4]
    omega = states[:, 5]
    chaser_x, chaser_y = CHASER_PORT_BODY
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    offset_x = -cos_psi * chaser_x + sin_psi * chaser_y
    offset_y = -sin_psi * chaser_x - cos_psi * chaser_y
    rotational_velocity = np.column_stack((-omega * offset_y, omega * offset_x))
    return states[:, 2:4] + rotational_velocity


def sampled_geometry(record: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the seed's P2 state and sampled P3 target without propagation."""
    seed = int(record["seed"])
    env = Docking3DoFEnv(
        phase="closing", seed=seed, dt=0.1, max_time=12000.0
    )
    p2_state = env.reset(mode="testing")
    return p2_state.copy(), env.p3_target.copy()


def select_geometry_spanning_subset(
    records: list[dict[str, str]], count: int
) -> tuple[list[dict[str, str]], int]:
    """Select a deterministic farthest-point subset over mission geometry.

    Features are P2 and P3 port positions plus closing duration.  The first
    point is the campaign medoid (the representative time-history mission),
    after which greedy farthest-point sampling fills out the geometric spread.
    """
    if count <= 0:
        raise ValueError("--num-trajectories must be positive")
    count = min(count, len(records))

    features = []
    for record in records:
        p2_state, p3_state = sampled_geometry(record)
        p2_port = port_position(p2_state[None, :])[0]
        p3_port = port_position(p3_state[None, :])[0]
        duration = float(record["closing_elapsed_time_s"])
        features.append([*p2_port, *p3_port, duration])

    feature_array = np.asarray(features, dtype=float)
    scales = feature_array.std(axis=0)
    scales[scales == 0.0] = 1.0
    normalized = (feature_array - feature_array.mean(axis=0)) / scales

    selected = [int(np.argmin(np.linalg.norm(normalized, axis=1)))]
    while len(selected) < count:
        distances = np.linalg.norm(
            normalized[:, None, :] - normalized[selected][None, :, :], axis=2
        )
        nearest_selected = distances.min(axis=1)
        nearest_selected[selected] = -np.inf
        selected.append(int(np.argmax(nearest_selected)))

    return [records[index] for index in selected], int(records[selected[0]]["seed"])


def cache_path(cache_dir: Path, seed: int) -> Path:
    return cache_dir / f"seed{seed:04d}_full_mission.npz"


def create_history_cache(record: dict[str, str], path: Path) -> None:
    seed = int(record["seed"])
    print(f"rerunning successful campaign seed {seed} ...", flush=True)
    runner, result = run_full_mission(
        seed=seed,
        mode="testing",
        closing_max_time=12000.0,
        final_max_time=5000.0,
        integrator="rk4",
        closing_dt=0.1,
        final_dt=1.0,
        retain_history=True,
    )
    if not result["mission_success"]:
        raise RuntimeError(
            f"Seed {seed} no longer reproduces its recorded successful outcome."
        )

    closing_times, closing_states, *_ = runner.closing_env.get_history_arrays()
    final_times, final_states, *_ = runner.final_env.get_history_arrays()
    metadata = {
        "seed": seed,
        "mission_success": True,
        "closing_dt_s": 0.1,
        "final_dt_s": 1.0,
        "closing_elapsed_time_s": float(closing_times[-1]),
        "final_elapsed_time_s": float(final_times[-1]),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        closing_times=closing_times,
        closing_states=closing_states,
        final_times=final_times,
        final_states=final_states,
        p3_target=np.asarray(runner.closing_env.p3_target, dtype=float),
        transition_state=np.asarray(result["transition_state"], dtype=float),
        metadata_json=json.dumps(metadata),
    )
    print(f"cached {path}", flush=True)


def ensure_histories(
    records: list[dict[str, str]], cache_dir: Path, force: bool
) -> list[Path]:
    paths = []
    for record in records:
        path = cache_path(cache_dir, int(record["seed"]))
        if force or not path.exists():
            create_history_cache(record, path)
        paths.append(path)
    return paths


def load_history(path: Path) -> dict[str, np.ndarray | dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        history: dict[str, np.ndarray | dict[str, object]] = {
            key: archive[key].copy() for key in archive.files if key != "metadata_json"
        }
        history["metadata"] = json.loads(str(archive["metadata_json"]))
    return history


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.35,
        }
    )


def make_xy_figure(histories: list[dict[str, object]], total_successes: int) -> plt.Figure:
    fig = plt.figure(figsize=(10.4, 7.0))
    ax = fig.add_axes([0.09, 0.10, 0.88, 0.81])
    # The upper-middle portion of the full-scale axes is free of trajectories.
    inset = ax.inset_axes([0.39, 0.63, 0.44, 0.30])
    if len(histories) <= len(TRAJECTORY_COLORS):
        colors = TRAJECTORY_COLORS[: len(histories)]
    else:
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(histories)))

    for color, history in zip(colors, histories):
        closing_states = np.asarray(history["closing_states"])
        final_states = np.asarray(history["final_states"])
        p3_target = np.asarray(history["p3_target"])[None, :]
        transition = np.asarray(history["transition_state"])[None, :]

        closing_xy = port_position(closing_states)
        final_xy = port_position(final_states)
        p3_xy = port_position(p3_target)[0]
        transition_xy = port_position(transition)[0]

        ax.plot(closing_xy[:, 0], closing_xy[:, 1], color=color, alpha=0.82)
        ax.plot(final_xy[:, 0], final_xy[:, 1], color=color, alpha=0.82)
        ax.scatter(*closing_xy[0], marker="o", s=24, facecolor=color, edgecolor="white", linewidth=0.45, zorder=4)
        ax.scatter(*p3_xy, marker="D", s=23, facecolor="none", edgecolor=color, linewidth=0.9, zorder=4)
        ax.scatter(*transition_xy, marker="^", s=28, facecolor=color, edgecolor="white", linewidth=0.45, zorder=5)

        inset.plot(closing_xy[:, 0], closing_xy[:, 1], color=color, alpha=0.8)
        inset.plot(final_xy[:, 0], final_xy[:, 1], color=color, alpha=0.8)
        inset.scatter(*p3_xy, marker="D", s=18, facecolor="none", edgecolor=color, linewidth=0.8, zorder=4)
        inset.scatter(*transition_xy, marker="^", s=22, facecolor=color, edgecolor="white", linewidth=0.35, zorder=5)

    for axis in (ax, inset):
        axis.scatter(0.0, 0.0, marker="*", s=100 if axis is ax else 70, color="#CC3311", edgecolor="white", linewidth=0.5, zorder=6)
        axis.axhline(0.0, color="0.45", linewidth=0.6, zorder=0)

    ax.set_xlabel("Axial docking-port position, $r_{p,x}$ [m]")
    ax.set_ylabel("Lateral docking-port position, $r_{p,y}$ [m]")
    fig.suptitle(
        f"{len(histories)} representative successful full-mission trajectories",
        y=0.975,
    )
    ax.margins(x=0.035, y=0.10)
    # Present the mission in the natural reading direction: approach from the
    # left and converge on the docking point at the right.  Keep the physical
    # axial coordinate unchanged; only reverse its display direction.
    ax.invert_xaxis()

    inset.set_xlim(350.0, -15.0)
    inset.set_ylim(-6.0, 6.0)
    inset.text(
        0.03,
        0.96,
        "$P_3$ handover to docking",
        transform=inset.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )
    inset.tick_params(labelsize=8)
    inset.set_xlabel("$r_{p,x}$ [m]", fontsize=8)
    inset.set_ylabel("$r_{p,y}$ [m]", fontsize=8)

    seed_handles = [
        Line2D(
            [],
            [],
            color=color,
            linewidth=2.2,
            label=f"Seed {int(history['metadata']['seed'])}",
        )
        for color, history in zip(colors, histories)
    ]
    marker_handles = [
        Line2D([], [], marker="o", linestyle="None", color="0.25", markerfacecolor="0.25", markersize=6, label="$P_2$ initial state"),
        Line2D([], [], marker="D", linestyle="None", color="0.25", markerfacecolor="none", markersize=6, label="$P_3$ reference"),
        Line2D([], [], marker="^", linestyle="None", color="0.25", markerfacecolor="0.25", markersize=6, label="Controller transition"),
        Line2D([], [], marker="*", linestyle="None", color="#CC3311", markersize=10, label="Docking point"),
    ]
    ax.legend(handles=seed_handles + marker_handles, loc="lower right", framealpha=0.95)
    return fig


def make_time_history_figure(history: dict[str, object]) -> plt.Figure:
    closing_times = np.asarray(history["closing_times"])
    final_times = np.asarray(history["final_times"])
    closing_states = np.asarray(history["closing_states"])
    final_states = np.asarray(history["final_states"])
    metadata = history["metadata"]
    seed = int(metadata["seed"])

    transition_time = float(closing_times[-1])
    mission_final_times = final_times + transition_time
    closing_position = port_position(closing_states)
    final_position = port_position(final_states)
    closing_velocity = port_velocity(closing_states)
    final_velocity = port_velocity(final_states)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.2, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )
    series = (
        (
            closing_position[:, 0],
            final_position[:, 0],
            "a) Axial position",
            "Axial docking-port position, $r_{p,x}$ [m]",
        ),
        (
            closing_position[:, 1],
            final_position[:, 1],
            "b) Lateral position",
            "Lateral docking-port position, $r_{p,y}$ [m]",
        ),
        (
            closing_velocity[:, 0],
            final_velocity[:, 0],
            "c) Axial velocity",
            "Axial docking-port velocity, $v_{p,x}$ [m/s]",
        ),
        (
            closing_velocity[:, 1],
            final_velocity[:, 1],
            "d) Lateral velocity",
            "Lateral docking-port velocity, $v_{p,y}$ [m/s]",
        ),
    )

    for axis, (closing_values, final_values, title, ylabel) in zip(axes.flat, series):
        axis.plot(closing_times, closing_values, color=PHASE_COLORS["closing"], label="Closing")
        axis.plot(mission_final_times, final_values, color=PHASE_COLORS["final"], label="Final approach")
        axis.axvline(transition_time, color="0.15", linestyle="--", linewidth=1.1)
        axis.axhline(0.0, color="0.45", linewidth=0.6)
        # Both phases contain the exact same propagated state at handover.
        axis.scatter(
            transition_time,
            closing_values[-1],
            s=20,
            color="0.12",
            edgecolor="white",
            linewidth=0.45,
            zorder=5,
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)

    # On the full mission scale, the final controller's initial velocity
    # transient occupies only a few pixels and can look like a state jump.
    # These local views expose the identical closing/final sample at t - t_P3=0.
    for axis, closing_values, final_values, inset_bounds in (
        (
            axes[1, 0],
            closing_velocity[:, 0],
            final_velocity[:, 0],
            [0.16, 0.60, 0.39, 0.36],
        ),
        (
            axes[1, 1],
            closing_velocity[:, 1],
            final_velocity[:, 1],
            [0.17, 0.60, 0.39, 0.36],
        ),
    ):
        # Use a curve-free white region within each velocity panel.
        detail = axis.inset_axes(inset_bounds)
        closing_mask = closing_times >= transition_time - 5.0
        final_mask = final_times <= 50.0
        detail.plot(
            closing_times[closing_mask] - transition_time,
            closing_values[closing_mask],
            color=PHASE_COLORS["closing"],
        )
        detail.plot(
            final_times[final_mask],
            final_values[final_mask],
            color=PHASE_COLORS["final"],
        )
        detail.axvline(0.0, color="0.15", linestyle="--", linewidth=0.8)
        detail.scatter(
            [0.0, 0.0],
            [closing_values[-1], final_values[0]],
            s=13,
            color="0.12",
            edgecolor="white",
            linewidth=0.35,
            zorder=5,
        )
        detail.set_xlim(-5.0, 50.0)
        detail.set_title("Transition detail", fontsize=8, pad=2)
        detail.set_xlabel("$t-t_{P_3}$ [s]", fontsize=7, labelpad=1)
        detail.tick_params(labelsize=7, pad=1)
        detail.grid(True, alpha=0.2)

    for axis in axes[-1, :]:
        axis.set_xlabel("Mission time [s]")

    axes[0, 0].legend(loc="best", framealpha=0.95)
    axes[0, 1].annotate(
        "$P_3$ controller\ntransition",
        xy=(transition_time, 0.88),
        xycoords=("data", "axes fraction"),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.suptitle(f"Representative successful full mission (seed {seed})")
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.085,
        top=0.90,
        hspace=0.32,
        wspace=0.22,
    )
    return fig


def save_figure(figure: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    path = output_base.with_suffix(".png")
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"saved {path}")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_plot_style()
    success_records = read_success_records(args.campaign_csv)
    selected_records, representative_seed = select_geometry_spanning_subset(
        success_records, args.num_trajectories
    )
    selected_seeds = [int(record["seed"]) for record in selected_records]
    success_record_by_seed = {
        int(record["seed"]): record for record in success_records
    }
    additional_seed = args.additional_time_history_seed
    if additional_seed not in success_record_by_seed:
        raise ValueError(
            "--additional-time-history-seed must identify a successful mission "
            f"in {args.campaign_csv}; got seed {additional_seed}."
        )
    if additional_seed == representative_seed:
        raise ValueError(
            "--additional-time-history-seed must differ from the representative "
            f"seed {representative_seed}."
        )

    print(f"successful missions in campaign: {len(success_records)}")
    print(f"selected representative seeds: {selected_seeds}")
    print(f"time-history representative seed: {representative_seed}")
    print(f"additional time-history seed: {additional_seed}")

    history_records = list(selected_records)
    if additional_seed not in selected_seeds:
        history_records.append(success_record_by_seed[additional_seed])
    paths = ensure_histories(history_records, args.cache_dir, args.force_rerun)
    histories = [load_history(path) for path in paths]
    history_by_seed = {
        int(history["metadata"]["seed"]): history for history in histories
    }

    xy_histories = [history_by_seed[seed] for seed in selected_seeds]
    xy_figure = make_xy_figure(xy_histories, len(success_records))
    save_figure(
        xy_figure, args.output_dir / "full_mission_xy_trajectories"
    )

    time_figure = make_time_history_figure(history_by_seed[representative_seed])
    save_figure(
        time_figure,
        args.output_dir / "representative_full_mission_time_histories",
    )

    additional_time_figure = make_time_history_figure(
        history_by_seed[additional_seed]
    )
    save_figure(
        additional_time_figure,
        args.output_dir
        / f"representative_full_mission_time_histories_seed{additional_seed:04d}",
    )


if __name__ == "__main__":
    main()
