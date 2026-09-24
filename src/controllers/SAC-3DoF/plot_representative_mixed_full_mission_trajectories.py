"""Rerun and plot three retained LQR-closing + SAC-final missions.

The controller/environment construction is imported from
``run_mixed_full_mission_evaluations.py`` so the policy artifacts,
normalization statistics, LQR settings, dynamics, and handover criterion are
identical to the committed 600-mission campaign.  No trajectory is inferred
from campaign terminal data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
CONTROLLERS = HERE.parents[0]
if str(CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS))

import run_mixed_full_mission_evaluations as campaign


SEEDS = (803, 306, 403)
CONFIG_KEYS = {
    "seed12_100k": "lqr_final_r1_s12_100k",
    "seed11_600k_buffer600k": "lqr_final_r5_s11_600k",
}
COLORS = ("#0072B2", "#D55E00", "#CC79A7")
PHASE_COLORS = {"closing": "#0072B2", "final": "#D55E00"}
DEFAULT_CACHE = HERE / "results/full_mission_mixed/representative_histories_lqr_sac_seed12_100k"
DEFAULT_OUTPUT = HERE / "figures" / "full_mission"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--final-policy", choices=CONFIG_KEYS, default="seed12_100k",
        help="Retained final-approach SAC policy to use after LQR handover.",
    )
    parser.add_argument("--state-evolution-only", action="store_true",
                        help="Generate only the seed-803 2x2 time-history figure.")
    parser.add_argument("--compare-final-policies", action="store_true",
                        help="Overlay the two retained final SAC policies for seed 803.")
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def history_arrays(core) -> tuple[np.ndarray, np.ndarray]:
    times, states, *_ = core.get_history_arrays()
    return np.asarray(times, dtype=float), np.asarray(states, dtype=float)


def rerun(seed: int, config: campaign.MissionConfiguration,
          final_sac: campaign.LoadedSAC) -> dict[str, object]:
    closing_env = campaign.make_env("closing", seed, config.closing_dt)
    observation, _ = closing_env.reset(seed=seed)
    lqr = campaign.build_lqr(closing_env)

    def closing_action(_observation: np.ndarray) -> np.ndarray:
        physical = lqr.compute_action(closing_env.core_env, closing_env.core_env.state)
        return closing_env.physical_action_to_normalized(physical)

    closing = campaign.run_phase(
        closing_env, observation, closing_action, math.ceil(12000.0 / config.closing_dt)
    )
    if not closing["success"]:
        raise RuntimeError(f"Seed {seed} did not reproduce its recorded closing success")
    closing_times, closing_states = history_arrays(closing_env.core_env)
    p3_target = closing_env.core_env.p3_target.copy()
    transition_state = closing["terminal_state"].copy()
    closing_env.close()

    final_env = campaign.make_env("final", seed, 1.0)
    observation, _ = final_env.reset(
        seed=seed, options={"initial_state": transition_state, "mode": "testing"}
    )
    final = campaign.run_phase(final_env, observation, final_sac.action, math.ceil(5000.0))
    if not final["success"]:
        raise RuntimeError(f"Seed {seed} did not reproduce its recorded final success")
    final_times, final_states = history_arrays(final_env.core_env)
    terminal_q = final_env.core_env.get_derived_quantities(final["terminal_state"])
    geometry = {
        "cone_length_m": float(final_env.core_env.cone_length),
        "cone_slope": float(final_env.core_env.cone_slope),
        "corridor_radius_m": float(final_env.core_env.corridor_radius),
        "d_p3_m": float(final_env.core_env.d_P3),
    }
    final_env.close()

    return {
        "seed": seed,
        "configuration": config.key,
        "closing_times": closing_times,
        "closing_states": closing_states,
        "final_times": final_times,
        "final_states": final_states,
        "p3_target": p3_target,
        "transition_state": transition_state,
        "handover_time_s": float(closing["elapsed_time_s"]),
        "final_phase_time_s": float(final["elapsed_time_s"]),
        "total_time_s": float(closing["elapsed_time_s"] + final["elapsed_time_s"]),
        "terminal_state": final["terminal_state"],
        "terminal_port_position": np.asarray(terminal_q["r_p"], dtype=float),
        "terminal_port_velocity": np.asarray(terminal_q["v_p"], dtype=float),
        "terminal_port_distance_m": float(terminal_q["port_distance"]),
        "terminal_port_speed_mps": float(terminal_q["port_speed"]),
        "terminal_attitude_error_deg": float(np.rad2deg(terminal_q["psi_e"])),
        "terminal_angular_rate_error_deg_s": float(np.rad2deg(terminal_q["omega"])),
        "geometry": geometry,
    }


def save_cache(path: Path, history: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in history.items() if isinstance(value, np.ndarray)}
    metadata = {key: value for key, value in history.items() if not isinstance(value, np.ndarray)}
    np.savez_compressed(path, **arrays, metadata_json=json.dumps(metadata))


def load_cache(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files if key != "metadata_json"}
        result.update(json.loads(str(archive["metadata_json"])))
    return result


def port_positions(states: np.ndarray) -> np.ndarray:
    # Use the environment implementation directly to avoid duplicating port geometry.
    env = campaign.make_env("final", 0, 1.0)
    try:
        return np.asarray([env.core_env.get_port_relative_position(state) for state in states])
    finally:
        env.close()


def port_velocities(states: np.ndarray) -> np.ndarray:
    """Evaluate docking-port velocities with the environment implementation."""
    env = campaign.make_env("final", 0, 1.0)
    try:
        return np.asarray([env.core_env.get_port_relative_velocity(state) for state in states])
    finally:
        env.close()


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 8.5, "lines.linewidth": 1.35,
        "figure.dpi": 130, "savefig.dpi": 400,
        "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def make_figure(histories: list[dict[str, object]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.2, 6.7))
    fig.subplots_adjust(left=0.095, right=0.975, bottom=0.105, top=0.925)
    inset = ax.inset_axes([0.42, 0.61, 0.43, 0.28])
    ax.axhline(0.0, color="0.35", linewidth=0.65, zorder=0)
    inset.axhline(0.0, color="0.35", linewidth=0.65, zorder=0)

    for history, color in zip(histories, COLORS):
        closing_xy = port_positions(np.asarray(history["closing_states"]))
        final_xy = port_positions(np.asarray(history["final_states"]))
        p3_xy = port_positions(np.asarray(history["p3_target"])[None, :])[0]
        transition_xy = port_positions(np.asarray(history["transition_state"])[None, :])[0]
        label = f"Seed {int(history['seed'])}"
        for axis in (ax, inset):
            axis.plot(closing_xy[:, 0], closing_xy[:, 1], color=color, alpha=0.88,
                      label=label if axis is ax else None)
            axis.plot(final_xy[:, 0], final_xy[:, 1], color=color, alpha=0.96)
            axis.scatter(*closing_xy[0], marker="o", s=18, facecolor=color,
                         edgecolor="white", linewidth=0.35, zorder=5)
            axis.scatter(*p3_xy, marker="D", s=17, facecolor="none",
                         edgecolor=color, linewidth=0.8, zorder=5)
            axis.scatter(*transition_xy, marker="^", s=18, facecolor=color,
                         edgecolor="white", linewidth=0.35, zorder=5)

    for axis in (ax, inset):
        axis.scatter(0.0, 0.0, marker="*", s=55, facecolor="#CC3311",
                     edgecolor="white", linewidth=0.4, zorder=6)

    ax.set_xlabel(r"Axial docking-port position, $r_{p,x}$ [m]")
    ax.set_ylabel(r"Lateral docking-port position, $r_{p,y}$ [m]")
    ax.set_title("Representative successful full-mission trajectories", pad=10)
    ax.margins(x=0.025, y=0.08)
    ax.invert_xaxis()

    inset.set_xlim(350.0, -10.0)
    inset.set_ylim(-6.0, 6.0)
    inset.set_xlabel(r"$r_{p,x}$ [m]", fontsize=8)
    inset.set_ylabel(r"$r_{p,y}$ [m]", fontsize=8)
    inset.tick_params(labelsize=8)
    inset.text(0.03, 0.95, r"$P_3$ handover to docking", transform=inset.transAxes,
               ha="left", va="top", fontsize=8.5,
               bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0})

    marker_handles = [
        Line2D([], [], marker="o", linestyle="None", color="0.25",
               markerfacecolor="0.25", markersize=5, label=r"$P_2$ initial state"),
        Line2D([], [], marker="D", linestyle="None", color="0.25",
               markerfacecolor="none", markersize=5, label=r"$P_3$ reference"),
        Line2D([], [], marker="^", linestyle="None", color="0.25",
               markerfacecolor="0.25", markersize=5, label="Controller transition"),
        Line2D([], [], marker="*", linestyle="None", color="#CC3311",
               markerfacecolor="#CC3311", markersize=8, label="Docking point"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + marker_handles, labels + [h.get_label() for h in marker_handles],
              loc="lower right", frameon=True, framealpha=0.96, ncol=1,
              borderpad=0.35, labelspacing=0.28, handlelength=1.8,
              handletextpad=0.55)
    return fig


def make_final_policy_comparison_figure(
    seed12: dict[str, object], seed11: dict[str, object]
) -> plt.Figure:
    """Overlay both final policies after their common seed-803 LQR handover."""
    closing_times = np.asarray(seed12["closing_times"], dtype=float)
    closing_states = np.asarray(seed12["closing_states"], dtype=float)
    handover_time = float(seed12["handover_time_s"])
    if not np.array_equal(closing_states, np.asarray(seed11["closing_states"])):
        raise RuntimeError("The two cached policies do not share an identical LQR history")
    if handover_time != float(seed11["handover_time_s"]):
        raise RuntimeError("The two cached policies do not share an identical handover time")

    closing_position = port_positions(closing_states)
    closing_velocity = port_velocities(closing_states)
    policies = (
        (seed12, "#D55E00", "SAC seed 12, 100k"),
        (seed11, "#009E73", "SAC seed 11, 600k/600k"),
    )
    final_data = []
    for history, color, label in policies:
        final_states = np.asarray(history["final_states"], dtype=float)
        final_times = np.asarray(history["final_times"], dtype=float)
        final_data.append({
            "times": final_times,
            "mission_times": final_times + handover_time,
            "position": port_positions(final_states),
            "velocity": port_velocities(final_states),
            "color": color, "label": label,
        })

    fig, axes = plt.subplots(
        2, 2, figsize=(11.2, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )
    fig.subplots_adjust(left=0.095, right=0.975, bottom=0.09, top=0.92,
                        wspace=0.22, hspace=0.32)
    panels = (
        (closing_position[:, 0], "position", 0, "a) Axial position",
         r"Axial docking-port position, $r_{p,x}$ [m]"),
        (closing_position[:, 1], "position", 1, "b) Lateral position",
         r"Lateral docking-port position, $r_{p,y}$ [m]"),
        (closing_velocity[:, 0], "velocity", 0, "c) Axial velocity",
         r"Axial docking-port velocity, $v_{p,x}$ [m/s]"),
        (closing_velocity[:, 1], "velocity", 1, "d) Lateral velocity",
         r"Lateral docking-port velocity, $v_{p,y}$ [m/s]"),
    )
    for axis, (closing_values, quantity, component, title, ylabel) in zip(axes.flat, panels):
        axis.plot(closing_times, closing_values, color="#0072B2", label="LQR closing")
        for data in final_data:
            axis.plot(data["mission_times"], data[quantity][:, component],
                      color=data["color"], label=data["label"])
        axis.axvline(handover_time, color="0.20", linestyle="--", linewidth=0.9)
        axis.axhline(0.0, color="0.45", linewidth=0.6)
        axis.scatter(handover_time, closing_values[-1], s=15, color="0.18",
                     edgecolor="white", linewidth=0.35, zorder=5)
        axis.set_title(title, loc="left")
        axis.set_ylabel(ylabel)
        axis.ticklabel_format(axis="both", style="plain", useOffset=False)

    axes[0, 0].legend(loc="upper right", framealpha=0.95, fontsize=8.2)
    axes[0, 1].text(
        handover_time + 0.018 * max(float(h["total_time_s"]) for h, _, _ in policies),
        0.94, r"$P_3$ handover", transform=axes[0, 1].get_xaxis_transform(),
        ha="left", va="top", fontsize=8.5,
    )
    for axis in axes[1, :]:
        axis.set_xlabel("Mission time [s]")

    for axis, closing_values, quantity, component, bounds in (
        (axes[1, 0], closing_velocity[:, 0], "velocity", 0, [0.16, 0.60, 0.39, 0.36]),
        (axes[1, 1], closing_velocity[:, 1], "velocity", 1, [0.17, 0.60, 0.39, 0.36]),
    ):
        detail = axis.inset_axes(bounds)
        closing_mask = closing_times >= handover_time - 5.0
        detail.plot(closing_times[closing_mask] - handover_time,
                    closing_values[closing_mask], color="#0072B2")
        for data in final_data:
            mask = data["times"] <= 50.0
            detail.plot(data["times"][mask], data[quantity][mask, component],
                        color=data["color"])
        detail.axvline(0.0, color="0.20", linestyle="--", linewidth=0.7)
        detail.scatter(0.0, closing_values[-1], s=10, color="0.18", zorder=5)
        detail.set_title("Transition detail", fontsize=7.5, pad=2)
        detail.set_xlabel(r"$t-t_{P_3}$ [s]", fontsize=7)
        detail.tick_params(labelsize=6.5)
        detail.ticklabel_format(axis="both", style="plain", useOffset=False)

    fig.suptitle("Final-policy comparison after the common LQR handover (seed 803)", y=0.975)
    return fig


def print_summary(history: dict[str, object]) -> None:
    state = np.asarray(history["terminal_state"])
    position = np.asarray(history["terminal_port_position"])
    velocity = np.asarray(history["terminal_port_velocity"])
    print(
        f"Seed {int(history['seed'])}: handover={history['handover_time_s']:.1f} s, "
        f"final phase={history['final_phase_time_s']:.1f} s, total={history['total_time_s']:.1f} s\n"
        f"  terminal port state: r_p=[{position[0]:.12f}, {position[1]:.12f}] m, "
        f"v_p=[{velocity[0]:.12f}, {velocity[1]:.12f}] m/s\n"
        f"  terminal CoM state: [{state[0]:.12f}, {state[1]:.12f}, {state[2]:.12f}, "
        f"{state[3]:.12f}, {np.rad2deg(state[4]):.12f} deg, "
        f"{np.rad2deg(state[5]):.12f} deg/s]\n"
        f"  port distance={history['terminal_port_distance_m']:.12f} m, "
        f"port speed={history['terminal_port_speed_mps']:.12f} m/s, "
        f"attitude error={history['terminal_attitude_error_deg']:.12f} deg, "
        f"angular-rate error={history['terminal_angular_rate_error_deg_s']:.12f} deg/s"
    )
    handover = np.asarray(history["transition_state"], dtype=float)
    print(
        "  handover CoM state [rx, ry, vx, vy, psi, omega] = "
        f"[{handover[0]:.12f}, {handover[1]:.12f}, {handover[2]:.12f}, "
        f"{handover[3]:.12f}, {np.rad2deg(handover[4]):.12f} deg, "
        f"{np.rad2deg(handover[5]):.12f} deg/s]"
    )


def main() -> None:
    args = parse_args()
    config = campaign.CONFIGURATIONS[CONFIG_KEYS[args.final_policy]]
    campaign.validate([config])
    final_sac = campaign.LoadedSAC(config.final_policy, "final", 1.0)
    histories = []
    try:
        selected_seeds = (803,) if args.state_evolution_only else SEEDS
        for seed in selected_seeds:
            path = args.cache_dir / f"seed{seed:04d}_lqr_{args.final_policy}.npz"
            if args.force_rerun or not path.exists():
                print(f"Rerunning seed {seed} with retained campaign configuration...", flush=True)
                history = rerun(seed, config, final_sac)
                save_cache(path, history)
            histories.append(load_cache(path))
    finally:
        final_sac.close()

    for history in histories:
        print_summary(history)
    set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = None
    if not args.state_evolution_only:
        figure = make_figure(histories)
        stem = args.output_dir / f"representative_successful_full_mission_trajectories_lqr_{args.final_policy}"
        figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
        plt.close(figure)
    comparison_stem = None
    if args.compare_final_policies:
        seed12_path = args.cache_dir / "seed0803_lqr_seed12_100k.npz"
        if not seed12_path.exists():
            legacy = args.cache_dir / "seed0803_lqr_sac12_100k.npz"
            seed12_path = legacy if legacy.exists() else seed12_path
        seed11_path = args.cache_dir / "seed0803_lqr_seed11_600k_buffer600k.npz"
        if not seed12_path.exists() or not seed11_path.exists():
            raise FileNotFoundError(
                "Comparison requires cached seed-803 histories for both retained policies"
            )
        comparison = make_final_policy_comparison_figure(
            load_cache(seed12_path), load_cache(seed11_path)
        )
        comparison_stem = args.output_dir / "full_mission_state_evolution_seed0803_final_policy_comparison"
        comparison.savefig(comparison_stem.with_suffix(".png"), dpi=400,
                           bbox_inches="tight", facecolor="white")
        plt.close(comparison)
    if stem is not None:
        print(f"Saved {stem.with_suffix('.png')}")
    if comparison_stem is not None:
        print(f"Saved {comparison_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
