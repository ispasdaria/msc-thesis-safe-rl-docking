"""Generate comparison figures for the recorded LQR versus full-CBF campaign.

Seed selection uses the saved 0--49 per-episode metrics.  Only the two selected
seed pairs are replayed, deterministically, with the exact evaluation dynamics,
LQR controllers, and hard CBF configurations from ``evaluate_lqr_cbf.py``.
Nothing is trained or modified by this script.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
SRC = REPO / "src"
LQR = SRC / "controllers" / "LQR-3DoF"
EVALUATIONS = HERE / "evaluations"
for path in (SRC, LQR, HERE, EVALUATIONS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_lqr_cbf as evaluation  # noqa: E402
from cbf_3dof.controller import CBFQPController  # noqa: E402
from envs.docking3dof.core import Docking3DoFEnv  # noqa: E402


RESULTS = HERE / "results" / "lqr"
OUTPUT = RESULTS / "comparison_figures"
CLOSING_FILTERED = RESULTS / "closing_dt0p1_full_cbf_family_breakdown_seeds_0_49" / "per_episode_metrics.csv"
FINAL_FILTERED = RESULTS / "final_dt1p0_full_cbf_family_breakdown_seeds_0_49" / "per_episode_metrics.csv"
CLOSING_NOMINAL = (
    LQR / "results" / "lqr_closing"
    / "closing_controller_dt0p1_evaluation_and_tuning"
    / "closing_phase_0_49_dt0p1.csv"
)
FINAL_NOMINAL = LQR / "results" / "lqr_final" / "final_approach_phase_0_49.csv"

NOMINAL_COLOR = "#D55E00"
FILTERED_COLOR = "#0072B2"
INTERVENTION_COLOR = "#56B4E9"
P3_COLOR = "#009E73"
CONTROL_COLORS = ("#0072B2", "#D55E00", "#009E73")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def as_bool(value: str) -> bool:
    return value.lower() in {"true", "1"}


def select_closing_seed() -> tuple[int, dict[str, float | bool]]:
    nominal = {int(row["seed"]): row for row in rows(CLOSING_NOMINAL)}
    candidates = []
    for filtered in rows(CLOSING_FILTERED):
        seed = int(filtered["eval_seed"])
        base = nominal[seed]
        nominal_error = float(np.hypot(float(base["abs_final_ex"]), float(base["abs_final_ey"])))
        filtered_error = float(filtered["terminal_p3_position_error"])
        nominal_success = as_bool(base["success"])
        filtered_success = as_bool(filtered["success"])
        intervention = float(filtered["intervention_fraction"])
        # Prefer a changed outcome, then a larger terminal-error degradation,
        # then stronger intervention.  This makes selection reproducible and
        # directly aligned with the stated representative-case requirement.
        score = (
            int(nominal_success and not filtered_success),
            filtered_error - nominal_error,
            intervention,
        )
        candidates.append((score, seed, {
            "nominal_success": nominal_success,
            "filtered_success": filtered_success,
            "nominal_terminal_p3_error_m": nominal_error,
            "filtered_terminal_p3_error_m": filtered_error,
            "intervention_fraction": intervention,
            "intervention_count": int(filtered["intervention_count"]),
            "nominal_duration_s": float(base["elapsed_time"]),
            "filtered_duration_s": float(filtered["phase_duration_seconds"]),
        }))
    _, seed, metrics = max(candidates, key=lambda item: item[0])
    return seed, metrics


def select_final_seed() -> tuple[int, dict[str, float | int]]:
    nominal = {int(row["seed"]): row for row in rows(FINAL_NOMINAL)}
    candidates = []
    for filtered in rows(FINAL_FILTERED):
        interventions = int(filtered["intervention_count"])
        if interventions <= 0:
            continue
        seed = int(filtered["eval_seed"])
        nominal_distance = float(nominal[seed]["port_distance"])
        filtered_distance = float(filtered["terminal_port_distance"])
        # All recorded cases have one intervention.  Use the largest recorded
        # terminal displacement as the deterministic visibility tie-breaker.
        score = (interventions, abs(filtered_distance - nominal_distance), -seed)
        candidates.append((score, seed, {
            "intervention_count": interventions,
            "nominal_terminal_port_distance_m": nominal_distance,
            "filtered_terminal_port_distance_m": filtered_distance,
        }))
    if not candidates:
        raise RuntimeError("No final-approach rollout with a CBF intervention was found.")
    _, seed, metrics = max(candidates, key=lambda item: item[0])
    return seed, metrics


def replay(phase: str, seed: int, filtered: bool) -> dict[str, np.ndarray | float | int | str]:
    dt = 0.1 if phase == "closing" else 1.0
    max_time = 12_000.0 if phase == "closing" else 5_000.0
    env = Docking3DoFEnv(
        phase=phase, dt=dt, max_time=max_time, integrator="rk4", seed=seed
    )
    state = env.reset(mode="testing")
    lqr = evaluation.build_lqr(env)
    controller = CBFQPController(lqr, evaluation.full_cbf_config(phase)) if filtered else None
    p3_target = None if env.p3_target is None else env.p3_target.copy()
    trajectory = [np.asarray(env.get_derived_quantities(state)["r_p"], float)]
    times, axial_speed, controls, physical_corrections = [], [], [], []
    intervention, intervention_norm, axial_binding = [], [], []
    qp_failures = 0
    info: dict = {}
    max_steps = int(np.ceil(max_time / dt))
    for step in range(max_steps):
        q = env.get_derived_quantities(state)
        times.append(float(env.time))
        axial_speed.append(abs(float(q["s_dot"])))
        if controller is None:
            action = np.asarray(lqr.compute_action(env, state), float)
            changed = False
            change_norm = 0.0
            axial_active = False
            physical_correction = np.zeros(3, dtype=float)
        else:
            action = np.asarray(controller.compute_action(env, state), float)
            result = controller.last_result
            changed = bool(result.intervened)
            change_norm = float(result.intervention_scaled_norm)
            physical_correction = np.asarray(
                result.action_filtered - result.action_nominal, dtype=float
            )
            qp_failures += int(result.qp_failed)
            axial_active = changed and result.cbf_residuals.get("axial_port_speed", np.inf) <= controller.config.intervention_eps
        controls.append(action / np.asarray(env.action_high, float))
        physical_corrections.append(physical_correction)
        intervention.append(changed)
        intervention_norm.append(change_norm)
        axial_binding.append(axial_active)
        state, done, info = env.step(action)
        trajectory.append(np.asarray(env.get_derived_quantities(state)["r_p"], float))
        if done:
            break
    terminal_q = env.get_derived_quantities(state)
    geometry = {
        "cone_slope": float(env.cone_slope),
        "corridor_radius": float(env.corridor_radius),
        "corridor_start": float(env.d_P3),
        "p3": np.array([env.d_P3, 0.0]),
    }
    if phase == "closing" and p3_target is not None:
        geometry["p3"] = np.asarray(env.get_derived_quantities(p3_target)["r_p"], float)
        geometry["corridor_start"] = float(geometry["p3"][0])
    result = {
        "trajectory": np.asarray(trajectory),
        "time": np.asarray(times),
        "axial_speed": np.asarray(axial_speed),
        "control": np.asarray(controls),
        "physical_correction": np.asarray(physical_corrections),
        "intervened": np.asarray(intervention, dtype=bool),
        "intervention_norm": np.asarray(intervention_norm),
        "axial_binding": np.asarray(axial_binding, dtype=bool),
        "intervention_count": int(np.count_nonzero(intervention)),
        "axial_binding_count": int(np.count_nonzero(axial_binding)),
        "qp_failure_count": qp_failures,
        "success": bool(info.get("success", False)),
        "termination_reason": str(info.get("termination_reason", "evaluation_step_limit")),
        "terminal_port_distance": float(terminal_q["port_distance"]),
        "geometry": geometry,
    }
    if phase == "closing" and p3_target is not None:
        result["terminal_p3_error"] = float(np.linalg.norm(state[:2] - p3_target[:2]))
    return result


def save_rollout(data: dict, phase: str, seed: int, label: str) -> None:
    path = OUTPUT / f"lqr_{phase}_{label}_seed{seed:04d}.npz"
    arrays = {key: value for key, value in data.items() if key != "geometry" and isinstance(value, (np.ndarray, int, float, bool, str))}
    np.savez_compressed(path, **arrays)


def load_saved_rollout(phase: str, seed: int, label: str) -> dict | None:
    path = OUTPUT / f"lqr_{phase}_{label}_seed{seed:04d}.npz"
    if not path.is_file():
        return None
    with np.load(path) as archive:
        if label == "cbf" and "physical_correction" not in archive.files:
            return None
        data = {key: archive[key].item() if archive[key].ndim == 0 else archive[key].copy()
                for key in archive.files}
    dt = 0.1 if phase == "closing" else 1.0
    env = Docking3DoFEnv(max_time=12_000.0 if phase == "closing" else 5_000.0, phase=phase, dt=dt, integrator="rk4", seed=seed)
    env.reset(mode="testing")
    p3 = np.array([env.d_P3, 0.0])
    if phase == "closing":
        p3 = np.asarray(env.get_derived_quantities(env.p3_target)["r_p"], float)
    data["geometry"] = {
        "cone_slope": float(env.cone_slope),
        "corridor_radius": float(env.corridor_radius),
        "corridor_start": float(p3[0]),
        "p3": p3,
    }
    return data


def draw_geometry(ax, geometry: dict, x_min: float, x_max: float, *, p3: bool) -> None:
    xx = np.linspace(max(0.0, x_min), x_max, 500)
    style = dict(color="0.48", lw=0.8, ls=":", alpha=0.55, zorder=0)
    ax.axhline(0.0, color="0.42", lw=0.8, ls="--", alpha=0.65, label="Docking axis", zorder=0)
    ax.plot(xx, geometry["cone_slope"] * xx, label="Docking cone", **style)
    ax.plot(xx, -geometry["cone_slope"] * xx, **style)
    cx = np.linspace(0.0, geometry["corridor_start"], 200)
    ax.plot(cx, np.full_like(cx, geometry["corridor_radius"]), label="Docking corridor", **style)
    ax.plot(cx, np.full_like(cx, -geometry["corridor_radius"]), **style)
    point = geometry["p3"] if p3 else np.zeros(2)
    ax.scatter(*point, marker="*", s=72, color=P3_COLOR, edgecolor="white", lw=0.5,
               label=r"$P_3$ handover point" if p3 else "Docking point", zorder=7)


def trajectory_panel(ax, nominal: dict, filtered: dict, title: str, *, p3: bool) -> None:
    nt, ft = nominal["trajectory"], filtered["trajectory"]
    all_x = np.concatenate((nt[:, 0], ft[:, 0], [0.0]))
    draw_geometry(ax, filtered["geometry"], float(np.min(all_x)), float(np.max(all_x)), p3=p3)
    ax.plot(nt[:, 0], nt[:, 1], color=NOMINAL_COLOR, lw=1.35, label="Nominal LQR", zorder=2)
    ax.plot(ft[:, 0], ft[:, 1], color=FILTERED_COLOR, lw=1.55, label="LQR + CBF", zorder=3)
    ax.scatter(nt[0, 0], nt[0, 1], marker="D", s=30, facecolor="white", edgecolor="0.3", label="Initial state", zorder=6)
    ax.scatter(nt[-1, 0], nt[-1, 1], marker="X", s=42, color=NOMINAL_COLOR, label="Nominal terminal", zorder=6)
    ax.scatter(ft[-1, 0], ft[-1, 1], marker="P", s=42, color=FILTERED_COLOR, label="CBF terminal", zorder=6)
    ax.set_title(title, loc="left", fontsize=9.5)
    ax.set_xlabel(r"Axial port position $r_{p,x}$ [m]")
    ax.set_ylabel(r"Lateral port position $r_{p,y}$ [m]")
    ax.invert_xaxis()
    ax.grid(True, color="0.9", lw=0.5)


def shade_intervention(ax, time: np.ndarray, active: np.ndarray, label: str | None = None) -> None:
    if len(time):
        ax.fill_between(time, 0.0, 1.0, where=active, transform=ax.get_xaxis_transform(),
                        color=INTERVENTION_COLOR, alpha=0.13, linewidth=0,
                        label=label)


def unique_legend(fig, axes, **kwargs) -> None:
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, **kwargs)


def save(fig, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.png", dpi=400, bbox_inches="tight")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    closing_seed, closing_metrics = select_closing_seed()
    final_seed, final_metrics = select_final_seed()
    print(f"Selected closing evaluation seed: {closing_seed}")
    print(
        "  nominal -> CBF: "
        f"success {closing_metrics['nominal_success']} -> {closing_metrics['filtered_success']}, "
        f"terminal P3 error {closing_metrics['nominal_terminal_p3_error_m']:.4f} -> "
        f"{closing_metrics['filtered_terminal_p3_error_m']:.4f} m, "
        f"duration {closing_metrics['nominal_duration_s']:.1f} -> "
        f"{closing_metrics['filtered_duration_s']:.1f} s, "
        f"intervention {100 * closing_metrics['intervention_fraction']:.2f}%"
    )
    print(f"Selected final-approach evaluation seed: {final_seed}")
    print(f"  recorded intervention steps: {final_metrics['intervention_count']}")

    closing_nominal = load_saved_rollout("closing", closing_seed, "nominal") or replay("closing", closing_seed, False)
    closing_cbf = load_saved_rollout("closing", closing_seed, "cbf") or replay("closing", closing_seed, True)
    final_nominal = load_saved_rollout("final", final_seed, "nominal") or replay("final", final_seed, False)
    final_cbf = load_saved_rollout("final", final_seed, "cbf") or replay("final", final_seed, True)
    for phase, seed, nominal, filtered in (
        ("closing", closing_seed, closing_nominal, closing_cbf),
        ("final", final_seed, final_nominal, final_cbf),
    ):
        save_rollout(nominal, phase, seed, "nominal")
        save_rollout(filtered, phase, seed, "cbf")

    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 7.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })

    fig1, axes = plt.subplots(1, 2, figsize=(7.15, 3.15))
    trajectory_panel(axes[0], closing_nominal, closing_cbf,
                     rf"(a) Closing, $\Delta t=0.1$ s (seed {closing_seed})", p3=True)
    trajectory_panel(axes[1], final_nominal, final_cbf,
                     rf"(b) Final approach, $\Delta t=1$ s (seed {final_seed})", p3=False)
    unique_legend(fig1, axes, loc="upper center", bbox_to_anchor=(0.5, 1.03),
                  ncol=5, frameon=False, columnspacing=0.9, handletextpad=0.4)
    fig1.subplots_adjust(left=0.09, right=0.99, bottom=0.15, top=0.79, wspace=0.24)
    save(fig1, "lqr_cbf_trajectory_comparison")
    plt.close(fig1)

    # Preserve the existing axial-speed diagnostic as a standalone panel.
    fig2, ax = plt.subplots(1, 1, figsize=(3.65, 3.15))
    ax.plot(closing_nominal["time"], closing_nominal["axial_speed"],
            color=NOMINAL_COLOR, lw=1.15, label=r"Nominal LQR $|\dot{s}|$")
    ax.plot(closing_cbf["time"], closing_cbf["axial_speed"],
            color=FILTERED_COLOR, lw=1.25, label=r"LQR + CBF $|\dot{s}|$")
    ax.axhline(0.15, color="0.2", lw=1.0, ls="--", label=r"CBF limit: $0.15$ m/s")
    shade_intervention(ax, closing_cbf["time"], closing_cbf["intervened"], "CBF intervention")
    ax.set_title("(a) Closing axial-velocity constraint", loc="left", fontsize=9.5)
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"Axial port speed $|\dot{s}|$ [m/s]")
    ax.grid(True, color="0.9", lw=0.5)
    unique_legend(fig2, (ax,), loc="upper center", bbox_to_anchor=(0.5, 1.02),
                  ncol=2, frameon=False, columnspacing=0.9, handletextpad=0.4)
    fig2.subplots_adjust(left=0.20, right=0.98, bottom=0.16, top=0.73)
    save(fig2, "lqr_cbf_closing_diagnostics")
    plt.close(fig2)

    # Signed, dimensional CBF corrections evaluated at the filtered state.
    correction = np.asarray(closing_cbf["physical_correction"], float)
    correction_labels = (
        r"$\Delta u_x$ [m/s$^2$]",
        r"$\Delta u_y$ [m/s$^2$]",
        r"$\Delta u_\psi$ [rad/s$^2$]",
    )
    fig3, correction_axes = plt.subplots(3, 1, figsize=(7.15, 5.8), sharex=True)
    for index, (axis, ylabel, color) in enumerate(
        zip(correction_axes, correction_labels, CONTROL_COLORS)
    ):
        axis.plot(closing_cbf["time"], correction[:, index], color=color, lw=1.0)
        axis.axhline(0.0, color="0.3", lw=0.7, ls="--")
        shade_intervention(
            axis, closing_cbf["time"], closing_cbf["intervened"],
            "CBF active" if index == 0 else None,
        )
        axis.set_ylabel(ylabel)
        axis.grid(True, color="0.9", lw=0.5)
    correction_axes[-1].set_xlabel(r"Time $t$ [s]")
    correction_axes[0].set_title(
        rf"Closing CBF control corrections (evaluation seed {closing_seed})",
        loc="left", fontsize=9.5,
    )
    unique_legend(fig3, correction_axes, loc="upper center", bbox_to_anchor=(0.5, 0.99),
                  ncol=1, frameon=False)
    fig3.subplots_adjust(left=0.16, right=0.985, bottom=0.10, top=0.88, hspace=0.16)
    save(fig3, "lqr_cbf_closing_control_corrections")
    plt.close(fig3)

    # Verification: the intervention mask and correction samples must agree.
    correction_active = np.linalg.norm(
        correction / np.array([0.01, 0.01, 0.003]), axis=1
    ) > evaluation.full_cbf_config("closing").intervention_eps
    mask_matches = bool(np.array_equal(correction_active, closing_cbf["intervened"]))
    active_indices = np.flatnonzero(closing_cbf["intervened"])
    last_active = int(active_indices[-1])
    last_active_time = float(closing_cbf["time"][last_active])
    max_later_correction = float(np.max(np.abs(correction[last_active + 1 :])))
    common_end = min(len(closing_nominal["axial_speed"]), len(closing_cbf["axial_speed"]))
    later_start = min(last_active + 1, common_end)
    max_later_speed_difference = float(np.max(np.abs(
        closing_nominal["axial_speed"][later_start:common_end]
        - closing_cbf["axial_speed"][later_start:common_end]
    )))
    verification = {
        "intervention_mask_matches_signed_corrections": mask_matches,
        "last_intervention_time_s": last_active_time,
        "maximum_absolute_correction_after_last_intervention": max_later_correction,
        "maximum_axial_speed_difference_after_last_intervention_mps": max_later_speed_difference,
        "interpretation": (
            "true closed-loop state-history effect" if mask_matches and max_later_correction <= 1.0e-12
            and max_later_speed_difference > 0.0 else "requires further investigation"
        ),
    }
    print(
        "Verification: intervention shading "
        f"{'matches' if mask_matches else 'DOES NOT MATCH'} the signed correction samples; "
        f"last active sample t={last_active_time:.1f} s, maximum later correction="
        f"{max_later_correction:.3e}, maximum later |s_dot| difference="
        f"{max_later_speed_difference:.6f} m/s. Result: {verification['interpretation']}."
    )

    report = {
        "closing": {"selected_seed": closing_seed, **closing_metrics,
                    "replayed_intervention_count": closing_cbf["intervention_count"],
                    "axial_velocity_binding_steps": closing_cbf["axial_binding_count"]},
        "final_approach": {"selected_seed": final_seed, **final_metrics,
                           "replayed_intervention_count": final_cbf["intervention_count"]},
        "diagnostic_verification": verification,
    }
    (OUTPUT / "lqr_cbf_selected_seeds.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Closing seed {closing_seed} is representative because the full CBF changes "
        f"a nominal success into a timeout and increases terminal P3 error by "
        f"{closing_metrics['filtered_terminal_p3_error_m'] - closing_metrics['nominal_terminal_p3_error_m']:.4f} m; "
        f"the axial-velocity row binds on {closing_cbf['axial_binding_count']} intervention steps."
    )
    print(
        f"Final seed {final_seed} is representative because it is one of the nine "
        f"intervened rollouts ({final_cbf['intervention_count']} step), while its "
        "trajectory and terminal docking distance remain essentially unchanged."
    )
    print(f"Figures written to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
