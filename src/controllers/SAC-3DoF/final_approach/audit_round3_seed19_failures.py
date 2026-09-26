"""Audit every Round-3 seed-19 evaluation failure on testing seeds 0--49."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))
from envs.docking3dof.gym_env import Docking3DoFGymEnv


HERE = Path(__file__).resolve().parent
STEM = "sac_final_approach_portobs_rewardv9_300ksteps_seed19_optuna12_buffer300k_best"
MODEL_DIR = HERE / "models" / "optuna12_300ksteps_buffer300k"
MODEL_PATH = MODEL_DIR / f"{STEM}.zip"
VECNORM_PATH = MODEL_DIR / f"{STEM}_vecnormalize.pkl"
OUT = HERE / "results" / "round3_seed19_failure_audit"
FIGURE_OUT = HERE / "results" / "thesis_selected_policy_figures"
MAX_STEPS = 5_000


def make_env(seed: int) -> Docking3DoFGymEnv:
    return Docking3DoFGymEnv(
        phase="final", mode="testing", dt=1.0, max_time=None,
        integrator="rk4", normalize_action=True, seed=seed,
    )


def state_record(env: Docking3DoFGymEnv, info: dict, eval_seed: int, step: int) -> dict:
    core = env.core_env
    q = core.get_derived_quantities(core.state)
    r_com = np.asarray(q["r_com"], dtype=float)
    v_com = np.asarray(q["v_com"], dtype=float)
    r_p = np.asarray(q["r_p"], dtype=float)
    v_p = np.asarray(q["v_p"], dtype=float)
    s, rho = float(q["s"]), float(q["rho"])
    cone_lateral_margin = float(core.cone_slope * s - rho)
    corridor_lateral_margin = float(core.corridor_radius - rho)
    pos = float(q["port_distance"])
    vel = float(q["port_speed"])
    att_deg = abs(float(np.rad2deg(q["psi_e"])))
    rate_deg_s = abs(float(np.rad2deg(q["omega"])))
    return {
        "eval_seed": eval_seed,
        "success": bool(info.get("success", False)),
        "failure": bool(info.get("failure", False)),
        "termination_reason": info.get("termination_reason"),
        "violated_constraints": ";".join(info.get("violated_constraints", [])),
        "failure_timestep": step,
        "failure_simulation_time_s": float(core.time),
        "com_relative_x_m": float(r_com[0]),
        "com_relative_y_m": float(r_com[1]),
        "com_relative_vx_mps": float(v_com[0]),
        "com_relative_vy_mps": float(v_com[1]),
        "port_relative_x_m": float(r_p[0]),
        "port_relative_y_m": float(r_p[1]),
        "port_relative_vx_mps": float(v_p[0]),
        "port_relative_vy_mps": float(v_p[1]),
        "port_position_norm_m": pos,
        "port_velocity_norm_mps": vel,
        "attitude_error_signed_deg": float(np.rad2deg(q["psi_e"])),
        "attitude_error_abs_deg": att_deg,
        "angular_rate_error_signed_deg_s": float(np.rad2deg(q["omega"])),
        "angular_rate_error_abs_deg_s": rate_deg_s,
        "cone_axial_lower_margin_m": s,
        "cone_axial_upper_margin_m": float(core.cone_length - s),
        "cone_allowed_lateral_radius_m": float(core.cone_slope * s),
        "cone_lateral_margin_m": cone_lateral_margin,
        "corridor_axial_lower_margin_m": s,
        "corridor_axial_upper_margin_m": float(core.d_P3 - s),
        "corridor_lateral_margin_m": corridor_lateral_margin,
        "in_docking_cone": bool(core.is_in_C_dock(q)),
        "in_final_corridor": bool(core.is_in_C_corr(q)),
        "position_requirement_satisfied": pos <= 0.1,
        "velocity_requirement_satisfied": vel <= 0.1,
        "attitude_requirement_satisfied": att_deg <= 4.0,
        "angular_rate_requirement_satisfied": rate_deg_s <= 0.2,
        "all_four_terminal_requirements_satisfied": bool(core.is_in_X_dock(core.state)),
        # check_done returns immediately on unsafe; the evaluator also breaks.
        "logged_states_after_failure": 0,
        "later_entered_terminal_region_after_failure": False,
    }


def plot_failure_mechanism(rows: list[dict]) -> None:
    FIGURE_OUT.mkdir(parents=True, exist_ok=True)
    seeds = np.array([r["eval_seed"] for r in rows])
    success = np.array([r["success"] for r in rows], dtype=bool)
    cone = 1000.0 * np.array([r["cone_lateral_margin_m"] for r in rows])
    distance = np.array([r["port_position_norm_m"] for r in rows])
    blue, orange, green = "#0072B2", "#D55E00", "#009E73"
    plt.rcParams.update({
        "font.size": 10.5, "axes.labelsize": 12, "axes.titlesize": 12,
        "legend.fontsize": 10, "figure.dpi": 130, "savefig.dpi": 400,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    for mask, color, marker in ((success, blue, "o"), (~success, orange, "x")):
        axes[0].scatter(seeds[mask], cone[mask], c=color, marker=marker, s=38, linewidths=1.4, zorder=3)
        axes[1].scatter(seeds[mask], distance[mask], c=color, marker=marker, s=38, linewidths=1.4, zorder=3)
    axes[0].axhline(0.0, color=green, ls="--", lw=1.3)
    axes[0].set_ylabel("Cone lateral margin [mm]")
    axes[0].set_title(r"Failure metric: $1000(0.1s-\rho)$", loc="left", fontweight="normal")
    note_transform = blended_transform_factory(axes[0].transAxes, axes[0].transData)
    axes[0].text(.01, 0.18, "Negative margin = outside docking cone",
                 transform=note_transform, color=green, ha="left", va="bottom",
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88, "pad": 1.0})
    axes[1].axhline(0.1, color=green, ls="--", lw=1.3)
    axes[1].set_ylabel(r"Terminal $\|r_p\|$ [m]")
    axes[1].set_xlabel("Evaluation seed")
    axes[1].set_title("Scalar docking-position requirement", loc="left", fontweight="normal")
    axes[1].set_xticks(np.arange(0, 50, 5))
    for ax in axes:
        ax.grid(True, color="#D9D9D9", lw=.6, alpha=.75)
        ax.set_axisbelow(True)
    fig.suptitle("Round 3 seed 19: termination geometry at the final logged state",
                 fontsize=14, fontweight="normal")
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color=blue, label="Successful rollout"),
        Line2D([], [], marker="x", ls="", color=orange, label="Failed: cone-geometry violation"),
        Line2D([], [], color=green, ls="--", label="Requirement boundary"),
    ], loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(left=.11, right=.98, top=.89, bottom=.13, hspace=.28)
    stem = FIGURE_OUT / "r3_s19_b300k_failure_mechanism"
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dummy = DummyVecEnv([lambda: Monitor(make_env(999))])
    vecnorm = VecNormalize.load(str(VECNORM_PATH), dummy)
    vecnorm.training = False
    vecnorm.norm_reward = False
    model = SAC.load(str(MODEL_PATH))
    rows = []
    for seed in range(50):
        env = make_env(seed)
        obs, info = env.reset(seed=seed)
        for step in range(1, MAX_STEPS + 1):
            action, _ = model.predict(vecnorm.normalize_obs(obs.reshape(1, -1)), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action[0])
            if terminated or truncated:
                break
        rows.append(state_record(env, info, seed, step))
        env.close()
    dummy.close()

    failed = [r for r in rows if not r["success"]]
    with (OUT / "all_termination_states.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (OUT / "failed_rollouts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(failed[0]))
        writer.writeheader(); writer.writerows(failed)
    summary = {
        "model": str(MODEL_PATH), "vecnormalize": str(VECNORM_PATH),
        "evaluation_seeds": "0-49", "deterministic": True,
        "n_success": sum(r["success"] for r in rows), "n_failure": len(failed),
        "failure_reasons": sorted(set(r["termination_reason"] for r in failed)),
        "failure_constraints": sorted(set(r["violated_constraints"] for r in failed)),
        "all_failures_satisfy_four_terminal_requirements": all(r["all_four_terminal_requirements_satisfied"] for r in failed),
        "all_failures_have_negative_cone_lateral_margin": all(r["cone_lateral_margin_m"] < 0 for r in failed),
        "any_post_failure_logged_state": any(r["logged_states_after_failure"] for r in failed),
        "logic": "check_done evaluates monitor['unsafe'] first and returns safety_violation; is_in_X_dock is evaluated only if no hard safety violation",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_failure_mechanism(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
