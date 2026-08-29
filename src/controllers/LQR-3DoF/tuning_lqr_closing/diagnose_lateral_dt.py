"""Diagnostic-only analysis of closing-phase lateral convergence.

This script deliberately reuses the selected closing CARE controller without
changing Q, R, K, actuator limits, velocity scheduling, or the physical time
horizon.  It stores compact numerical arrays instead of the environment's
per-step dictionaries so dt=0.1 histories remain manageable.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


LQR_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LQR_ROOT.parents[1]
for path in (str(LQR_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from envs.docking3dof.core import Docking3DoFEnv
from lqr_3dof_closing import (
    LQRClosingController,
    build_Q_closing,
    build_R_closing,
    build_linearized_3dof_matrices,
    compute_lqr_gain,
)


MAX_TIME = 12_000.0
MODE = "testing"
SELECTED_DT01 = {
    28: "slight_lateral_miss",
    49: "typical_lateral_timeout",
    5: "largest_lateral_miss",
    41: "successful_reference",
    1: "matched_dt_regression",
}
MATCHED_SEED = 1


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (complex, np.complexfloating)):
        if abs(value.imag) < 1e-14:
            return float(value.real)
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def build_selected_controller():
    design_env = Docking3DoFEnv(
        phase="closing", dt=0.1, max_time=MAX_TIME, seed=0
    )
    # Reproduce the original dt=1.0-selected baseline for this historical diagnostic.
    Q = build_Q_closing(dt=1.0)
    R = build_R_closing(design_env)
    A, B = build_linearized_3dof_matrices(design_env)
    K, _ = compute_lqr_gain(A, B, Q, R)
    return LQRClosingController(K), A, B, Q, R, K


def run_compact_history(controller, seed, dt):
    env = Docking3DoFEnv(
        phase="closing",
        dt=dt,
        max_time=MAX_TIME,
        seed=seed,
    )

    # Logging is diagnostic-only and does not participate in dynamics or
    # termination.  Suppress dictionary history and retain compact arrays here.
    env.log_step = lambda **_: None
    state = env.reset(mode=MODE)

    max_steps = int(np.ceil(MAX_TIME / dt)) + 1
    times = np.empty(max_steps + 1, dtype=float)
    states = np.empty((max_steps + 1, 6), dtype=float)
    raw_body = np.empty((max_steps, 3), dtype=float)
    applied_body = np.empty((max_steps, 3), dtype=float)
    requested_orbital = np.empty((max_steps, 3), dtype=float)
    applied_orbital = np.empty((max_steps, 3), dtype=float)

    times[0] = env.time
    states[0] = state
    done = False
    info = {}
    steps = 0

    while not done and steps < max_steps:
        requested_body = controller.compute_action(env, state)
        clipped_body = env.clip_action(requested_body)
        requested_orbit = env.transform_action_to_orbital_frame(
            requested_body, state[4]
        )
        applied_orbit = env.transform_action_to_orbital_frame(
            clipped_body, state[4]
        )

        raw_body[steps] = requested_body
        applied_body[steps] = clipped_body
        requested_orbital[steps] = requested_orbit
        applied_orbital[steps] = applied_orbit

        state, done, info = env.step(requested_body)
        steps += 1
        times[steps] = env.time
        states[steps] = state

    return {
        "times": times[: steps + 1],
        "states": states[: steps + 1],
        "control_times": times[:steps],
        "raw_body": raw_body[:steps],
        "applied_body": applied_body[:steps],
        "requested_orbital": requested_orbital[:steps],
        "applied_orbital": applied_orbital[:steps],
        "p3_target": np.asarray(env.p3_target, dtype=float),
        "p3_tolerance": np.asarray(env.p3_target_tolerance, dtype=float),
        "metadata": {
            "seed": seed,
            "dt": dt,
            "steps": steps,
            "elapsed_time": env.time,
            "success": bool(info.get("success", False)),
            "termination_reason": info.get("termination_reason"),
            "mode": MODE,
            "max_time": MAX_TIME,
        },
        "mu": env.mu,
        "r0": env.r0,
        "n": env.n,
    }


def save_history(path, history, K):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        times=history["times"],
        states=history["states"],
        control_times=history["control_times"],
        raw_body=history["raw_body"],
        applied_body=history["applied_body"],
        requested_orbital=history["requested_orbital"],
        applied_orbital=history["applied_orbital"],
        p3_target=history["p3_target"],
        p3_tolerance=history["p3_tolerance"],
        K=K,
        mu=history["mu"],
        r0=history["r0"],
        n=history["n"],
        metadata_json=json.dumps(json_safe(history["metadata"]), indent=2),
    )


def load_history(path):
    data = np.load(path, allow_pickle=False)
    return {
        "times": data["times"],
        "states": data["states"],
        "control_times": data["control_times"],
        "raw_body": data["raw_body"],
        "applied_body": data["applied_body"],
        "requested_orbital": data["requested_orbital"],
        "applied_orbital": data["applied_orbital"],
        "p3_target": data["p3_target"],
        "p3_tolerance": data["p3_tolerance"],
        "K": data["K"],
        "mu": float(data["mu"]),
        "r0": float(data["r0"]),
        "n": float(data["n"]),
        "metadata": json.loads(str(data["metadata_json"])),
    }


def sample_state_values(times, values, requested_times):
    result = []
    for requested_time in requested_times:
        index = int(np.argmin(np.abs(times - requested_time)))
        result.append(
            {
                "time": float(times[index]),
                "value": float(values[index]),
            }
        )
    return result


def reversal_statistics(signal, dt, threshold=1e-8):
    signal = np.asarray(signal, dtype=float)
    valid = (np.abs(signal[:-1]) > threshold) & (
        np.abs(signal[1:]) > threshold
    )
    reversals = valid & (signal[:-1] * signal[1:] < 0.0)
    valid_count = int(np.count_nonzero(valid))
    reversal_count = int(np.count_nonzero(reversals))
    return {
        "count": reversal_count,
        "fraction_of_nontrivial_pairs": (
            reversal_count / valid_count if valid_count else 0.0
        ),
        "rate_per_physical_second": reversal_count
        / max(dt * (len(signal) - 1), dt),
        "total_variation_per_second": float(np.sum(np.abs(np.diff(signal))))
        / max(dt * (len(signal) - 1), dt),
    }


def summarize_history(history):
    times = history["times"]
    states = history["states"]
    control_times = history["control_times"]
    raw_body = history["raw_body"]
    applied_body = history["applied_body"]
    requested_orbital = history["requested_orbital"]
    applied_orbital = history["applied_orbital"]
    target = history["p3_target"]
    K = history["K"]
    dt = float(history["metadata"]["dt"])

    ey = states[:, 1] - target[1]
    evy = states[:, 3] - target[3]
    ey_control = ey[:-1]
    final_time = float(times[-1])
    late_start = max(float(times[0]), final_time - 2000.0)
    late_state = times >= late_start
    late_control = control_times >= late_start

    slope, intercept = np.polyfit(times[late_state], ey[late_state], 1)
    mean_sign = np.sign(np.median(ey[late_state]))
    d_abs_ey_dt = float(mean_sign * slope)
    final_abs_ey = abs(float(ey[-1]))
    if final_abs_ey <= 2.0:
        extrapolated_seconds = 0.0
    elif d_abs_ey_dt < -1e-12:
        extrapolated_seconds = (final_abs_ey - 2.0) / (-d_abs_ey_dt)
    else:
        extrapolated_seconds = None

    saturated_channels = np.abs(raw_body - applied_body) > 1e-12
    saturated_any = np.any(saturated_channels, axis=1)
    nontrivial_lateral = np.abs(requested_orbital[:, 1]) > 1e-12
    position_directed = requested_orbital[:, 1] * ey_control < 0.0

    # Relative to nominal alignment, orbital ay would equal -body ay.  This
    # isolates the extra redistribution caused by nonzero attitude.
    rotation_delta_requested = requested_orbital[:, 1] + raw_body[:, 1]
    clipping_delta_orbital_y = applied_orbital[:, 1] - requested_orbital[:, 1]

    x = states[:-1, 0]
    y = states[:-1, 1]
    rc = np.sqrt(x * x + y * y + history["r0"] ** 2)
    nonlinear_gravity_y = -history["mu"] * y / rc**3
    linear_gravity_y = -(history["n"] ** 2) * y
    nonlinear_model_delta_y = nonlinear_gravity_y - linear_gravity_y

    ky = float(K[1, 1])
    kvy = float(K[1, 3])
    y_ref = float(target[1])
    vy_ref = float(target[3])
    y_equilibrium = (ky * y_ref + kvy * vy_ref) / (
        ky + history["n"] ** 2
    )
    ey_equilibrium = y_equilibrium - y_ref
    position_term_final = -ky * float(ey[-1])
    velocity_term_final = -kvy * float(evy[-1])

    checkpoints = sorted(
        set(
            max(0.0, final_time - offset)
            for offset in (4000.0, 3000.0, 2000.0, 1000.0, 500.0, 0.0)
        )
    )

    return {
        **history["metadata"],
        "initial": {
            "y": float(states[0, 1]),
            "vy": float(states[0, 3]),
            "ey": float(ey[0]),
            "evy": float(evy[0]),
        },
        "target": {"y": y_ref, "vy": vy_ref},
        "final": {
            "y": float(states[-1, 1]),
            "vy": float(states[-1, 3]),
            "ey": float(ey[-1]),
            "evy": float(evy[-1]),
            "psi_deg": float(np.rad2deg(states[-1, 4])),
        },
        "ey_checkpoints": sample_state_values(times, ey, checkpoints),
        "late_2000s": {
            "ey_linear_slope_mps": float(slope),
            "d_abs_ey_dt_mps": d_abs_ey_dt,
            "ey_abs_change_m": float(abs(ey[-1]) - abs(ey[late_state][0])),
            "fraction_steps_abs_ey_nonincreasing": float(
                np.mean(np.diff(np.abs(ey[late_state])) <= 1e-10)
            ),
            "evy_final_mps": float(evy[-1]),
            "evy_mean_mps": float(np.mean(evy[late_state])),
            "requested_orbital_ay_mean_abs_mps2": float(
                np.mean(np.abs(requested_orbital[late_control, 1]))
            ),
            "requested_orbital_ay_max_abs_mps2": float(
                np.max(np.abs(requested_orbital[late_control, 1]))
            ),
            "applied_orbital_ay_mean_abs_mps2": float(
                np.mean(np.abs(applied_orbital[late_control, 1]))
            ),
            "raw_body_ax_mean_abs_mps2": float(
                np.mean(np.abs(raw_body[late_control, 0]))
            ),
            "raw_body_ay_mean_abs_mps2": float(
                np.mean(np.abs(raw_body[late_control, 1]))
            ),
            "any_saturation_fraction": float(np.mean(saturated_any[late_control])),
            "ay_body_saturation_fraction": float(
                np.mean(saturated_channels[late_control, 1])
            ),
            "position_directed_command_fraction": float(
                np.mean(position_directed[late_control & nontrivial_lateral])
            )
            if np.any(late_control & nontrivial_lateral)
            else None,
            "attitude_mean_abs_deg": float(
                np.mean(np.abs(np.rad2deg(states[:-1][late_control, 4])))
            ),
            "attitude_max_abs_deg": float(
                np.max(np.abs(np.rad2deg(states[:-1][late_control, 4])))
            ),
            "frame_rotation_delta_ay_rms_mps2": float(
                np.sqrt(np.mean(rotation_delta_requested[late_control] ** 2))
            ),
            "clipping_delta_orbital_ay_rms_mps2": float(
                np.sqrt(np.mean(clipping_delta_orbital_y[late_control] ** 2))
            ),
            "nonlinear_minus_linear_y_accel_max_abs_mps2": float(
                np.max(np.abs(nonlinear_model_delta_y[late_control]))
            ),
        },
        "whole_rollout": {
            "any_saturation_fraction": float(np.mean(saturated_any)),
            "ax_body_saturation_fraction": float(np.mean(saturated_channels[:, 0])),
            "ay_body_saturation_fraction": float(np.mean(saturated_channels[:, 1])),
            "alphaz_body_saturation_fraction": float(
                np.mean(saturated_channels[:, 2])
            ),
            "requested_orbital_ay_reversals": reversal_statistics(
                requested_orbital[:, 1], dt
            ),
            "applied_orbital_ay_reversals": reversal_statistics(
                applied_orbital[:, 1], dt
            ),
            "net_lateral_motion_m": float(states[-1, 1] - states[0, 1]),
            "absolute_lateral_motion_m": float(
                np.sum(np.abs(np.diff(states[:, 1])))
            ),
            "frame_rotation_delta_ay_max_abs_mps2": float(
                np.max(np.abs(rotation_delta_requested))
            ),
            "clipping_delta_orbital_ay_max_abs_mps2": float(
                np.max(np.abs(clipping_delta_orbital_y))
            ),
        },
        "predicted_lateral_equilibrium": {
            "y_m": float(y_equilibrium),
            "ey_m": float(ey_equilibrium),
            "final_minus_predicted_ey_m": float(ey[-1] - ey_equilibrium),
            "k_y": ky,
            "k_vy": kvy,
            "k_vy_over_k_y_seconds": kvy / ky,
            "position_feedback_term_final_mps2": position_term_final,
            "velocity_feedback_term_final_mps2": velocity_term_final,
        },
        "extrapolated_seconds_to_abs_ey_2m_from_linear_late_slope": extrapolated_seconds,
    }


def make_late_plot(history, summary, path, label):
    times = history["times"]
    control_times = history["control_times"]
    states = history["states"]
    target = history["p3_target"]
    start = max(0.0, float(times[-1]) - 4000.0)
    state_mask = times >= start
    control_mask = control_times >= start
    ey = states[:, 1] - target[1]
    evy = states[:, 3] - target[3]
    saturated = np.any(
        np.abs(history["raw_body"] - history["applied_body"]) > 1e-12,
        axis=1,
    )

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Closing lateral diagnostic | seed {summary['seed']} | dt={summary['dt']} s | {label}"
    )

    axes[0, 0].plot(times[state_mask], ey[state_mask])
    axes[0, 0].axhline(2.0, color="black", linestyle=":", label="P3 tolerance")
    axes[0, 0].axhline(-2.0, color="black", linestyle=":")
    axes[0, 0].axhline(
        summary["predicted_lateral_equilibrium"]["ey_m"],
        color="tab:red",
        linestyle="--",
        label="predicted equilibrium",
    )
    axes[0, 0].set_ylabel(r"$e_y$ [m]")
    axes[0, 0].legend()

    axes[0, 1].plot(times[state_mask], evy[state_mask])
    axes[0, 1].axhline(0.01, color="black", linestyle=":", label="P3 tolerance")
    axes[0, 1].axhline(-0.01, color="black", linestyle=":")
    axes[0, 1].set_ylabel(r"$e_{v_y}$ [m/s]")
    axes[0, 1].legend()

    axes[1, 0].plot(
        control_times[control_mask],
        history["requested_orbital"][control_mask, 1],
        linestyle="--",
        label="requested orbital",
    )
    axes[1, 0].plot(
        control_times[control_mask],
        history["applied_orbital"][control_mask, 1],
        label="applied orbital",
    )
    axes[1, 0].scatter(
        control_times[control_mask & saturated],
        history["applied_orbital"][control_mask & saturated, 1],
        s=8,
        color="tab:red",
        label="clipped update",
    )
    orbital_y_max = float(
        np.max(np.abs(history["requested_orbital"][control_mask, 1]))
    )
    orbital_y_span = max(1.2 * orbital_y_max, 1e-7)
    axes[1, 0].set_ylim(-orbital_y_span, orbital_y_span)
    axes[1, 0].set_ylabel(r"$u_{y,O}$ [m/s$^2$]")
    axes[1, 0].legend()

    for channel, name, axis in (
        (0, r"$a_{x,b_c}$", axes[1, 1]),
        (1, r"$a_{y,b_c}$", axes[2, 0]),
    ):
        axis.plot(
            control_times[control_mask],
            history["raw_body"][control_mask, channel],
            linestyle="--",
            label="requested",
        )
        axis.plot(
            control_times[control_mask],
            history["applied_body"][control_mask, channel],
            label="applied",
        )
        axis.axhline(0.01, color="black", linestyle=":", label="actuator limit")
        axis.axhline(-0.01, color="black", linestyle=":")
        axis.set_ylabel(name + r" [m/s$^2$]")
        axis.legend()

    axes[2, 1].plot(times[state_mask], np.rad2deg(states[state_mask, 4]))
    axes[2, 1].set_ylabel(r"$\psi$ [deg]")
    axes[2, 1].ticklabel_format(axis="y", style="plain", useOffset=False)

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("physical time [s]")
    axes[2, 1].set_xlabel("physical time [s]")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_matched_plot(dt1, dt01, path):
    target1 = dt1["p3_target"]
    target01 = dt01["p3_target"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Matched seed 1: dt=1.0 s versus dt=0.1 s")

    axes[0, 0].plot(dt1["times"], dt1["states"][:, 1] - target1[1], label="dt=1.0")
    axes[0, 0].plot(dt01["times"], dt01["states"][:, 1] - target01[1], label="dt=0.1")
    axes[0, 0].axhline(2.0, color="black", linestyle=":")
    axes[0, 0].axhline(-2.0, color="black", linestyle=":")
    axes[0, 0].set_ylabel(r"$e_y$ [m]")
    axes[0, 0].legend()

    axes[0, 1].plot(dt1["times"], dt1["states"][:, 3] - target1[3], label="dt=1.0")
    axes[0, 1].plot(dt01["times"], dt01["states"][:, 3] - target01[3], label="dt=0.1")
    axes[0, 1].set_ylabel(r"$e_{v_y}$ [m/s]")
    axes[0, 1].legend()

    for history, column, dt_label in ((dt1, 0, "dt=1.0"), (dt01, 1, "dt=0.1")):
        axes[1, column].plot(
            history["control_times"],
            history["requested_orbital"][:, 1],
            linestyle="--",
            linewidth=0.8,
            label="requested",
        )
        axes[1, column].plot(
            history["control_times"],
            history["applied_orbital"][:, 1],
            linewidth=0.8,
            label="applied",
        )
        axes[1, column].set_title(dt_label + " orbital lateral command")
        axes[1, column].set_ylabel(r"$u_{y,O}$ [m/s$^2$]")
        axes[1, column].legend()

        axes[2, column].plot(
            history["control_times"],
            history["raw_body"][:, 1],
            linestyle="--",
            linewidth=0.8,
            label="requested body ay",
        )
        axes[2, column].plot(
            history["control_times"],
            history["applied_body"][:, 1],
            linewidth=0.8,
            label="applied body ay",
        )
        axes[2, column].axhline(0.01, color="black", linestyle=":")
        axes[2, column].axhline(-0.01, color="black", linestyle=":")
        axes[2, column].set_ylabel(r"$a_{y,b_c}$ [m/s$^2$]")
        axes[2, column].set_xlabel("physical time [s]")
        axes[2, column].legend()

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_switching_detail(dt1, dt01, path):
    end = float(dt1["times"][-1])
    start = end - 30.0
    state_masks = {
        "dt=1.0": (dt1, dt1["times"] >= start),
        "dt=0.1": (dt01, (dt01["times"] >= start) & (dt01["times"] <= end)),
    }
    control_masks = {
        "dt=1.0": (dt1, dt1["control_times"] >= start),
        "dt=0.1": (
            dt01,
            (dt01["control_times"] >= start) & (dt01["control_times"] <= end),
        ),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(
        "Matched seed 1: final 30 s before dt=1.0 success "
        "(step-to-step switching detail)"
    )

    for label, (history, mask) in state_masks.items():
        ey = history["states"][:, 1] - history["p3_target"][1]
        axes[0, 0].plot(history["times"][mask], ey[mask], marker=".", label=label)
    axes[0, 0].axhline(2.0, color="black", linestyle=":")
    axes[0, 0].axhline(-2.0, color="black", linestyle=":")
    axes[0, 0].set_ylabel(r"$e_y$ [m]")
    axes[0, 0].legend()

    for label, (history, mask) in control_masks.items():
        axes[0, 1].plot(
            history["control_times"][mask],
            history["applied_orbital"][mask, 1],
            marker=".",
            label=label,
        )
        axes[1, 0].plot(
            history["control_times"][mask],
            history["applied_body"][mask, 1],
            marker=".",
            label=label,
        )
        axes[1, 1].plot(
            history["control_times"][mask],
            history["applied_body"][mask, 0],
            marker=".",
            label=label,
        )

    axes[0, 1].set_ylabel(r"applied $u_{y,O}$ [m/s$^2$]")
    axes[1, 0].set_ylabel(r"applied $a_{y,b_c}$ [m/s$^2$]")
    axes[1, 1].set_ylabel(r"applied $a_{x,b_c}$ [m/s$^2$]")
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.legend()
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("physical time [s]")
    axes[1, 1].set_xlabel("physical time [s]")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def lateral_failure_seeds(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    return [
        {
            "seed": int(row["seed"]),
            "abs_final_ey": float(row["abs_final_ey"]),
            "status": row["status"],
        }
        for row in records
        if row["fail_ry_tolerance"].lower() == "true"
    ]


def population_equilibrium_check(csv_path, K):
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    ky = float(K[1, 1])
    kvy = float(K[1, 3])
    records = []
    for row in rows:
        seed = int(row["seed"])
        env = Docking3DoFEnv(
            phase="closing", dt=0.1, max_time=MAX_TIME, seed=seed
        )
        env.log_step = lambda **_: None
        env.reset(mode=MODE)
        y_ref = float(env.p3_target[1])
        vy_ref = float(env.p3_target[3])
        y_equilibrium = (ky * y_ref + kvy * vy_ref) / (ky + env.n**2)
        ey_equilibrium = y_equilibrium - y_ref
        observed_abs_ey = float(row["abs_final_ey"])
        observed_failure = row["fail_ry_tolerance"].lower() == "true"
        predicted_failure = abs(ey_equilibrium) > float(
            env.p3_target_tolerance[1]
        )
        records.append(
            {
                "seed": seed,
                "target_y": y_ref,
                "target_vy": vy_ref,
                "predicted_equilibrium_ey": ey_equilibrium,
                "observed_abs_terminal_ey": observed_abs_ey,
                "predicted_lateral_failure": predicted_failure,
                "observed_lateral_failure": observed_failure,
            }
        )

    observed_failures = [item for item in records if item["observed_lateral_failure"]]
    return {
        "predicted_failure_count": sum(
            item["predicted_lateral_failure"] for item in records
        ),
        "observed_failure_count": sum(
            item["observed_lateral_failure"] for item in records
        ),
        "classification_matches_all_50": all(
            item["predicted_lateral_failure"]
            == item["observed_lateral_failure"]
            for item in records
        ),
        "maximum_abs_terminal_vs_equilibrium_error_among_lateral_failures_m": max(
            abs(
                item["observed_abs_terminal_ey"]
                - abs(item["predicted_equilibrium_ey"])
            )
            for item in observed_failures
        ),
        "approximate_target_vy_threshold_for_2m_offset_mps": 2.0 * ky / kvy,
        "records": records,
    }


def write_summary_csv(path, summaries):
    rows = []
    for item in summaries:
        late = item["late_2000s"]
        equilibrium = item["predicted_lateral_equilibrium"]
        rows.append(
            {
                "seed": item["seed"],
                "dt": item["dt"],
                "success": item["success"],
                "termination_reason": item["termination_reason"],
                "initial_y": item["initial"]["y"],
                "initial_vy": item["initial"]["vy"],
                "target_y": item["target"]["y"],
                "target_vy": item["target"]["vy"],
                "final_ey": item["final"]["ey"],
                "final_evy": item["final"]["evy"],
                "predicted_equilibrium_ey": equilibrium["ey_m"],
                "final_minus_predicted_ey": equilibrium["final_minus_predicted_ey_m"],
                "late_d_abs_ey_dt": late["d_abs_ey_dt_mps"],
                "late_requested_ay_mean_abs": late[
                    "requested_orbital_ay_mean_abs_mps2"
                ],
                "late_applied_ay_mean_abs": late[
                    "applied_orbital_ay_mean_abs_mps2"
                ],
                "late_ay_saturation_fraction": late[
                    "ay_body_saturation_fraction"
                ],
                "whole_ay_saturation_fraction": item["whole_rollout"][
                    "ay_body_saturation_fraction"
                ],
                "late_attitude_max_abs_deg": late["attitude_max_abs_deg"],
                "late_frame_rotation_delta_ay_rms": late[
                    "frame_rotation_delta_ay_rms_mps2"
                ],
                "late_nonlinear_model_delta_y_max_abs": late[
                    "nonlinear_minus_linear_y_accel_max_abs_mps2"
                ],
                "extrapolated_seconds_to_2m": item[
                    "extrapolated_seconds_to_abs_ey_2m_from_linear_late_slope"
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    output_root = (
        LQR_ROOT / "results" / "lqr_closing" / "dt_0p1" / "lateral_diagnostics"
    )
    history_dir = output_root / "histories"
    plot_dir = output_root / "plots"
    controller, A, B, Q, R, K = build_selected_controller()

    cases = [(seed, 0.1, label) for seed, label in SELECTED_DT01.items()]
    cases.append((MATCHED_SEED, 1.0, "matched_dt1_success"))
    histories = {}
    summaries = []

    for seed, dt, label in cases:
        dt_tag = str(dt).replace(".", "p")
        history_path = history_dir / f"seed{seed:04d}_dt{dt_tag}_{label}.npz"
        if history_path.exists():
            print(f"loading seed={seed}, dt={dt:g}, label={label}", flush=True)
        else:
            print(f"running seed={seed}, dt={dt:g}, label={label}", flush=True)
            history = run_compact_history(controller, seed, dt)
            history["K"] = K
            save_history(history_path, history, K)
        loaded = load_history(history_path)
        summary = summarize_history(loaded)
        summary["label"] = label
        histories[(seed, dt)] = loaded
        summaries.append(summary)
        make_late_plot(
            loaded,
            summary,
            plot_dir / f"seed{seed:04d}_dt{dt_tag}_{label}_late.png",
            label,
        )
        print(
            f"  {summary['termination_reason']}: final ey={summary['final']['ey']:.9f} m, "
            f"predicted equilibrium ey={summary['predicted_lateral_equilibrium']['ey_m']:.9f} m",
            flush=True,
        )

    make_matched_plot(
        histories[(MATCHED_SEED, 1.0)],
        histories[(MATCHED_SEED, 0.1)],
        plot_dir / "seed0001_dt1_vs_dt0p1.png",
    )
    make_switching_detail(
        histories[(MATCHED_SEED, 1.0)],
        histories[(MATCHED_SEED, 0.1)],
        plot_dir / "seed0001_switching_detail_last30s.png",
    )

    failures = lateral_failure_seeds(
        LQR_ROOT
        / "results"
        / "lqr_closing"
        / "dt_0p1"
        / "closing_phase_0_49_dt0p1.csv"
    )
    population_check = population_equilibrium_check(
        LQR_ROOT
        / "results"
        / "lqr_closing"
        / "dt_0p1"
        / "closing_phase_0_49_dt0p1.csv",
        K,
    )
    report = {
        "controller_unchanged": {
            "Q": Q,
            "R": R,
            "K": K,
            "continuous_closed_loop_eigenvalues": np.linalg.eigvals(A - B @ K),
            "orbital_lateral_gain_cross_channel_max_abs": float(
                np.max(np.abs(K[1, [0, 2, 4, 5]]))
            ),
        },
        "lateral_failure_seeds": sorted(failures, key=lambda item: item["seed"]),
        "population_equilibrium_check": population_check,
        "representative_summaries": summaries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "lateral_diagnostic_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(json_safe(report), handle, indent=2)
    write_summary_csv(output_root / "representative_summary.csv", summaries)
    print(f"saved diagnostics under: {output_root}")


if __name__ == "__main__":
    main()
