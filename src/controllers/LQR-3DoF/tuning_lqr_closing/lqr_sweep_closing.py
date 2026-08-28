import argparse
import csv
import json
import os
import sys

import numpy as np

# Allow imports from the repo's src/ directory when running from the project root.
LQR_ROOT = os.path.abspath(os.path.dirname(__file__))
while not os.path.exists(os.path.join(LQR_ROOT, "lqr_3dof_closing.py")):
    parent_dir = os.path.dirname(LQR_ROOT)
    if parent_dir == LQR_ROOT:
        break
    LQR_ROOT = parent_dir
SRC_PATH = os.path.abspath(os.path.join(LQR_ROOT, "..", ".."))
if LQR_ROOT not in sys.path:
    sys.path.insert(0, LQR_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from lqr_3dof_closing import (
    LQRClosingController,
    build_linearized_3dof_matrices,
    build_Q_closing,
    build_R_closing,
    compute_lqr_gain,
)
from envs.docking3dof.core import Docking3DoFEnv


def parse_scales(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_multipliers(text, expected_len, name):
    if not text:
        return None

    values = parse_scales(text)

    if len(values) != expected_len:
        raise ValueError(f"{name} must have {expected_len} values, got {len(values)}.")

    return np.asarray(values, dtype=float)


def apply_diag_multipliers(base, multipliers):
    """
    Applies per-state or per-input multipliers to the diagonal entries of Q/R.
    """
    base = np.asarray(base, dtype=float)

    if multipliers is None:
        return base

    diag = np.diag(base)
    return np.diag(diag * multipliers)


def saturation_fraction(env, atol=1e-12):
    """
    Fraction of logged timesteps where at least one body-frame action channel
    was clipped by the environment.
    """
    _, _, actions_body_raw, actions_body_clipped, _ = env.get_history_arrays()

    if actions_body_raw.size == 0 or actions_body_clipped.size == 0:
        return 0.0

    diff = actions_body_raw - actions_body_clipped
    saturated_per_channel = np.abs(diff) > atol
    saturated_any = np.any(saturated_per_channel, axis=1)

    return float(np.mean(saturated_any))


def translational_delta_v_proxy(env):
    """Return ``sum ||a_xy,applied|| dt`` for the completed rollout."""
    times, _, _, actions_body_clipped, _ = env.get_history_arrays()

    if actions_body_clipped.size == 0:
        return 0.0

    if times.size >= 2:
        dt = float(np.median(np.diff(times)))
    else:
        dt = float(getattr(env, "dt", 1.0))

    a_trans = actions_body_clipped[:, :2]
    return float(np.sum(np.linalg.norm(a_trans, axis=1)) * dt)


def angular_effort_proxy(env):
    """Return ``sum |alpha_z,applied| dt`` for the completed rollout."""
    times, _, _, actions_body_clipped, _ = env.get_history_arrays()

    if actions_body_clipped.size == 0:
        return 0.0

    if times.size >= 2:
        dt = float(np.median(np.diff(times)))
    else:
        dt = float(getattr(env, "dt", 1.0))

    return float(np.sum(np.abs(actions_body_clipped[:, 2])) * dt)


def trajectory_mean_vx(env):
    """
    Mean vx over the whole stored trajectory.

    This is different from the final vx. It is useful as a diagnostic only,
    not as a success criterion.
    """
    times, states, _, _, _ = env.get_history_arrays()

    if states.size == 0:
        return float("nan")

    return float(np.mean(states[:, 2]))


def make_json_safe(value):
    """
    Converts NumPy values/arrays to JSON-serializable Python objects.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    return value


def rollout_status(info):
    if info.get("success"):
        return "success"
    return str(info.get("termination_reason", "unknown"))


def compute_final_error(env, state):
    """
    Computes final error relative to the sampled P3 target.
    Returns (norm, component_error). If no target exists, returns NaNs.
    """
    if env.p3_target is None:
        return float("nan"), np.full(6, np.nan)

    e = state - env.p3_target
    e[4] = env.wrap_to_pi(state[4] - env.p3_target[4])
    return float(np.linalg.norm(e)), e


def run_single_rollout(controller, max_time, seed, mode="testing"):
    """
    Runs one closing-phase rollout and returns the completed environment,
    final state, final info dictionary, and number of integration steps.
    """
    env = Docking3DoFEnv(
        phase="closing",
        max_time=max_time,
        seed=seed,
    )

    state = env.reset(mode=mode)

    done = False
    info = {}
    steps = 0
    max_steps = int(max_time / env.dt) + 1

    while not done and steps < max_steps:
        action = controller.compute_action(env, state)
        state, done, info = env.step(action)
        steps += 1

    return env, state, info, steps


def build_rollout_record(env, state, info, seed, steps):
    """
    Builds a lightweight per-rollout record used for selecting representative
    histories. This record does not contain the full trajectory.
    """
    final_error, component_error = compute_final_error(env, state)
    status = rollout_status(info)
    sat = saturation_fraction(env)
    delta_v_proxy = translational_delta_v_proxy(env)
    angular_effort = angular_effort_proxy(env)

    violated_constraints = info.get("violated_constraints", [])
    if violated_constraints is None:
        violated_constraints = []

    return {
        "seed": int(seed),
        "status": status,
        "success": bool(info.get("success", False)),
        "termination_reason": str(info.get("termination_reason", status)),
        "violated_constraints": list(violated_constraints),
        "steps": int(steps),
        "elapsed_time": float(env.time),
        "saturation_fraction": float(sat),
        "delta_v_proxy": float(delta_v_proxy),
        "angular_effort": float(angular_effort),
        "final_error": float(final_error),
        "abs_final_ex": float(abs(component_error[0])) if component_error.size else float("nan"),
        "abs_final_ey": float(abs(component_error[1])) if component_error.size else float("nan"),
        "abs_final_evx": float(abs(component_error[2])) if component_error.size else float("nan"),
        "abs_final_evy": float(abs(component_error[3])) if component_error.size else float("nan"),
        "abs_final_epsi": float(abs(component_error[4])) if component_error.size else float("nan"),
        "abs_final_eomega": float(abs(component_error[5])) if component_error.size else float("nan"),
    }


def select_representative_rollouts(records):
    """
    Selects representative rollouts without saving all histories:
      - first successful rollout
      - median successful rollout, if at least 3 successes exist
      - typical timeout rollout
      - worst final-error rollout
      - highest-saturation rollout
      - first safety-failure rollout, if any exists

    Returns a list of (label, record). If two categories point to the same seed,
    the labels are combined and only one history is saved.
    """
    selected_by_seed = {}

    def add(label, record):
        if record is None:
            return
        seed = record["seed"]
        if seed not in selected_by_seed:
            selected_by_seed[seed] = {"labels": [label], "record": record}
        else:
            selected_by_seed[seed]["labels"].append(label)

    finite_error_records = [
        r for r in records if np.isfinite(r.get("final_error", float("nan")))
    ]

    success_records = [r for r in records if r["success"]]
    timeout_records = [r for r in records if r["termination_reason"] == "timeout"]
    safety_records = [r for r in records if r["termination_reason"] == "safety_violation"]

    if success_records:
        add("first_success", success_records[0])

        if len(success_records) >= 3:
            ordered_success = sorted(success_records, key=lambda r: r["final_error"])
            add("median_success", ordered_success[len(ordered_success) // 2])

    if timeout_records:
        ordered_timeout = sorted(timeout_records, key=lambda r: r["final_error"])
        add("typical_timeout", ordered_timeout[len(ordered_timeout) // 2])

    if finite_error_records:
        add("worst_final_error", max(finite_error_records, key=lambda r: r["final_error"]))

    if records:
        add("highest_saturation", max(records, key=lambda r: r["saturation_fraction"]))

    if safety_records:
        add("first_safety_failure", safety_records[0])

    selected = []
    for item in selected_by_seed.values():
        label = "__".join(item["labels"])
        selected.append((label, item["record"]))

    return selected


def save_rollout_history(
    controller,
    q_scale,
    r_scale,
    Q,
    R,
    args,
    record,
    label,
    output_dir,
):
    """
    Re-runs one selected rollout and saves its full history as a compressed .npz.

    The re-run avoids storing every rollout trajectory during the sweep. Since the
    seed and controller are identical, the saved history corresponds to the same
    rollout selected from the sweep.
    """
    env, final_state, info, steps = run_single_rollout(
        controller=controller,
        max_time=args.max_time,
        seed=record["seed"],
        mode=args.mode,
    )

    times, states, actions_body_raw, actions_body_clipped, actions_orbital = (
        env.get_history_arrays()
    )

    final_error, component_error = compute_final_error(env, final_state)

    metadata = {
        "label": label,
        "seed": int(record["seed"]),
        "status": rollout_status(info),
        "success": bool(info.get("success", False)),
        "termination_reason": str(info.get("termination_reason", "")),
        "violated_constraints": list(info.get("violated_constraints", []) or []),
        "steps": int(steps),
        "elapsed_time": float(env.time),
        "saturation_fraction": saturation_fraction(env),
        "final_error": float(final_error),
        "final_error_components": component_error.tolist(),
        "q_scale": float(q_scale),
        "r_scale": float(r_scale),
        "q_multipliers": None if args.q_mults == "" else args.q_mults,
        "r_multipliers": None if args.r_mults == "" else args.r_mults,
        "v_fast": make_json_safe(getattr(controller, "v_fast", None)),
        "d_fast": make_json_safe(getattr(controller, "d_fast", None)),
        "d_slow": make_json_safe(getattr(controller, "d_slow", None)),
        "max_time": float(args.max_time),
        "mode": args.mode,
        "dt": float(env.dt),
        "phase": "closing",
        "selected_from_record": record,
    }

    safe_label = label.replace("/", "_").replace(" ", "_")
    filename = (
        f"q{q_scale:g}_r{r_scale:g}_seed{record['seed']:04d}_{safe_label}.npz"
    )
    path = os.path.join(output_dir, filename)

    np.savez_compressed(
        path,
        times=times,
        states=states,
        actions_body_raw=actions_body_raw,
        actions_body_clipped=actions_body_clipped,
        actions_orbital=actions_orbital,
        p3_target=np.asarray(env.p3_target, dtype=float)
        if env.p3_target is not None
        else np.full(6, np.nan),
        p3_target_tolerance=np.asarray(env.p3_target_tolerance, dtype=float)
        if env.p3_target_tolerance is not None
        else np.full(6, np.nan),
        final_state=np.asarray(final_state, dtype=float),
        Q=np.asarray(Q, dtype=float),
        R=np.asarray(R, dtype=float),
        metadata_json=json.dumps(make_json_safe(metadata), indent=2),
    )

    return path


def write_rollout_records_csv(records, output_path):
    """Write one analysis-ready row per phase rollout."""
    if not records:
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def run_rollouts(
    controller,
    num_rollouts,
    max_time,
    seed_start,
    mode="testing",
):
    successes = 0
    timeouts = 0
    safety_violations = 0
    violation_geometry = 0
    violation_velocity = 0
    violation_alignment = 0
    unsafe_at_reset = 0
    fail_rx = 0
    fail_ry = 0
    fail_vx = 0
    fail_vy = 0
    fail_psi = 0
    fail_omega = 0

    sat_fractions = []
    delta_v_proxies = []
    angular_effort_proxies = []
    rollout_records = []

    # Final error diagnostics relative to the sampled P3 target.
    final_errors = []
    final_ex = []
    final_ey = []
    final_evx = []
    final_evy = []
    final_epsi = []
    final_eomega = []

    # Final-state diagnostics.
    final_port_speeds = []
    final_port_distances = []
    final_com_distances = []
    final_rho = []
    final_psi_abs = []
    final_omega_abs = []
    final_vy = []
    final_vy_abs = []
    final_s_dot_abs = []
    final_rx_dot = []

    # Timing / distance diagnostics.
    initial_s_delta = []
    elapsed_times = []
    mean_vx_trajectory = []

    for seed in range(seed_start, seed_start + num_rollouts):
        env = Docking3DoFEnv(
            phase="closing",
            max_time=max_time,
            seed=seed,
        )

        state = env.reset(mode=mode)

        if env.p3_target is not None:
            initial_s_delta.append(abs(state[0] - env.p3_target[0]))

        initial_monitor = env.monitor_safety(state)
        if initial_monitor.get("unsafe"):
            unsafe_at_reset += 1

        done = False
        info = {}
        steps = 0
        max_steps = int(max_time / env.dt) + 1

        while not done and steps < max_steps:
            action = controller.compute_action(env, state)
            state, done, info = env.step(action)
            steps += 1

        if info.get("success"):
            successes += 1
        elif info.get("termination_reason") == "timeout":
            timeouts += 1
        elif info.get("termination_reason") == "safety_violation":
            safety_violations += 1
            violated = info.get("violated_constraints", [])
            if "geometry" in violated:
                violation_geometry += 1
            if "velocity" in violated:
                violation_velocity += 1
            if "alignment" in violated:
                violation_alignment += 1

        # Final error relative to the exact sampled P3 target.
        if env.p3_target is not None:
            e = state - env.p3_target
            e[4] = env.wrap_to_pi(state[4] - env.p3_target[4])

            final_errors.append(float(np.linalg.norm(e)))
            final_ex.append(abs(e[0]))
            final_ey.append(abs(e[1]))
            final_evx.append(abs(e[2]))
            final_evy.append(abs(e[3]))
            final_epsi.append(abs(e[4]))
            final_eomega.append(abs(e[5]))

            tol = env.p3_target_tolerance
            failed_components = np.abs(e) > tol

            fail_rx += int(failed_components[0])
            fail_ry += int(failed_components[1])
            fail_vx += int(failed_components[2])
            fail_vy += int(failed_components[3])
            fail_psi += int(failed_components[4])
            fail_omega += int(failed_components[5])

        sat_fractions.append(saturation_fraction(env))
        delta_v_proxies.append(translational_delta_v_proxy(env))
        angular_effort_proxies.append(angular_effort_proxy(env))

        q = env.get_derived_quantities(state)
        final_port_speeds.append(q["port_speed"])
        final_port_distances.append(q["port_distance"])
        final_com_distances.append(q["com_distance"])
        final_rho.append(q["rho"])
        final_psi_abs.append(abs(q["psi_e"]))
        final_omega_abs.append(abs(q["omega"]))
        final_vy.append(state[3])
        final_vy_abs.append(abs(state[3]))
        final_s_dot_abs.append(abs(q["s_dot"]))
        final_rx_dot.append(state[2])

        elapsed_times.append(env.time)
        mean_vx_trajectory.append(trajectory_mean_vx(env))

        rollout_records.append(build_rollout_record(env, state, info, seed, steps))

    success_rate = successes / float(num_rollouts)
    timeout_rate = timeouts / float(num_rollouts)
    safety_rate = safety_violations / float(num_rollouts)
    unsafe_reset_rate = unsafe_at_reset / float(num_rollouts)
    violation_geometry_rate = violation_geometry / float(num_rollouts)
    violation_velocity_rate = violation_velocity / float(num_rollouts)
    violation_alignment_rate = violation_alignment / float(num_rollouts)

    sat_mean = float(np.mean(sat_fractions)) if sat_fractions else 0.0

    final_error_mean = float(np.mean(final_errors)) if final_errors else float("nan")
    final_ex_mean = float(np.mean(final_ex)) if final_ex else float("nan")
    final_ey_mean = float(np.mean(final_ey)) if final_ey else float("nan")
    final_evx_mean = float(np.mean(final_evx)) if final_evx else float("nan")
    final_evy_mean = float(np.mean(final_evy)) if final_evy else float("nan")
    final_epsi_mean = float(np.mean(final_epsi)) if final_epsi else float("nan")
    final_eomega_mean = float(np.mean(final_eomega)) if final_eomega else float("nan")

    port_speed_mean = float(np.mean(final_port_speeds)) if final_port_speeds else float("nan")
    port_distance_mean = float(np.mean(final_port_distances)) if final_port_distances else float("nan")
    com_distance_mean = float(np.mean(final_com_distances)) if final_com_distances else float("nan")
    rho_mean = float(np.mean(final_rho)) if final_rho else float("nan")
    psi_abs_mean = float(np.mean(final_psi_abs)) if final_psi_abs else float("nan")
    omega_abs_mean = float(np.mean(final_omega_abs)) if final_omega_abs else float("nan")
    vy_mean = float(np.mean(final_vy)) if final_vy else float("nan")
    vy_abs_mean = float(np.mean(final_vy_abs)) if final_vy_abs else float("nan")
    s_dot_abs_mean = float(np.mean(final_s_dot_abs)) if final_s_dot_abs else float("nan")
    rx_dot_final_mean = float(np.mean(final_rx_dot)) if final_rx_dot else float("nan")

    s_delta_mean = float(np.mean(initial_s_delta)) if initial_s_delta else float("nan")
    elapsed_time_mean = float(np.mean(elapsed_times)) if elapsed_times else float("nan")
    vx_traj_mean = float(np.mean(mean_vx_trajectory)) if mean_vx_trajectory else float("nan")

    # This keeps your old diagnostic, but note that it uses final vx, not trajectory-average vx.
    if np.isfinite(rx_dot_final_mean) and abs(rx_dot_final_mean) > 1e-9:
        est_time_from_final_vx = s_delta_mean / abs(rx_dot_final_mean)
    else:
        est_time_from_final_vx = float("nan")

    # This is usually the more meaningful velocity-based estimate.
    if np.isfinite(vx_traj_mean) and abs(vx_traj_mean) > 1e-9:
        est_time_from_mean_vx = s_delta_mean / abs(vx_traj_mean)
    else:
        est_time_from_mean_vx = float("nan")

    result = {
        "success_rate": success_rate,
        "timeout_rate": timeout_rate,
        "safety_rate": safety_rate,
        "unsafe_reset_rate": unsafe_reset_rate,
        "geom_rate": violation_geometry_rate,
        "vel_rate": violation_velocity_rate,
        "align_rate": violation_alignment_rate,
        "sat_fraction": sat_mean,
        "delta_v_proxy": (
            float(np.mean(delta_v_proxies)) if delta_v_proxies else float("nan")
        ),
        "angular_effort": (
            float(np.mean(angular_effort_proxies))
            if angular_effort_proxies
            else float("nan")
        ),
        "final_err": final_error_mean,
        "ex": final_ex_mean,
        "ey": final_ey_mean,
        "evx": final_evx_mean,
        "evy": final_evy_mean,
        "epsi": final_epsi_mean,
        "eomega": final_eomega_mean,
        "vport_mean": port_speed_mean,
        "vport_dist": port_distance_mean,
        "com_dist": com_distance_mean,
        "rho_mean": rho_mean,
        "psi_abs": psi_abs_mean,
        "omega_abs": omega_abs_mean,
        "vy_mean": vy_mean,
        "vy_abs": vy_abs_mean,
        "sdot_mean": s_dot_abs_mean,
        "rx_dot_final": rx_dot_final_mean,
        "vx_traj_mean": vx_traj_mean,
        "elapsed_time": elapsed_time_mean,
        "est_time_final_vx": est_time_from_final_vx,
        "est_time_mean_vx": est_time_from_mean_vx,
        "fail_rx_rate": fail_rx / float(num_rollouts),
        "fail_ry_rate": fail_ry / float(num_rollouts),
        "fail_vx_rate": fail_vx / float(num_rollouts),
        "fail_vy_rate": fail_vy / float(num_rollouts),
        "fail_psi_rate": fail_psi / float(num_rollouts),
        "fail_omega_rate": fail_omega / float(num_rollouts),
    }

    return result, rollout_records


def print_header():
    header = (
        "q_scale r_scale success timeout safety unsafe0 geom vel align sat "
        "dv effort "
        "final_err ex ey evx evy epsi eomega "
        "vport vport_dist com_dist rho psi_abs omega_abs vy vy_abs sdot "
        "vx_final vx_mean elapsed est_final_vx est_mean_vx "
        "fail_rx fail_ry fail_vx fail_vy fail_psi fail_omega"
    )
    print(header)
    print("-" * len(header))


def print_result_row(q_scale, r_scale, result):
    print(
        f"{q_scale:7.3f} {r_scale:7.3f}"
        f" {result['success_rate']:7.3f}"
        f" {result['timeout_rate']:7.3f}"
        f" {result['safety_rate']:6.3f}"
        f" {result['unsafe_reset_rate']:7.3f}"
        f" {result['geom_rate']:5.3f}"
        f" {result['vel_rate']:5.3f}"
        f" {result['align_rate']:5.3f}"
        f" {result['sat_fraction']:5.3f}"
        f" {result['delta_v_proxy']:7.3f}"
        f" {result['angular_effort']:7.4f}"
        f" {result['final_err']:9.3f}"
        f" {result['ex']:7.3f}"
        f" {result['ey']:7.3f}"
        f" {result['evx']:7.4f}"
        f" {result['evy']:7.4f}"
        f" {np.rad2deg(result['epsi']):7.3f}"
        f" {np.rad2deg(result['eomega']):8.4f}"
        f" {result['vport_mean']:7.3f}"
        f" {result['vport_dist']:10.1f}"
        f" {result['com_dist']:8.1f}"
        f" {result['rho_mean']:6.2f}"
        f" {np.rad2deg(result['psi_abs']):7.3f}"
        f" {np.rad2deg(result['omega_abs']):9.4f}"
        f" {result['vy_mean']:7.3f}"
        f" {result['vy_abs']:7.3f}"
        f" {result['sdot_mean']:7.3f}"
        f" {result['rx_dot_final']:8.3f}"
        f" {result['vx_traj_mean']:7.3f}"
        f" {result['elapsed_time']:8.1f}"
        f" {result['est_time_final_vx']:12.1f}"
        f" {result['est_time_mean_vx']:11.1f}"
        f" {result['fail_rx_rate']:7.3f}"
        f" {result['fail_ry_rate']:7.3f}"
        f" {result['fail_vx_rate']:7.3f}"
        f" {result['fail_vy_rate']:7.3f}"
        f" {result['fail_psi_rate']:8.3f}"
        f" {result['fail_omega_rate']:10.3f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sweep Q/R scales for closing-phase LQR and report success/saturation."
    )
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--max-time", type=float, default=12000.0)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--mode", type=str, default="testing", choices=["nominal", "testing", "training"])
    parser.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="Write per-rollout records for a single candidate to this CSV.",
    )

    parser.add_argument("--q-scales", type=str, default="1.0")
    parser.add_argument("--r-scales", type=str, default="1.0")
    parser.add_argument(
        "--q-mults",
        type=str,
        default="",
        help="Per-state Q multipliers in order rx,ry,vx,vy,psi,omega. Six comma-separated values.",
    )
    parser.add_argument(
        "--r-mults",
        type=str,
        default="",
        help="Per-input R multipliers in order ax,ay,alpha. Three comma-separated values.",
    )

    # Controller schedule parameters.
    # parser.add_argument("--v-fast", type=float, default=-0.12)
    # parser.add_argument("--d-fast", type=float, default=200.0)
    # parser.add_argument("--d-slow", type=float, default=30.0)

    # Optional rollout-history saving for later diagnostic plotting.
    parser.add_argument(
        "--save-rollout-histories",
        action="store_true",
        help=(
            "Save only selected representative rollout histories per tuning. "
            "The sweep still prints summary statistics as before."
        ),
    )
    parser.add_argument(
        "--history-dir",
        type=str,
        default=os.path.join(LQR_ROOT, "results", "lqr_closing", "rollout_histories"),
        help="Directory where selected rollout .npz histories are saved.",
    )

    args = parser.parse_args()

    q_scales = parse_scales(args.q_scales)
    r_scales = parse_scales(args.r_scales)
    q_mults = parse_multipliers(args.q_mults, 6, "--q-mults")
    r_mults = parse_multipliers(args.r_mults, 3, "--r-mults")

    env_for_gain = Docking3DoFEnv(phase="closing", max_time=args.max_time, seed=0)
    A, B = build_linearized_3dof_matrices(env_for_gain)
    Q_base = apply_diag_multipliers(build_Q_closing(), q_mults)
    R_base = apply_diag_multipliers(build_R_closing(env_for_gain), r_mults)

    if q_mults is not None:
        print("q_multipliers:", q_mults)
    if r_mults is not None:
        print("r_multipliers:", r_mults)

    #print(
        #"controller schedule: "
        #f"v_fast={args.v_fast}, d_fast={args.d_fast}, d_slow={args.d_slow}"
    #)

    if args.save_rollout_histories:
        os.makedirs(args.history_dir, exist_ok=True)
        print(f"selected rollout histories will be saved to: {args.history_dir}")

    print_header()

    safe_candidates = []

    for q_scale in q_scales:
        for r_scale in r_scales:
            Q = Q_base * q_scale
            R = R_base * r_scale

            K, _ = compute_lqr_gain(A, B, Q, R)
            controller = LQRClosingController(K)

            result, rollout_records = run_rollouts(
                controller,
                num_rollouts=args.rollouts,
                max_time=args.max_time,
                seed_start=args.seed_start,
                mode=args.mode,
            )

            if args.results_csv:
                if len(q_scales) != 1 or len(r_scales) != 1:
                    raise ValueError(
                        "--results-csv requires exactly one q-scale and one r-scale."
                    )
                write_rollout_records_csv(rollout_records, args.results_csv)
                print(f"rollout records written to: {args.results_csv}")

            print_result_row(q_scale, r_scale, result)

            if args.save_rollout_histories:
                selected = select_representative_rollouts(rollout_records)
                saved_paths = []
                for label, record in selected:
                    path = save_rollout_history(
                        controller=controller,
                        q_scale=q_scale,
                        r_scale=r_scale,
                        Q=Q,
                        R=R,
                        args=args,
                        record=record,
                        label=label,
                        output_dir=args.history_dir,
                    )
                    saved_paths.append(path)

                if saved_paths:
                    print(
                        f"  saved {len(saved_paths)} selected histories for "
                        f"q_scale={q_scale:g}, r_scale={r_scale:g}"
                    )
                    for path in saved_paths:
                        print(f"    {path}")

            if result["safety_rate"] == 0.0:
                safe_candidates.append((q_scale, r_scale, result))

    if safe_candidates:
        print("\nSafe candidates (safety_rate=0):")
        for q_scale, r_scale, result in safe_candidates:
            print(
                f"  q_scale={q_scale:.3f}, r_scale={r_scale:.3f}, "
                f"success={result['success_rate']:.3f}, "
                f"timeout={result['timeout_rate']:.3f}, "
                f"final_err={result['final_err']:.3f}, "
                f"ex={result['ex']:.3f}, ey={result['ey']:.3f}, "
                f"evx={result['evx']:.4f}, evy={result['evy']:.4f}, "
                f"est_mean_vx={result['est_time_mean_vx']:.1f}s"
            )


if __name__ == "__main__":
    main()
