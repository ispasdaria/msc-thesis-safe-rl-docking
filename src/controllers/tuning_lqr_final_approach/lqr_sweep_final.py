import argparse
import os
import sys

import numpy as np

# Allow imports from the repo's src/ directory when running from the project root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from controllers.lqr_3dof_final import (
    LQRFinalController,
    build_linearized_3dof_matrices,
    build_port_error_mapping,
    build_Q_final,
    build_R_final,
    compute_lqr_gain,
    final_error,
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
    Applies per-state/per-port-error or per-input multipliers to the diagonal
    entries of Q/R.

    For final approach, Q multipliers are applied to Q_port before the mapping

        Q_com = H.T @ Q_port @ H

    is formed.
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
    _, _, actions_body_raw, actions_body_clipped, _, _ = env.get_history_arrays()

    if actions_body_raw.size == 0 or actions_body_clipped.size == 0:
        return 0.0

    diff = actions_body_raw - actions_body_clipped
    saturated_per_channel = np.abs(diff) > atol
    saturated_any = np.any(saturated_per_channel, axis=1)

    return float(np.mean(saturated_any))


def translational_delta_v_proxy(env):
    """
    Simple control-effort proxy: sum ||a_trans,clipped|| dt.

    This is not exact propellant use, but it is useful for comparing controllers.
    """
    times, _, _, actions_body_clipped, _, _ = env.get_history_arrays()

    if actions_body_clipped.size == 0:
        return 0.0

    if times.size >= 2:
        dt = float(np.median(np.diff(times)))
    else:
        dt = float(getattr(env, "dt", 1.0))

    a_trans = actions_body_clipped[:, :2]
    return float(np.sum(np.linalg.norm(a_trans, axis=1)) * dt)


def angular_effort_proxy(env):
    """
    Simple angular-effort proxy: sum |alpha_z,clipped| dt.
    """
    times, _, _, actions_body_clipped, _, _ = env.get_history_arrays()

    if actions_body_clipped.size == 0:
        return 0.0

    if times.size >= 2:
        dt = float(np.median(np.diff(times)))
    else:
        dt = float(getattr(env, "dt", 1.0))

    return float(np.sum(np.abs(actions_body_clipped[:, 2])) * dt)


def run_rollouts(
    controller,
    num_rollouts,
    max_time,
    seed_start,
    mode="testing",
    dock_dist_tol=0.1,
    dock_speed_tol=0.1,
    psi_tol=np.deg2rad(4.0),
    omega_tol=np.deg2rad(0.2),
):
    successes = 0
    timeouts = 0
    safety_violations = 0
    violation_geometry = 0
    violation_velocity = 0
    violation_alignment = 0
    unsafe_at_reset = 0

    fail_dist = 0
    fail_speed = 0
    fail_psi = 0
    fail_omega = 0

    sat_fractions = []
    elapsed_times = []
    delta_v_proxies = []
    angular_effort_proxies = []

    final_port_distances = []
    final_port_speeds = []
    final_rpx_abs = []
    final_rpy_abs = []
    final_vpx_abs = []
    final_vpy_abs = []
    final_psi_abs = []
    final_omega_abs = []

    for seed in range(seed_start, seed_start + num_rollouts):
        env = Docking3DoFEnv(
            phase="final",
            max_time=max_time,
            seed=seed,
        )

        state = env.reset(mode=mode)

        initial_monitor = env.monitor_safety(state)
        if initial_monitor.get("unsafe"):
            unsafe_at_reset += 1

        done = False
        info = {}
        steps = 0
        max_steps = int(max_time / env.dt) + 1

        while not done and steps < max_steps:
            action = controller.compute_action(env, state)
            state, _, done, info = env.step(action)
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

        # Final port-level error diagnostics.
        e = final_error(env, state)
        rpx, rpy, vpx, vpy, psi_e, omega = e

        port_distance = float(np.linalg.norm(e[:2]))
        port_speed = float(np.linalg.norm(e[2:4]))

        final_port_distances.append(port_distance)
        final_port_speeds.append(port_speed)
        final_rpx_abs.append(abs(rpx))
        final_rpy_abs.append(abs(rpy))
        final_vpx_abs.append(abs(vpx))
        final_vpy_abs.append(abs(vpy))
        final_psi_abs.append(abs(psi_e))
        final_omega_abs.append(abs(omega))

        fail_dist += int(port_distance > dock_dist_tol)
        fail_speed += int(port_speed > dock_speed_tol)
        fail_psi += int(abs(psi_e) > psi_tol)
        fail_omega += int(abs(omega) > omega_tol)

        sat_fractions.append(saturation_fraction(env))
        elapsed_times.append(env.time)
        delta_v_proxies.append(translational_delta_v_proxy(env))
        angular_effort_proxies.append(angular_effort_proxy(env))

    success_rate = successes / float(num_rollouts)
    timeout_rate = timeouts / float(num_rollouts)
    safety_rate = safety_violations / float(num_rollouts)
    unsafe_reset_rate = unsafe_at_reset / float(num_rollouts)
    violation_geometry_rate = violation_geometry / float(num_rollouts)
    violation_velocity_rate = violation_velocity / float(num_rollouts)
    violation_alignment_rate = violation_alignment / float(num_rollouts)

    return {
        "success_rate": success_rate,
        "timeout_rate": timeout_rate,
        "safety_rate": safety_rate,
        "unsafe_reset_rate": unsafe_reset_rate,
        "geom_rate": violation_geometry_rate,
        "vel_rate": violation_velocity_rate,
        "align_rate": violation_alignment_rate,
        "sat_fraction": float(np.mean(sat_fractions)) if sat_fractions else 0.0,
        "port_dist": float(np.mean(final_port_distances)) if final_port_distances else float("nan"),
        "port_speed": float(np.mean(final_port_speeds)) if final_port_speeds else float("nan"),
        "rpx_abs": float(np.mean(final_rpx_abs)) if final_rpx_abs else float("nan"),
        "rpy_abs": float(np.mean(final_rpy_abs)) if final_rpy_abs else float("nan"),
        "vpx_abs": float(np.mean(final_vpx_abs)) if final_vpx_abs else float("nan"),
        "vpy_abs": float(np.mean(final_vpy_abs)) if final_vpy_abs else float("nan"),
        "psi_abs": float(np.mean(final_psi_abs)) if final_psi_abs else float("nan"),
        "omega_abs": float(np.mean(final_omega_abs)) if final_omega_abs else float("nan"),
        "elapsed_time": float(np.mean(elapsed_times)) if elapsed_times else float("nan"),
        "delta_v_proxy": float(np.mean(delta_v_proxies)) if delta_v_proxies else float("nan"),
        "angular_effort": float(np.mean(angular_effort_proxies)) if angular_effort_proxies else float("nan"),
        "fail_dist_rate": fail_dist / float(num_rollouts),
        "fail_speed_rate": fail_speed / float(num_rollouts),
        "fail_psi_rate": fail_psi / float(num_rollouts),
        "fail_omega_rate": fail_omega / float(num_rollouts),
    }


def print_header():
    header = (
        "q_scale r_scale success timeout safety unsafe0 geom vel align sat "
        "pdist pspeed rpx rpy vpx vpy psi omega elapsed dv effort "
        "fail_dist fail_speed fail_psi fail_omega"
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
        f" {result['port_dist']:6.3f}"
        f" {result['port_speed']:6.3f}"
        f" {result['rpx_abs']:6.3f}"
        f" {result['rpy_abs']:6.3f}"
        f" {result['vpx_abs']:6.4f}"
        f" {result['vpy_abs']:6.4f}"
        f" {np.rad2deg(result['psi_abs']):6.3f}"
        f" {np.rad2deg(result['omega_abs']):7.4f}"
        f" {result['elapsed_time']:7.1f}"
        f" {result['delta_v_proxy']:7.3f}"
        f" {result['angular_effort']:7.3f}"
        f" {result['fail_dist_rate']:9.3f}"
        f" {result['fail_speed_rate']:10.3f}"
        f" {result['fail_psi_rate']:8.3f}"
        f" {result['fail_omega_rate']:10.3f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sweep Q/R scales for final-approach LQR and report success/saturation."
    )
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--max-time", type=float, default=5000.0)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--mode", type=str, default="testing", choices=["nominal", "testing", "training"])

    parser.add_argument("--q-scales", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--r-scales", type=str, default="1.0,2.0,5.0")
    parser.add_argument(
        "--q-mults",
        type=str,
        default="",
        help=(
            "Per-port-error Q multipliers in order "
            "r_px,r_py,v_px,v_py,psi,omega. Six comma-separated values."
        ),
    )
    parser.add_argument(
        "--r-mults",
        type=str,
        default="",
        help="Per-input R multipliers in order ax,ay,alpha. Three comma-separated values.",
    )

    parser.add_argument("--dock-dist", type=float, default=0.1)
    parser.add_argument("--dock-speed", type=float, default=0.1)
    parser.add_argument("--psi-tol-deg", type=float, default=4.0)
    parser.add_argument("--omega-tol-deg", type=float, default=0.2)

    args = parser.parse_args()

    q_scales = parse_scales(args.q_scales)
    r_scales = parse_scales(args.r_scales)
    q_mults = parse_multipliers(args.q_mults, 6, "--q-mults")
    r_mults = parse_multipliers(args.r_mults, 3, "--r-mults")

    env_for_gain = Docking3DoFEnv(phase="final", max_time=args.max_time, seed=0)
    A, B = build_linearized_3dof_matrices(env_for_gain)
    H = build_port_error_mapping(env_for_gain)

    Q_port_base = apply_diag_multipliers(build_Q_final(), q_mults)
    R_base = apply_diag_multipliers(build_R_final(env_for_gain), r_mults)

    if q_mults is not None:
        print("q_multipliers:", q_mults)
    if r_mults is not None:
        print("r_multipliers:", r_mults)

    print(
        "terminal tolerances: "
        f"dock_dist={args.dock_dist}, dock_speed={args.dock_speed}, "
        f"psi_tol={args.psi_tol_deg} deg, omega_tol={args.omega_tol_deg} deg/s"
    )

    print_header()

    safe_candidates = []

    for q_scale in q_scales:
        for r_scale in r_scales:
            Q_port = Q_port_base * q_scale
            Q_com = H.T @ Q_port @ H
            R = R_base * r_scale

            K, _ = compute_lqr_gain(A, B, Q_com, R)
            controller = LQRFinalController(K=K, H=H)

            result = run_rollouts(
                controller,
                num_rollouts=args.rollouts,
                max_time=args.max_time,
                seed_start=args.seed_start,
                mode=args.mode,
                dock_dist_tol=args.dock_dist,
                dock_speed_tol=args.dock_speed,
                psi_tol=np.deg2rad(args.psi_tol_deg),
                omega_tol=np.deg2rad(args.omega_tol_deg),
            )

            print_result_row(q_scale, r_scale, result)

            if result["safety_rate"] == 0.0:
                safe_candidates.append((q_scale, r_scale, result))

    if safe_candidates:
        print("\nSafe candidates (safety_rate=0):")
        for q_scale, r_scale, result in safe_candidates:
            print(
                f"  q_scale={q_scale:.3f}, r_scale={r_scale:.3f}, "
                f"success={result['success_rate']:.3f}, "
                f"timeout={result['timeout_rate']:.3f}, "
                f"pdist={result['port_dist']:.3f}, "
                f"pspeed={result['port_speed']:.3f}, "
                f"psi={np.rad2deg(result['psi_abs']):.3f} deg, "
                f"omega={np.rad2deg(result['omega_abs']):.4f} deg/s, "
                f"sat={result['sat_fraction']:.3f}"
            )


if __name__ == "__main__":
    main()
