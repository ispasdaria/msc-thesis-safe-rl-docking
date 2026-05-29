import argparse
import os
import sys

import numpy as np

# Allow imports from the repo's src/ directory when running from the project root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from controllers.lqr_3dof_closing import (
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
    if multipliers is None:
        return base

    diag = np.diag(base)
    return np.diag(diag * multipliers)


def saturation_fraction(env, atol=1e-12):
    _, _, actions_body_raw, actions_body_clipped, _, _ = env.get_history_arrays()

    if actions_body_raw.size == 0 or actions_body_clipped.size == 0:
        return 0.0

    diff = actions_body_raw - actions_body_clipped
    saturated_per_channel = np.abs(diff) > atol
    saturated_any = np.any(saturated_per_channel, axis=1)

    return float(np.mean(saturated_any))


def run_rollouts(controller, num_rollouts, max_time, seed_start):
    successes = 0
    timeouts = 0
    safety_violations = 0
    violation_geometry = 0
    violation_velocity = 0
    violation_alignment = 0
    unsafe_at_reset = 0
    sat_fractions = []
    final_errors = []
    mean_port_speeds = []
    mean_s_dot_abs = []
    mean_rx_dot = []
    mean_s_delta = []

    for seed in range(seed_start, seed_start + num_rollouts):
        env = Docking3DoFEnv(
            phase="closing",
            max_time=max_time,
            seed=seed,
        )
        state = env.reset(mode="testing")
        if env.p3_target is not None:
            mean_s_delta.append(abs(state[0] - env.p3_target[0]))

        initial_monitor = env.monitor_safety(state)
        if initial_monitor.get("unsafe"):
            unsafe_at_reset += 1

        done = False
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

        if env.p3_target is not None:
            final_errors.append(np.linalg.norm(state - env.p3_target))

        sat_fractions.append(saturation_fraction(env))
        q = env.get_derived_quantities(state)
        mean_port_speeds.append(q["port_speed"])
        mean_s_dot_abs.append(abs(q["s_dot"]))
        mean_rx_dot.append(state[2])

    success_rate = successes / float(num_rollouts)
    timeout_rate = timeouts / float(num_rollouts)
    safety_rate = safety_violations / float(num_rollouts)
    unsafe_reset_rate = unsafe_at_reset / float(num_rollouts)
    violation_geometry_rate = violation_geometry / float(num_rollouts)
    violation_velocity_rate = violation_velocity / float(num_rollouts)
    violation_alignment_rate = violation_alignment / float(num_rollouts)
    sat_mean = float(np.mean(sat_fractions)) if sat_fractions else 0.0
    final_error_mean = float(np.mean(final_errors)) if final_errors else float("nan")
    port_speed_mean = float(np.mean(mean_port_speeds)) if mean_port_speeds else float("nan")
    s_dot_abs_mean = float(np.mean(mean_s_dot_abs)) if mean_s_dot_abs else float("nan")
    rx_dot_mean = float(np.mean(mean_rx_dot)) if mean_rx_dot else float("nan")
    s_delta_mean = float(np.mean(mean_s_delta)) if mean_s_delta else float("nan")

    if np.isfinite(rx_dot_mean) and abs(rx_dot_mean) > 1e-9:
        est_time_mean = s_delta_mean / abs(rx_dot_mean)
    else:
        est_time_mean = float("nan")

    return (
        success_rate,
        timeout_rate,
        safety_rate,
        unsafe_reset_rate,
        violation_geometry_rate,
        violation_velocity_rate,
        violation_alignment_rate,
        sat_mean,
        final_error_mean,
        port_speed_mean,
        s_dot_abs_mean,
        rx_dot_mean,
        est_time_mean,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sweep Q/R scales for closing-phase LQR and report success/saturation."
    )
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--max-time", type=float, default=12000.0)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--q-scales", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--r-scales", type=str, default="1.0,2.0,4.0")
    parser.add_argument(
        "--q-mults",
        type=str,
        default="",
        help="Per-state Q multipliers (6 values, comma-separated).",
    )
    parser.add_argument(
        "--r-mults",
        type=str,
        default="50,50,50",
        help="Per-input R multipliers (3 values, comma-separated).",
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

    header = (
        "q_scale r_scale success_rate timeout_rate safety_rate unsafe0_rate "
        "geom_rate vel_rate align_rate sat_fraction final_err vport_mean sdot_mean rx_dot est_time"
    )
    print(header)
    print("-" * len(header))

    safe_candidates = []

    for q_scale in q_scales:
        for r_scale in r_scales:
            Q = Q_base * q_scale
            R = R_base * r_scale

            K, _ = compute_lqr_gain(A, B, Q, R)
            controller = LQRClosingController(K)

            (
                success_rate,
                timeout_rate,
                safety_rate,
                unsafe_reset_rate,
                geom_rate,
                vel_rate,
                align_rate,
                sat_mean,
                final_err,
                vport_mean,
                sdot_mean,
                rx_dot_mean,
                est_time_mean,
            ) = run_rollouts(
                controller,
                num_rollouts=args.rollouts,
                max_time=args.max_time,
                seed_start=args.seed_start,
            )

            print(
                f"{q_scale:7.3f} {r_scale:7.3f}"
                f" {success_rate:12.3f} {timeout_rate:12.3f}"
                f" {safety_rate:11.3f} {unsafe_reset_rate:12.3f}"
                f" {geom_rate:9.3f} {vel_rate:8.3f}"
                f" {align_rate:9.3f} {sat_mean:12.3f} {final_err:9.3f}"
                f" {vport_mean:10.3f} {sdot_mean:9.3f} {rx_dot_mean:7.3f}"
                f" {est_time_mean:9.1f}"
            )

            if safety_rate == 0.0:
                safe_candidates.append(
                    (q_scale, r_scale, success_rate, timeout_rate, est_time_mean)
                )

    if safe_candidates:
        print("\nSafe candidates (safety_rate=0):")
        for q_scale, r_scale, success_rate, timeout_rate, est_time in safe_candidates:
            print(
                f"  q_scale={q_scale:.3f}, r_scale={r_scale:.3f}, "
                f"success={success_rate:.3f}, timeout={timeout_rate:.3f}, "
                f"est_time={est_time:.1f}s"
            )


if __name__ == "__main__":
    main()
