import argparse
import os
import sys

import numpy as np

# Allow imports from the repo's src/ directory when running from the project root.
LQR_ROOT = os.path.dirname(__file__)
while not os.path.exists(os.path.join(LQR_ROOT, "lqr_3dof_closing_error_sat.py")):
    parent_dir = os.path.dirname(LQR_ROOT)
    if parent_dir == LQR_ROOT:
        break
    LQR_ROOT = parent_dir
SRC_PATH = os.path.abspath(os.path.join(LQR_ROOT, "..", ".."))
if LQR_ROOT not in sys.path:
    sys.path.insert(0, LQR_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from lqr_3dof_closing_error_sat import (
    LQRClosingControllerErrorSat,
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


def parse_error_limits(text):
    if not text:
        return None

    values = parse_scales(text)

    if len(values) != 6:
        raise ValueError(
            "--error-limits must have 6 values in order "
            "rx,ry,vx,vy,psi_deg,omega_deg_s."
        )

    # User enters attitude limits in degrees for convenience.
    return np.array(
        [
            values[0],
            values[1],
            values[2],
            values[3],
            np.deg2rad(values[4]),
            np.deg2rad(values[5]),
        ],
        dtype=float,
    )


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
    Computes actuator saturation fractions from the environment action logs.

    Returns:
        sat_any   : fraction of timesteps where at least one channel saturated
        sat_ax    : fraction of timesteps where ax_Bc saturated
        sat_ay    : fraction of timesteps where ay_Bc saturated
        sat_alpha : fraction of timesteps where alphaz_Bc saturated
    """
    _, _, actions_body_raw, actions_body_clipped, _, _ = env.get_history_arrays()

    if actions_body_raw.size == 0 or actions_body_clipped.size == 0:
        return {
            "sat_any": 0.0,
            "sat_ax": 0.0,
            "sat_ay": 0.0,
            "sat_alpha": 0.0,
        }

    diff = actions_body_raw - actions_body_clipped

    saturated_per_channel = np.abs(diff) > atol
    saturated_any = np.any(saturated_per_channel, axis=1)

    return {
        "sat_any": float(np.mean(saturated_any)),
        "sat_ax": float(np.mean(saturated_per_channel[:, 0])),
        "sat_ay": float(np.mean(saturated_per_channel[:, 1])),
        "sat_alpha": float(np.mean(saturated_per_channel[:, 2])),
    }


def trajectory_mean_vx(env):
    """
    Mean vx over the whole stored trajectory.

    This is different from the final vx. It is useful as a diagnostic only,
    not as a success criterion.
    """
    _, states, _, _, _, _ = env.get_history_arrays()

    if states.size == 0:
        return float("nan")

    return float(np.mean(states[:, 2]))


def print_safety_violation_debug(seed, state, info, derived):
    """
    Optional compact debug print for safety-violation rollouts.
    """
    print("\nSafety violation")
    print("----------------")
    print(f"seed:      {seed}")
    print(f"violated:  {info.get('violated_constraints', [])}")
    print(
        "state:     "
        f"rx={state[0]:.3f}, ry={state[1]:.3f}, "
        f"vx={state[2]:.4f}, vy={state[3]:.4f}, "
        f"psi={np.rad2deg(state[4]):.3f} deg, "
        f"omega={np.rad2deg(state[5]):.4f} deg/s"
    )
    print(
        "derived:   "
        f"s={derived['s']:.3f}, rho={derived['rho']:.3f}, "
        f"s_dot={derived['s_dot']:.4f}, "
        f"vport={derived['port_speed']:.4f}, "
        f"psi_e={np.rad2deg(derived['psi_e']):.3f} deg, "
        f"omega={np.rad2deg(derived['omega']):.4f} deg/s"
    )


def run_rollouts(
    controller,
    num_rollouts,
    max_time,
    seed_start,
    mode="testing",
    debug_failures=False,
    max_debug_failures=3,
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

    debug_failure_count = 0

    sat_any_values = []
    sat_ax_values = []
    sat_ay_values = []
    sat_alpha_values = []

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

            if debug_failures and debug_failure_count < max_debug_failures:
                print_safety_violation_debug(
                    seed=seed,
                    state=state,
                    info=info,
                    derived=env.get_derived_quantities(state),
                )
                debug_failure_count += 1

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

        sat = saturation_fraction(env)
        sat_any_values.append(sat["sat_any"])
        sat_ax_values.append(sat["sat_ax"])
        sat_ay_values.append(sat["sat_ay"])
        sat_alpha_values.append(sat["sat_alpha"])

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

    success_rate = successes / float(num_rollouts)
    timeout_rate = timeouts / float(num_rollouts)
    safety_rate = safety_violations / float(num_rollouts)
    unsafe_reset_rate = unsafe_at_reset / float(num_rollouts)
    violation_geometry_rate = violation_geometry / float(num_rollouts)
    violation_velocity_rate = violation_velocity / float(num_rollouts)
    violation_alignment_rate = violation_alignment / float(num_rollouts)

    sat_mean = float(np.mean(sat_any_values)) if sat_any_values else 0.0
    sat_ax_mean = float(np.mean(sat_ax_values)) if sat_ax_values else 0.0
    sat_ay_mean = float(np.mean(sat_ay_values)) if sat_ay_values else 0.0
    sat_alpha_mean = float(np.mean(sat_alpha_values)) if sat_alpha_values else 0.0

    final_error_mean = float(np.mean(final_errors)) if final_errors else float("nan")
    final_ex_mean = float(np.mean(final_ex)) if final_ex else float("nan")
    final_ey_mean = float(np.mean(final_ey)) if final_ey else float("nan")
    final_evx_mean = float(np.mean(final_evx)) if final_evx else float("nan")
    final_evy_mean = float(np.mean(final_evy)) if final_evy else float("nan")
    final_epsi_mean = float(np.mean(final_epsi)) if final_epsi else float("nan")
    final_eomega_mean = float(np.mean(final_eomega)) if final_eomega else float("nan")

    port_speed_mean = (
        float(np.mean(final_port_speeds)) if final_port_speeds else float("nan")
    )
    port_distance_mean = (
        float(np.mean(final_port_distances)) if final_port_distances else float("nan")
    )
    com_distance_mean = (
        float(np.mean(final_com_distances)) if final_com_distances else float("nan")
    )
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

    if np.isfinite(rx_dot_final_mean) and abs(rx_dot_final_mean) > 1e-9:
        est_time_from_final_vx = s_delta_mean / abs(rx_dot_final_mean)
    else:
        est_time_from_final_vx = float("nan")

    if np.isfinite(vx_traj_mean) and abs(vx_traj_mean) > 1e-9:
        est_time_from_mean_vx = s_delta_mean / abs(vx_traj_mean)
    else:
        est_time_from_mean_vx = float("nan")

    return {
        "success_rate": success_rate,
        "timeout_rate": timeout_rate,
        "safety_rate": safety_rate,
        "unsafe_reset_rate": unsafe_reset_rate,
        "geom_rate": violation_geometry_rate,
        "vel_rate": violation_velocity_rate,
        "align_rate": violation_alignment_rate,
        "sat_fraction": sat_mean,
        "sat_ax": sat_ax_mean,
        "sat_ay": sat_ay_mean,
        "sat_alpha": sat_alpha_mean,
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


def print_configuration(args, q_scales, r_scales, q_mults, r_mults, error_limits):
    print()
    print("Closing LQR sweep configuration")
    print("=" * 40)
    print(f"rollouts:            {args.rollouts}")
    print(f"max_time:            {args.max_time:.1f} s")
    print(f"mode:                {args.mode}")
    print(f"seed_start:          {args.seed_start}")
    print(f"q_scales:            {q_scales}")
    print(f"r_scales:            {r_scales}")
    print(f"q_multipliers:       {q_mults if q_mults is not None else 'base'}")
    print(f"r_multipliers:       {r_mults if r_mults is not None else 'base'}")
    print(f"v_fast:              {args.v_fast:.3f} m/s")
    print(f"d_fast:              {args.d_fast:.1f} m")
    print(f"d_slow:              {args.d_slow:.1f} m")
    print(f"error_limit_mode:    {args.error_limit_mode}")

    if error_limits is None:
        print("error_limits:        controller default")
    else:
        print(
            "error_limits:        "
            f"rx={error_limits[0]:.1f} m, "
            f"ry={error_limits[1]:.1f} m, "
            f"vx={error_limits[2]:.3f} m/s, "
            f"vy={error_limits[3]:.3f} m/s, "
            f"psi={np.rad2deg(error_limits[4]):.2f} deg, "
            f"omega={np.rad2deg(error_limits[5]):.3f} deg/s"
        )

    print("velocity schedule:   enabled")
    print(f"debug_failures:      {args.debug_failures}")
    if args.debug_failures:
        print(f"max_debug_failures:  {args.max_debug_failures} per Q/R case")


def print_controller_settings(controller):
    print()
    print("Controller settings")
    print("=" * 40)
    print("velocity schedule: enabled")
    print(f"v_fast:          {controller.v_fast:.3f} m/s")
    print(f"d_fast:          {controller.d_fast:.1f} m")
    print(f"d_slow:          {controller.d_slow:.1f} m")
    print(f"error_limit_mode: {controller.error_limit_mode}")
    print(
        "error_limits:     "
        f"rx={controller.error_limits[0]:.1f} m, "
        f"ry={controller.error_limits[1]:.1f} m, "
        f"vx={controller.error_limits[2]:.3f} m/s, "
        f"vy={controller.error_limits[3]:.3f} m/s, "
        f"psi={np.rad2deg(controller.error_limits[4]):.2f} deg, "
        f"omega={np.rad2deg(controller.error_limits[5]):.3f} deg/s"
    )


def print_header():
    print()
    print("Sweep results")
    print("=" * 150)

    header = (
        f"{'q':>5} {'r':>5} | "
        f"{'succ':>5} {'tout':>5} {'safe':>5} | "
        f"{'geom':>5} {'vel':>5} {'aln':>5} | "
        f"{'sat':>5} {'sax':>5} {'say':>5} {'sal':>5} | "
        f"{'ex[m]':>8} {'ey[m]':>8} | "
        f"{'evx':>7} {'evy':>7} | "
        f"{'vport':>7} {'vy':>7} {'sdot':>7} | "
        f"{'psi[deg]':>8} {'omg[deg/s]':>10} | "
        f"{'t[s]':>8}"
    )

    print(header)
    print("-" * len(header))


def print_result_row(q_scale, r_scale, result):
    print(
        f"{q_scale:5.2f} {r_scale:5.2f} | "
        f"{result['success_rate']:5.2f} "
        f"{result['timeout_rate']:5.2f} "
        f"{result['safety_rate']:5.2f} | "
        f"{result['geom_rate']:5.2f} "
        f"{result['vel_rate']:5.2f} "
        f"{result['align_rate']:5.2f} | "
        f"{result['sat_fraction']:5.2f} "
        f"{result['sat_ax']:5.2f} "
        f"{result['sat_ay']:5.2f} "
        f"{result['sat_alpha']:5.2f} | "
        f"{result['ex']:8.1f} "
        f"{result['ey']:8.1f} | "
        f"{result['evx']:7.4f} "
        f"{result['evy']:7.4f} | "
        f"{result['vport_mean']:7.3f} "
        f"{result['vy_abs']:7.3f} "
        f"{result['sdot_mean']:7.3f} | "
        f"{np.rad2deg(result['epsi']):8.3f} "
        f"{np.rad2deg(result['eomega']):10.4f} | "
        f"{result['elapsed_time']:8.1f}"
    )


def print_safe_candidates(safe_candidates):
    if not safe_candidates:
        print()
        print("No fully safe candidates found in this sweep.")
        return

    print()
    print("Safe candidates")
    print("=" * 110)

    header = (
        f"{'q':>5} {'r':>5} | "
        f"{'succ':>5} {'tout':>5} {'safe':>5} | "
        f"{'sat':>5} {'sax':>5} {'say':>5} {'sal':>5} | "
        f"{'ex[m]':>8} {'ey[m]':>8} | "
        f"{'evx':>7} {'evy':>7} | "
        f"{'vport':>7} {'t[s]':>8}"
    )
    print(header)
    print("-" * len(header))

    for q_scale, r_scale, result in safe_candidates:
        print(
            f"{q_scale:5.2f} {r_scale:5.2f} | "
            f"{result['success_rate']:5.2f} "
            f"{result['timeout_rate']:5.2f} "
            f"{result['safety_rate']:5.2f} | "
            f"{result['sat_fraction']:5.2f} "
            f"{result['sat_ax']:5.2f} "
            f"{result['sat_ay']:5.2f} "
            f"{result['sat_alpha']:5.2f} | "
            f"{result['ex']:8.1f} "
            f"{result['ey']:8.1f} | "
            f"{result['evx']:7.4f} "
            f"{result['evy']:7.4f} | "
            f"{result['vport_mean']:7.3f} "
            f"{result['elapsed_time']:8.1f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep Q/R scales for the closing-phase LQR using both velocity "
            "scheduling and error saturation."
        )
    )

    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--max-time", type=float, default=12000.0)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--mode",
        type=str,
        default="testing",
        choices=[
            "none",
            "hard",
            "smooth",
            "norm",
            "smooth_norm",
            "mixed_norm",
            "grouped_smooth_norm",
        ]
    )

    parser.add_argument("--q-scales", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--r-scales", type=str, default="1.0,2.0,4.0")
    parser.add_argument(
        "--q-mults",
        type=str,
        default="",
        help=(
            "Per-state Q multipliers in order rx,ry,vx,vy,psi,omega. "
            "Six comma-separated values."
        ),
    )
    parser.add_argument(
        "--r-mults",
        type=str,
        default="50,50,50",
        help=(
            "Per-input R multipliers in order ax,ay,alpha. "
            "Three comma-separated values."
        ),
    )

    # Velocity schedule parameters.
    parser.add_argument("--v-fast", type=float, default=-0.15)
    parser.add_argument("--d-fast", type=float, default=100.0)
    parser.add_argument("--d-slow", type=float, default=10.0)

    # Error saturation parameters.
    parser.add_argument(
        "--error-limit-mode",
        type=str,
        default="smooth",
        choices=[
            "none",
            "hard",
            "smooth",
            "norm",
            "smooth_norm",
            "mixed_norm",
            "grouped_smooth_norm",
        ],
        help=(
            "Error saturation mode used before applying the LQR gain. "
            "Modes: none, hard, smooth, norm, smooth_norm, mixed_norm, "
            "grouped_smooth_norm."
        ),
    )
    parser.add_argument(
        "--error-limits",
        type=str,
        default="",
        help=(
            "Optional apparent-error limits in order "
            "rx,ry,vx,vy,psi_deg,omega_deg_s. "
            "Example: 100,20,0.05,0.03,5,0.2"
        ),
    )

    parser.add_argument(
        "--debug-failures",
        action="store_true",
        help="Print detailed information for safety-violation rollouts.",
    )
    parser.add_argument(
        "--max-debug-failures",
        type=int,
        default=3,
        help=(
            "Maximum number of safety-violation rollouts to print per Q/R case "
            "when --debug-failures is enabled."
        ),
    )

    args = parser.parse_args()

    if args.max_debug_failures < 0:
        raise ValueError("--max-debug-failures must be non-negative.")

    if args.d_fast <= args.d_slow:
        raise ValueError("--d-fast must be greater than --d-slow.")

    q_scales = parse_scales(args.q_scales)
    r_scales = parse_scales(args.r_scales)
    q_mults = parse_multipliers(args.q_mults, 6, "--q-mults")
    r_mults = parse_multipliers(args.r_mults, 3, "--r-mults")
    error_limits = parse_error_limits(args.error_limits)

    env_for_gain = Docking3DoFEnv(
        phase="closing",
        max_time=args.max_time,
        seed=0,
    )

    A, B = build_linearized_3dof_matrices(env_for_gain)
    Q_base = apply_diag_multipliers(build_Q_closing(), q_mults)
    R_base = apply_diag_multipliers(build_R_closing(env_for_gain), r_mults)

    print_configuration(args, q_scales, r_scales, q_mults, r_mults, error_limits)

    safe_candidates = []
    printed_controller_info = False

    for q_scale in q_scales:
        for r_scale in r_scales:
            Q = Q_base * q_scale
            R = R_base * r_scale

            K, _ = compute_lqr_gain(A, B, Q, R)

            controller = LQRClosingControllerErrorSat(
                K,
                v_fast=args.v_fast,
                d_fast=args.d_fast,
                d_slow=args.d_slow,
                error_limits=error_limits,
                error_limit_mode=args.error_limit_mode,
            )

            if not printed_controller_info:
                print_controller_settings(controller)
                print_header()
                printed_controller_info = True

            result = run_rollouts(
                controller,
                num_rollouts=args.rollouts,
                max_time=args.max_time,
                seed_start=args.seed_start,
                mode=args.mode,
                debug_failures=args.debug_failures,
                max_debug_failures=args.max_debug_failures,
            )

            print_result_row(q_scale, r_scale, result)

            if result["safety_rate"] == 0.0:
                safe_candidates.append((q_scale, r_scale, result))

    print_safe_candidates(safe_candidates)


if __name__ == "__main__":
    main()
