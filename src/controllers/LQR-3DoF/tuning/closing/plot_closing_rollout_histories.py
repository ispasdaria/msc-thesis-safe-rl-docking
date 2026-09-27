import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

LQR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def load_rollout(path):
    data = np.load(path, allow_pickle=False)

    metadata = {}
    if "metadata_json" in data.files:
        metadata = json.loads(str(data["metadata_json"]))

    return data, metadata


def get_basename_without_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


def compute_errors(states, p3_target):
    errors = states - p3_target.reshape(1, -1)
    errors[:, 4] = wrap_to_pi(errors[:, 4])
    return errors


def scheduled_vx_from_metadata(states, p3_target, metadata):
    """Approximate the closing velocity schedule from saved metadata.

    The reconstruction uses the standard closing schedule: ``v_fast`` beyond
    ``d_fast``, the P3 target velocity within ``d_slow``, and linear
    interpolation between the two distances. It is used only for plotting.
    """
    v_fast = float(metadata.get("v_fast", np.nan))
    d_fast = float(metadata.get("d_fast", np.nan))
    d_slow = float(metadata.get("d_slow", np.nan))

    if not np.isfinite(v_fast) or not np.isfinite(d_fast) or not np.isfinite(d_slow):
        return None

    v_slow = float(p3_target[2])
    distance_to_p3_x = np.abs(states[:, 0] - p3_target[0])

    vx_ref = np.empty(states.shape[0], dtype=float)

    far = distance_to_p3_x >= d_fast
    close = distance_to_p3_x <= d_slow
    mid = ~(far | close)

    vx_ref[far] = v_fast
    vx_ref[close] = v_slow

    if np.any(mid):
        alpha = (distance_to_p3_x[mid] - d_slow) / max(d_fast - d_slow, 1e-12)
        vx_ref[mid] = v_slow + alpha * (v_fast - v_slow)

    return vx_ref


def choose_time_for_actions(times, actions):
    if actions.size == 0:
        return np.asarray([])

    n = min(len(times), len(actions))
    return times[:n]


def plot_rollout(path, save_dir=None, show=True):
    data, metadata = load_rollout(path)

    times = np.asarray(data["times"], dtype=float)
    states = np.asarray(data["states"], dtype=float)
    p3_target = np.asarray(data["p3_target"], dtype=float)
    p3_tol = np.asarray(data["p3_target_tolerance"], dtype=float)

    actions_body_raw = np.asarray(data["actions_body_raw"], dtype=float)
    actions_body_clipped = np.asarray(data["actions_body_clipped"], dtype=float)

    errors = compute_errors(states, p3_target)
    vx_ref = scheduled_vx_from_metadata(states, p3_target, metadata)

    title = (
        f"{metadata.get('label', 'rollout')} | "
        f"seed={metadata.get('seed', '?')} | "
        f"q={metadata.get('q_scale', '?')} r={metadata.get('r_scale', '?')} | "
        f"{metadata.get('termination_reason', metadata.get('status', '?'))}"
    )

    # Mission overview
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(title)

    ax = axes[0, 0]
    ax.plot(states[:, 0], states[:, 1], label="CoM trajectory")
    ax.scatter(states[0, 0], states[0, 1], marker="o", label="initial")
    ax.scatter(p3_target[0], p3_target[1], marker="x", label="P3 target")
    ax.scatter(states[-1, 0], states[-1, 1], marker="s", label="final")
    ax.set_xlabel("rx [m]")
    ax.set_ylabel("ry [m]")
    ax.set_title("Trajectory in docking plane")
    ax.grid(True)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(times, errors[:, 0], label="rx - rx_P3")
    ax.plot(times, errors[:, 1], label="ry - ry_P3")
    if np.all(np.isfinite(p3_tol[:2])):
        ax.axhline(p3_tol[0], linestyle="--", linewidth=1)
        ax.axhline(-p3_tol[0], linestyle="--", linewidth=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title("Position error")
    ax.grid(True)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(times, states[:, 2], label="vx")
    if vx_ref is not None:
        ax.plot(times, vx_ref, linestyle="--", label="vx_ref schedule")
    ax.plot(times, states[:, 3], label="vy")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("velocity [m/s]")
    ax.set_title("Velocity behavior")
    ax.grid(True)
    ax.legend()

    ax = axes[1, 1]
    speed = np.linalg.norm(states[:, 2:4], axis=1)
    pos_err_norm = np.linalg.norm(errors[:, 0:2], axis=1)
    ax.plot(times, pos_err_norm, label="||position error||")
    ax2 = ax.twinx()
    ax2.plot(times, speed, linestyle="--", label="||velocity||")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance to P3 [m]")
    ax2.set_ylabel("speed [m/s]")
    ax.set_title("Distance and speed")
    ax.grid(True)
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2)

    ax = axes[2, 0]
    psi_deg = np.rad2deg(errors[:, 4])
    ax.plot(times, psi_deg, label="psi error")
    ax.axhline(4.0, linestyle="--", linewidth=1, label="+/-4 deg")
    ax.axhline(-4.0, linestyle="--", linewidth=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("angle [deg]")
    ax.set_title("Attitude error")
    ax.grid(True)
    ax.legend()

    ax = axes[2, 1]
    omega_deg_s = np.rad2deg(states[:, 5])
    ax.plot(times, omega_deg_s, label="omega")
    ax.axhline(0.2, linestyle="--", linewidth=1, label="+/-0.2 deg/s")
    ax.axhline(-0.2, linestyle="--", linewidth=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("angular rate [deg/s]")
    ax.set_title("Angular rate")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    output_paths = []
    base = get_basename_without_ext(path)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        overview_path = os.path.join(save_dir, f"{base}_overview.png")
        fig.savefig(overview_path, dpi=200)
        output_paths.append(overview_path)

    # Actions and saturation behavior
    if actions_body_raw.size > 0 and actions_body_clipped.size > 0:
        n = min(len(times), len(actions_body_raw), len(actions_body_clipped))
        action_time = times[:n]
        raw = actions_body_raw[:n]
        clipped = actions_body_clipped[:n]

        fig2, axes2 = plt.subplots(3, 2, figsize=(15, 10))
        fig2.suptitle(
            title
            + " | body-frame control actions\n"
            + "red markers identify updates where the requested command was clipped"
        )

        action_names = [
            r"$a_{x,b_c}$",
            r"$a_{y,b_c}$",
            r"$\alpha_{z,b_c}$",
        ]
        action_units = [r"m/s$^2$", r"m/s$^2$", r"rad/s$^2$"]
        action_limits = np.array([0.01, 0.01, 0.003], dtype=float)
        saturated = np.abs(raw - clipped) > 1e-12
        dt = float(metadata.get("dt", np.median(np.diff(action_time))))
        control_update_count = int(metadata.get("steps", n))

        saturated_any = np.any(saturated, axis=1)
        if np.any(saturated_any):
            last_saturation_time = float(action_time[np.flatnonzero(saturated_any)[-1]])
            startup_end = min(float(action_time[-1]), max(30.0, last_saturation_time + 5.0))
        else:
            startup_end = min(float(action_time[-1]), 30.0)
        startup = action_time <= startup_end

        for i in range(3):
            if i >= raw.shape[1]:
                continue

            limit = action_limits[i]
            requested_percent = 100.0 * raw[:, i] / limit
            applied_percent = 100.0 * clipped[:, i] / limit
            channel_saturated = saturated[:, i]
            count = int(np.count_nonzero(channel_saturated))
            fraction_percent = 100.0 * count / control_update_count
            saturated_duration = count * dt

            startup_ax = axes2[i, 0]
            startup_ax.plot(
                action_time[startup],
                requested_percent[startup],
                linestyle="--",
                linewidth=1.2,
                color="tab:blue",
                label="requested",
            )
            startup_ax.plot(
                action_time[startup],
                applied_percent[startup],
                linewidth=1.5,
                color="tab:orange",
                label="applied",
            )
            startup_saturated = startup & channel_saturated
            startup_ax.scatter(
                action_time[startup_saturated],
                applied_percent[startup_saturated],
                s=10,
                color="tab:red",
                label="actually clipped",
                zorder=3,
            )
            startup_ax.axhline(100.0, color="black", linestyle=":", linewidth=1.0)
            startup_ax.axhline(-100.0, color="black", linestyle=":", linewidth=1.0)
            startup_ax.set_yscale("symlog", linthresh=5.0)
            startup_ax.set_ylabel(
                f"{action_names[i]}\n[% of limit]"
            )
            startup_ax.set_title(
                f"Startup detail (0-{startup_end:g} s); limit = +/-{limit:g} {action_units[i]}"
            )
            startup_ax.grid(True, which="both", alpha=0.35)

            full_ax = axes2[i, 1]
            full_ax.plot(
                action_time,
                applied_percent,
                linewidth=0.8,
                color="tab:orange",
                rasterized=True,
                label="applied",
            )
            full_ax.scatter(
                action_time[channel_saturated],
                applied_percent[channel_saturated],
                s=8,
                color="tab:red",
                label="actually clipped",
                zorder=3,
            )
            full_ax.axhline(100.0, color="black", linestyle=":", linewidth=1.0)
            full_ax.axhline(-100.0, color="black", linestyle=":", linewidth=1.0)
            full_ax.set_ylim(-110.0, 110.0)
            full_ax.set_ylabel(f"{action_names[i]}\n[% of limit]")
            full_ax.set_title(
                f"Full rollout: {count}/{control_update_count} updates clipped "
                f"({fraction_percent:.4g}%, {saturated_duration:.3g} s)"
            )
            full_ax.grid(True, alpha=0.35)

        axes2[0, 0].legend(loc="best")
        axes2[0, 1].legend(loc="best")
        axes2[-1, 0].set_xlabel("time [s]")
        axes2[-1, 1].set_xlabel("time [s]")
        fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

        if save_dir is not None:
            actions_path = os.path.join(save_dir, f"{base}_actions_clear.png")
            fig2.savefig(actions_path, dpi=200)
            output_paths.append(actions_path)

    # Phase-plane diagnostics
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))
    fig3.suptitle(title + " | phase-plane diagnostics")

    axes3[0].plot(errors[:, 0], errors[:, 2])
    axes3[0].scatter(errors[0, 0], errors[0, 2], marker="o", label="initial")
    axes3[0].scatter(errors[-1, 0], errors[-1, 2], marker="s", label="final")
    axes3[0].set_xlabel("rx error [m]")
    axes3[0].set_ylabel("vx error [m/s]")
    axes3[0].set_title("Longitudinal phase plane")
    axes3[0].grid(True)
    axes3[0].legend()

    axes3[1].plot(errors[:, 1], errors[:, 3])
    axes3[1].scatter(errors[0, 1], errors[0, 3], marker="o", label="initial")
    axes3[1].scatter(errors[-1, 1], errors[-1, 3], marker="s", label="final")
    axes3[1].set_xlabel("ry error [m]")
    axes3[1].set_ylabel("vy error [m/s]")
    axes3[1].set_title("Lateral phase plane")
    axes3[1].grid(True)
    axes3[1].legend()

    axes3[2].plot(np.rad2deg(errors[:, 4]), np.rad2deg(states[:, 5]))
    axes3[2].scatter(np.rad2deg(errors[0, 4]), np.rad2deg(states[0, 5]), marker="o", label="initial")
    axes3[2].scatter(np.rad2deg(errors[-1, 4]), np.rad2deg(states[-1, 5]), marker="s", label="final")
    axes3[2].set_xlabel("psi error [deg]")
    axes3[2].set_ylabel("omega [deg/s]")
    axes3[2].set_title("Attitude phase plane")
    axes3[2].grid(True)
    axes3[2].legend()

    fig3.tight_layout()

    if save_dir is not None:
        phase_path = os.path.join(save_dir, f"{base}_phase_plane.png")
        fig3.savefig(phase_path, dpi=200)
        output_paths.append(phase_path)

    if show:
        plt.show()
    else:
        plt.close("all")

    return output_paths


def collect_input_files(args):
    files = []

    if args.files:
        files.extend(args.files)
    elif args.history_dir:
        files.extend(glob.glob(os.path.join(args.history_dir, "*.npz")))

    # Keep order stable and remove duplicates.
    unique_files = []
    seen = set()
    for path in sorted(files):
        abs_path = os.path.abspath(path)
        if abs_path not in seen:
            unique_files.append(path)
            seen.add(abs_path)

    return unique_files


def main():
    parser = argparse.ArgumentParser(
        description="Plot selected closing-phase LQR rollout histories saved by sweep_closing_lqr_weights.py."
    )
    parser.add_argument(
        "--history-dir",
        type=str,
        default=os.path.join(LQR_ROOT, "results", "lqr_closing", "rollout_histories"),
        help="Directory containing selected rollout .npz files.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional explicit .npz files to plot. If omitted, all histories in --history-dir are used.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help=(
            "Directory where figures are saved. If omitted, figures are only shown "
            "and no plot files are written."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display figures. Useful when using --save-dir in batch mode.",
    )

    args = parser.parse_args()

    files = collect_input_files(args)
    if not files:
        raise FileNotFoundError(
            "No rollout history .npz files found. Check --history-dir or pass files with --files."
        )

    for path in files:
        print(f"plotting: {path}")
        output_paths = plot_rollout(
            path=path,
            save_dir=args.save_dir,
            show=not args.no_show,
        )

        for output_path in output_paths:
            print(f"  saved: {output_path}")


if __name__ == "__main__":
    main()
