import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_history(path):
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    return data, metadata


def title_from_metadata(metadata):
    return (
        f"{metadata.get('label', '')} | seed={metadata.get('seed')} | "
        f"mission_success={metadata.get('mission_success')} | failed_phase={metadata.get('failed_phase')}"
    )


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def final_time_axis(closing_times, final_times):
    if final_times.size == 0:
        return final_times
    offset = float(closing_times[-1]) if closing_times.size else 0.0
    return final_times + offset


def get_port_errors_from_data(data):
    """
    For final approach, the plotted state is assumed to already be compatible
    with the final controller's port-error interpretation. If your env stores
    only CoM states, this still gives a useful state-level approximation.
    """
    final_states = data["final_states"]
    if final_states.size == 0:
        return np.empty((0, 6))
    return final_states[:, :6]


def plot_mission_overview(path, save_dir=None, show=True):
    data, metadata = load_history(path)
    base_title = title_from_metadata(metadata)

    tc = data["closing_times"]
    xc = data["closing_states"]
    tf_raw = data["final_times"]
    xf = data["final_states"]
    tf = final_time_axis(tc, tf_raw)
    p3 = data["closing_p3_target"]
    transition = data["transition_state"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(base_title)

    # 1. Full CoM trajectory
    ax = axes[0, 0]
    if xc.size:
        ax.plot(xc[:, 0], xc[:, 1], label="closing CoM")
        ax.scatter(xc[0, 0], xc[0, 1], marker="o", label="initial")
        ax.scatter(xc[-1, 0], xc[-1, 1], marker="s", label="end closing")
    if xf.size:
        ax.plot(xf[:, 0], xf[:, 1], linestyle="--", label="final CoM")
        ax.scatter(xf[-1, 0], xf[-1, 1], marker="x", label="final")
    if np.isfinite(p3).all():
        ax.scatter(p3[0], p3[1], marker="*", label="P3 target")
    ax.scatter(0.0, 0.0, marker="+", label="dock target")
    ax.set_title("Full CoM trajectory")
    ax.set_xlabel("rx [m]")
    ax.set_ylabel("ry [m]")
    ax.grid(True)
    ax.legend()

    # 2. Position errors / states over full time
    ax = axes[0, 1]
    if xc.size:
        if np.isfinite(p3).all():
            ax.plot(tc, xc[:, 0] - p3[0], label="closing rx-rx_P3")
            ax.plot(tc, xc[:, 1] - p3[1], label="closing ry-ry_P3")
        else:
            ax.plot(tc, xc[:, 0], label="closing rx")
            ax.plot(tc, xc[:, 1], label="closing ry")
    if xf.size:
        ax.plot(tf, xf[:, 0], linestyle="--", label="final rx/port x approx")
        ax.plot(tf, xf[:, 1], linestyle="--", label="final ry/port y approx")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title("Position evolution")
    ax.set_xlabel("mission time [s]")
    ax.set_ylabel("position [m]")
    ax.grid(True)
    ax.legend()

    # 3. Velocity evolution
    ax = axes[1, 0]
    if xc.size:
        ax.plot(tc, xc[:, 2], label="closing vx")
        ax.plot(tc, xc[:, 3], label="closing vy")
    if xf.size:
        ax.plot(tf, xf[:, 2], linestyle="--", label="final vx/port vx approx")
        ax.plot(tf, xf[:, 3], linestyle="--", label="final vy/port vy approx")
    ax.axhline(0.1, linestyle="--", linewidth=0.8, label="+/-0.1 m/s")
    ax.axhline(-0.1, linestyle="--", linewidth=0.8)
    ax.set_title("Velocity evolution")
    ax.set_xlabel("mission time [s]")
    ax.set_ylabel("velocity [m/s]")
    ax.grid(True)
    ax.legend()

    # 4. Distance and speed norms
    ax = axes[1, 1]
    if xc.size:
        if np.isfinite(p3).all():
            closing_pos_norm = np.linalg.norm(xc[:, :2] - p3[:2], axis=1)
        else:
            closing_pos_norm = np.linalg.norm(xc[:, :2], axis=1)
        closing_speed = np.linalg.norm(xc[:, 2:4], axis=1)
        ax.plot(tc, closing_pos_norm, label="closing position error norm")
        ax2 = ax.twinx()
        ax2.plot(tc, closing_speed, linestyle="--", label="closing speed norm")
        ax2.set_ylabel("speed [m/s]")
    else:
        ax2 = ax.twinx()
    if xf.size:
        final_pos_norm = np.linalg.norm(xf[:, :2], axis=1)
        final_speed = np.linalg.norm(xf[:, 2:4], axis=1)
        ax.plot(tf, final_pos_norm, label="final position/port error norm")
        ax2.plot(tf, final_speed, linestyle="--", label="final speed/port speed norm")
    ax.set_title("Distance and speed norms")
    ax.set_xlabel("mission time [s]")
    ax.set_ylabel("distance [m]")
    ax.grid(True)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    # 5. Attitude error
    ax = axes[2, 0]
    if xc.size:
        ax.plot(tc, np.rad2deg(xc[:, 4]), label="closing psi")
    if xf.size:
        ax.plot(tf, np.rad2deg(xf[:, 4]), linestyle="--", label="final psi")
    ax.axhline(4.0, linestyle="--", linewidth=0.8, label="+/-4 deg")
    ax.axhline(-4.0, linestyle="--", linewidth=0.8)
    ax.set_title("Attitude")
    ax.set_xlabel("mission time [s]")
    ax.set_ylabel("psi [deg]")
    ax.grid(True)
    ax.legend()

    # 6. Angular rate
    ax = axes[2, 1]
    if xc.size:
        ax.plot(tc, np.rad2deg(xc[:, 5]), label="closing omega")
    if xf.size:
        ax.plot(tf, np.rad2deg(xf[:, 5]), linestyle="--", label="final omega")
    ax.axhline(0.2, linestyle="--", linewidth=0.8, label="+/-0.2 deg/s")
    ax.axhline(-0.2, linestyle="--", linewidth=0.8)
    ax.set_title("Angular rate")
    ax.set_xlabel("mission time [s]")
    ax.set_ylabel("omega [deg/s]")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    save_or_show(fig, path, save_dir, "overview", show)


def plot_mission_actions(path, save_dir=None, show=True):
    data, metadata = load_history(path)
    base_title = title_from_metadata(metadata)

    tc = data["closing_times"]
    tf_raw = data["final_times"]
    tf = final_time_axis(tc, tf_raw)

    c_raw = data["closing_actions_body_raw"]
    c_clip = data["closing_actions_body_clipped"]
    f_raw = data["final_actions_body_raw"]
    f_clip = data["final_actions_body_clipped"]

    names = ["ax command", "ay command", "alpha command"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(base_title + " | full-mission body-frame control actions")

    for i, ax in enumerate(axes):
        if c_raw.size:
            ax.plot(tc, c_raw[:, i], linestyle="--", label="closing raw")
            ax.plot(tc, c_clip[:, i], label="closing clipped")
        if f_raw.size:
            ax.plot(tf, f_raw[:, i], linestyle=":", label="final raw")
            ax.plot(tf, f_clip[:, i], linestyle="-.", label="final clipped")
        ax.axhline(0.0, linewidth=0.8)
        ax.set_ylabel(names[i])
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("mission time [s]")
    fig.tight_layout()
    save_or_show(fig, path, save_dir, "actions", show)


def plot_mission_phase_planes(path, save_dir=None, show=True):
    data, metadata = load_history(path)
    base_title = title_from_metadata(metadata)

    xc = data["closing_states"]
    xf = data["final_states"]
    p3 = data["closing_p3_target"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(base_title + " | phase-plane diagnostics")

    # Closing longitudinal
    ax = axes[0, 0]
    if xc.size:
        if np.isfinite(p3).all():
            ex = xc[:, 0] - p3[0]
            evx = xc[:, 2] - p3[2]
        else:
            ex = xc[:, 0]
            evx = xc[:, 2]
        ax.plot(ex, evx)
        ax.scatter(ex[0], evx[0], label="initial")
        ax.scatter(ex[-1], evx[-1], marker="s", label="final")
    ax.set_title("Closing longitudinal")
    ax.set_xlabel("rx error [m]")
    ax.set_ylabel("vx error [m/s]")
    ax.grid(True)
    ax.legend()

    # Closing lateral
    ax = axes[0, 1]
    if xc.size:
        if np.isfinite(p3).all():
            ey = xc[:, 1] - p3[1]
            evy = xc[:, 3] - p3[3]
        else:
            ey = xc[:, 1]
            evy = xc[:, 3]
        ax.plot(ey, evy)
        ax.scatter(ey[0], evy[0], label="initial")
        ax.scatter(ey[-1], evy[-1], marker="s", label="final")
    ax.set_title("Closing lateral")
    ax.set_xlabel("ry error [m]")
    ax.set_ylabel("vy error [m/s]")
    ax.grid(True)
    ax.legend()

    # Closing attitude
    ax = axes[0, 2]
    if xc.size:
        ax.plot(np.rad2deg(xc[:, 4]), np.rad2deg(xc[:, 5]))
        ax.scatter(np.rad2deg(xc[0, 4]), np.rad2deg(xc[0, 5]), label="initial")
        ax.scatter(np.rad2deg(xc[-1, 4]), np.rad2deg(xc[-1, 5]), marker="s", label="final")
    ax.set_title("Closing attitude")
    ax.set_xlabel("psi [deg]")
    ax.set_ylabel("omega [deg/s]")
    ax.grid(True)
    ax.legend()

    # Final longitudinal
    ax = axes[1, 0]
    if xf.size:
        ax.plot(xf[:, 0], xf[:, 2])
        ax.scatter(xf[0, 0], xf[0, 2], label="initial")
        ax.scatter(xf[-1, 0], xf[-1, 2], marker="s", label="final")
    ax.set_title("Final longitudinal / port approx")
    ax.set_xlabel("r_px error [m]")
    ax.set_ylabel("v_px error [m/s]")
    ax.grid(True)
    ax.legend()

    # Final lateral
    ax = axes[1, 1]
    if xf.size:
        ax.plot(xf[:, 1], xf[:, 3])
        ax.scatter(xf[0, 1], xf[0, 3], label="initial")
        ax.scatter(xf[-1, 1], xf[-1, 3], marker="s", label="final")
    ax.set_title("Final lateral / port approx")
    ax.set_xlabel("r_py error [m]")
    ax.set_ylabel("v_py error [m/s]")
    ax.grid(True)
    ax.legend()

    # Final attitude
    ax = axes[1, 2]
    if xf.size:
        ax.plot(np.rad2deg(xf[:, 4]), np.rad2deg(xf[:, 5]))
        ax.scatter(np.rad2deg(xf[0, 4]), np.rad2deg(xf[0, 5]), label="initial")
        ax.scatter(np.rad2deg(xf[-1, 4]), np.rad2deg(xf[-1, 5]), marker="s", label="final")
    ax.set_title("Final attitude")
    ax.set_xlabel("psi [deg]")
    ax.set_ylabel("omega [deg/s]")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    save_or_show(fig, path, save_dir, "phase_plane", show)


def save_or_show(fig, source_path, save_dir, suffix, show):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(source_path))[0]
        output_path = os.path.join(save_dir, f"{base}_{suffix}.png")
        fig.savefig(output_path, dpi=200)
        print(f"saved {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot selected full-mission LQR histories.")
    parser.add_argument(
        "--history-dir",
        type=str,
        default=os.path.join("results", "lqr_mission", "mission_histories"),
    )
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--overview-only", action="store_true")
    args, unknown_args = parser.parse_known_args()

    if unknown_args:
        print(
            "warning: ignoring unrecognized arguments intended for mission generation: "
            + " ".join(unknown_args)
        )
        print("use src/controllers/run_lqr_full_mission.py to generate histories")

    if args.files:
        files = args.files
    else:
        files = sorted(glob.glob(os.path.join(args.history_dir, "*.npz")))

    if not files:
        raise FileNotFoundError(
            "No mission history .npz files found. Check --history-dir or pass files with --files."
        )

    show = not args.no_show
    for path in files:
        print(f"plotting {path}")
        plot_mission_overview(path, save_dir=args.save_dir, show=show)
        if not args.overview_only:
            plot_mission_actions(path, save_dir=args.save_dir, show=show)
            plot_mission_phase_planes(path, save_dir=args.save_dir, show=show)


if __name__ == "__main__":
    main()
