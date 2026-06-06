import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_rollout(path):
    data = np.load(path, allow_pickle=False)

    metadata = {}
    if "metadata_json" in data.files:
        metadata = json.loads(str(data["metadata_json"]))

    return data, metadata


def get_basename_without_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_final_error_history(data):
    if "final_error_history" not in data.files:
        raise KeyError(
            "This .npz file does not contain final_error_history. "
            "Regenerate it with the modified final-approach sweep script."
        )
    return np.asarray(data["final_error_history"], dtype=float)


def plot_rollout(path, save_dir=None, show=True):
    data, metadata = load_rollout(path)

    times = np.asarray(data["times"], dtype=float)
    states = np.asarray(data["states"], dtype=float)
    errors = get_final_error_history(data)

    actions_body_raw = np.asarray(data["actions_body_raw"], dtype=float)
    actions_body_clipped = np.asarray(data["actions_body_clipped"], dtype=float)

    n = min(len(times), len(states), len(errors))
    times = times[:n]
    states = states[:n]
    errors = errors[:n]

    rpx = errors[:, 0]
    rpy = errors[:, 1]
    vpx = errors[:, 2]
    vpy = errors[:, 3]
    psi_e = errors[:, 4]
    omega = errors[:, 5]

    port_distance = np.linalg.norm(errors[:, 0:2], axis=1)
    port_speed = np.linalg.norm(errors[:, 2:4], axis=1)

    dock_dist_tol = float(metadata.get("dock_dist_tol", 0.1))
    dock_speed_tol = float(metadata.get("dock_speed_tol", 0.1))
    psi_tol_deg = float(metadata.get("psi_tol_deg", 4.0))
    omega_tol_deg = float(metadata.get("omega_tol_deg", 0.2))

    title = (
        f"{metadata.get('label', 'rollout')} | "
        f"seed={metadata.get('seed', '?')} | "
        f"q={metadata.get('q_scale', '?')} r={metadata.get('r_scale', '?')} | "
        f"{metadata.get('termination_reason', metadata.get('status', '?'))}"
    )

    output_paths = []
    base = get_basename_without_ext(path)

    # -------------------------------------------------------------------------
    # Figure 1: mission overview
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(title)

    ax = axes[0, 0]
    ax.plot(states[:, 0], states[:, 1], label="CoM trajectory")
    ax.scatter(states[0, 0], states[0, 1], marker="o", label="initial CoM")
    ax.scatter(states[-1, 0], states[-1, 1], marker="s", label="final CoM")
    ax.set_xlabel("rx [m]")
    ax.set_ylabel("ry [m]")
    ax.set_title("CoM trajectory in docking plane")
    ax.grid(True)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(rpx, rpy, label="docking-port trajectory")
    ax.scatter(rpx[0], rpy[0], marker="o", label="initial port error")
    ax.scatter(0.0, 0.0, marker="x", label="dock target")
    ax.scatter(rpx[-1], rpy[-1], marker="s", label="final port error")
    ax.set_xlabel("r_px error [m]")
    ax.set_ylabel("r_py error [m]")
    ax.set_title("Docking-port trajectory in error plane")
    ax.grid(True)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(times, rpx, label="r_px")
    ax.plot(times, rpy, label="r_py")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title("Port position error components")
    ax.grid(True)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(times, vpx, label="v_px")
    ax.plot(times, vpy, label="v_py")
    ax.axhline(dock_speed_tol, linestyle="--", linewidth=1, label="speed tol")
    ax.axhline(-dock_speed_tol, linestyle="--", linewidth=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("velocity error [m/s]")
    ax.set_title("Port velocity error components")
    ax.grid(True)
    ax.legend()

    ax = axes[2, 0]
    ax.plot(times, port_distance, label="||port position error||")
    ax.axhline(dock_dist_tol, linestyle="--", linewidth=1, label="distance tol")
    ax2 = ax.twinx()
    ax2.plot(times, port_speed, linestyle="--", label="||port velocity||")
    ax2.axhline(dock_speed_tol, linestyle=":", linewidth=1, label="speed tol")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance [m]")
    ax2.set_ylabel("speed [m/s]")
    ax.set_title("Port distance and speed")
    ax.grid(True)
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2)

    ax = axes[2, 1]
    ax.plot(times, np.rad2deg(psi_e), label="psi error")
    ax.plot(times, np.rad2deg(omega), label="omega")
    ax.axhline(psi_tol_deg, linestyle="--", linewidth=1, label="+/- psi tol")
    ax.axhline(-psi_tol_deg, linestyle="--", linewidth=1)
    ax.axhline(omega_tol_deg, linestyle=":", linewidth=1, label="+/- omega tol")
    ax.axhline(-omega_tol_deg, linestyle=":", linewidth=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("deg or deg/s")
    ax.set_title("Attitude and angular-rate error")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        overview_path = os.path.join(save_dir, f"{base}_overview.png")
        fig.savefig(overview_path, dpi=200)
        output_paths.append(overview_path)

    # -------------------------------------------------------------------------
    # Figure 2: actions and saturation behavior
    # -------------------------------------------------------------------------
    if actions_body_raw.size > 0 and actions_body_clipped.size > 0:
        n_action = min(len(times), len(actions_body_raw), len(actions_body_clipped))
        action_time = times[:n_action]
        raw = actions_body_raw[:n_action]
        clipped = actions_body_clipped[:n_action]

        fig2, axes2 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig2.suptitle(title + " | body-frame control actions")

        action_names = ["ax command", "ay command", "alpha command"]
        for i, ax in enumerate(axes2):
            if i >= raw.shape[1]:
                continue
            ax.plot(action_time, raw[:, i], linestyle="--", label="raw")
            ax.plot(action_time, clipped[:, i], label="clipped")
            ax.set_ylabel(action_names[i])
            ax.grid(True)
            ax.legend()

        axes2[-1].set_xlabel("time [s]")
        fig2.tight_layout()

        if save_dir is not None:
            actions_path = os.path.join(save_dir, f"{base}_actions.png")
            fig2.savefig(actions_path, dpi=200)
            output_paths.append(actions_path)

    # -------------------------------------------------------------------------
    # Figure 3: phase-plane diagnostics
    # -------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))
    fig3.suptitle(title + " | phase-plane diagnostics")

    axes3[0].plot(rpx, vpx)
    axes3[0].scatter(rpx[0], vpx[0], marker="o", label="initial")
    axes3[0].scatter(rpx[-1], vpx[-1], marker="s", label="final")
    axes3[0].set_xlabel("r_px error [m]")
    axes3[0].set_ylabel("v_px error [m/s]")
    axes3[0].set_title("Longitudinal port phase plane")
    axes3[0].grid(True)
    axes3[0].legend()

    axes3[1].plot(rpy, vpy)
    axes3[1].scatter(rpy[0], vpy[0], marker="o", label="initial")
    axes3[1].scatter(rpy[-1], vpy[-1], marker="s", label="final")
    axes3[1].set_xlabel("r_py error [m]")
    axes3[1].set_ylabel("v_py error [m/s]")
    axes3[1].set_title("Lateral port phase plane")
    axes3[1].grid(True)
    axes3[1].legend()

    axes3[2].plot(np.rad2deg(psi_e), np.rad2deg(omega))
    axes3[2].scatter(np.rad2deg(psi_e[0]), np.rad2deg(omega[0]), marker="o", label="initial")
    axes3[2].scatter(np.rad2deg(psi_e[-1]), np.rad2deg(omega[-1]), marker="s", label="final")
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

    if args.history_dir:
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
        description="Plot selected final-approach LQR rollout histories saved by lqr_sweep_final.py."
    )
    parser.add_argument(
        "--history-dir",
        type=str,
        default=os.path.join("results", "lqr_final", "rollout_histories"),
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
