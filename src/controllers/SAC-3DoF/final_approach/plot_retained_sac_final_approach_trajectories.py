"""Plot retained-SAC final-approach trajectories in the LQR figure style."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from envs.docking3dof.gym_env import Docking3DoFGymEnv


MODEL_NAME = "sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12"
MODEL_DIR = SCRIPT_DIR / "models" / "optuna12_100ksteps_buffer100k"
MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}.zip"
VECNORM_PATH = MODEL_DIR / f"{MODEL_NAME}_vecnormalize.pkl"
RESULT_DIR = (
    SCRIPT_DIR / "results" / "eval_lqr_phase_50" / MODEL_NAME
    / "testing_seeds_0_49"
)
CACHE_PATH = RESULT_DIR / "trajectory_histories.npz"
OUTPUT_BASE = RESULT_DIR / "final_approach_port_xy_trajectories"

TRAJECTORY_COLOR = "#0072B2"
REPRESENTATIVE_COLOR = "#003B5C"
TOLERANCE_COLOR = "#009E73"
REPRESENTATIVE_SEED = 24
N_ROLLOUTS = 50
MAX_STEPS = 5000
DOCK_TOLERANCE_M = 0.1


def make_env(seed: int) -> Docking3DoFGymEnv:
    return Docking3DoFGymEnv(
        phase="final",
        mode="testing",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
    )


def collect_histories() -> dict[str, np.ndarray]:
    dummy_env = DummyVecEnv([lambda: Monitor(make_env(999))])
    vecnorm = VecNormalize.load(str(VECNORM_PATH), dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    model = SAC.load(str(MODEL_PATH))

    times: list[np.ndarray] = []
    port_x: list[np.ndarray] = []
    port_y: list[np.ndarray] = []
    port_speed: list[np.ndarray] = []
    attitude_error_deg: list[np.ndarray] = []
    angular_rate_error_deg_s: list[np.ndarray] = []
    successes: list[bool] = []
    reasons: list[str] = []
    terminal_distances: list[float] = []

    for seed in range(N_ROLLOUTS):
        env = make_env(seed)
        obs, _ = env.reset(seed=seed)
        initial = env.core_env.get_derived_quantities(env.core_env.state)
        rollout_times = [0.0]
        rollout_x = [float(initial["r_p"][0])]
        rollout_y = [float(initial["r_p"][1])]
        rollout_speed = [float(np.linalg.norm(initial["v_p"]))]
        rollout_attitude = [float(abs(np.rad2deg(initial["psi_e"])))]
        rollout_rate = [float(abs(np.rad2deg(initial["omega"])))]
        final_info = None

        for _ in range(MAX_STEPS):
            normalized_obs = vecnorm.normalize_obs(obs.reshape(1, -1))
            action, _ = model.predict(normalized_obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action[0])
            final_info = info
            rollout_times.append(float(env.core_env.time))
            rollout_x.append(float(info["port_relative_position"][0]))
            rollout_y.append(float(info["port_relative_position"][1]))
            rollout_speed.append(float(info["port_speed"]))
            rollout_attitude.append(float(abs(np.rad2deg(info["psi_e"]))))
            rollout_rate.append(float(abs(np.rad2deg(info["omega"]))))
            if terminated or truncated:
                break

        if final_info is None:
            raise RuntimeError(f"Seed {seed} produced no evaluation step")
        times.append(np.asarray(rollout_times))
        port_x.append(np.asarray(rollout_x))
        port_y.append(np.asarray(rollout_y))
        port_speed.append(np.asarray(rollout_speed))
        attitude_error_deg.append(np.asarray(rollout_attitude))
        angular_rate_error_deg_s.append(np.asarray(rollout_rate))
        successes.append(bool(final_info.get("success", False)))
        reasons.append(str(final_info.get("termination_reason", "unknown")))
        terminal_distances.append(float(final_info["port_distance"]))
        env.close()
        if (seed + 1) % 5 == 0:
            print(f"Completed {seed + 1}/{N_ROLLOUTS} rollouts")

    dummy_env.close()
    max_length = max(len(values) for values in times)

    def pad(rows: list[np.ndarray]) -> np.ndarray:
        result = np.full((len(rows), max_length), np.nan)
        for index, row in enumerate(rows):
            result[index, : len(row)] = row
        return result

    return {
        "seeds": np.arange(N_ROLLOUTS, dtype=int),
        "times": pad(times),
        "port_x_positions": pad(port_x),
        "port_y_positions": pad(port_y),
        "port_position_errors": pad(
            [np.hypot(x_values, y_values) for x_values, y_values in zip(port_x, port_y)]
        ),
        "port_velocity_errors": pad(port_speed),
        "attitude_errors_deg": pad(attitude_error_deg),
        "angular_rate_errors_deg_s": pad(angular_rate_error_deg_s),
        "success": np.asarray(successes, dtype=bool),
        "termination_reasons": np.asarray(reasons),
        "terminal_port_distances": np.asarray(terminal_distances),
        "dock_tolerance": np.asarray(DOCK_TOLERANCE_M),
        "representative_seed": np.asarray(REPRESENTATIVE_SEED),
        "model_name": np.asarray(MODEL_NAME),
        "deterministic": np.asarray(True),
    }


def make_figure(histories: dict[str, np.ndarray]) -> plt.Figure:
    seeds = histories["seeds"].astype(int)
    times = histories["times"]
    x = histories["port_x_positions"]
    y = histories["port_y_positions"]
    valid_counts = np.sum(np.isfinite(times), axis=1)
    representative_index = int(np.flatnonzero(seeds == REPRESENTATIVE_SEED)[0])

    plt.rcParams.update({
        "font.size": 10.5,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 11.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(9.6, 5.6))

    for index in range(seeds.size):
        count = valid_counts[index]
        ax.plot(x[index, :count], y[index, :count], color=TRAJECTORY_COLOR,
                linewidth=0.85, alpha=0.18, zorder=2)
        ax.scatter(x[index, count - 1], y[index, count - 1], s=9,
                   color=TRAJECTORY_COLOR, alpha=0.28, edgecolor="none", zorder=3)

    count = valid_counts[representative_index]
    ax.plot(x[representative_index, :count], y[representative_index, :count],
            color=REPRESENTATIVE_COLOR, linewidth=2.2, alpha=0.95, zorder=4)
    ax.scatter(x[representative_index, count - 1], y[representative_index, count - 1],
               s=42, marker="o", facecolor=REPRESENTATIVE_COLOR, edgecolor="white",
               linewidth=0.6, zorder=5)
    ax.axhline(0.0, color="#AAAAAA", linewidth=0.8, linestyle=":", zorder=0)
    ax.axvline(0.0, color="#AAAAAA", linewidth=0.8, linestyle=":", zorder=0)
    ax.grid(True, color="#D9D9D9", linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)
    ax.margins(x=0.035, y=0.10)
    ax.invert_xaxis()
    ax.set_xlabel(r"Axial target docking-port-relative position $r_{p,x}$ [m]")
    ax.set_ylabel(r"Lateral target docking-port-relative position $r_{p,y}$ [m]")
    ax.set_title("Final-approach trajectories in the docking plane across 50 rollouts",
                 loc="left", fontweight="bold")
    ax.scatter(0.0, 0.0, s=190, marker="o", facecolor=TOLERANCE_COLOR,
               edgecolor="#007A59", linewidth=1.5, alpha=0.30, zorder=6)

    handles = [
        Line2D([], [], color=TRAJECTORY_COLOR, linewidth=1.2, alpha=0.3,
               label="Individual rollouts"),
        Line2D([], [], color=REPRESENTATIVE_COLOR, linewidth=2.2,
               label="Representative rollout (seed 24)"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=TOLERANCE_COLOR,
               markeredgecolor="#007A59", alpha=0.45, markersize=9,
               label="0.1 m docking tolerance"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.94),
               ncol=3, frameon=False)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.14, top=0.82)
    return fig


def main() -> None:
    if not MODEL_PATH.exists() or not VECNORM_PATH.exists():
        raise FileNotFoundError("Retained SAC model or VecNormalize file is missing")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    required_fields = {
        "port_position_errors", "port_velocity_errors", "attitude_errors_deg",
        "angular_rate_errors_deg_s",
    }
    if CACHE_PATH.exists():
        histories = dict(np.load(CACHE_PATH, allow_pickle=False))
    else:
        histories = {}
    if required_fields.issubset(histories):
        print(f"Loaded {CACHE_PATH}")
    else:
        histories = collect_histories()
        np.savez_compressed(CACHE_PATH, **histories)
        print(f"Wrote {CACHE_PATH}")

    representative_index = int(
        np.flatnonzero(histories["seeds"].astype(int) == REPRESENTATIVE_SEED)[0]
    )
    if not bool(histories["success"][representative_index]):
        raise RuntimeError("Representative seed 24 was not successful")

    figure = make_figure(histories)
    png_path = OUTPUT_BASE.with_suffix(".png")
    figure.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    distance = float(histories["terminal_port_distances"][representative_index])
    print(f"Successful rollouts: {int(np.sum(histories['success']))}/{N_ROLLOUTS}")
    print(f"Representative seed: {REPRESENTATIVE_SEED}")
    print(f"Representative terminal port distance: {distance:.9f} m")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
