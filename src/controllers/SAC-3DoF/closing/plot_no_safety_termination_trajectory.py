"""Plot a representative rollout from the no-safety-termination experiment."""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.closing_safety_exploration_env import (  # noqa: E402
    ClosingSafetyExplorationGymEnv,
)


EXPERIMENT = (
    "sac_closing_v15_no_safety_termination_from_trial16_"
    "400k_add300k_seed0_20260813_220234"
)
MODEL_DIR = os.path.join(
    SCRIPT_DIR, "models", "no_safety_termination", EXPERIMENT
)
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.zip")
VECNORM_PATH = os.path.join(MODEL_DIR, "final_vecnormalize.pkl")
SEED = 105
OUTPUT_DIR = os.path.join(
    SCRIPT_DIR, "results", "no_safety_termination", EXPERIMENT, "plots"
)
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR, f"representative_trajectory_seed{SEED}.png"
)
PDF_PATH = os.path.join(
    OUTPUT_DIR, f"representative_trajectory_seed{SEED}.pdf"
)


def main():
    env = ClosingSafetyExplorationGymEnv(
        phase="closing",
        mode="testing",
        max_time=12_000.0,
        seed=SEED,
        closing_reward_version="v15",
    )
    dummy_env = DummyVecEnv(
        [
            lambda: Monitor(
                ClosingSafetyExplorationGymEnv(
                    phase="closing",
                    mode="testing",
                    max_time=12_000.0,
                    seed=999,
                    closing_reward_version="v15",
                )
            )
        ]
    )
    vecnormalize = VecNormalize.load(VECNORM_PATH, dummy_env)
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    model = SAC.load(MODEL_PATH, device="cpu")

    observation, _ = env.reset(seed=SEED)
    axial = []
    lateral = []
    unsafe = []
    times = []
    terminal_info = {}

    for step in range(12_000):
        position_error = env.core_env.state[:2] - env.core_env.p3_target[:2]
        axial.append(float(position_error[0]))
        lateral.append(float(position_error[1]))
        times.append(float(step))

        normalized = vecnormalize.normalize_obs(
            np.asarray(observation, dtype=np.float32)[None, :]
        )[0]
        action, _ = model.predict(normalized, deterministic=True)
        observation, _, terminated, truncated, terminal_info = env.step(action)
        unsafe.append(bool(terminal_info.get("unsafe", False)))
        if terminated or truncated:
            break

    position_error = env.core_env.state[:2] - env.core_env.p3_target[:2]
    axial.append(float(position_error[0]))
    lateral.append(float(position_error[1]))
    times.append(float(len(times)))

    axial = np.asarray(axial)
    lateral = np.asarray(lateral)
    times = np.asarray(times)
    unsafe = np.asarray([False] + unsafe, dtype=bool)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10.0, 5.5))
    axis.plot(axial, lateral, color="#1f77b4", linewidth=1.8,
              label="Trajectory")
    axis.scatter(axial[0], lateral[0], s=55, color="#1f77b4", marker="o",
                 label="Start", zorder=4)
    axis.scatter(axial[-1], lateral[-1], s=65, color="#ff7f0e", marker="x",
                 linewidth=1.5, label="End", zorder=4)
    axis.scatter(0.0, 0.0, s=60, color="#2ca02c", marker="+",
                 linewidth=1.5, label=r"$P_3$ target", zorder=5)

    axis.set_xlabel(
        r"Axial $P_3$-relative position error, $e_{r_{p,x}}$ [m]"
    )
    axis.set_ylabel(
        "Lateral $P_3$-relative position\n"
        r"error, $e_{r_{p,y}}$ [m]"
    )
    axis.set_title(
        "Representative trajectory of the closing policy trained without "
        "hard-safety termination\n"
        f"(reward version 15, testing seed {SEED})"
    )
    axis.grid(True, alpha=0.3)
    axis.margins(x=0.06, y=0.18)
    axis.invert_xaxis()
    axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4,
        frameon=False,
    )
    figure.subplots_adjust(left=0.14, right=0.97, top=0.80, bottom=0.22)
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    figure.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(figure)

    env.close()
    dummy_env.close()
    print(f"plot={OUTPUT_PATH}")
    print(f"pdf={PDF_PATH}")
    print(f"steps={len(times) - 1}")
    print(f"unsafe_fraction={np.mean(unsafe):.6f}")
    print(f"termination_reason={terminal_info.get('termination_reason')}")


if __name__ == "__main__":
    main()
