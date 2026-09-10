"""Plot a representative replay-pre-fill SAC rollout and an LQR demonstration.

The SAC rollout is regenerated deterministically from the retained 600k
checkpoint.  The LQR trajectory is read directly from the demonstration
archive used to pre-fill replay.  No training is performed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_sac_closing_lqr_replay import make_closing_env


MODEL_DIR = (
    HERE / "models" / "lqr_replay"
    / "sac_closing_rewardv15_lqr_replay_600000steps" / "seed_0"
)
MODEL_PATH = MODEL_DIR / "model_0600000_steps.zip"
VECNORMALIZE_PATH = MODEL_DIR / "model_0600000_steps_vecnormalize.pkl"
DEMONSTRATION_PATH = (
    HERE / "demonstrations" / "lqr_closing_training_rewardv15_28successes.npz"
)
OUTPUT_DIR = HERE / "results" / "publication_figures" / "demonstration_replay"
OUTPUT_STEM = OUTPUT_DIR / "replay_prefill_sac_vs_successful_lqr"

SAC_SEED = 1
LQR_SEED = 20_020
MAX_STEPS = 12_000


def load_sac_trajectory():
    """Regenerate the selected SAC rollout with deterministic policy actions."""
    normalization_env = DummyVecEnv(
        [lambda: make_closing_env(
            mode="testing", seed=SAC_SEED, reward_version="v15", monitor=False
        )]
    )
    vecnormalize = VecNormalize.load(VECNORMALIZE_PATH, normalization_env)
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    model = SAC.load(MODEL_PATH, device="cpu")
    env = make_closing_env(
        mode="testing", seed=SAC_SEED, reward_version="v15", monitor=False
    )
    obs, _ = env.reset(seed=SAC_SEED)
    core = env.core_env
    errors = [np.asarray(core.state[:2] - core.p3_target[:2], dtype=float)]
    info = {}
    for _ in range(MAX_STEPS):
        normalized_obs = vecnormalize.normalize_obs(
            np.asarray(obs, dtype=np.float32).reshape(1, -1)
        )
        action, _ = model.predict(normalized_obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(
            np.asarray(action[0], dtype=np.float32)
        )
        errors.append(np.asarray(core.state[:2] - core.p3_target[:2], dtype=float))
        if terminated or truncated:
            break
    env.close()
    vecnormalize.close()
    return np.asarray(errors), info


def load_lqr_demonstration():
    """Load one complete successful LQR episode from the replay archive."""
    with np.load(DEMONSTRATION_PATH, allow_pickle=False) as data:
        seeds = np.asarray(data["episode_seeds"], dtype=int)
        matches = np.flatnonzero(seeds == LQR_SEED)
        if len(matches) != 1:
            raise ValueError(f"Expected one demonstration for seed {LQR_SEED}.")
        episode = int(matches[0])
        start = int(data["episode_offsets"][episode])
        stop = int(data["episode_offsets"][episode + 1])
        # Closing observations contain [port state, P3-relative error].
        trajectory = np.asarray(data["observations"][start:stop, 6:8], dtype=float)
        final_error = np.asarray(
            data["final_states"][episode, :2] - data["p3_targets"][episode, :2],
            dtype=float,
        )
    return np.vstack((trajectory, final_error))


def add_geometry(ax, x_min, x_max):
    """Add the idealized docking cone, near-P3 corridor, and docking axis."""
    cone_x = np.linspace(0.0, x_max, 800)
    ax.plot(cone_x, 0.1 * cone_x, color="#E69F00", ls=":", lw=1.15)
    ax.plot(cone_x, -0.1 * cone_x, color="#E69F00", ls=":", lw=1.15)
    # The final docking corridor begins at P3 and continues into the
    # post-P3 region, rather than extending backward along the closing leg.
    ax.hlines((5.0, -5.0), x_min, 0.0, colors="#9467BD",
              linestyles=":", linewidth=1.15)
    ax.axhline(0.0, color="0.55", ls="-.", lw=0.85)


def main():
    sac, sac_info = load_sac_trajectory()
    lqr = load_lqr_demonstration()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.55))

    x_max = 1.04 * max(float(np.max(sac[:, 0])), float(np.max(lqr[:, 0])))
    x_min = min(-20.0, float(np.min(sac[:, 0])), float(np.min(lqr[:, 0])))
    plot_x_min = x_min - 0.02 * x_max
    add_geometry(ax, plot_x_min, x_max)

    sac_color = "#D55E00"
    lqr_color = "#0072B2"
    ax.plot(sac[:, 0], sac[:, 1], color=sac_color, lw=1.8,
            label=f"Replay-pre-fill SAC (testing seed {SAC_SEED})", zorder=4)
    ax.plot(lqr[:, 0], lqr[:, 1], color=lqr_color, lw=1.8,
            label=f"Successful LQR demonstration (seed {LQR_SEED})", zorder=4)

    for trajectory, color in ((sac, sac_color), (lqr, lqr_color)):
        ax.scatter(*trajectory[0], marker="D", s=45, facecolor="white",
                   edgecolor=color, linewidth=1.25, zorder=6)
        ax.scatter(*trajectory[-1], marker="X", s=62, color=color,
                   edgecolor="white", linewidth=0.55, zorder=7)
    ax.scatter(0.0, 0.0, marker="*", s=125, color="#009E73",
               edgecolor="white", linewidth=0.55, zorder=8)

    all_y = np.concatenate((sac[:, 1], lqr[:, 1], 0.1 * np.array([x_max, -x_max])))
    y_extent = 1.06 * float(np.max(np.abs(all_y)))
    ax.set_xlim(plot_x_min, x_max)
    ax.set_ylim(-y_extent, y_extent)
    ax.invert_xaxis()
    ax.set_xlabel(r"Axial position error relative to $P_3$ [m]")
    ax.set_ylabel(r"Lateral position error relative to $P_3$ [m]")
    ax.set_title(
        "Representative replay-pre-fill SAC rollout versus successful LQR demonstration",
        fontweight="normal", pad=52,
    )
    ax.grid(True, color="0.85", lw=0.6, alpha=0.75)
    ax.set_axisbelow(True)

    handles = [
        Line2D([], [], color=sac_color, lw=1.8,
               label=f"Replay-pre-fill SAC (testing seed {SAC_SEED})"),
        Line2D([], [], color=lqr_color, lw=1.8,
               label=f"Successful LQR demonstration (seed {LQR_SEED})"),
        Line2D([], [], marker="D", ls="none", mfc="white", mec="0.25",
               label="Initial state"),
        Line2D([], [], marker="X", ls="none", color="0.25", label="Terminal state"),
        Line2D([], [], marker="*", ls="none", color="#009E73", label=r"$P_3$ target"),
        Line2D([], [], color="#E69F00", ls=":", label="Docking cone"),
        Line2D([], [], color="#9467BD", ls=":", label="Docking corridor"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.055),
              ncol=4, frameon=False, fontsize=7.0, handlelength=2.2,
              columnspacing=1.25)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=400,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"SAC seed: {SAC_SEED} ({sac_info.get('termination_reason', 'timeout')})")
    print(f"SAC terminal position error: {np.linalg.norm(sac[-1]):.3f} m")
    print(f"LQR demonstration seed: {LQR_SEED} (successful handover)")
    print(f"LQR terminal position error: {np.linalg.norm(lqr[-1]):.3f} m")
    print(f"PNG: {OUTPUT_STEM.with_suffix('.png').resolve()}")
    print(f"PDF: {OUTPUT_STEM.with_suffix('.pdf').resolve()}")


if __name__ == "__main__":
    main()
