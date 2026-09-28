"""Plot Round 5, seed 3 SAC actor and critic losses versus episode number."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


HERE = Path(__file__).resolve().parents[1]
RUN = "sac_final_approach_portobs_rewardv9_600ksteps_seed3_optuna12_buffer600k"
EVENT_DIR = (
    HERE / "logs" / "retained_policy_training"
    / "optuna12_600ksteps_buffer600k" / f"{RUN}_1"
)
LEARNING_CURVE = (
    HERE / "results" / "retained_policy_evaluations"
    / "optuna12_600ksteps_buffer600k"
    / "learning_curves" / RUN / "learning_curve.csv"
)
OUT = HERE / "results" / "retained_figures" / "selected_policy_analysis"
BLUE = "#0072B2"


def completed_episode_numbers(training_steps: np.ndarray, episode_end_steps: np.ndarray) -> np.ndarray:
    """Return the number of episodes completed at each TensorBoard training step."""
    return np.searchsorted(episode_end_steps, training_steps, side="right")


def main() -> None:
    learning_curve = pd.read_csv(LEARNING_CURVE)
    accumulator = EventAccumulator(str(EVENT_DIR))
    accumulator.Reload()

    series = {}
    for tag in ("train/actor_loss", "train/critic_loss"):
        events = accumulator.Scalars(tag)
        steps = np.asarray([event.step for event in events], dtype=int)
        values = np.asarray([event.value for event in events], dtype=float)
        episodes = completed_episode_numbers(
            steps, learning_curve["timestep"].to_numpy(dtype=int)
        )
        series[tag] = (episodes, values)

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    panels = (
        ("train/actor_loss", "Actor loss"),
        ("train/critic_loss", "Critic loss"),
    )
    for ax, (tag, title) in zip(axes, panels):
        episodes, values = series[tag]
        ax.plot(episodes, values, color=BLUE, linewidth=1.6)
        ax.set_xlabel("Episode number")
        ax.set_ylabel(title)
        ax.set_title(title, loc="left", fontweight="normal")
        ax.grid(True, color="#D9D9D9", linewidth=.65, alpha=.75)
        ax.set_axisbelow(True)
        ax.margins(x=.01)

    fig.suptitle("SAC training losses — Round 5, seed 3", fontweight="normal")
    fig.subplots_adjust(left=.09, right=.985, bottom=.15, top=.84, wspace=.27)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / "r5_s3_b600k_actor_critic_losses.png",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
