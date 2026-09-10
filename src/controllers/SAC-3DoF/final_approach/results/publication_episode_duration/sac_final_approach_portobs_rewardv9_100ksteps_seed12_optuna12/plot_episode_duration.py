"""Plot reconstructed training-episode durations for the retained seed-12 SAC run.

The source learning curve records the cumulative training timestep at the end
of every completed episode. Since training used one environment, differencing
successive terminal timesteps gives each individual episode length exactly.
The environment timestep was 1.0 s, so duration = episode length * 1.0 s.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RUN_NAME = "sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12"
DT_SECONDS = 1.0
ROLLING_WINDOW = 10

SCRIPT_PATH = Path(__file__).resolve()
FINAL_APPROACH_DIR = SCRIPT_PATH.parents[3]
SOURCE_CSV = (
    FINAL_APPROACH_DIR
    / "results"
    / "optuna12_100ksteps_buffer100k"
    / "learning_curves"
    / RUN_NAME
    / "learning_curve.csv"
)
OUTPUT_PNG = SCRIPT_PATH.with_name("episode_duration_vs_training_episode.png")
OUTPUT_PDF = SCRIPT_PATH.with_name("episode_duration_vs_training_episode.pdf")


def load_episode_durations(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    required = {"episode", "timestep"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if not data["episode"].equals(pd.Series(range(1, len(data) + 1))):
        raise ValueError("Episodes are not stored in chronological order")
    if not data["timestep"].is_monotonic_increasing:
        raise ValueError("Episode-end timesteps are not monotonically increasing")

    data = data.copy()
    data["episode_length_steps"] = data["timestep"].diff().fillna(data["timestep"])
    data["episode_length_steps"] = data["episode_length_steps"].astype(int)
    data["episode_duration_seconds"] = data["episode_length_steps"] * DT_SECONDS
    data["rolling_duration_seconds"] = data["episode_duration_seconds"].rolling(
        ROLLING_WINDOW, min_periods=1
    ).mean()
    return data


def make_plot(data: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)

    ax.plot(
        data["episode"],
        data["episode_duration_seconds"],
        color="#2C7FB8",
        linewidth=0.8,
        marker="o",
        markersize=2.1,
        markeredgewidth=0,
        alpha=0.48,
        label="Individual episode",
        zorder=2,
    )
    ax.plot(
        data["episode"],
        data["rolling_duration_seconds"],
        color="#D95F0E",
        linewidth=1.8,
        label=f"{ROLLING_WINDOW}-episode rolling mean",
        zorder=3,
    )

    ax.set(
        xlabel="Episode number",
        ylabel="Episode duration (s)",
        title="Retained SAC final-approach policy (training seed 12)",
        xlim=(1, int(data["episode"].iloc[-1])),
        ylim=(0, None),
    )
    ax.grid(True, which="major", color="0.88", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = load_episode_durations(SOURCE_CSV)
    make_plot(data)
    duration = data["episode_duration_seconds"]
    print(f"Source: {SOURCE_CSV}")
    print(f"Completed episodes: {len(data)}")
    print(f"Last completed episode timestep: {int(data['timestep'].iloc[-1])}")
    print(f"Duration min/mean/max (s): {duration.min():.2f} / {duration.mean():.2f} / {duration.max():.2f}")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
