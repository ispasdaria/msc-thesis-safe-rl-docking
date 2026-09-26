"""Compare the seed-3 SAC histories from training rounds 2 and 5.

The plot deliberately starts with the first episode that began after SAC updates
were enabled.  Model metadata stores ``learning_starts=1000`` for both runs.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RUNS = (
    {
        "name": "Round 2: 300k steps, buffer size 100k",
        "color": "#0072B2",
        "linestyle": "-",
        "csv": HERE / "results/optuna12_300ksteps_buffer100k/learning_curves/sac_final_approach_portobs_rewardv9_300ksteps_seed3_optuna12_buffer100k/learning_curve.csv",
        "model": HERE / "models/optuna12_300ksteps_buffer100k/sac_final_approach_portobs_rewardv9_300ksteps_seed3_optuna12_buffer100k.zip",
    },
    {
        "name": "Round 5: 600k steps, buffer size 600k",
        "color": "#D55E00",
        "linestyle": "--",
        "csv": HERE / "results/optuna12_600ksteps_buffer600k/learning_curves/sac_final_approach_portobs_rewardv9_600ksteps_seed3_optuna12_buffer600k/learning_curve.csv",
        "model": HERE / "models/optuna12_600ksteps_buffer600k/sac_final_approach_portobs_rewardv9_600ksteps_seed3_optuna12_buffer600k.zip",
    },
)
OUT_DIR = HERE / "results/thesis_selected_policy_figures"
CHECKPOINT_STEP = 100_000
SMOOTHING_WINDOW = 10


def model_metadata(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read("data"))
    return {key: data[key] for key in ("learning_starts", "buffer_size", "num_timesteps", "seed")}


def episode_coordinate_at_step(frame: pd.DataFrame, step: int) -> tuple[float, int, int, int]:
    """Map a step inside an episode to the continuous episode-axis coordinate."""
    index = int(np.searchsorted(frame["timestep"].to_numpy(), step, side="left"))
    episode = int(frame.iloc[index]["episode"])
    end_step = int(frame.iloc[index]["timestep"])
    start_step = 0 if index == 0 else int(frame.iloc[index - 1]["timestep"])
    fraction = (step - start_step) / (end_step - start_step)
    return (episode - 1 + fraction), episode, start_step, end_step


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit: dict[str, object] = {
        "smoothing": f"rolling mean over {SMOOTHING_WINDOW} episodes",
        "checkpoint_step": CHECKPOINT_STEP,
        "runs": [],
    }
    loaded = []
    for run in RUNS:
        metadata = model_metadata(run["model"])
        if metadata["learning_starts"] != 1_000:
            raise ValueError(f"Unexpected learning_starts in {run['model']}: {metadata}")
        frame = pd.read_csv(run["csv"])
        episode_starts = np.r_[0, frame["timestep"].to_numpy()[:-1]]
        # Exclude even the episode that straddles learning_starts: every plotted
        # episode therefore began after gradient updates were enabled.
        first_index = int(np.flatnonzero(episode_starts >= metadata["learning_starts"])[0])
        plotted = frame.iloc[first_index:].copy()
        plotted["smoothed_return"] = plotted["episode_return"].rolling(
            SMOOTHING_WINDOW, min_periods=1
        ).mean()
        checkpoint_x, checkpoint_episode, previous_end, episode_end = episode_coordinate_at_step(
            frame, CHECKPOINT_STEP
        )
        run_audit = {
            "label": run["name"],
            "model": str(run["model"].relative_to(HERE)),
            "learning_curve_csv": str(run["csv"].relative_to(HERE)),
            "model_metadata": metadata,
            "first_fully_post_learning_episode": int(plotted.iloc[0]["episode"]),
            "first_fully_post_learning_episode_start_step": int(episode_starts[first_index]),
            "checkpoint_episode": checkpoint_episode,
            "checkpoint_previous_episode_end_step": previous_end,
            "checkpoint_episode_end_step": episode_end,
            "checkpoint_continuous_episode_coordinate": checkpoint_x,
        }
        audit["runs"].append(run_audit)
        loaded.append((run, frame, plotted, checkpoint_x))

    a, b = loaded[0][1], loaded[1][1]
    common = min(len(a), len(b))
    comparison_columns = ["episode", "timestep", "episode_return", "termination_reason"]
    equal_rows = (a.iloc[:common][comparison_columns].reset_index(drop=True)
                  == b.iloc[:common][comparison_columns].reset_index(drop=True)).all(axis=1)
    mismatches = np.flatnonzero(~equal_rows.to_numpy())
    completed_before_checkpoint = a["timestep"] <= CHECKPOINT_STEP
    audit["history_comparison"] = {
        "completed_episodes_at_or_before_100k": int(completed_before_checkpoint.sum()),
        "all_logged_fields_identical_for_those_episodes": bool(equal_rows[completed_before_checkpoint].all()),
        "last_identical_episode_before_100k": int(a.loc[completed_before_checkpoint, "episode"].iloc[-1]),
        "last_identical_episode_end_step": int(a.loc[completed_before_checkpoint, "timestep"].iloc[-1]),
        "first_mismatching_episode": None if not len(mismatches) else int(a.iloc[mismatches[0]]["episode"]),
    }

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    for ax, early_only in zip(axes, (False, True)):
        for run, _frame, plotted, checkpoint_x in loaded:
            visible = plotted[plotted["timestep"] <= CHECKPOINT_STEP] if early_only else plotted
            ax.plot(
                visible["episode"], visible["episode_return"],
                color=run["color"], linestyle=run["linestyle"], alpha=0.18, linewidth=0.8,
            )
            ax.plot(
                visible["episode"], visible["smoothed_return"],
                color=run["color"], linestyle=run["linestyle"], linewidth=1.8,
                label=run["name"],
            )
            ax.axvline(checkpoint_x, color=run["color"], linestyle="--", linewidth=1.1, alpha=0.9)
        ax.set_xlabel("Episode number")
        ax.set_ylabel("Episode return")
        ax.grid(True, alpha=0.25, linewidth=0.5)
    axes[0].set_title("Full post-warm-up training histories")
    axes[1].set_title("Early training through 100k environment steps")
    checkpoint_midpoint = np.mean([item[3] for item in loaded])
    axes[1].annotate(
        "100k checkpoint\n(during episode 247)",
        xy=(checkpoint_midpoint, 0.96), xycoords=("data", "axes fraction"),
        xytext=(-7, -2), textcoords="offset points", ha="right", va="top",
        fontsize=8, bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.5", "alpha": 0.9},
    )
    axes[1].text(
        0.03, 0.04, "Raw returns: faint lines\nSmoothed: 10-episode rolling mean",
        transform=axes[1].transAxes, fontsize=8, va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.9},
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("SAC seed-3 learning curves: effect of replay-buffer capacity", fontsize=11)

    stem = OUT_DIR / "sac_seed3_round2_vs_round5_learning_curves"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    audit_path = stem.with_name(stem.name + "_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {audit_path}")


if __name__ == "__main__":
    main()
