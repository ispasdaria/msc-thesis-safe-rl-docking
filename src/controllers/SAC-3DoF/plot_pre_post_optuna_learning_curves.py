"""Create scientifically matched pre-/post-Optuna learning-curve overlays.

The source CSV files are never modified.  Each panel shows every raw episode
return (low opacity) and a trailing moving average computed here with exactly
the same window and edge convention for the two runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "pre_post_optuna_comparison"


@dataclass(frozen=True)
class Comparison:
    slug: str
    title: str
    x_label: str
    show_caption: bool
    pre_csv: Path
    post_csv: Path
    window: int
    x_max: int
    pre_learning_starts: int
    post_learning_starts: int
    caption: str
    metadata: dict


COMPARISONS = (
    Comparison(
        slug="final_approach_reward9_pre_vs_post_optuna",
        title="Final-approach retained reward: Pre- vs Post-Optuna",
        x_label="Episode number",
        show_caption=False,
        pre_csv=ROOT / "final_approach" / "results" / "reward_shaping_trials"
        / "learning_curves" / "sac_final_approach_portobs_rewardv9_100ksteps_seed0"
        / "learning_curve.csv",
        post_csv=ROOT / "final_approach" / "results" / "optuna12_100ksteps_buffer100k"
        / "learning_curves"
        / "sac_final_approach_portobs_rewardv9_100ksteps_seed0_optuna12"
        / "learning_curve.csv",
        window=10,
        x_max=100_000,
        pre_learning_starts=1_000,
        post_learning_starts=1_000,
        caption=(
            "Training seed 0; 100,000 environment steps; docking-port observation; "
            "Reward 9; dt=1 s; RK4; normalized actions; post-Optuna uses trial 12 "
            "and a 100,000-transition replay buffer. Episodes ending before "
            "learning_starts=1,000 steps are omitted."
        ),
        metadata={
            "seed": 0,
            "phase": "final approach",
            "reward_version": 9,
            "observation": "docking-port",
            "training_budget_steps": 100_000,
            "post_optuna_trial": 12,
            "post_replay_buffer_size": 100_000,
            "dt_seconds": 1.0,
            "integrator": "RK4",
        },
    ),
    Comparison(
        slug="closing_phase_reward15_pre_vs_post_optuna",
        title="Analyzed closing-phase reward: Pre- vs Post-Optuna",
        x_label="Episode number",
        show_caption=False,
        pre_csv=ROOT / "closing" / "results" / "reward_shaping_trials"
        / "episode_return_learning_curves"
        / "sac_closing_p3augobs_rewardv15_300ksteps_seed0_buffer300k"
        / "learning_curve.csv",
        post_csv=ROOT / "closing" / "results" / "reward_shaping_trials"
        / "episode_return_learning_curves"
        / "sac_closing_p3augobs_rewardv15_optuna_trial16_400ksteps_seed0_buffer300k"
        / "learning_curve.csv",
        window=10,
        x_max=300_000,
        pre_learning_starts=1_000,
        post_learning_starts=10_000,
        caption=(
            "Training seed 0; common first 300,000 environment steps; augmented P3 "
            "observation; Reward 15; 300,000-transition replay buffer; dt=1 s; RK4; "
            "normalized actions. Post-Optuna uses trial 16 (the source run continued "
            "to 400,000 steps). Replay-only prefixes are omitted: learning_starts is "
            "1,000 steps pre-Optuna and 10,000 steps post-Optuna."
        ),
        metadata={
            "seed": 0,
            "phase": "closing",
            "reward_version": 15,
            "observation": "augmented docking-port safety state plus P3 error",
            "comparison_horizon_steps": 300_000,
            "pre_training_budget_steps": 300_000,
            "post_source_training_budget_steps": 400_000,
            "post_optuna_trial": 16,
            "replay_buffer_size": 300_000,
            "dt_seconds": 1.0,
            "integrator": "RK4",
        },
    ),
)


def load_raw(path: Path, x_max: int, learning_starts: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"episode", "timestep", "episode_return"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.loc[:, ["episode", "timestep", "episode_return"]].copy()
    frame = frame.apply(pd.to_numeric, errors="raise")
    if frame.empty or not frame["timestep"].is_monotonic_increasing:
        raise ValueError(f"{path} has an empty or non-monotonic timestep series")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path} contains non-finite values")
    # An episode is retained when its terminal timestep reaches the first point
    # at which gradient learning is enabled. Its return can straddle the exact
    # threshold, which is preferable to inventing a partial-episode return.
    keep = (frame["timestep"] >= learning_starts) & (frame["timestep"] <= x_max)
    return frame.loc[keep].reset_index(drop=True)


def trailing_mean(values: pd.Series, window: int) -> pd.Series:
    # min_periods=1 preserves an aligned value for every raw episode.
    return values.rolling(window=window, min_periods=1).mean()


def plot_one(spec: Comparison) -> dict:
    pre = load_raw(spec.pre_csv, spec.x_max, spec.pre_learning_starts)
    post = load_raw(spec.post_csv, spec.x_max, spec.post_learning_starts)
    if pre.empty or post.empty:
        raise ValueError(f"No shared-horizon observations for {spec.slug}")

    colors = {"Pre-Optuna": "#0072B2", "Post-Optuna": "#D55E00"}
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for label, frame in (("Pre-Optuna", pre), ("Post-Optuna", post)):
        color = colors[label]
        ax.plot(
            frame["episode"], frame["episode_return"], color=color,
            alpha=0.18, linewidth=0.75, label=f"{label} (raw)", rasterized=False,
        )
        ax.plot(
            frame["episode"], trailing_mean(frame["episode_return"], spec.window),
            color=color, linewidth=2.0,
            label=f"{label} ({spec.window}-episode moving average)",
        )

    ax.set(xlabel=spec.x_label, ylabel="Episode return", title=spec.title)
    first_episode = min(pre["episode"].min(), post["episode"].min())
    last_episode = max(pre["episode"].max(), post["episode"].max())
    ax.set_xlim(max(1, first_episode - 1), last_episode + 1)
    ax.grid(True, which="major", color="0.85", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="best")
    ax.tick_params(direction="out")
    bottom_margin = 0.23 if spec.show_caption else 0.12
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=bottom_margin)
    if spec.show_caption:
        fig.text(
            0.5, 0.035, spec.caption, ha="center", va="bottom", fontsize=7.5,
            wrap=True,
        )

    common_metadata = {
        "Title": spec.title,
        "Author": "msc-thesis-safe-rl-docking",
        "Subject": "Pre-Optuna versus post-Optuna learning curve",
    }
    png = OUTPUT_DIR / f"{spec.slug}.png"
    fig.savefig(
        png,
        dpi=300,
        metadata={**common_metadata, "Description": spec.caption},
    )
    plt.close(fig)

    return {
        **spec.metadata,
        "metric": "raw training episode return",
        "x_axis": "original completed training episode number",
        "replay_fill_filter": {
            "rule": "retain completed episodes whose terminal timestep is at or after learning_starts",
            "pre_learning_starts_steps": spec.pre_learning_starts,
            "post_learning_starts_steps": spec.post_learning_starts,
        },
        "smoothing": {
            "type": "trailing moving average",
            "window_episodes": spec.window,
            "min_periods": 1,
            "identical_for_both_runs": True,
        },
        "raw_data_plotted": True,
        "pre_source": str(spec.pre_csv.relative_to(ROOT)),
        "post_source": str(spec.post_csv.relative_to(ROOT)),
        "pre_points_plotted": len(pre),
        "post_points_plotted": len(post),
        "caption": spec.caption,
        "outputs": [png.name],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    manifest = {spec.slug: plot_one(spec) for spec in COMPARISONS}
    manifest_path = OUTPUT_DIR / "comparison_metadata.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(COMPARISONS)} PNG/PDF figure pairs and {manifest_path}")


if __name__ == "__main__":
    main()
