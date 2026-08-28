"""Evaluate and plot per-channel actuator saturation over closing rollouts.

The figure combines a seed-by-channel heatmap with a distribution view.  The
underlying CSV is cached so subsequent plotting does not rerun the simulations.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent.parent
SWEEP_DIR = SCRIPT_DIR / "tuning_lqr_closing"
for import_path in (SCRIPT_DIR, SRC_DIR, SWEEP_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from envs.docking3dof.core import Docking3DoFEnv
from lqr_3dof_closing import (
    LQRClosingController,
    build_linearized_3dof_matrices,
    build_Q_closing,
    build_R_closing,
    compute_lqr_gain,
)
from lqr_sweep_closing import run_single_rollout


RESULT_DIR = SCRIPT_DIR / "results" / "lqr_closing"
DEFAULT_CSV = RESULT_DIR / "closing_phase_0_49_actuator_saturation.csv"
DEFAULT_OUTPUT = RESULT_DIR / "closing_actuator_saturation"
CHANNEL_COLUMNS = (
    "saturation_fraction_ax",
    "saturation_fraction_ay",
    "saturation_fraction_alphaz",
)
CHANNEL_LABELS = (r"$a_x$", r"$a_y$", r"$\alpha_z$")

SUCCESS_COLOR = "#0072B2"
TIMEOUT_COLOR = "#D55E00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-channel saturation over closing-phase rollouts."
    )
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=12000.0)
    parser.add_argument(
        "--mode", choices=("nominal", "testing", "training"), default="testing"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Rerun simulations even when the saturation CSV already exists.",
    )
    return parser.parse_args()


def build_controller(max_time: float) -> LQRClosingController:
    env = Docking3DoFEnv(phase="closing", max_time=max_time, seed=0)
    A, B = build_linearized_3dof_matrices(env)
    gain, _ = compute_lqr_gain(A, B, build_Q_closing(), build_R_closing(env))
    return LQRClosingController(gain)


def collect_rollouts(args: argparse.Namespace) -> list[dict[str, object]]:
    controller = build_controller(args.max_time)
    records: list[dict[str, object]] = []
    for rollout_index in range(args.rollouts):
        seed = args.seed_start + rollout_index
        env, _, info, steps = run_single_rollout(
            controller=controller,
            max_time=args.max_time,
            seed=seed,
            mode=args.mode,
        )
        _, _, requested, applied, _ = env.get_history_arrays()
        clipped = np.abs(requested - applied) > 1e-12
        channel_fraction = np.mean(clipped, axis=0)
        any_fraction = float(np.mean(np.any(clipped, axis=1)))
        records.append(
            {
                "seed": seed,
                "success": bool(info.get("success", False)),
                "termination_reason": str(info.get("termination_reason", "unknown")),
                "steps": int(steps),
                "saturation_fraction_ax": float(channel_fraction[0]),
                "saturation_fraction_ay": float(channel_fraction[1]),
                "saturation_fraction_alphaz": float(channel_fraction[2]),
                "saturation_fraction_any": any_fraction,
            }
        )
        if (rollout_index + 1) % 5 == 0 or rollout_index + 1 == args.rollouts:
            print(f"Completed {rollout_index + 1}/{args.rollouts} rollouts")
    return records


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def load_records(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No saturation records found in {path}")
    records: list[dict[str, object]] = []
    for row in rows:
        records.append(
            {
                "seed": int(row["seed"]),
                "success": row["success"].strip().lower() == "true",
                "termination_reason": row["termination_reason"],
                "steps": int(row["steps"]),
                **{column: float(row[column]) for column in CHANNEL_COLUMNS},
                "saturation_fraction_any": float(row["saturation_fraction_any"]),
            }
        )
    return records


def make_figure(records: list[dict[str, object]]) -> plt.Figure:
    records = sorted(records, key=lambda record: int(record["seed"]))
    success = np.asarray([record["success"] for record in records], dtype=bool)
    values = 100.0 * np.asarray(
        [[record[column] for column in CHANNEL_COLUMNS] for record in records],
        dtype=float,
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 11.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.8))
    rng = np.random.default_rng(20260828)
    panel_titles = (
        r"(a) Body-axis acceleration $a_{x,b_c}$",
        r"(b) Body-axis acceleration $a_{y,b_c}$",
        r"(c) Body-axis angular acceleration $\alpha_{z,b_c}$",
    )
    y_limits = ((89.5, 91.15), (92.4, 99.5), (-0.001, 0.031))

    for channel_index, ax in enumerate(axes):
        successful_values = values[success, channel_index]
        timeout_values = values[~success, channel_index]
        groups = (successful_values, timeout_values)
        positions = (0.0, 1.0)
        boxes = ax.boxplot(
            groups,
            positions=positions,
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.5},
            whiskerprops={"color": "#777777", "linewidth": 1.0},
            capprops={"color": "#777777", "linewidth": 1.0},
            boxprops={"edgecolor": "#777777", "linewidth": 1.0},
        )
        boxes["boxes"][0].set_facecolor(SUCCESS_COLOR)
        boxes["boxes"][0].set_alpha(0.12)
        boxes["boxes"][1].set_facecolor(TIMEOUT_COLOR)
        boxes["boxes"][1].set_alpha(0.12)

        success_jitter = rng.uniform(-0.16, 0.16, size=successful_values.size)
        timeout_jitter = rng.uniform(-0.16, 0.16, size=timeout_values.size)
        ax.scatter(
            positions[0] + success_jitter,
            successful_values,
            s=25,
            marker="o",
            facecolor=SUCCESS_COLOR,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.8,
            zorder=3,
        )
        ax.scatter(
            positions[1] + timeout_jitter,
            timeout_values,
            s=43,
            marker="X",
            facecolor=TIMEOUT_COLOR,
            edgecolor="#7F2704",
            linewidth=0.45,
            alpha=0.9,
            zorder=4,
        )
        for position, group in zip(positions, groups):
            ax.scatter(
                position,
                np.mean(group),
                s=68,
                marker="D",
                facecolor="white",
                edgecolor="#111111",
                linewidth=1.1,
                zorder=5,
            )

        ax.set_xticks(
            positions,
            (f"Successful\n(n={successful_values.size})", f"Timeout\n(n={timeout_values.size})"),
        )
        ax.set_xlim(-0.48, 1.48)
        ax.set_ylim(*y_limits[channel_index])
        ax.set_title(panel_titles[channel_index], loc="left", fontweight="bold")
        ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.text(
            0.5,
            0.96,
            (
                f"All-rollout mean: {np.mean(values[:, channel_index]):.3f}%"
                if channel_index == 2
                else f"All-rollout mean: {np.mean(values[:, channel_index]):.2f}%"
            ),
            transform=ax.transAxes,
            fontsize=10.5,
            ha="center",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )

    axes[0].set_ylabel("Saturated timesteps per rollout [%]")

    success_count = int(np.sum(success))
    timeout_count = int(np.sum(~success))
    legend_handles = [
        Line2D(
            [], [], marker="o", linestyle="none", markerfacecolor=SUCCESS_COLOR,
            markeredgecolor="white", markersize=7, label=f"Successful ({success_count})"
        ),
        Line2D(
            [], [], marker="X", linestyle="none", markerfacecolor=TIMEOUT_COLOR,
            markeredgecolor="#7F2704", markersize=7, label=f"Timeout ({timeout_count})"
        ),
        Line2D(
            [], [], marker="D", linestyle="none", markerfacecolor="white",
            markeredgecolor="#111111", markersize=7, label="Mean"
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.84, wspace=0.28)
    return fig


def print_summary(records: list[dict[str, object]]) -> None:
    values = 100.0 * np.asarray(
        [[record[column] for column in CHANNEL_COLUMNS] for record in records],
        dtype=float,
    )
    for label, channel_values in zip(CHANNEL_LABELS, values.T):
        print(
            f"{label}: mean={np.mean(channel_values):.3f}%, "
            f"median={np.median(channel_values):.3f}%, "
            f"range=[{np.min(channel_values):.3f}%, {np.max(channel_values):.3f}%]"
        )


def main() -> None:
    args = parse_args()
    if args.recompute or not args.csv.exists():
        records = collect_rollouts(args)
        write_records(args.csv, records)
        print(f"Wrote {args.csv}")
    else:
        records = load_records(args.csv)
        print(f"Loaded {args.csv}")

    figure = make_figure(records)
    output_base = args.output.with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    figure.savefig(png_path, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print_summary(records)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
