"""Plot closing-LQR sampled-data spectral radius versus control interval.

The continuous-time CARE gain is held fixed.  For each candidate control
interval, only the linear plant is discretized using an exact zero-order hold:

    x[k + 1] = (A_d - B_d K) x[k].

This distinction matters: redesigning K at every interval would compare
different controllers and would not isolate the sampled-data stability limit.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from scipy.linalg import expm


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent.parent
for import_path in (SCRIPT_DIR, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from envs.docking3dof.core import Docking3DoFEnv
from lqr_3dof_closing import (
    build_Q_closing,
    build_R_closing,
    build_linearized_3dof_matrices,
    compute_lqr_gain,
)


RESULT_DIR = SCRIPT_DIR / "results" / "lqr_closing"
DEFAULT_OUTPUT = RESULT_DIR / "closing_sampled_data_stability"
DEFAULT_INTERVALS = (1.0, 0.5, 0.2, 0.1, 0.05)

STABLE_COLOR = "#0072B2"
UNSTABLE_COLOR = "#D55E00"
SELECTED_COLOR = "#009E73"
LINE_COLOR = "#555555"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot rho(A_d - B_d K) for candidate closing-LQR control intervals."
        )
    )
    parser.add_argument(
        "--intervals",
        type=float,
        nargs="+",
        default=DEFAULT_INTERVALS,
        metavar="DT",
        help="Candidate control intervals in seconds (default: 1 .5 .2 .1 .05).",
    )
    parser.add_argument(
        "--gain-design-dt",
        type=float,
        choices=(1.0, 0.1),
        default=0.1,
        help=(
            "Selected Q matrix used to form the fixed continuous-time gain K "
            "(default: 0.1, the final closing controller)."
        ),
    )
    parser.add_argument(
        "--selected-dt",
        type=float,
        default=0.1,
        help="Control interval highlighted as selected (default: 0.1 s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without an extension; PNG, PDF, and CSV are written.",
    )
    args = parser.parse_args()
    if any(dt <= 0.0 for dt in args.intervals):
        parser.error("all --intervals values must be positive")
    if len(set(args.intervals)) != len(args.intervals):
        parser.error("--intervals must not contain duplicates")
    if args.selected_dt not in args.intervals:
        parser.error("--selected-dt must be one of --intervals")
    return args


def exact_zoh(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact zero-order-hold discretization of (A, B)."""
    n, m = B.shape
    block = np.zeros((n + m, n + m), dtype=float)
    block[:n, :n] = A
    block[:n, n:] = B
    discrete = expm(block * dt)
    return discrete[:n, :n], discrete[:n, n:]


def compute_stability_records(
    intervals: tuple[float, ...] | list[float], gain_design_dt: float
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Compute sampled eigenvalues while keeping the continuous gain fixed."""
    env = Docking3DoFEnv(
        phase="closing", dt=gain_design_dt, max_time=12_000.0, seed=0
    )
    A, B = build_linearized_3dof_matrices(env)
    gain, _ = compute_lqr_gain(
        A,
        B,
        build_Q_closing(dt=gain_design_dt),
        build_R_closing(env),
    )

    records: list[dict[str, object]] = []
    for dt in sorted(intervals):
        Ad, Bd = exact_zoh(A, B, dt)
        eigenvalues = np.linalg.eigvals(Ad - Bd @ gain)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        records.append(
            {
                "dt_seconds": float(dt),
                "spectral_radius": spectral_radius,
                "sampled_data_stable": spectral_radius < 1.0,
            }
        )
    return records, gain


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _draw_points(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    stable = y < 1.0
    ax.plot(x, y, color=LINE_COLOR, linewidth=1.5, zorder=2)
    ax.scatter(
        x[stable],
        y[stable],
        s=64,
        marker="o",
        facecolor=STABLE_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.scatter(
        x[~stable],
        y[~stable],
        s=72,
        marker="X",
        facecolor=UNSTABLE_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )


def make_figure(
    records: list[dict[str, object]], selected_dt: float, gain_design_dt: float
) -> plt.Figure:
    x = np.asarray([record["dt_seconds"] for record in records], dtype=float)
    y = np.asarray([record["spectral_radius"] for record in records], dtype=float)
    selected_index = int(np.flatnonzero(np.isclose(x, selected_dt))[0])

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "legend.fontsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.25))
    y_max = max(1.12, 1.08 * float(np.max(y)))
    ax.axhspan(0.0, 1.0, color=STABLE_COLOR, alpha=0.055, zorder=0)
    ax.axhspan(1.0, y_max, color=UNSTABLE_COLOR, alpha=0.045, zorder=0)
    ax.axhline(1.0, color="#222222", linewidth=1.35, linestyle="--", zorder=1)
    _draw_points(ax, x, y)

    # Ring the chosen point without changing the stable/unstable color coding.
    ax.scatter(
        [x[selected_index]],
        [y[selected_index]],
        s=155,
        marker="o",
        facecolor="none",
        edgecolor=SELECTED_COLOR,
        linewidth=2.2,
        zorder=5,
    )

    for dt, rho in zip(x, y):
        if np.isclose(dt, selected_dt) or dt <= 0.1:
            continue
        label = f"{rho:.3f}"
        offset = (-4, 9) if dt >= 0.5 else (0, 10)
        ax.annotate(
            label,
            (dt, rho),
            xytext=offset,
            textcoords="offset points",
            ha="right" if dt >= 0.5 else "center",
            va="bottom",
            fontsize=9.3,
            color="#333333",
        )

    ax.annotate(
        f"selected: {selected_dt:g} s\n(largest tested stable interval)",
        xy=(x[selected_index], y[selected_index]),
        xytext=(0.28, 2.15),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": SELECTED_COLOR, "linewidth": 1.35},
        fontsize=10,
        color="#006B50",
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": SELECTED_COLOR,
            "alpha": 0.95,
        },
    )

    ax.set_xscale("log")
    ax.set_xlim(0.04, 1.2)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{dt:g}" for dt in x])
    ax.set_xlabel(r"Control interval $\Delta t$ [s]")
    ax.set_ylabel(r"Spectral radius $\rho(A_d-B_dK)$")
    ax.set_title("Sampled-data stability of the closing-phase LQR", loc="left", fontweight="bold")
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.75, alpha=0.85)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)

    ax.text(
        0.043,
        1.12,
        r"unstable: $\rho>1$",
        color=UNSTABLE_COLOR,
        fontsize=9.5,
        ha="left",
        va="bottom",
    )
    ax.text(
        0.043,
        0.60,
        r"stable: $\rho<1$",
        color=STABLE_COLOR,
        fontsize=9.5,
        ha="left",
        va="center",
    )

    # The dominant slow mode places both stable values extremely close to one;
    # this inset makes their strict location below the threshold legible.
    inset = ax.inset_axes([0.11, 0.49, 0.38, 0.34])
    zoom_mask = x <= 0.2 + 1e-12
    inset.axhspan(0.94, 1.0, color=STABLE_COLOR, alpha=0.07, zorder=0)
    inset.axhspan(1.0, 1.62, color=UNSTABLE_COLOR, alpha=0.055, zorder=0)
    inset.axhline(1.0, color="#222222", linewidth=1.0, linestyle="--", zorder=1)
    _draw_points(inset, x[zoom_mask], y[zoom_mask])
    inset.scatter(
        [x[selected_index]],
        [y[selected_index]],
        s=112,
        marker="o",
        facecolor="none",
        edgecolor=SELECTED_COLOR,
        linewidth=1.8,
        zorder=5,
    )
    inset.set_xscale("log")
    inset.set_xlim(0.045, 0.23)
    inset.set_ylim(0.94, 1.62)
    inset.xaxis.set_major_locator(FixedLocator(x[zoom_mask]))
    inset.xaxis.set_major_formatter(
        FixedFormatter([f"{dt:g}" for dt in x[zoom_mask]])
    )
    inset.xaxis.set_minor_formatter(NullFormatter())
    inset.xaxis.offsetText.set_visible(False)
    inset.tick_params(axis="x", labelsize=8)
    inset.tick_params(axis="y", labelsize=8)
    inset.grid(True, which="major", color="#E0E0E0", linewidth=0.6)
    inset.grid(False, which="minor")
    inset.set_title(r"Stability-boundary detail", fontsize=9.2, loc="left", pad=3)
    for dt, rho in zip(x[zoom_mask], y[zoom_mask]):
        label = f"{rho:.6f}" if rho < 1.01 else f"{rho:.3f}"
        inset.annotate(
            label,
            (dt, rho),
            xytext=(0, -10 if rho >= 1.01 else 7),
            textcoords="offset points",
            ha="center",
            va="top" if rho >= 1.01 else "bottom",
            fontsize=7.5,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=STABLE_COLOR,
            markeredgecolor="white",
            markersize=8,
            label=r"Stable ($\rho<1$)",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor=UNSTABLE_COLOR,
            markeredgecolor="white",
            markersize=8,
            label=r"Unstable ($\rho>1$)",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.95)
    fig.text(
        0.01,
        0.01,
        (
            "Linear, unsaturated model with exact zero-order-hold discretization; "
            "fixed continuous-time CARE gain "
            f"using the selected Q for $\\Delta t={gain_design_dt:g}$ s."
        ),
        fontsize=8.3,
        color="#555555",
    )
    fig.subplots_adjust(left=0.105, right=0.98, top=0.91, bottom=0.15)
    return fig


def main() -> None:
    args = parse_args()
    records, gain = compute_stability_records(args.intervals, args.gain_design_dt)
    figure = make_figure(records, args.selected_dt, args.gain_design_dt)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output.with_suffix(".png")
    csv_path = args.output.with_suffix(".csv")
    figure.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    write_records(csv_path, records)

    print("Fixed continuous-time gain K:")
    print(gain)
    for record in sorted(records, key=lambda item: item["dt_seconds"], reverse=True):
        status = "stable" if record["sampled_data_stable"] else "unstable"
        print(
            f"dt={record['dt_seconds']:g} s: "
            f"rho={record['spectral_radius']:.12f} ({status})"
        )
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
