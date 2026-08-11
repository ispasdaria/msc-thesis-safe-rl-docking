"""Plot saved final-approach SAC + CBF evaluation trajectories.

The script consumes the ``step_timeseries_*.npz`` files produced by
``evaluate_final_approach_cbf.py``.  It never reruns the simulator, so every
figure corresponds exactly to the metrics already stored in a result folder.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


VARIANT_LABELS = {
    "baseline": "Unfiltered SAC",
    "cone": "Cone CBF",
    "speed": "Speed CBF",
    "alignment": "Alignment CBF",
    "both": "Cone + speed CBF",
    "all": "Full CBF (cone + speed + alignment)",
}
VARIANT_COLORS = {
    "baseline": "#6b7280",
    "cone": "#d97706",
    "speed": "#7c3aed",
    "alignment": "#0891b2",
    "both": "#2563eb",
    "all": "#dc2626",
}
VARIANT_LINESTYLES = {
    "baseline": "--",
    "cone": "-.",
    "speed": ":",
    "alignment": "-.",
    "both": "-",
    "all": "-",
}
DEFAULT_VARIANTS = ("baseline", "both", "all")


@dataclass(frozen=True)
class Episode:
    variant: str
    eval_seed: int
    metrics: dict[str, str]
    arrays: dict[str, np.ndarray]

    @property
    def label(self) -> str:
        outcome = "success" if _csv_bool(self.metrics.get("success", "")) else self.metrics.get(
            "termination_reason", "failure"
        )
        return f"{VARIANT_LABELS.get(self.variant, self.variant)} ({outcome})"

    @property
    def color(self) -> str:
        return VARIANT_COLORS.get(self.variant, "#111827")

    @property
    def linestyle(self) -> str:
        return VARIANT_LINESTYLES.get(self.variant, "-")


def _csv_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _load_configuration(result_dir: Path) -> dict:
    path = result_dir / "run_configuration.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run configuration: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_metric_rows(result_dir: Path) -> list[dict[str, str]]:
    path = result_dir / "per_episode_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing episode metrics: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _available_pairs(rows: list[dict[str, str]]) -> str:
    pairs = sorted({(row["variant"], int(row["eval_seed"])) for row in rows})
    return ", ".join(f"{variant}/seed{seed}" for variant, seed in pairs)


def _load_episode(
    result_dir: Path,
    rows: list[dict[str, str]],
    variant: str,
    eval_seed: int,
) -> Episode:
    matching = [
        row
        for row in rows
        if row.get("variant") == variant and int(row.get("eval_seed", -1)) == eval_seed
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Expected exactly one metrics row for {variant}/seed{eval_seed}; "
            f"available pairs: {_available_pairs(rows)}"
        )

    row = matching[0]
    step_log = row.get("step_log", "")
    if not step_log:
        raise ValueError(f"Metrics row for {variant}/seed{eval_seed} has no step_log.")
    path = result_dir / step_log
    if not path.exists():
        raise FileNotFoundError(f"Missing step log named by the metrics CSV: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    if arrays.get("step", np.empty(0)).size == 0:
        raise ValueError(f"Step log is empty: {path}")
    return Episode(variant=variant, eval_seed=eval_seed, metrics=row, arrays=arrays)


def _require(episode: Episode, name: str) -> np.ndarray:
    if name not in episode.arrays:
        raise KeyError(
            f"{name!r} is absent from {episode.variant}/seed{episode.eval_seed}. "
            "Regenerate the evaluation with the current evaluator schema."
        )
    return episode.arrays[name]


def _time(episode: Episode) -> np.ndarray:
    if "time" in episode.arrays:
        return episode.arrays["time"].astype(float)
    return episode.arrays["step"].astype(float)


def _minimum_cone_barrier(episode: Episode) -> np.ndarray:
    upper = _require(episode, "post_h_cone_upper").astype(float)
    lower = _require(episode, "post_h_cone_lower").astype(float)
    return np.minimum(upper, lower)


def _thin_indices(mask: np.ndarray, maximum: int = 250) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size <= maximum:
        return indices
    return indices[np.linspace(0, indices.size - 1, maximum, dtype=int)]


def _add_direction_arrow(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if x.size < 3:
        return
    index = max(1, min(x.size - 2, int(0.65 * x.size)))
    previous = max(0, index - max(1, x.size // 100))
    if np.hypot(x[index] - x[previous], y[index] - y[previous]) <= 1.0e-12:
        return
    ax.annotate(
        "",
        xy=(x[index], y[index]),
        xytext=(x[previous], y[previous]),
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5},
    )


def _finish_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#d1d5db", alpha=0.65, linewidth=0.6)
    ax.set_axisbelow(True)


def _save_figure(
    figure: plt.Figure,
    output_stem: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    paths = []
    for extension in formats:
        path = output_stem.with_suffix(f".{extension}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def plot_spatial_trajectory(
    episodes: list[Episode],
    config: dict,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    figure, ax = plt.subplots(figsize=(11.0, 6.2), constrained_layout=True)
    cbf = config["cbf"]
    kappa = float(cbf["kappa"])

    s_values = np.concatenate([_require(ep, "s").astype(float) for ep in episodes])
    maximum_s = max(0.0, float(np.nanmax(s_values)))
    cone_s = np.linspace(0.0, maximum_s * 1.02 + 1.0e-9, 500)
    ax.fill_between(
        cone_s,
        -kappa * cone_s,
        kappa * cone_s,
        color="#dcfce7",
        alpha=0.55,
        label=r"Protected docking cone $|y_p| \leq \kappa s$",
        zorder=0,
    )
    ax.plot(cone_s, kappa * cone_s, color="#15803d", lw=1.2)
    ax.plot(cone_s, -kappa * cone_s, color="#15803d", lw=1.2)
    ax.axhline(5.0, color="#9ca3af", ls=":", lw=1.0, label="Simulator corridor (±5 m)")
    ax.axhline(-5.0, color="#9ca3af", ls=":", lw=1.0)
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            0.1,
            facecolor="#fde68a",
            edgecolor="#92400e",
            lw=1.2,
            zorder=4,
            label="Docking-distance target (0.1 m)",
        )
    )

    for episode in episodes:
        s = _require(episode, "s").astype(float)
        y_p = _require(episode, "y_p").astype(float)
        ax.plot(
            s,
            y_p,
            color=episode.color,
            ls=episode.linestyle,
            lw=1.8,
            label=episode.label,
            zorder=3,
        )
        ax.scatter(s[0], y_p[0], marker="o", s=35, color=episode.color, zorder=5)
        ax.scatter(s[-1], y_p[-1], marker="X", s=55, color=episode.color, zorder=5)
        _add_direction_arrow(ax, s, y_p, episode.color)

        intervention = episode.arrays.get("intervened", np.zeros(s.size, dtype=bool)).astype(bool)
        indices = _thin_indices(intervention)
        if indices.size:
            ax.scatter(
                s[indices],
                y_p[indices],
                marker=".",
                s=13,
                color=episode.color,
                alpha=0.55,
                zorder=4,
            )
        qp_failed = episode.arrays.get("qp_failed", np.zeros(s.size, dtype=bool)).astype(bool)
        indices = _thin_indices(qp_failed)
        if indices.size:
            ax.scatter(
                s[indices],
                y_p[indices],
                marker="x",
                s=38,
                linewidths=1.3,
                color="#111827",
                zorder=6,
            )

    ax.set_xlabel(r"Along-axis port separation $s$ [m]")
    ax.set_ylabel(r"Lateral port displacement $y_p$ [m]")
    ax.set_title(f"Final-approach spatial trajectory — evaluation seed {episodes[0].eval_seed}")
    ax.set_aspect("equal", adjustable="datalim")
    _finish_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="best", fontsize=8)
    ax.text(
        0.01,
        0.01,
        "○ start    ✕ end    small colored points: CBF intervention    black ×: failed QP",
        transform=ax.transAxes,
        fontsize=8,
        color="#374151",
    )
    return _save_figure(
        figure,
        output_dir / "spatial_docking_trajectory_comparison",
        formats,
        dpi,
    )


def plot_safety_envelopes(
    episodes: list[Episode],
    config: dict,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    figure, axes = plt.subplots(4, 1, figsize=(11.0, 10.5), sharex=True, constrained_layout=True)
    cbf = config["cbf"]
    attitude_safe = np.rad2deg(float(cbf["attitude_max"]))
    rate_safe = np.rad2deg(float(cbf["angular_rate_max"]))
    speed_safe = float(cbf["port_speed_max"])

    for episode in episodes:
        time = _time(episode)
        style = {
            "color": episode.color,
            "ls": episode.linestyle,
            "lw": 1.45,
            "label": episode.label,
        }
        axes[0].plot(time, _minimum_cone_barrier(episode), **style)
        axes[1].plot(time, _require(episode, "port_speed"), **style)
        axes[2].plot(time, np.rad2deg(_require(episode, "psi_error")), **style)
        axes[3].plot(time, np.rad2deg(_require(episode, "angular_rate")), **style)

    axes[0].axhline(0.0, color="#111827", lw=1.0)
    axes[0].set_ylabel("Minimum cone\nbarrier [m]")

    axes[1].axhline(speed_safe, color="#15803d", lw=1.0, label="CBF envelope")
    axes[1].axhline(0.2, color="#b91c1c", ls=":", lw=1.0, label="Simulator limit")
    axes[1].set_ylabel("Port speed\n[m/s]")

    for sign in (-1.0, 1.0):
        axes[2].axhline(sign * attitude_safe, color="#15803d", lw=1.0)
        axes[2].axhline(sign * 4.0, color="#b91c1c", ls=":", lw=1.0)
        axes[3].axhline(sign * rate_safe, color="#15803d", lw=1.0)
        axes[3].axhline(sign * 0.2, color="#b91c1c", ls=":", lw=1.0)
    axes[2].set_ylabel("Attitude error\n[deg]")
    axes[3].set_ylabel("Angular rate\n[deg/s]")
    axes[3].set_xlabel("Episode time [s]")

    for ax in axes:
        _finish_axis(ax)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)
    figure.suptitle(
        f"Protected-state envelopes — evaluation seed {episodes[0].eval_seed}\n"
        "green: CBF boundary; dotted red: simulator failure boundary",
        fontsize=13,
    )
    return _save_figure(
        figure,
        output_dir / "protected_state_envelope_comparison",
        formats,
        dpi,
    )


def plot_intervention_and_action_corrections(
    episodes: list[Episode],
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    figure, axes = plt.subplots(4, 1, figsize=(11.0, 10.5), sharex=True, constrained_layout=True)
    correction_labels = (
        r"$\Delta a_{x,B}$ [m/s²]",
        r"$\Delta a_{y,B}$ [m/s²]",
        r"$\Delta \alpha$ [deg/s²]",
    )
    for episode in episodes:
        time = _time(episode)
        nominal = _require(episode, "action_nominal").astype(float)
        filtered = _require(episode, "action_filtered").astype(float)
        correction = filtered - nominal
        correction[:, 2] = np.rad2deg(correction[:, 2])
        style = {
            "color": episode.color,
            "ls": episode.linestyle,
            "lw": 1.2,
            "label": episode.label,
        }
        axes[0].plot(time, _require(episode, "intervention_scaled_norm"), **style)
        for index in range(3):
            axes[index + 1].plot(time, correction[:, index], **style)

    axes[0].set_ylabel("Scaled action\ncorrection norm")
    for index, label in enumerate(correction_labels, start=1):
        axes[index].axhline(0.0, color="#111827", lw=0.7)
        axes[index].set_ylabel(label)
    axes[-1].set_xlabel("Episode time [s]")
    for ax in axes:
        _finish_axis(ax)
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle(
        f"Safety-filter intervention — evaluation seed {episodes[0].eval_seed}\n"
        r"$\Delta u = u_{filtered} - u_{SAC}$",
        fontsize=13,
    )
    return _save_figure(
        figure,
        output_dir / "safety_filter_intervention_and_action_correction",
        formats,
        dpi,
    )


def plot_qp_diagnostics(
    episodes: list[Episode],
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    figure, axes = plt.subplots(2, 1, figsize=(11.0, 7.2), sharex=True, constrained_layout=True)
    any_failure = False
    reason_lines = []

    for episode in episodes:
        time = _time(episode)
        qp_failed = _require(episode, "qp_failed").astype(bool)
        barrier = _minimum_cone_barrier(episode)
        axes[0].plot(
            time,
            barrier,
            color=episode.color,
            ls=episode.linestyle,
            lw=1.3,
            label=episode.label,
        )
        failure_indices = np.flatnonzero(qp_failed)
        if failure_indices.size:
            any_failure = True
            axes[0].scatter(
                time[failure_indices],
                barrier[failure_indices],
                marker="x",
                s=42,
                linewidths=1.5,
                color=episode.color,
                zorder=5,
            )
            margin = _require(episode, "qp_failure_hard_feasibility_margin").astype(float)
            axes[1].scatter(
                time[failure_indices],
                margin[failure_indices],
                marker="x",
                s=42,
                linewidths=1.5,
                color=episode.color,
                label=episode.label,
            )
            reasons = _require(episode, "qp_failure_reason").astype(str)[failure_indices]
            unique, counts = np.unique(reasons, return_counts=True)
            summary = "; ".join(f"{reason}: {count}" for reason, count in zip(unique, counts))
            reason_lines.append(f"{VARIANT_LABELS.get(episode.variant, episode.variant)} — {summary}")

    axes[0].axhline(0.0, color="#111827", lw=0.9)
    axes[0].set_ylabel("Minimum post-step\ncone barrier [m]")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].axhline(0.0, color="#111827", lw=0.9)
    axes[1].set_ylabel("Hard-feasibility\nmargin (diagnostic units)")
    axes[1].set_xlabel("Episode time [s]")
    if any_failure:
        axes[1].legend(loc="best", fontsize=8)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No QP failures in the selected trajectories",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=11,
        )
    for ax in axes:
        _finish_axis(ax)
    subtitle = "\n".join(reason_lines) if reason_lines else "All QPs solved successfully"
    figure.suptitle(
        f"QP diagnostics — evaluation seed {episodes[0].eval_seed}\n{subtitle}",
        fontsize=12,
    )
    return _save_figure(
        figure,
        output_dir / "hard_qp_feasibility_diagnostics",
        formats,
        dpi,
    )


def plot_numerical_boundary_zoom(
    episodes: list[Episode],
    tolerance: float,
    zoom_steps: int,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    figure, axes = plt.subplots(
        len(episodes),
        1,
        figsize=(11.0, max(3.2, 2.8 * len(episodes))),
        squeeze=False,
        constrained_layout=True,
    )
    half_window = max(2, zoom_steps // 2)
    for ax, episode in zip(axes[:, 0], episodes):
        barrier = _minimum_cone_barrier(episode)
        minimum_index = int(np.nanargmin(barrier))
        start = max(0, minimum_index - half_window)
        stop = min(barrier.size, minimum_index + half_window + 1)
        indices = np.arange(start, stop)
        relative_step = indices - minimum_index
        values = barrier[indices]
        ax.plot(relative_step, values, color=episode.color, lw=1.5)
        ax.scatter([0], [barrier[minimum_index]], color=episode.color, marker="X", s=55, zorder=4)
        ax.axhline(0.0, color="#111827", lw=1.0, label="Strict boundary: h = 0")
        ax.axhline(
            -tolerance,
            color="#7c3aed",
            ls="--",
            lw=1.0,
            label=fr"Reporting tolerance: h = -{tolerance:.0e}",
        )
        strict = values < 0.0
        beyond = values < -tolerance
        if np.any(strict):
            ax.scatter(relative_step[strict], values[strict], color="#f59e0b", s=24, label="Strict h < 0")
        if np.any(beyond):
            ax.scatter(
                relative_step[beyond],
                values[beyond],
                facecolors="none",
                edgecolors="#dc2626",
                linewidths=1.4,
                s=54,
                label="h below numerical tolerance",
            )
        # A symmetric-log scale keeps metre-scale approach values visible while
        # resolving the distinction between h = 0 and h = -1e-6 m.
        ax.set_yscale("symlog", linthresh=tolerance, linscale=1.0, base=10)
        ax.set_ylabel("Minimum cone\nbarrier [m]")
        ax.set_title(
            f"{episode.label}: min h = {barrier[minimum_index]:.4e} m "
            f"at step {int(_require(episode, 'step')[minimum_index])}",
            fontsize=10,
        )
        _finish_axis(ax)
        ax.legend(loc="best", fontsize=7)
    axes[-1, 0].set_xlabel("Step relative to minimum cone barrier")
    figure.suptitle(
        f"Sampled cone-boundary zoom — evaluation seed {episodes[0].eval_seed}",
        fontsize=13,
    )
    return _save_figure(
        figure,
        output_dir / "sampled_cone_boundary_tolerance_zoom",
        formats,
        dpi,
    )


def plot_evaluation_seed(
    result_dir: Path,
    rows: list[dict[str, str]],
    config: dict,
    variants: list[str],
    eval_seed: int,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    zoom_steps: int,
) -> list[Path]:
    episodes = [_load_episode(result_dir, rows, variant, eval_seed) for variant in variants]
    seed_output_dir = output_dir / f"eval_seed{eval_seed:04d}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)
    tolerance = float(config.get("protected_barrier_tolerance", 1.0e-6))

    created = []
    created.extend(plot_spatial_trajectory(episodes, config, seed_output_dir, formats, dpi))
    created.extend(plot_safety_envelopes(episodes, config, seed_output_dir, formats, dpi))
    created.extend(
        plot_intervention_and_action_corrections(episodes, seed_output_dir, formats, dpi)
    )
    created.extend(plot_qp_diagnostics(episodes, seed_output_dir, formats, dpi))
    created.extend(
        plot_numerical_boundary_zoom(
            episodes,
            tolerance,
            zoom_steps,
            seed_output_dir,
            formats,
            dpi,
        )
    )
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create thesis-ready plots from saved final-approach CBF trajectories."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Evaluation folder containing run_configuration.json and step NPZ files.",
    )
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        required=True,
        help="One or more evaluation seeds already present in the result folder.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        choices=sorted(VARIANT_LABELS),
        help="Saved filter variants to overlay (default: baseline both all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output root (default: a central plots/RESULT_FOLDER directory "
            "beside the evaluation result folders)."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=["png", "pdf"],
        help="Figure formats (default: PNG and PDF).",
    )
    parser.add_argument(
        "--zoom-steps",
        type=int,
        default=80,
        help="Number of steps shown around each minimum cone barrier (default: 80).",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Raster DPI (default: 220).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    if not result_dir.is_dir():
        raise NotADirectoryError(f"Result directory does not exist: {result_dir}")
    if args.zoom_steps < 4:
        raise ValueError("--zoom-steps must be at least 4.")
    final_approach_results_dir = (
        result_dir.parent.parent
        if result_dir.parent.name == "evaluations"
        else result_dir.parent
    )
    default_output_dir = final_approach_results_dir / "plots" / result_dir.name
    output_dir = (args.output_dir or default_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_configuration(result_dir)
    rows = _load_metric_rows(result_dir)

    created = []
    for eval_seed in args.eval_seeds:
        created.extend(
            plot_evaluation_seed(
                result_dir=result_dir,
                rows=rows,
                config=config,
                variants=args.variants,
                eval_seed=eval_seed,
                output_dir=output_dir,
                formats=args.formats,
                dpi=args.dpi,
                zoom_steps=args.zoom_steps,
            )
        )

    print(f"Created {len(created)} figure files in: {output_dir}")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
