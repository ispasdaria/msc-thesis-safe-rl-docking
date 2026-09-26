"""Create the article trajectory comparisons from the recorded SAC/CBF campaigns.

The script scans the per-seed campaign results, selects representative seeds,
replays only the six selected deterministic *nominal* SAC rollouts, and overlays
those trajectories on the already-recorded full-CBF trajectories.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from stable_baselines3 import SAC


REPO = Path(__file__).resolve().parents[3]
CBF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CBF_DIR))

import evaluate_closing_cbf as closing_eval  # noqa: E402
import evaluate_final_approach_cbf as final_eval  # noqa: E402
from controllers.cbf_3DoF.closing_barriers import ClosingCBFConfig  # noqa: E402
from controllers.cbf_3DoF.barriers import FinalApproachCBFConfig  # noqa: E402


OUT = CBF_DIR / "results" / "trajectory_figures"
NOMINAL = OUT / "selected_nominal_rollouts"

CLOSING_NOMINAL = {
    "phase-balanced": REPO / "src/controllers/SAC-3DoF/closing/results/timestep_transfer/phase_balanced_guarded100k_seeds0_49_horizon12000/native_1hz/rollout_metrics.csv",
    "vanilla": REPO / "src/controllers/SAC-3DoF/closing/results/timestep_transfer/article_trial16_400k_seeds0_49/native_1hz/rollout_metrics.csv",
}
CLOSING_FILTERED = {
    "phase-balanced": CBF_DIR / "results/closing/evaluations/campaign_phase_balanced_100k_seeds_0_49",
    "vanilla": CBF_DIR / "results/closing/evaluations/campaign_v15_optuna_selected_seeds_0_49",
}
FINAL_NOMINAL_SELECTED = REPO / "src/controllers/SAC-3DoF/final_approach/results/reconstructed_selected_policies_50rollouts/all_rollouts.csv"
FINAL_NOMINAL_SUPPLEMENTAL = REPO / "src/controllers/SAC-3DoF/final_approach/results/reconstructed_supplemental_policies_50rollouts/all_rollouts.csv"
FINAL_CASES = {
    "round1": (1, 12, CBF_DIR / "results/final_approach/evaluations/campaign_round1_seed12_final100k_seeds_0_49"),
    "round3": (3, 19, CBF_DIR / "results/final_approach/evaluations/campaign_round3_seed19_best160k_seeds_0_49"),
    "round5_seed10": (5, 10, CBF_DIR / "results/final_approach/evaluations/campaign_round5_seed10_best490k_seeds_0_49"),
    "round5_seed14": (5, 14, CBF_DIR / "results/final_approach/evaluations/campaign_round5_seed14_best_seeds_0_49"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_config(campaign: Path) -> dict:
    with (campaign / "run_configuration.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def dataclass_from_dict(cls, values: dict):
    allowed = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def filtered_row(campaign: Path, seed: int) -> dict[str, str]:
    return next(row for row in read_csv(campaign / "per_episode_metrics.csv") if int(row["eval_seed"]) == seed)


def filtered_path(campaign: Path, seed: int, closing: bool) -> Path:
    label = "cone_speed_and_alignment_cbfs" if closing else "all_cbf"
    return campaign / f"step_timeseries_{label}_eval_seed{seed:04d}.npz"


def select_closing() -> dict[str, int]:
    selected = {}
    for key, nominal_csv in CLOSING_NOMINAL.items():
        repaired = [
            row for row in read_csv(nominal_csv)
            if row["cone_exit_ever"] == "1"
            and filtered_row(CLOSING_FILTERED[key], int(row["seed"]))["docking_cone_violation"] == "False"
        ]
        if not repaired:
            raise RuntimeError(f"No repaired closing seed found for {key}.")
        # The vanilla historical CSV was produced with the unsuffixed final
        # checkpoint, whereas its CBF campaign used the selected ``_best``
        # checkpoint. Validate its flagged candidates by deterministic replay
        # so that the plotted pair always uses exactly the same controller.
        if key == "vanilla":
            selected[key] = -1
            for row in sorted(repaired, key=lambda item: int(item["seed"])):
                seed = int(row["seed"])
                with np.load(run_nominal(CLOSING_FILTERED[key], seed, True)) as data:
                    minimum_h = min(
                        float(np.min(data["post_h_cone_upper"])),
                        float(np.min(data["post_h_cone_lower"])),
                        float(np.min(data["post_h_cone_axial_upper"])),
                    )
                if minimum_h < 0.0:
                    selected[key] = seed
                    break
            if selected[key] < 0:
                raise RuntimeError("No paired cone-violating vanilla seed found among the historically flagged candidates.")
        else:
            # Largest recorded lateral excursion gives the clearest geometric contrast.
            selected[key] = int(max(repaired, key=lambda row: float(row["max_cone_lateral_excess_m"]))["seed"])
    return selected


def final_nominal_rows(round_number: int, training_seed: int) -> list[dict[str, str]]:
    paths = [FINAL_NOMINAL_SELECTED, FINAL_NOMINAL_SUPPLEMENTAL]
    matches = []
    for path in paths:
        matches.extend(
            row for row in read_csv(path)
            if int(row["round"]) == round_number and int(row["training_seed"]) == training_seed
        )
    if len(matches) != 50:
        raise RuntimeError(f"Expected 50 nominal records for round {round_number}, seed {training_seed}; found {len(matches)}.")
    return matches


def physical_cone_violation(row: dict[str, str]) -> bool:
    return row["failure"] == "True" and "geometry" in row["violated_constraints"].split(";")


def select_final() -> tuple[dict[str, int], dict[str, bool]]:
    selected, repaired_available = {}, {}
    for key, (round_number, training_seed, campaign) in FINAL_CASES.items():
        nominal = final_nominal_rows(round_number, training_seed)
        filtered = {int(row["eval_seed"]): row for row in read_csv(campaign / "per_episode_metrics.csv")}
        if key == "round1":
            # All nominal rollouts are safe. Choose the largest intervention count,
            # breaking ties by proximity to the median terminal docking distance.
            distances = np.array([float(row["final_port_distance"]) for row in filtered.values()])
            median = float(np.median(distances))
            selected[key] = max(
                filtered,
                key=lambda seed: (int(filtered[seed]["intervention_count"]), -abs(float(filtered[seed]["final_port_distance"]) - median)),
            )
            repaired_available[key] = False
            continue
        candidates = [
            int(row["eval_seed"]) for row in nominal
            if physical_cone_violation(row)
            and filtered[int(row["eval_seed"])]["docking_cone_violation"] == "False"
        ]
        repaired_available[key] = bool(candidates)
        if candidates:
            # Strong intervention activity generally gives the clearest visible repair.
            selected[key] = max(candidates, key=lambda seed: int(filtered[seed]["intervention_count"]))
        else:
            # Choose the filtered trajectory closest to the boundary (smallest cone h).
            def boundary_score(seed: int) -> float:
                with np.load(filtered_path(campaign, seed, False)) as data:
                    return float(min(np.min(data["post_h_cone_upper"]), np.min(data["post_h_cone_lower"])))
            selected[key] = min(filtered, key=boundary_score)
    return selected, repaired_available


def run_nominal(campaign: Path, seed: int, closing: bool) -> Path:
    config_json = read_config(campaign)
    destination = NOMINAL / campaign.name
    destination.mkdir(parents=True, exist_ok=True)
    label = "unfiltered"
    path = destination / f"step_timeseries_{label}_eval_seed{seed:04d}.npz"
    if path.is_file():
        return path
    model_path = Path(config_json["model"])
    vec_path = Path(config_json["vecnormalize"])
    evaluator = closing_eval if closing else final_eval
    config_cls = ClosingCBFConfig if closing else FinalApproachCBFConfig
    model = SAC.load(str(model_path))
    vecnormalize, dummy = evaluator.load_vecnormalize(vec_path)
    try:
        kwargs = dict(
            model=model,
            vecnormalize=vecnormalize,
            eval_seed=seed,
            mode=config_json["mode"],
            max_steps=config_json["max_steps"],
            variant="baseline",
            base_config=dataclass_from_dict(config_cls, config_json["cbf"]),
            output_dir=destination,
        )
        evaluator.run_rollout(**kwargs)
    finally:
        dummy.close()
    return path


def xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return np.asarray(data["s"], float), np.asarray(data["y_p"], float)


def geometry(campaign: Path, seed: int, closing: bool) -> dict[str, float | np.ndarray]:
    evaluator = closing_eval if closing else final_eval
    env = evaluator.make_env(seed, "testing")
    try:
        env.reset(seed=seed)
        core = env.core_env
        result = {
            "cone_slope": float(core.cone_slope),
            "corridor_radius": float(core.corridor_radius),
            "corridor_start": float(core.d_P3),
        }
        if closing:
            p3 = np.asarray(core.get_derived_quantities(core.p3_target)["r_p"], float)
            result.update(
                p3=p3,
                p3_tolerance=np.asarray(core.p3_target_tolerance[:2], float),
                corridor_start=float(p3[0]),
            )
        return result
    finally:
        env.close()


def draw_panel(
    ax,
    nominal_path: Path,
    filtered_file: Path,
    geom: dict,
    title: str,
    mark_p3: bool,
    *,
    stronger_boundaries: bool = False,
    external_zoom: bool = False,
):
    nx, ny = xy(nominal_path)
    fx, fy = xy(filtered_file)
    x_all = np.concatenate((nx, fx))
    xmin, xmax = min(0.0, float(np.min(x_all))), max(0.0, float(np.max(x_all)))
    pad = max(1.0, 0.035 * (xmax - xmin))
    xx = np.linspace(xmin - pad, xmax + pad, 500)
    valid = xx >= 0.0
    boundary = geom["cone_slope"] * xx[valid]
    faint = dict(
        color="0.48" if stronger_boundaries else "0.55",
        lw=1.15 if stronger_boundaries else 0.8,
        ls=":",
        alpha=0.72 if stronger_boundaries else 0.38,
        zorder=0,
    )
    ax.plot(xx[valid], boundary, label="Docking cone", **faint)
    ax.plot(xx[valid], -boundary, **faint)
    corridor_x = np.linspace(0.0, geom["corridor_start"], 160)
    ax.plot(
        corridor_x,
        np.full_like(corridor_x, geom["corridor_radius"]),
        label="Docking corridor",
        **faint,
    )
    ax.plot(
        corridor_x,
        np.full_like(corridor_x, -geom["corridor_radius"]),
        **faint,
    )
    ax.plot(nx, ny, color="#D55E00", lw=1.35, label="Nominal SAC", zorder=2)
    ax.plot(fx, fy, color="#0072B2", lw=1.55, label="SAC + CBF", zorder=3)
    ax.scatter(nx[0], ny[0], marker="D", s=34, facecolor="white", edgecolor="0.35", lw=1.1, label="Initial state", zorder=5)
    ax.scatter(nx[-1], ny[-1], marker="X", s=42, color="0.3", label="Terminal state", zorder=5)
    ax.scatter(fx[0], fy[0], marker="o", s=22, facecolor="#0072B2", edgecolor="white", lw=0.5, zorder=6)
    ax.scatter(fx[-1], fy[-1], marker="X", s=42, color="#0072B2", zorder=6)
    if mark_p3:
        p3 = np.asarray(geom["p3"])
        tol = np.asarray(geom["p3_tolerance"])
        ax.scatter(p3[0], p3[1], marker="*", s=75, color="#009E73", edgecolor="white", lw=0.5, label="$P_3$ handover point", zorder=7)
        ax.add_patch(Ellipse(p3, 2 * tol[0], 2 * tol[1], facecolor="#009E73", edgecolor="none", alpha=0.09, zorder=1))
    else:
        ax.scatter(0.0, 0.0, marker="*", s=75, color="#009E73", edgecolor="white", lw=0.5, label="Docking point", zorder=7)
        if external_zoom:
            zoom = ax.inset_axes([0.52, 0.66, 0.42, 0.25], facecolor="white")
            zoom_x = min(float(geom["corridor_start"]), max(0.15, 0.02 * xmax))
            mask_n, mask_f = nx <= zoom_x, fx <= zoom_x
            zoom.plot(nx[mask_n], ny[mask_n], color="#D55E00", lw=1.05)
            zoom.plot(fx[mask_f], fy[mask_f], color="#0072B2", lw=1.15)
            zx = np.linspace(0.0, zoom_x, 100)
            zoom.plot(zx, geom["cone_slope"] * zx, color="0.55", lw=0.8, ls=":", alpha=0.55)
            zoom.plot(zx, -geom["cone_slope"] * zx, color="0.55", lw=0.8, ls=":", alpha=0.55)
            zoom.scatter(0.0, 0.0, marker="*", s=30, color="#009E73", zorder=7)
            zoom.set_xlim(-0.02 * zoom_x, zoom_x)
            zoom.invert_xaxis()
            zoom.set_ylim(-1.15 * geom["cone_slope"] * zoom_x, 1.15 * geom["cone_slope"] * zoom_x)
            zoom.text(
                0.03,
                0.94,
                r"$P_3$ handover to docking",
                transform=zoom.transAxes,
                ha="left",
                va="top",
                fontsize=7.2,
            )
            zoom.tick_params(labelsize=6.5)
            zoom.grid(True, color="0.91", lw=0.45)
            for spine in zoom.spines.values():
                spine.set_visible(True)
                spine.set_color("0.2")
                spine.set_linewidth(0.7)
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.invert_xaxis()
    ax.grid(True, color="0.9", lw=0.55)


def add_shared_labels_and_legend(
    fig,
    axes,
    *,
    label_fontsize: float = 9.5,
    legend_fontsize: float = 7.2,
):
    handles, labels = axes.flat[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(unique),
        frameon=False,
        fontsize=legend_fontsize,
        handletextpad=0.45,
        columnspacing=1.1,
    )
    fig.supxlabel(r"Axial port position $r_{p,x}$ [m]", fontsize=label_fontsize)
    fig.supylabel(r"Lateral port position $r_{p,y}$ [m]", fontsize=label_fontsize)


def save_figure(fig, stem: str):
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    closing_seeds = select_closing()
    final_seeds, repaired = select_final()

    close_nominal = {
        key: run_nominal(CLOSING_FILTERED[key], seed, True)
        for key, seed in closing_seeds.items()
    }
    final_nominal = {
        key: run_nominal(FINAL_CASES[key][2], seed, False)
        for key, seed in final_seeds.items()
    }

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
    fig1, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    fig1.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.80, wspace=0.14)
    for ax, key, title in zip(axes, ("phase-balanced", "vanilla"), ("Closing phase-balanced policy", "Closing vanilla policy")):
        seed = closing_seeds[key]
        draw_panel(ax, close_nominal[key], filtered_path(CLOSING_FILTERED[key], seed, True), geometry(CLOSING_FILTERED[key], seed, True), f"{title} (evaluation seed {seed})", True, stronger_boundaries=True)
    add_shared_labels_and_legend(fig1, axes, label_fontsize=11.5)
    save_figure(fig1, "closing_sac_vs_sac_cbf_representative_trajectories")
    plt.close(fig1)

    titles = {
        "round1": "Round 1, seed 12, final 100k (eval. seed 0)",
        "round3": "Round 3, seed 18, best 160k (eval. seed 39)",
        "round5_seed10": "Round 5, seed 10, best 490k (eval. seed 7)",
        "round5_seed14": "Round 5, seed 14, best 270k (eval. seed 34)",
    }
    fig2, axes = plt.subplots(2, 2, figsize=(10.2, 7.5))
    fig2.subplots_adjust(left=0.075, right=0.985, bottom=0.09, top=0.86, wspace=0.12, hspace=0.25)
    for ax, key in zip(axes.flat, titles):
        seed = final_seeds[key]
        campaign = FINAL_CASES[key][2]
        draw_panel(
            ax,
            final_nominal[key],
            filtered_path(campaign, seed, False),
            geometry(campaign, seed, False),
            titles[key],
            False,
            stronger_boundaries=True,
            external_zoom=True,
        )
    add_shared_labels_and_legend(fig2, axes, label_fontsize=11.5)
    save_figure(fig2, "final_approach_sac_vs_sac_cbf_representative_trajectories")
    plt.close(fig2)

    panel_order = {
        "combined": ("phase-balanced", "round1", "round3", "round5_seed10"),
    }
    combined_titles = {
        "round1": "Round 1, seed 12, final 100k,\ninitial 100% success rate (eval. seed 0)",
        "round3": "Round 3, seed 18, best 160k,\ninitial 42% success rate (eval. seed 39)",
        "round5_seed10": "Round 5, seed 10, best 490k,\ninitial 66% success rate (eval. seed 7)",
    }
    fig3, axes = plt.subplots(2, 2, figsize=(10.2, 7.5))
    fig3.suptitle("Representative nominal vs. CBF-filtered trajectories", fontsize=13, fontweight="bold", x=0.075, y=0.935, ha="left")
    fig3.subplots_adjust(left=0.075, right=0.985, bottom=0.09, top=0.86, wspace=0.12, hspace=0.25)
    for ax, key in zip(axes.flat, panel_order["combined"]):
        if key == "phase-balanced":
            seed = closing_seeds[key]
            draw_panel(
                ax,
                close_nominal[key],
                filtered_path(CLOSING_FILTERED[key], seed, True),
                geometry(CLOSING_FILTERED[key], seed, True),
                f"Closing phase-balanced policy (evaluation seed {seed})",
                True,
                stronger_boundaries=True,
            )
            continue

        seed = final_seeds[key]
        campaign = FINAL_CASES[key][2]
        draw_panel(
            ax,
            final_nominal[key],
            filtered_path(campaign, seed, False),
            geometry(campaign, seed, False),
            combined_titles.get(key, titles[key]),
            False,
            stronger_boundaries=True,
            external_zoom=True,
        )
    add_shared_labels_and_legend(fig3, axes, label_fontsize=11.5, legend_fontsize=8.2)
    save_figure(fig3, "closing_and_final_approach_representative_trajectories_2x2")
    plt.close(fig3)

    report = {"closing": {}, "final_approach": {}}
    for key, seed in closing_seeds.items():
        report["closing"][key] = {"selected_seed": seed, "nominal_cone_violation": True, "filtered_repaired_cone_violation": True}
        print(f"Closing {key}: seed {seed}; nominal cone violation: yes; filtered repair: yes")
    for key, seed in final_seeds.items():
        nominal_violation = any(int(row["eval_seed"]) == seed and physical_cone_violation(row) for row in final_nominal_rows(*FINAL_CASES[key][:2]))
        report["final_approach"][key] = {"selected_seed": seed, "nominal_cone_violation": nominal_violation, "filtered_repaired_cone_violation": repaired[key]}
        note = " representative safe rollout with near-median terminal distance and maximal intervention count" if key == "round1" else ""
        print(f"Final {key}: seed {seed}; nominal cone violation: {'yes' if nominal_violation else 'no'}; filtered repair: {'yes' if repaired[key] else 'no'};{note}")
    if not repaired["round3"]:
        print("Round 3 note: no nominal cone-violation seed was repaired by the full CBF filter in seeds 0-49; selected the filtered case closest to the cone boundary.")
    with (OUT / "selected_seeds.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Figures written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
