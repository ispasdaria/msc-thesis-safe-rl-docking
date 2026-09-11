"""Numerical audit of the retained mixed LQR/SAC full-mission campaigns.

This script creates no figures.  It combines the committed seed blocks,
recomputes Wilson intervals, audits closing safety failures against the current
environment definitions, and measures closing-SAC terminal states relative to
the final-approach testing box.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from envs.docking3dof.core import Docking3DoFEnv

RESULTS = ROOT / "src/controllers/SAC-3DoF/results/full_mission_mixed"
BLOCKS = ("seeds_300_349", "seeds_350_899")
LQR_CONFIGS = ("lqr_final_r1_s12_100k", "lqr_final_r5_s11_600k")
CLOSING_CONFIGS = ("balanced100k_final_r1_s12_100k", "vanilla400k_final_r1_s12_100k")

STATE_FIELDS = (
    "closing_terminal_rx_m", "closing_terminal_ry_m",
    "closing_terminal_vx_mps", "closing_terminal_vy_mps",
    "closing_terminal_psi_deg", "closing_terminal_omega_deg_s",
)
BOX_LOW = np.array([280.0, -3.0, -0.05, -0.01, -1.0, -0.05])
BOX_HIGH = np.array([320.0, 3.0, -0.01, 0.01, 1.0, 0.05])
BOX_HALF_WIDTH = (BOX_HIGH - BOX_LOW) / 2.0
P3_TOLERANCE = np.array([20.0, 2.0, 0.01, 0.01, 0.5, 0.05])


def rows_for(config: str) -> list[dict]:
    blocks = BLOCKS if config in LQR_CONFIGS else BLOCKS[:1]
    rows = []
    for block in blocks:
        with (RESULTS / block / f"{config}.csv").open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def numbers(rows: list[dict], field: str) -> np.ndarray:
    return np.array([float(row[field]) for row in rows if row[field] != ""], dtype=float)


def stats(values: np.ndarray) -> dict:
    return {
        "n": int(values.size), "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()), "max": float(values.max()),
    }


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return [center - radius, center + radius]


def closing_sac_audit(config: str) -> dict:
    rows = rows_for(config)
    states = np.array([[float(row[field]) for field in STATE_FIELDS] for row in rows])
    below = np.maximum(BOX_LOW - states, 0.0)
    above = np.maximum(states - BOX_HIGH, 0.0)
    excursions = below + above
    inside_components = (states >= BOX_LOW) & (states <= BOX_HIGH)
    inside_all = inside_components.all(axis=1)
    position_distance = np.linalg.norm(excursions[:, :2], axis=1)
    normalized_distance = np.linalg.norm(excursions / BOX_HALF_WIDTH, axis=1)
    error_fields = (
        "closing_axial_position_error_m", "closing_lateral_position_error_m",
        "closing_axial_velocity_error_mps", "closing_lateral_velocity_error_mps",
        "closing_attitude_error_deg", "closing_angular_rate_error_deg_s",
    )
    labels = ("rx", "ry", "vx", "vy", "psi", "omega")
    error_summary = {}
    for field in error_fields:
        values = numbers(rows, field)
        error_summary[field] = {"signed": stats(values), "absolute": stats(np.abs(values))}
    return {
        "n": len(rows),
        "termination_counts": dict(Counter(row["closing_termination_reason"] for row in rows)),
        "p3_target_error": error_summary,
        "position_error_norm_to_sampled_p3_m": stats(numbers(rows, "closing_position_error_norm_m")),
        "testing_box": {label: [float(lo), float(hi)] for label, lo, hi in zip(labels, BOX_LOW, BOX_HIGH)},
        "inside_all_components": int(inside_all.sum()),
        "inside_all_percent": float(100.0 * inside_all.mean()),
        "outside_all_components": int((~inside_all).sum()),
        "component_inside_counts": {label: int(inside_components[:, i].sum()) for i, label in enumerate(labels)},
        "component_outside_counts": {label: int((~inside_components[:, i]).sum()) for i, label in enumerate(labels)},
        "component_outside_percent": {label: float(100.0 * (~inside_components[:, i]).mean()) for i, label in enumerate(labels)},
        "component_terminal_values": {label: stats(states[:, i]) for i, label in enumerate(labels)},
        "component_absolute_excursion_beyond_box": {label: stats(excursions[:, i]) for i, label in enumerate(labels)},
        "position_distance_to_testing_rectangle_m": stats(position_distance),
        "normalized_6d_distance_to_testing_box": stats(normalized_distance),
    }


def safety_details(row: dict) -> dict:
    seed = int(row["seed"])
    state = np.array([float(row[field]) for field in STATE_FIELDS])
    state[4:] = np.deg2rad(state[4:])
    env = Docking3DoFEnv(phase="closing", dt=0.1, max_time=12000.0, seed=seed)
    env.reset(mode="testing")
    monitor = env.monitor_safety(state)
    q = monitor["derived"]
    return {
        "seed": seed, "violated_constraints": monitor["violated_constraints"],
        "in_E_AE": bool(monitor["geometry"]["in_E_AE"]),
        "in_S_KOS": bool(monitor["geometry"]["in_S_KOS"]),
        "in_C_dock": bool(monitor["geometry"]["in_C_dock"]),
        "port_speed_mps": float(q["port_speed"]), "abs_s_dot_mps": float(abs(q["s_dot"])),
        "com_rx_m": float(state[0]), "com_ry_m": float(state[1]),
        "s_m": float(q["s"]), "rho_m": float(q["rho"]),
        "ellipse_value": float((state[0] / env.ae_x) ** 2 + (state[1] / env.ae_y) ** 2),
    }


def lqr_campaign_audit(config: str) -> dict:
    rows = rows_for(config)
    successful = [row for row in rows if row["mission_success"] == "True"]
    handovers = [row for row in rows if row["closing_success"] == "True"]
    failed = [row for row in rows if row["mission_success"] != "True"]
    safety_rows = [row for row in failed if row["closing_termination_reason"] == "safety_violation"]
    timeout_rows = [row for row in failed if row["closing_termination_reason"] == "timeout"]
    safety = [safety_details(row) for row in safety_rows]
    result = {
        "n": len(rows), "successes": len(successful), "success_rate": len(successful) / len(rows),
        "wilson_95": wilson(len(successful), len(rows)), "unsuccessful": len(failed),
        "failed_phase_counts": dict(Counter(row["failed_phase"] or "none" for row in rows)),
        "closing_termination_counts": dict(Counter(row["closing_termination_reason"] for row in rows)),
        "final_termination_counts": dict(Counter(row["final_termination_reason"] for row in handovers)),
        "handover_count": len(handovers), "final_success_count": sum(row["final_success"] == "True" for row in handovers),
        "final_success_given_handover_percent": 100.0 * sum(row["final_success"] == "True" for row in handovers) / len(handovers),
        "post_handover_failures": sum(row["final_success"] != "True" for row in handovers),
        "duration_s": {
            "mission_all": stats(numbers(rows, "mission_elapsed_time_s")),
            "closing_all": stats(numbers(rows, "closing_elapsed_time_s")),
            "final_given_handover": stats(numbers(handovers, "final_elapsed_time_s")),
        },
        "final_terminal": {},
        "failed_terminal": {},
        "safety_failure_rows": safety,
    }
    error_fields = (
        "closing_axial_position_error_m", "closing_lateral_position_error_m",
        "closing_axial_velocity_error_mps", "closing_lateral_velocity_error_mps",
        "closing_attitude_error_deg", "closing_angular_rate_error_deg_s",
    )
    labels = ("rx", "ry", "vx", "vy", "psi", "omega")
    timeout_errors = np.array([[float(row[field]) for field in error_fields] for row in timeout_rows])
    result["timeout_handover_tolerance_failures"] = {
        label: {
            "outside_count": int((np.abs(timeout_errors[:, i]) > P3_TOLERANCE[i]).sum()),
            "outside_percent_of_timeouts": float(100.0 * (np.abs(timeout_errors[:, i]) > P3_TOLERANCE[i]).mean()),
            "tolerance": float(P3_TOLERANCE[i]),
        }
        for i, label in enumerate(labels)
    }
    result["handover_terminal_state"] = {
        label: stats(numbers(handovers, field)) for label, field in zip(labels, STATE_FIELDS)
    }
    for field in ("final_final_port_distance_m", "final_final_port_speed_mps", "final_axial_position_error_m",
                  "final_lateral_position_error_m", "final_attitude_error_deg", "final_angular_rate_error_deg_s",
                  "final_saturation_fraction"):
        result["final_terminal"][field] = stats(numbers(handovers, field))
    for reason, subset in (("timeout", timeout_rows), ("safety_violation", safety_rows)):
        result["failed_terminal"][reason] = {
            field: stats(numbers(subset, field)) for field in (
                "closing_elapsed_time_s", "closing_final_port_distance_m", "closing_final_port_speed_mps",
                "closing_axial_position_error_m", "closing_lateral_position_error_m",
                "closing_axial_velocity_error_mps", "closing_lateral_velocity_error_mps",
                "closing_attitude_error_deg", "closing_angular_rate_error_deg_s",
                *STATE_FIELDS,
            )
        }
    if safety:
        for field in ("port_speed_mps", "abs_s_dot_mps", "com_rx_m", "com_ry_m", "s_m", "rho_m", "ellipse_value"):
            result["failed_terminal"]["safety_violation"][f"recomputed_{field}"] = stats(
                np.array([item[field] for item in safety])
            )
    return result


def main() -> None:
    result = {
        "source_blocks": list(BLOCKS),
        "closing_sac": {config: closing_sac_audit(config) for config in CLOSING_CONFIGS},
        "lqr_plus_final_sac": {config: lqr_campaign_audit(config) for config in LQR_CONFIGS},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
