"""Validate zero-residual wrappers against the canonical dt=0.1 closing LQR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from her_closing_utils import make_her_env
from residual_lqr_config import (
    CLOSING_LQR_CONTROL_HZ,
    CLOSING_LQR_DT_SECONDS,
    steps_for_duration,
)


CLOSING_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE_CSV = (
    SRC_PATH
    / "controllers"
    / "LQR-3DoF"
    / "results"
    / "lqr_closing"
    / "residual_sac_lqr_reference_dt0p1"
    / "lqr_dt0p1_validation_seeds100_109.csv"
)
DEFAULT_OUTPUT_DIR = (
    CLOSING_DIR
    / "results"
    / "residual_lqr_baseline"
    / "dt0p1_validation_seeds100_109"
)
LQR_SOURCE = (
    SRC_PATH / "controllers" / "LQR-3DoF" / "lqr_3dof_closing.py"
)
WRAPPER_SOURCES = (
    CLOSING_DIR / "lqr_residual_env.py",
    CLOSING_DIR / "lqr_reference_residual_env.py",
    CLOSING_DIR / "residual_lqr_config.py",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run paired zero-action direct/reference residual wrappers and "
            "compare them with a pure dt=0.1 LQR rollout CSV."
        )
    )
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--max-time", type=float, default=12_000.0)
    parser.add_argument("--mode", choices=("nominal", "testing", "training"), default="testing")
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state-atol", type=float, default=1e-10)
    parser.add_argument("--reference-metric-atol", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_fingerprint(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_reference_rows(path, expected_seeds):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing pure-LQR reference CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {int(row["seed"]): row for row in rows}
    missing = [seed for seed in expected_seeds if seed not in indexed]
    if missing:
        raise ValueError(f"Reference CSV is missing seeds: {missing}")
    return indexed


def _status(info, terminated, truncated):
    if bool(info.get("success", False) or info.get("is_success", False)):
        return "success"
    if bool(info.get("failure", False)):
        return "failure"
    if truncated or info.get("termination_reason") == "timeout":
        return "timeout"
    if terminated:
        return str(info.get("termination_reason", "terminated"))
    return "step_limit"


def _physical_errors(env):
    state = np.asarray(env.core_env.state, dtype=np.float64)
    target = np.asarray(env.core_env.p3_target, dtype=np.float64)
    error = state - target
    error[4] = env.core_env.wrap_to_pi(error[4])
    return error


def _controller_payload(env):
    from lqr_3dof_closing import (
        build_Q_closing,
        build_R_closing,
    )

    controller = env.lqr_controller
    core = env.core_env
    payload = {
        "controller": "LQRClosingController",
        "dt_seconds": float(core.dt),
        "control_frequency_hz": CLOSING_LQR_CONTROL_HZ,
        "integrator": str(core.integrator),
        "Q": build_Q_closing(dt=core.dt).tolist(),
        "R": build_R_closing(core).tolist(),
        "K": np.asarray(controller.K, dtype=np.float64).tolist(),
        "schedule": {
            "v_fast_mps": float(controller.v_fast),
            "d_fast_m": float(controller.d_fast),
            "d_slow_m": float(controller.d_slow),
        },
        "physical_action_limits": np.asarray(
            env.env.get_physical_action_limits(), dtype=np.float64
        ).tolist(),
        "normalize_action": True,
    }
    return payload


def pure_lqr_rollout_equivalent(
    *,
    direct_status,
    pure_status,
    direct_steps,
    pure_steps,
    pure_metric_difference,
    metric_atol,
):
    """Require the wrapped zero residual to match the pure-LQR terminal event."""
    return bool(
        direct_status == pure_status
        and direct_steps == pure_steps
        and np.isfinite(pure_metric_difference)
        and pure_metric_difference <= metric_atol
    )


def run_paired_rollouts(*, seeds, max_time, mode, reference_rows, state_atol, metric_atol):
    maximum_steps = steps_for_duration(max_time)
    zero_action = np.zeros(3, dtype=np.float32)
    output_rows = []
    global_max_state_difference = 0.0
    global_max_action_difference = 0.0
    controller_payload = None

    for seed in seeds:
        direct = make_her_env(
            seed=seed,
            mode=mode,
            max_time=max_time,
            reward_version="lqr_residual",
            terminate_on_success=True,
        )
        reference = make_her_env(
            seed=seed,
            mode=mode,
            max_time=max_time,
            reward_version="lqr_reference_residual",
            terminate_on_success=True,
        )
        try:
            # The gate needs only running comparisons, not 120k-step histories.
            direct.core_env.log_step = lambda **_: None
            reference.core_env.log_step = lambda **_: None
            direct.reset(seed=seed)
            reference.reset(seed=seed)
            if controller_payload is None:
                controller_payload = _controller_payload(direct)

            initial_state_difference = float(
                np.max(np.abs(direct.core_env.state - reference.core_env.state))
            )
            target_difference = float(
                np.max(
                    np.abs(direct.core_env.p3_target - reference.core_env.p3_target)
                )
            )
            max_state_difference = initial_state_difference
            max_action_difference = 0.0
            direct_info = {}
            reference_info = {}
            direct_terminated = direct_truncated = False
            reference_terminated = reference_truncated = False
            steps = 0

            for steps in range(1, maximum_steps + 1):
                _, _, direct_terminated, direct_truncated, direct_info = direct.step(
                    zero_action
                )
                (
                    _,
                    _,
                    reference_terminated,
                    reference_truncated,
                    reference_info,
                ) = reference.step(zero_action)
                state_difference = float(
                    np.max(
                        np.abs(direct.core_env.state - reference.core_env.state)
                    )
                )
                action_difference = float(
                    np.max(
                        np.abs(
                            np.asarray(
                                direct_info["executed_normalized_action"],
                                dtype=np.float64,
                            )
                            - np.asarray(
                                reference_info["executed_normalized_action"],
                                dtype=np.float64,
                            )
                        )
                    )
                )
                max_state_difference = max(max_state_difference, state_difference)
                max_action_difference = max(max_action_difference, action_difference)
                direct_done = bool(direct_terminated or direct_truncated)
                reference_done = bool(reference_terminated or reference_truncated)
                if direct_done != reference_done:
                    break
                if direct_done:
                    break

            direct_status = _status(
                direct_info, direct_terminated, direct_truncated
            )
            reference_status = _status(
                reference_info, reference_terminated, reference_truncated
            )
            direct_error = _physical_errors(direct)
            reference_error = _physical_errors(reference)
            pure = reference_rows[seed]
            pure_status = "success" if pure["success"].lower() == "true" else pure["status"]
            pure_steps = int(pure["steps"])
            pure_abs_errors = np.asarray(
                [
                    float(pure["abs_final_ex"]),
                    float(pure["abs_final_ey"]),
                    float(pure["abs_final_evx"]),
                    float(pure["abs_final_evy"]),
                    float(pure["abs_final_epsi"]),
                    float(pure["abs_final_eomega"]),
                ],
                dtype=np.float64,
            )
            pure_metric_difference = float(
                np.max(np.abs(np.abs(direct_error) - pure_abs_errors))
            )
            wrapper_equivalent = bool(
                max_state_difference <= state_atol
                and max_action_difference <= state_atol
                and direct_status == reference_status
            )
            pure_equivalent = pure_lqr_rollout_equivalent(
                direct_status=direct_status,
                pure_status=pure_status,
                direct_steps=steps,
                pure_steps=pure_steps,
                pure_metric_difference=pure_metric_difference,
                metric_atol=metric_atol,
            )
            output_rows.append(
                {
                    "seed": seed,
                    "steps": steps,
                    "direct_status": direct_status,
                    "reference_status": reference_status,
                    "pure_lqr_status": pure_status,
                    "pure_lqr_steps": pure_steps,
                    "initial_state_max_abs_difference": initial_state_difference,
                    "target_max_abs_difference": target_difference,
                    "wrapper_max_state_abs_difference": max_state_difference,
                    "wrapper_max_action_abs_difference": max_action_difference,
                    "pure_lqr_final_metric_max_abs_difference": pure_metric_difference,
                    "wrapper_equivalent": wrapper_equivalent,
                    "pure_lqr_equivalent": pure_equivalent,
                    "direct_abs_ex": abs(float(direct_error[0])),
                    "direct_abs_ey": abs(float(direct_error[1])),
                    "direct_abs_evx": abs(float(direct_error[2])),
                    "direct_abs_evy": abs(float(direct_error[3])),
                    "direct_abs_epsi": abs(float(direct_error[4])),
                    "direct_abs_eomega": abs(float(direct_error[5])),
                    "reference_final_error_max_abs_difference": float(
                        np.max(np.abs(direct_error - reference_error))
                    ),
                }
            )
            global_max_state_difference = max(
                global_max_state_difference, max_state_difference
            )
            global_max_action_difference = max(
                global_max_action_difference, max_action_difference
            )
            print(
                f"seed={seed} direct={direct_status} reference={reference_status} "
                f"pure={pure_status} steps={steps} wrapper_diff={max_state_difference:.3e} "
                f"pure_metric_diff={pure_metric_difference:.3e}",
                flush=True,
            )
        finally:
            direct.close()
            reference.close()

    return output_rows, controller_payload, {
        "maximum_wrapper_state_abs_difference": global_max_state_difference,
        "maximum_wrapper_action_abs_difference": global_max_action_difference,
    }


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_saved_baseline_manifest(path, *, expected_seeds=None):
    """Fail fast if a saved baseline no longer matches the current controller."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing residual-LQR baseline manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    summary = manifest.get("summary", {})
    if not bool(summary.get("baseline_gate_passed", False)):
        raise ValueError("The residual-LQR baseline manifest did not pass its gate.")
    configuration = manifest.get("configuration", {})
    if float(configuration.get("dt_seconds", np.nan)) != CLOSING_LQR_DT_SECONDS:
        raise ValueError("Baseline manifest does not describe the dt=0.1 controller.")
    if expected_seeds is not None:
        expected = list(map(int, expected_seeds))
        recorded = list(
            range(
                int(configuration["seed_start"]),
                int(configuration["seed_end"]) + 1,
            )
        )
        if expected != recorded:
            raise ValueError(
                f"Validation seeds {expected} do not match baseline seeds {recorded}."
            )

    env = make_her_env(
        seed=int(configuration.get("seed_start", 0)),
        mode=str(configuration.get("mode", "testing")),
        max_time=CLOSING_LQR_DT_SECONDS,
        reward_version="lqr_residual",
        terminate_on_success=True,
    )
    try:
        current_payload = _controller_payload(env)
    finally:
        env.close()
    expected_fingerprint = manifest["controller"]["numeric_fingerprint_sha256"]
    current_fingerprint = _json_fingerprint(current_payload)
    if current_fingerprint != expected_fingerprint:
        raise ValueError(
            "Current closing-LQR numeric fingerprint does not match the saved "
            f"baseline ({current_fingerprint} != {expected_fingerprint})."
        )
    for relative, expected_hash in manifest["controller"].get(
        "source_sha256", {}
    ).items():
        source_path = REPO_ROOT / Path(relative)
        relocated_source = False
        if not source_path.exists():
            legacy_path = Path(relative)
            closing_prefix = Path("src/controllers/SAC-3DoF/closing")
            try:
                family_relative = legacy_path.relative_to(closing_prefix)
            except ValueError:
                pass
            else:
                source_path = closing_prefix / "residual_lqr" / family_relative
                source_path = REPO_ROOT / source_path
                relocated_source = source_path.exists()
        if not source_path.exists() or (
            not relocated_source and _sha256(source_path) != expected_hash
        ):
            raise ValueError(f"Baseline source fingerprint mismatch: {relative}")
    reference_path = Path(manifest["reference"]["csv"])
    if not reference_path.exists() or _sha256(reference_path) != manifest[
        "reference"
    ]["sha256"]:
        raise ValueError("Pure-LQR reference CSV fingerprint mismatch.")
    return manifest


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.rollouts <= 0 or args.max_time <= 0.0:
        parser.error("--rollouts and --max-time must be positive.")
    if args.state_atol < 0.0 or args.reference_metric_atol < 0.0:
        parser.error("equivalence tolerances must be non-negative.")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_start, args.seed_start + args.rollouts))
    reference_rows = _load_reference_rows(args.reference_csv, seeds)
    rows, controller, diagnostics = run_paired_rollouts(
        seeds=seeds,
        max_time=args.max_time,
        mode=args.mode,
        reference_rows=reference_rows,
        state_atol=args.state_atol,
        metric_atol=args.reference_metric_atol,
    )
    controller["numeric_fingerprint_sha256"] = _json_fingerprint(controller)
    controller["source_sha256"] = {
        str(path.relative_to(REPO_ROOT)): _sha256(path)
        for path in (LQR_SOURCE, *WRAPPER_SOURCES)
    }
    wrapper_pass = all(row["wrapper_equivalent"] for row in rows)
    pure_pass = all(row["pure_lqr_equivalent"] for row in rows)
    manifest = {
        "schema_version": 1,
        "purpose": "canonical dt=0.1 residual-SAC zero-residual baseline gate",
        "configuration": {
            "dt_seconds": CLOSING_LQR_DT_SECONDS,
            "control_frequency_hz": CLOSING_LQR_CONTROL_HZ,
            "mode": args.mode,
            "seed_start": args.seed_start,
            "seed_end": seeds[-1],
            "rollouts": args.rollouts,
            "max_time_seconds": args.max_time,
            "max_steps": steps_for_duration(args.max_time),
            "state_atol": args.state_atol,
            "reference_metric_atol": args.reference_metric_atol,
        },
        "reference": {
            "csv": str(args.reference_csv.resolve()),
            "sha256": _sha256(args.reference_csv),
        },
        "controller": controller,
        "summary": {
            "direct_successes": sum(row["direct_status"] == "success" for row in rows),
            "direct_timeouts": sum(row["direct_status"] == "timeout" for row in rows),
            "direct_failures": sum(row["direct_status"] == "failure" for row in rows),
            "wrapper_equivalence_passed": wrapper_pass,
            "pure_lqr_equivalence_passed": pure_pass,
            "baseline_gate_passed": bool(wrapper_pass and pure_pass),
            **diagnostics,
        },
    }
    _write_csv(output_dir / "paired_zero_residual_rollouts.csv", rows)
    with (output_dir / "baseline_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote baseline artifacts to: {output_dir}")
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    if not manifest["summary"]["baseline_gate_passed"]:
        raise RuntimeError("Residual-LQR baseline equivalence gate failed.")


if __name__ == "__main__":
    main()
