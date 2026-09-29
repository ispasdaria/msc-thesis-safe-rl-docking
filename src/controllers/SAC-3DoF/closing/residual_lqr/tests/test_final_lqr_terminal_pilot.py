from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


SRC_PATH = Path(__file__).resolve().parents[5]
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
CLOSING_PATH = Path(__file__).resolve().parents[1]
if str(CLOSING_PATH) not in sys.path:
    sys.path.insert(0, str(CLOSING_PATH))

from build_final_lqr_terminal_bank_dt01 import (
    DEFAULT_BASELINE_MANIFEST,
    build_parser as build_bank_parser,
)
from lqr_reference_residual_env import (
    ReferenceResidualLQRGoalConditionedClosingEnv,
)
from terminal_reference_residual_env import (
    EmpiricalTerminalReferenceResidualWrapper,
    load_terminal_reset_dataset,
    sha256_file,
    validate_terminal_reset_provenance,
)
from train_sac_closing_reference_residual_final_lqr_dt01 import (
    PILOT_J_DEFAULTS,
    main as pilot_j_main,
)
from train_sac_closing_reference_residual_lqr_terminal import (
    build_parser as build_training_parser,
    trained_checkpoint_passes_acceptance,
    validate_args as validate_training_args,
)


def _write_manifest(path, fingerprint="final-dt01-controller"):
    manifest = {
        "configuration": {"dt_seconds": 0.1},
        "controller": {
            "numeric_fingerprint_sha256": fingerprint,
            "integrator": "rk4",
        },
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _write_bank(path, manifest_path, *, fingerprint="final-dt01-controller"):
    target = np.asarray([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    targets = np.repeat(target[None, :], 5, axis=0)
    states = targets.copy()
    states[:, 0] += np.asarray([25.0, 30.0, 35.0, 40.0, 45.0])
    metadata = {
        "schema_version": 2,
        "dt_seconds": 0.1,
        "max_time_seconds": 12_000.0,
        "max_steps": 120_000,
        "integrator": "rk4",
        "controller_numeric_fingerprint_sha256": fingerprint,
        "baseline_manifest_sha256": sha256_file(manifest_path),
    }
    np.savez_compressed(
        path,
        states=states,
        p3_targets=targets,
        max_target_ratios=np.asarray([1.25, 1.5, 1.75, 1.8, 1.9]),
        source_seeds=np.asarray([20, 21, 22, 23, 30]),
        source_steps=np.asarray([100, 200, 300, 400, 500]),
        source_outcomes=np.asarray(
            ["success", "success", "success", "success", "time_limit"]
        ),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_bank_defaults_use_full_12000_second_ten_hz_horizon():
    args = build_bank_parser().parse_args([])
    assert args.max_time == 12_000.0
    assert args.snapshot_stride_steps == 25
    assert args.target_success_episodes > 0
    assert args.target_timeout_episodes > 0


def test_provenance_gate_accepts_matching_bank_and_requires_both_outcomes(tmp_path):
    manifest_path = tmp_path / "baseline.json"
    bank_path = tmp_path / "bank.npz"
    manifest = _write_manifest(manifest_path)
    _write_bank(bank_path, manifest_path)
    dataset = load_terminal_reset_dataset(bank_path)

    audit = validate_terminal_reset_provenance(
        dataset,
        dataset_path=bank_path,
        baseline_manifest=manifest,
        baseline_manifest_path=manifest_path,
    )

    assert audit["dt_seconds"] == 0.1
    assert audit["source_outcome_episode_counts"] == {
        "success": 4,
        "time_limit": 1,
    }


def test_provenance_gate_rejects_a_bank_from_another_controller(tmp_path):
    manifest_path = tmp_path / "baseline.json"
    bank_path = tmp_path / "bank.npz"
    manifest = _write_manifest(manifest_path)
    _write_bank(bank_path, manifest_path, fingerprint="old-controller")
    dataset = load_terminal_reset_dataset(bank_path)

    with pytest.raises(ValueError, match="controller fingerprint"):
        validate_terminal_reset_provenance(
            dataset,
            dataset_path=bank_path,
            baseline_manifest=manifest,
            baseline_manifest_path=manifest_path,
        )


def test_provenance_gate_rejects_a_timestep_inconsistent_horizon(tmp_path):
    manifest_path = tmp_path / "baseline.json"
    bank_path = tmp_path / "bank.npz"
    manifest = _write_manifest(manifest_path)
    _write_bank(bank_path, manifest_path)
    dataset = load_terminal_reset_dataset(bank_path)
    metadata = json.loads(str(dataset["metadata_json"].item()))
    metadata["max_steps"] = 12_000
    dataset["metadata_json"] = np.asarray(json.dumps(metadata))

    with pytest.raises(ValueError, match="horizon is inconsistent"):
        validate_terminal_reset_provenance(
            dataset,
            dataset_path=bank_path,
            baseline_manifest=manifest,
            baseline_manifest_path=manifest_path,
        )


def test_provenance_gate_rejects_the_old_unprovenanced_schema(tmp_path):
    manifest_path = tmp_path / "baseline.json"
    bank_path = tmp_path / "bank.npz"
    manifest = _write_manifest(manifest_path)
    _write_bank(bank_path, manifest_path)
    dataset = load_terminal_reset_dataset(bank_path)
    metadata = json.loads(str(dataset["metadata_json"].item()))
    metadata["schema_version"] = 1
    dataset["metadata_json"] = np.asarray(json.dumps(metadata))

    with pytest.raises(ValueError, match="predates the provenance-locked"):
        validate_terminal_reset_provenance(
            dataset,
            dataset_path=bank_path,
            baseline_manifest=manifest,
            baseline_manifest_path=manifest_path,
        )


def test_balanced_reset_sampling_does_not_follow_snapshot_imbalance(tmp_path):
    manifest_path = tmp_path / "baseline.json"
    bank_path = tmp_path / "bank.npz"
    _write_manifest(manifest_path)
    _write_bank(bank_path, manifest_path)
    base = ReferenceResidualLQRGoalConditionedClosingEnv(
        mode="training", seed=5, dt=0.1, max_time=20.0
    )
    env = EmpiricalTerminalReferenceResidualWrapper(
        base,
        bank_path,
        include_source_outcomes=("success", "time_limit"),
        balance_source_outcomes=True,
    )
    rng = np.random.default_rng(123)
    selected = [env._choose_candidate_index(rng) for _ in range(1_000)]
    outcomes = env.dataset["source_outcomes"][selected].astype(str)
    success_fraction = float(np.mean(outcomes == "success"))
    assert 0.45 < success_fraction < 0.55
    env.close()


def test_pilot_j_defaults_are_valid_and_use_disjoint_acceptance_seeds():
    parser = build_training_parser()
    args = parser.parse_args(PILOT_J_DEFAULTS)
    validate_training_args(parser, args)
    assert args.pilot_label == "J"
    assert args.dt == 0.1
    assert args.terminal_prefill is True
    assert args.terminal_residual_fade_full_ratio == 1.2
    assert set(args.validation_seeds).isdisjoint(args.acceptance_seeds)


def test_pilot_j_dry_run_accepts_a_bank_bound_to_the_current_final_lqr(
    tmp_path, capsys
):
    manifest = json.loads(DEFAULT_BASELINE_MANIFEST.read_text(encoding="utf-8"))
    fingerprint = manifest["controller"]["numeric_fingerprint_sha256"]
    bank_path = tmp_path / "current_final_lqr_bank.npz"
    _write_bank(bank_path, DEFAULT_BASELINE_MANIFEST, fingerprint=fingerprint)

    pilot_j_main(["--terminal-dataset", str(bank_path), "--dry-run"])

    config = json.loads(capsys.readouterr().out)
    assert config["pilot"] == "J"
    assert config["artifact_family"] == "lqr_reference_residual_final_dt01"
    assert config["controller_provenance"]["numeric_fingerprint_sha256"] == (
        fingerprint
    )
    assert not config["controller_provenance"][
        "lqr_actions_or_transitions_from_dt1_used"
    ]


def test_acceptance_requires_strict_success_gain_without_more_failures():
    baseline = {"success_rate": 0.7, "failure_rate": 0.0}
    assert trained_checkpoint_passes_acceptance(
        candidate_checkpoint=250_000,
        baseline_summary=baseline,
        candidate_summary={"success_rate": 0.8, "failure_rate": 0.0},
    )
    assert not trained_checkpoint_passes_acceptance(
        candidate_checkpoint=250_000,
        baseline_summary=baseline,
        candidate_summary={"success_rate": 0.7, "failure_rate": 0.0},
    )
    assert not trained_checkpoint_passes_acceptance(
        candidate_checkpoint=250_000,
        baseline_summary=baseline,
        candidate_summary={"success_rate": 0.8, "failure_rate": 0.1},
    )
