import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_residual_lqr_dt01_baseline import (
    DEFAULT_OUTPUT_DIR,
    pure_lqr_rollout_equivalent,
    validate_saved_baseline_manifest,
)


def test_successful_zero_residual_wrapper_must_match_lqr_success_checkpoint():
    assert pure_lqr_rollout_equivalent(
        direct_status="success",
        pure_status="success",
        direct_steps=112_993,
        pure_steps=112_993,
        pure_metric_difference=1e-12,
        metric_atol=1e-5,
    )
    assert not pure_lqr_rollout_equivalent(
        direct_status="success",
        pure_status="success",
        direct_steps=120_000,
        pure_steps=112_993,
        pure_metric_difference=10.3,
        metric_atol=1e-5,
    )


def test_timeout_requires_close_terminal_metrics():
    assert pure_lqr_rollout_equivalent(
        direct_status="timeout",
        pure_status="timeout",
        direct_steps=120_000,
        pure_steps=120_000,
        pure_metric_difference=1e-12,
        metric_atol=1e-5,
    )
    assert not pure_lqr_rollout_equivalent(
        direct_status="timeout",
        pure_status="timeout",
        direct_steps=120_000,
        pure_steps=120_000,
        pure_metric_difference=1e-2,
        metric_atol=1e-5,
    )


def test_saved_canonical_manifest_matches_current_controller():
    manifest = validate_saved_baseline_manifest(
        DEFAULT_OUTPUT_DIR / "baseline_manifest.json",
        expected_seeds=range(100, 110),
    )
    assert manifest["summary"]["baseline_gate_passed"] is True
    assert manifest["summary"]["direct_successes"] == 4
