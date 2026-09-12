import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "analyze_final_approach_control_effort.py"


def load_module():
    spec = importlib.util.spec_from_file_location("control_effort", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rollout_metrics_use_euclidean_translation_and_separate_rotation():
    module = load_module()
    times = np.array([0.0, 2.0, 4.0])
    positions = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 8.0]])
    actions = np.array([[3.0, 4.0, -2.0], [0.0, 5.0, 1.0]])

    result = module.compute_rollout_metrics(
        times, positions, actions, np.array([3.0, 5.0, 2.0])
    )

    assert result["num_steps"] == 2
    assert result["duration_s"] == pytest.approx(4.0)
    assert result["delta_v_eq_mps"] == pytest.approx(20.0)
    assert result["J_alpha_radps"] == pytest.approx(6.0)
    assert result["path_length_m"] == pytest.approx(9.0)
    assert result["path_efficiency"] == pytest.approx(np.sqrt(73.0) / 9.0)
    assert result["J_delta_a_mps2"] == pytest.approx(10.0**0.5)
    assert result["J_delta_alpha_radps2"] == pytest.approx(3.0)
    assert result["saturation_fraction"] == pytest.approx(1.0)


def test_rollout_metrics_reject_misaligned_history():
    module = load_module()
    with pytest.raises(ValueError, match="one more sample"):
        module.compute_rollout_metrics(
            np.array([0.0, 1.0]),
            np.zeros((2, 2)),
            np.zeros((2, 3)),
            np.ones(3),
        )


def test_distribution_stats_use_sample_standard_deviation_and_linear_quartiles():
    module = load_module()
    result = module.distribution_stats([1.0, 2.0, 3.0, 4.0])
    assert result == pytest.approx(
        {
            "mean": 2.5,
            "std": np.sqrt(5.0 / 3.0),
            "median": 2.5,
            "q1": 1.75,
            "q3": 3.25,
            "iqr": 1.5,
            "min": 1.0,
            "max": 4.0,
        }
    )


def test_paired_stats_define_difference_as_method_minus_lqr():
    module = load_module()
    result = module.paired_stats(
        np.array([2.0, 4.0, 6.0]), np.array([1.0, 5.0, 6.0])
    )
    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["median_difference"] == pytest.approx(0.0)
    assert result["std_difference"] == pytest.approx(1.0)
    assert result["lower_count"] == 1
    assert result["higher_count"] == 1
    assert result["tie_count"] == 1
    assert result["change_vs_lqr_percent"] == pytest.approx(0.0)


def test_canonical_loader_rejects_wrong_seed(tmp_path):
    module = load_module()
    path = tmp_path / "seed_0000.npz"
    np.savez_compressed(
        path,
        evaluation_seed=1,
        time=np.array([0.0, 1.0]),
        port_position=np.zeros((2, 2)),
        applied_action_body=np.zeros((1, 3)),
        success=True,
        termination_reason="success_docking",
        dt_seconds=1.0,
        action_limits=np.array([0.01, 0.01, 0.003]),
        controller_key="lqr",
        source="fixture",
        provenance_id=module.LQR_PROVENANCE,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        module._load_canonical(path, expected_seed=0, expected_controller="lqr")


def test_canonical_loader_rejects_nonunit_timestep(tmp_path):
    module = load_module()
    path = tmp_path / "seed_0000.npz"
    np.savez_compressed(
        path,
        evaluation_seed=0,
        time=np.array([0.0, 0.5]),
        port_position=np.zeros((2, 2)),
        applied_action_body=np.zeros((1, 3)),
        success=True,
        termination_reason="success_docking",
        dt_seconds=1.0,
        action_limits=np.array([0.01, 0.01, 0.003]),
        controller_key="lqr",
        source="fixture",
        provenance_id=module.LQR_PROVENANCE,
    )
    with pytest.raises(ValueError, match="timestep mismatch"):
        module._load_canonical(path, expected_seed=0, expected_controller="lqr")


def test_json_safe_converts_numpy_boolean_and_numeric_scalars():
    module = load_module()
    converted = module.json_safe(
        {"ok": np.bool_(True), "count": np.int64(3), "value": np.float64(1.25)}
    )
    assert json.dumps(converted) == '{"ok": true, "count": 3, "value": 1.25}'


def test_paired_plot_uses_noninteractive_backend(tmp_path):
    module = load_module()
    rows = []
    for index, controller in enumerate(module.CONTROLLERS):
        rows.append({
            "controller": controller,
            "evaluation_seed": 0,
            "delta_v_eq_mps": float(index + 1),
        })
    module._plot_paired(
        rows, {0}, "delta_v_eq_mps", "Delta-v", "fixture", tmp_path
    )
    assert (tmp_path / "fixture.png").is_file()
    assert (tmp_path / "fixture.pdf").is_file()
