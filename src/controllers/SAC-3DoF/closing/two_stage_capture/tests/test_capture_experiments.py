from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TRIAL_DIR = Path(__file__).resolve().parents[1]
SRC_ROOT = TRIAL_DIR.parents[4] / "src"
for path in (TRIAL_DIR, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capture_reset_env import CaptureResetWrapper  # noqa: E402
from capture_safety_terminal_env import CaptureSafetyTerminalEnv  # noqa: E402
from common import (  # noqa: E402
    CaptureEnvelope,
    handover_metrics,
    is_capture_envelope_entry,
    load_handover_dataset,
    save_handover_dataset,
)
from envs.docking3dof.gym_env import Docking3DoFGymEnv  # noqa: E402


TOLERANCE = np.array(
    [20.0, 2.0, 0.01, 0.01, np.deg2rad(0.5), np.deg2rad(0.05)]
)


def make_record(state, target, radius, seed=123):
    metrics = handover_metrics(state, target, TOLERANCE)
    return {
        "state": np.asarray(state, dtype=np.float64),
        "p3_target": np.asarray(target, dtype=np.float64),
        "radius_m": float(radius),
        "seed": int(seed),
        "step": 50,
        "time_s": 50.0,
        **metrics,
    }


def test_capture_envelope_requires_component_bounds_and_approach():
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    approaching = target + np.array(
        [60.0, 10.0, -0.01, 0.0, np.deg2rad(2.0), np.deg2rad(0.05)]
    )
    envelope = CaptureEnvelope(radius_m=100.0)
    metrics = handover_metrics(approaching, target, TOLERANCE)
    assert metrics["radial_rate_mps"] < 0.0
    assert is_capture_envelope_entry(metrics, envelope, {"unsafe": False})

    receding = approaching.copy()
    receding[2] = target[2] + 0.01
    metrics = handover_metrics(receding, target, TOLERANCE)
    assert metrics["radial_rate_mps"] > 0.0
    assert not is_capture_envelope_entry(metrics, envelope, {"unsafe": False})

    excessive_lateral = approaching.copy()
    excessive_lateral[1] = 16.0
    metrics = handover_metrics(excessive_lateral, target, TOLERANCE)
    assert not is_capture_envelope_entry(metrics, envelope, {"unsafe": False})


def test_handover_dataset_radius_selection(tmp_path):
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    records = [
        make_record(target + np.array([45.0, 1.0, -0.01, 0, 0, 0]), target, 50),
        make_record(target + np.array([90.0, 1.0, -0.01, 0, 0, 0]), target, 100),
    ]
    path = tmp_path / "handover.npz"
    save_handover_dataset(path, records, {"num_rollouts": 1})

    complete = load_handover_dataset(path)
    assert complete["states"].shape == (2, 6)
    selected = load_handover_dataset(path, radius_m=100)
    assert selected["states"].shape == (1, 6)
    np.testing.assert_allclose(selected["states"][0], records[1]["state"])


def test_empirical_reset_preserves_state_and_p3_target(tmp_path):
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    state = target + np.array([60.0, 2.0, -0.01, 0.0, np.deg2rad(1.0), 0.0])
    record = make_record(state, target, 100)
    dataset_path = tmp_path / "handover.npz"
    save_handover_dataset(dataset_path, [record], {"num_rollouts": 1})

    base = Docking3DoFGymEnv(
        phase="closing",
        mode="training",
        max_time=2_000.0,
        closing_reward_version="v20",
    )
    env = CaptureResetWrapper(
        base,
        dataset_path=dataset_path,
        radius_m=100,
        empirical_fraction=1.0,
    )
    try:
        _, info = env.reset(seed=5)
        np.testing.assert_allclose(env.core_env.state, state)
        np.testing.assert_allclose(env.core_env.p3_target, target)
        assert info["capture_reset_source"] == env.RESET_EMPIRICAL
        assert not info["unsafe"]
    finally:
        env.close()


def test_synthetic_reset_obeys_component_envelope(tmp_path):
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    state = target + np.array([60.0, 2.0, -0.01, 0.0, 0.0, 0.0])
    dataset_path = tmp_path / "handover.npz"
    save_handover_dataset(
        dataset_path,
        [make_record(state, target, 100)],
        {"num_rollouts": 1},
    )

    base = Docking3DoFGymEnv(
        phase="closing",
        mode="training",
        max_time=2_000.0,
        closing_reward_version="v20",
    )
    env = CaptureResetWrapper(
        base,
        dataset_path=dataset_path,
        radius_m=100,
        empirical_fraction=0.0,
    )
    try:
        _, info = env.reset(seed=17)
        metrics = handover_metrics(
            env.core_env.state,
            env.core_env.p3_target,
            env.core_env.p3_target_tolerance,
        )
        assert info["capture_reset_source"] == env.RESET_SYNTHETIC
        assert metrics["max_target_ratio"] >= env.minimum_max_target_ratio
        assert is_capture_envelope_entry(
            metrics,
            env.envelope,
            env.core_env.monitor_safety(env.core_env.state),
        )
        assert not info["unsafe"]
    finally:
        env.close()


def test_capture_reward_rewards_six_component_progress():
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    state = target + np.array([60.0, 2.0, 0.0, 0.0, np.deg2rad(1.0), 0.0])
    env = CaptureSafetyTerminalEnv(mode="training", max_time=2_000.0)
    try:
        env.reset(
            options={
                "initial_state": state,
                "p3_target": target,
                "mode": "training",
            }
        )
        closer_state = target + 0.9 * (state - target)
        monitor = env.core_env.monitor_safety(closer_state)
        info = {
            **monitor,
            "success": False,
            "failure": False,
        }
        reward, components = env._compute_sac_reward(
            state=closer_state,
            normalized_action=np.zeros(3),
            physical_action=np.zeros(3),
            info=info,
            terminated=False,
            truncated=False,
        )
        assert reward > 0.0
        assert components["reward_capture_progress"] > 0.0
        assert "reward_tolerance_state" not in components
    finally:
        env.close()


def test_capture_reward_keeps_hard_safety_termination():
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    state = np.array([400.0, 0.0, 0.349, 0.0, 0.0, 0.0])
    env = CaptureSafetyTerminalEnv(mode="training", max_time=2_000.0)
    try:
        env.reset(
            options={
                "initial_state": state,
                "p3_target": target,
                "mode": "training",
            }
        )
        _, reward, terminated, truncated, info = env.step(
            np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        assert terminated
        assert not truncated
        assert info["failure"]
        assert info["termination_reason"] == "safety_violation"
        assert info["reward_components"]["reward_safety_failure"] == -1_500.0
        assert reward < -1_000.0
    finally:
        env.close()


def test_capture_time_limit_is_shorter_than_final_default():
    capture = CaptureSafetyTerminalEnv(mode="training", max_time=2_000.0)
    final = Docking3DoFGymEnv(phase="final", mode="testing")
    try:
        assert capture.max_time == 2_000.0
        assert final.max_time == 5_000.0
        assert capture.max_time < final.max_time
    finally:
        capture.close()
        final.close()


def test_capture_alignment_warning_is_reported_and_penalized():
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    state = target.copy()
    state[4] = np.deg2rad(3.5)
    env = CaptureSafetyTerminalEnv(mode="training", max_time=2_000.0)
    try:
        _, info = env.reset(
            options={
                "initial_state": state,
                "p3_target": target,
                "mode": "training",
            }
        )
        assert info["warning"]
        assert not info["unsafe"]
        assert info["alignment"]["warn"]
        assert info["alignment_warning_reasons"] == ["attitude"]

        _, _, terminated, truncated, step_info = env.step(np.zeros(3))
        assert not terminated
        assert not truncated
        assert step_info["reward_components"]["warning_depth"] > 0.0
        assert step_info["reward_components"]["reward_warning_margin"] < 0.0
    finally:
        env.close()


def test_capture_attitude_violation_terminates_as_alignment_failure():
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    state = target.copy()
    state[4] = np.deg2rad(4.1)
    env = CaptureSafetyTerminalEnv(mode="training", max_time=2_000.0)
    try:
        env.reset(
            options={
                "initial_state": state,
                "p3_target": target,
                "mode": "training",
            }
        )
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        assert terminated
        assert not truncated
        assert info["failure"]
        assert info["termination_reason"] == "safety_violation"
        assert "alignment" in info["violated_constraints"]
        assert "attitude" in info["violated_constraints"]
        assert info["alignment_unsafe_reasons"] == ["attitude"]
        assert info["reward_components"]["reward_safety_failure"] == -1_500.0
    finally:
        env.close()


def test_capture_angular_rate_violation_terminates_as_alignment_failure():
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    state = target.copy()
    state[5] = np.deg2rad(0.21)
    env = CaptureSafetyTerminalEnv(mode="training", max_time=2_000.0)
    try:
        env.reset(
            options={
                "initial_state": state,
                "p3_target": target,
                "mode": "training",
            }
        )
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        assert terminated
        assert not truncated
        assert info["alignment"]["unsafe"]
        assert "alignment" in info["violated_constraints"]
        assert "angular_rate" in info["violated_constraints"]
        assert info["alignment_unsafe_reasons"] == ["angular_rate"]
    finally:
        env.close()
