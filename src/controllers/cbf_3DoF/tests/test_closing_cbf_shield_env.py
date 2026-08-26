import numpy as np
import pytest

from envs.docking3dof.closing_cbf_shield_env import (
    ClosingCBFShieldedGymEnv,
    build_closing_cbf_config,
)


INITIAL_STATE = np.array([1024.1, 99.0, -0.03, 0.04, 0.0, 0.0])
P3_TARGET = np.array([500.0, 0.0, -0.08, 0.0, 0.0, 0.0])
NOMINAL_PHYSICAL_ACTION = np.array([0.0, -0.01, 0.0])


def make_env(penalty_weight):
    return ClosingCBFShieldedGymEnv(
        phase="closing",
        mode="testing",
        dt=1.0,
        integrator="rk4",
        normalize_action=True,
        seed=0,
        cbf_variant="all",
        cbf_config=build_closing_cbf_config(
            "all",
            solver_backend="scipy",
        ),
        intervention_penalty_weight=penalty_weight,
    )


def reset_scenario(env):
    return env.reset(
        seed=0,
        options={
            "initial_state": INITIAL_STATE.copy(),
            "p3_target": P3_TARGET.copy(),
        },
    )


def test_training_variants_are_hard_and_select_alignment_explicitly():
    translational = build_closing_cbf_config("all", solver_backend="scipy")
    full = build_closing_cbf_config("full", solver_backend="scipy")

    assert not translational.allow_slack
    assert translational.enable_cone
    assert translational.enable_port_speed
    assert translational.enable_axial_port_speed
    assert not translational.enable_alignment
    assert full.enable_alignment


def test_shield_executes_filtered_action_and_reports_both_actions():
    env = make_env(penalty_weight=0.0)
    try:
        _, reset_info = reset_scenario(env)
        nominal = env.physical_action_to_normalized(NOMINAL_PHYSICAL_ACTION)

        _, _, terminated, truncated, info = env.step(nominal)

        assert reset_info["cbf_filter_enabled"]
        assert not terminated
        assert not truncated
        assert info["cbf_intervened"]
        assert not info["cbf_qp_failed"]
        assert info["cbf_intervention_count"] == 1
        assert np.allclose(
            info["nominal_physical_action"],
            NOMINAL_PHYSICAL_ACTION,
        )
        assert not np.allclose(
            info["filtered_physical_action"],
            info["nominal_physical_action"],
        )
        # The inherited action diagnostics describe the command actually sent
        # to the simulator, while the nominal command remains separately logged.
        assert np.allclose(
            info["physical_action"],
            info["filtered_physical_action"],
        )
        assert info["cbf_residuals"]["cone_upper"] >= -1.0e-6
    finally:
        env.close()


def test_intervention_penalty_is_exact_reward_difference():
    unpenalized = make_env(penalty_weight=0.0)
    penalized = make_env(penalty_weight=1.0)
    try:
        reset_scenario(unpenalized)
        reset_scenario(penalized)
        nominal = unpenalized.physical_action_to_normalized(
            NOMINAL_PHYSICAL_ACTION
        )

        _, reward_without_penalty, _, _, info_without = unpenalized.step(
            nominal
        )
        _, reward_with_penalty, _, _, info_with = penalized.step(nominal)

        expected_penalty = info_with["cbf_intervention_scaled_norm"] ** 2
        assert info_without["cbf_intervention_penalty"] == 0.0
        assert info_with["cbf_intervention_penalty"] == pytest.approx(
            expected_penalty
        )
        assert reward_without_penalty - reward_with_penalty == pytest.approx(
            expected_penalty
        )
        assert info_with["reward_components"][
            "reward_cbf_intervention"
        ] == pytest.approx(-expected_penalty)
        assert info_with["reward_components"]["reward_total"] == pytest.approx(
            reward_with_penalty
        )
    finally:
        unpenalized.close()
        penalized.close()


def test_negative_intervention_penalty_is_rejected():
    with pytest.raises(ValueError, match="finite and nonnegative"):
        make_env(penalty_weight=-0.1)
