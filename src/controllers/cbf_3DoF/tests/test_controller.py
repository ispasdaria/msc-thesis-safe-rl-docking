from dataclasses import replace

import numpy as np
import pytest

from controllers.cbf_3DoF import (
    CBFQPController,
    ClosingCBFConfig,
    FinalApproachCBFConfig,
    with_cbf_qp,
)
from envs.docking3dof.core import Docking3DoFEnv


class ConstantPhysicalController:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=float)
        self.calls = 0

    def compute_action(self, env, state):
        self.calls += 1
        return self.action.copy()


def scipy_config(config):
    return replace(config, solver_backend="scipy")


@pytest.mark.parametrize(
    ("phase", "config"),
    [
        ("closing", scipy_config(ClosingCBFConfig())),
        ("final", scipy_config(FinalApproachCBFConfig())),
    ],
)
def test_adapter_has_lqr_controller_contract_for_both_phases(phase, config):
    env = Docking3DoFEnv(phase=phase, seed=7)
    state = env.reset(mode="testing")
    nominal_controller = ConstantPhysicalController([0.0, 0.0, 0.0])
    controller = with_cbf_qp(nominal_controller, config)

    action = controller.compute_action(env, state)

    assert action.shape == (3,)
    assert nominal_controller.calls == 1
    assert controller.last_result is not None
    assert np.array_equal(action, controller.last_result.action_filtered)
    assert np.array_equal(
        controller.last_result.action_nominal, np.zeros(3, dtype=float)
    )


def test_adapter_clips_lqr_command_in_physical_units_before_filtering():
    env = Docking3DoFEnv(phase="final", seed=3)
    state = env.reset(mode="testing")
    raw = 2.0 * env.action_high
    controller = CBFQPController(
        ConstantPhysicalController(raw),
        scipy_config(FinalApproachCBFConfig()),
    )

    action = controller.compute_action(env, state)

    assert np.all(np.abs(action) <= env.action_high + 1.0e-12)
    assert np.array_equal(controller.last_result.action_nominal, env.action_high)


def test_adapter_rejects_config_for_the_other_phase():
    env = Docking3DoFEnv(phase="closing", seed=1)
    state = env.reset(mode="testing")
    controller = CBFQPController(
        ConstantPhysicalController(np.zeros(3)), FinalApproachCBFConfig()
    )

    with pytest.raises(ValueError, match="cannot be used in the closing phase"):
        controller.compute_action(env, state)
