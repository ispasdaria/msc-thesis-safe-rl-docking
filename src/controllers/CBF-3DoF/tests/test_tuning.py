from types import SimpleNamespace

import pytest

from cbf_3dof.barriers import FinalApproachCBFConfig
from tune_final_approach_cbf import (
    boundary_precheck,
    create_grid_search_space,
    validate_args,
)


def valid_args(**changes):
    values = {
        "num_eval_seeds": 1,
        "max_steps": 10,
        "dt": 1.0,
        "barrier_tolerance": 1.0e-6,
        "cone_k_values": [0.03, 0.1],
        "independent_cone_k2_values": None,
        "speed_alpha_values": [0.1, 0.5],
        "attitude_k_values": [0.3],
        "angular_rate_alpha_values": [0.5],
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_tied_gain_grid_has_expected_size():
    search = create_grid_search_space([0.03, 0.1], [0.1, 0.5])

    assert set(search) == {
        "cone_k",
        "speed_alpha",
        "attitude_k",
        "angular_rate_alpha",
    }
    assert len(search["cone_k"]) * len(search["speed_alpha"]) == 4


def test_independent_gain_grid_has_expected_size():
    search = create_grid_search_space([0.05, 0.1], [0.5], [0.1, 0.2, 0.3])

    assert set(search) == {
        "cone_k1",
        "cone_k2",
        "speed_alpha",
        "attitude_k",
        "angular_rate_alpha",
    }
    assert len(search["cone_k1"]) * len(search["cone_k2"]) == 6


def test_argument_validation_rejects_nonpositive_gains():
    with pytest.raises(ValueError):
        validate_args(valid_args(cone_k_values=[0.0]))


@pytest.mark.parametrize("cone_k", [0.03, 0.05, 0.1, 0.2, 0.3])
@pytest.mark.parametrize("speed_alpha", [0.1, 0.2, 0.5, 1.0])
def test_default_tuning_grid_boundary_precheck_passes(cone_k, speed_alpha):
    config = FinalApproachCBFConfig(
        cone_k1=cone_k,
        cone_k2=cone_k,
        speed_alpha=speed_alpha,
        allow_slack=False,
        solver_backend="osqp",
    )

    passed, reason, _, _ = boundary_precheck(config)

    assert passed, reason
