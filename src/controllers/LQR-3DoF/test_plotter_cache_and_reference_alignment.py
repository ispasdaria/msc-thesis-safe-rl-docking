from argparse import Namespace

import numpy as np

from envs.docking3dof.core import Docking3DoFEnv
from plot_final_approach_convergence import (
    CACHE_SCHEMA_VERSION,
    cache_matches,
)
from plot_lqr_closing_schedule_comparison import (
    run_rollout as run_closing_rollout,
    scheduled_axial_reference,
)
from plot_lqr_final_schedule_comparison import (
    run_rollout as run_final_rollout,
    scheduled_port_velocity_reference,
)


class ZeroActionController:
    v_fast = -0.08
    d_fast = 100.0
    d_slow = 10.0

    def compute_action(
        self, env: Docking3DoFEnv, state: np.ndarray
    ) -> np.ndarray:
        del env, state
        return np.zeros(3, dtype=float)


def compatible_cache() -> tuple[dict[str, np.ndarray], Namespace]:
    args = Namespace(
        rollouts=3,
        seed_start=7,
        max_time=5000.0,
        mode="testing",
        dock_tolerance=0.1,
    )
    histories = {
        "seeds": np.arange(7, 10, dtype=int),
        "times": np.zeros((3, 1)),
        "port_distances": np.zeros((3, 1)),
        "port_x_positions": np.zeros((3, 1)),
        "port_y_positions": np.zeros((3, 1)),
        "axial_port_velocities": np.zeros((3, 1)),
        "success": np.ones(3, dtype=bool),
        "termination_reasons": np.asarray(["success"] * 3),
        "dock_tolerance": np.asarray(0.1),
        "mode": np.asarray("testing"),
        "max_time": np.asarray(5000.0),
        "cache_schema_version": np.asarray(CACHE_SCHEMA_VERSION),
    }
    return histories, args


def test_final_convergence_cache_requires_matching_configuration() -> None:
    histories, args = compatible_cache()
    assert cache_matches(histories, args)

    for field, incompatible_value in (
        ("rollouts", 2),
        ("seed_start", 8),
        ("max_time", 4000.0),
        ("mode", "nominal"),
        ("dock_tolerance", 0.2),
    ):
        incompatible_args = Namespace(**vars(args))
        setattr(incompatible_args, field, incompatible_value)
        assert not cache_matches(histories, incompatible_args)

    legacy_cache = dict(histories)
    del legacy_cache["cache_schema_version"]
    assert not cache_matches(legacy_cache, args)


def test_final_schedule_reference_matches_every_logged_state() -> None:
    env = Docking3DoFEnv(phase="final", dt=1.0, max_time=3.0, seed=4)
    rollout = run_final_rollout(
        env, ZeroActionController(), mode="testing", reference_kind="scheduled"
    )
    expected = np.asarray(
        [
            scheduled_port_velocity_reference(state, env)
            for state in rollout["states"]
        ]
    )
    assert np.array_equal(rollout["vpx_reference"], expected)


def test_closing_schedule_reference_matches_every_logged_state() -> None:
    controller = ZeroActionController()
    env = Docking3DoFEnv(phase="closing", dt=0.1, max_time=0.3, seed=4)
    rollout = run_closing_rollout(
        env, controller, mode="testing", reference_kind="scheduled"
    )
    expected = np.asarray(
        [
            scheduled_axial_reference(state, rollout["p3_target"], controller)
            for state in rollout["states"]
        ]
    )
    assert np.array_equal(rollout["vx_reference"], expected)
