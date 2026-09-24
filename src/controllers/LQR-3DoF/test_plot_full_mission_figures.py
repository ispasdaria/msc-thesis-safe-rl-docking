import numpy as np

from envs.docking3dof.core import Docking3DoFEnv
from plot_full_mission_figures import port_position, port_velocity


def test_vectorized_port_quantities_match_environment() -> None:
    states = np.array(
        [
            [100.0, 50.0, -0.2, 0.1, 0.0, 0.1],
            [300.0, -2.0, -0.04, 0.003, np.pi / 4.0, -0.02],
            [24.2, 0.0, -0.005, 0.0, -np.pi / 3.0, 0.04],
        ]
    )
    env = Docking3DoFEnv(max_time=5_000.0, phase="final")

    expected_position = np.array(
        [env.get_port_relative_position(state) for state in states]
    )
    expected_velocity = np.array(
        [env.get_port_relative_velocity(state) for state in states]
    )

    assert np.allclose(port_position(states), expected_position)
    assert np.allclose(port_velocity(states), expected_velocity)
