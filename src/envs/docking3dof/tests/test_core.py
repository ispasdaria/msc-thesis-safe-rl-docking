import os
import sys
import pytest 

import numpy as np

# Allow imports from the repo's src/ directory when running pytest from the root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.core import Docking3DoFEnv

"""
Unit tests for the Docking3DoFEnv environment. These tests cover initialization, state resetting, action clipping, action transformation, and step output format.
"""

### Checks that the simulator can be created without errors and that all important constants are stored correctly. 
def test_environment_initialization():
    env = Docking3DoFEnv()
    assert env.dt > 0
    assert env.max_time > 0
    assert env.ax_max > 0
    assert env.ay_max > 0
    assert env.alphaz_max > 0


### Checks that the reset method returns a valid state vector of the correct shape and that all values are finite and that the time is reset to zero.
### It also checks that the initial state values are within the expected bounds for different phases and modes. 
def test_reset_returns_valid_state():
    bounds_by_phase = {
        "closing": {
            "nominal": {
                "rx": (1700.0, 1700.0),
                "ry": (0.0, 0.0),
                "vx": (-0.10, -0.10),
                "vy": (0.0, 0.0),
                "psi": (np.deg2rad(0.0), np.deg2rad(0.0)),
                "omegaz": (np.deg2rad(0.0), np.deg2rad(0.0)),
            },
            "testing": {
                "rx": (1600.0, 1800.0),
                "ry": (-100.0, 100.0),
                "vx": (-0.15, -0.05),
                "vy": (-0.02, 0.02),
                "psi": (np.deg2rad(-3.0), np.deg2rad(3.0)),
                "omegaz": (np.deg2rad(-0.05), np.deg2rad(0.05)),
            },
            "training": {
                "rx": (1500.0, 1900.0),
                "ry": (-150.0, 150.0),
                "vx": (-0.20, -0.03),
                "vy": (-0.03, 0.03),
                "psi": (np.deg2rad(-5.0), np.deg2rad(5.0)),
                "omegaz": (np.deg2rad(-0.10), np.deg2rad(0.10)),
            },
        },
        "final": {
            "nominal": {
                "rx": (300.0, 300.0),
                "ry": (0.0, 0.0),
                "vx": (-0.03, -0.03),
                "vy": (0.0, 0.0),
                "psi": (np.deg2rad(0.0), np.deg2rad(0.0)),
                "omegaz": (np.deg2rad(0.0), np.deg2rad(0.0)),
            },
            "testing": {
                "rx": (290.0, 310.0),
                "ry": (-3.0, 3.0),
                "vx": (-0.05, -0.01),
                "vy": (-0.005, 0.005),
                "psi": (np.deg2rad(-2.0), np.deg2rad(2.0)),
                "omegaz": (np.deg2rad(-0.03), np.deg2rad(0.03)),
            },
            "training": {
                "rx": (280.0, 320.0),
                "ry": (-5.0, 5.0),
                "vx": (-0.06, -0.01),
                "vy": (-0.01, 0.01),
                "psi": (np.deg2rad(-3.0), np.deg2rad(3.0)),
                "omegaz": (np.deg2rad(-0.05), np.deg2rad(0.05)),
            },
        },
    }

    for phase, bounds_by_mode in bounds_by_phase.items():
        for mode, bounds in bounds_by_mode.items():
            env = Docking3DoFEnv(seed=42, phase=phase)
            state = env.reset(mode=mode)

            assert state.shape == (6,)
            assert np.all(np.isfinite(state))
            assert env.time == 0.0

            rx, ry, vx, vy, psi, omegaz = state

            assert bounds["rx"][0] <= rx <= bounds["rx"][1]
            assert bounds["ry"][0] <= ry <= bounds["ry"][1]
            assert bounds["vx"][0] <= vx <= bounds["vx"][1]
            assert bounds["vy"][0] <= vy <= bounds["vy"][1]
            assert bounds["psi"][0] <= psi <= bounds["psi"][1]
            assert bounds["omegaz"][0] <= omegaz <= bounds["omegaz"][1]


### Checks that the clip method correctly limits the action values.
def test_action_clipping():
    env = Docking3DoFEnv(
        ax_max=0.01,
        ay_max=0.02,
        alphaz_max=0.003
    )

    # Case 1: action exceeds the bounds, so it should be clipped
    action_body = np.array([1.0, -1.0, 1.0])
    clipped = env.clip_action(action_body)

    expected = np.array([0.01, -0.02, 0.003])

    assert np.allclose(clipped, expected)


    # Case 2: action is already inside the bounds, so it should stay unchanged
    action_body_inside = np.array([0.005, -0.01, 0.001])
    clipped_inside = env.clip_action(action_body_inside)

    expected_inside = np.array([0.005, -0.01, 0.001])

    assert np.allclose(clipped_inside, expected_inside)


### Checks that the action transformation correctly converts chaser body-frame actions to target orbital-frame actions, for psi=0, pi/2. 
def test_action_transformation_zero_angle_and_pi_over_2():
    env = Docking3DoFEnv()

    action_body = np.array([0.01, 0.02, 0.003])

    # Case 1: zero relative attitude angle
    action_orbital_zero = env.transform_action_to_orbital_frame(
        action_body,
        psi=0.0
    )

    expected_zero = np.array([-0.01, -0.02, 0.003])

    assert np.allclose(action_orbital_zero, expected_zero)


    # Case 2: relative attitude angle of pi/2
    action_orbital_pi_over_2 = env.transform_action_to_orbital_frame(
        action_body,
        psi=np.pi / 2
    )

    expected_pi_over_2 = np.array([0.02, -0.01, 0.003])

    assert np.allclose(action_orbital_pi_over_2, expected_pi_over_2)


### Checks that the dynamics method correctly computes the state derivative, for psi=0. 
def test_dynamics_returns_correct_state_derivative():
    env = Docking3DoFEnv(
        ax_max=0.01,
        ay_max=0.02,
        alphaz_max=0.003
    )

    state = np.array([
        100.0,   # rx
        50.0,    # ry
        -0.2,    # vx
        0.1,     # vy
        0.0,     # psi
        0.01     # omegaz
    ])

    action_body = np.array([
        0.005,   # ax_Bc
        0.010,   # ay_Bc
        0.002    # alphaz_Bc
    ])

    state_dot = env.dynamics(state, action_body)

    rx, ry, vx, vy, psi, omegaz = state

    # psi = 0, transformation gives:
    # ax_O = -ax_Bc
    # ay_O = -ay_Bc
    # alphaz_O = alphaz_Bc
    ax = -0.005
    ay = -0.010
    alphaz = 0.002

    rc = np.sqrt(rx**2 + ry**2 + env.r0**2)

    expected = np.array([
        vx,
        vy,
        -env.mu * rx / rc**3 + env.n**2 * rx + ax,
        -env.mu * ry / rc**3 + ay,
        omegaz,
        alphaz
    ])

    assert state_dot.shape == (6,)
    assert np.all(np.isfinite(state_dot))
    assert np.allclose(state_dot, expected)


### Checks that the dynamics method correctly clips the action before computing the state derivative, for psi=0.
def test_dynamics_clips_action_before_computing_derivative():
    env = Docking3DoFEnv(
        ax_max=0.01,
        ay_max=0.02,
        alphaz_max=0.003
    )

    state = np.array([
        100.0,   # rx
        50.0,    # ry
        -0.2,    # vx
        0.1,     # vy
        0.0,     # psi
        0.01     # omegaz
    ])

    # This action is much larger than the allowed limits
    action_body = np.array([
        1.0,
        -1.0,
        1.0
    ])

    state_dot = env.dynamics(state, action_body)

    rx, ry, vx, vy, psi, omegaz = state

    # First the action should be clipped in the body frame:
    # action_body_clipped = [0.01, -0.02, 0.003]
    #
    # Then, because psi = 0:
    # ax_O = -ax_Bc = -0.01
    # ay_O = -ay_Bc = 0.02
    # alphaz_O = alphaz_Bc = 0.003
    ax = -0.01
    ay = 0.02
    alphaz = 0.003

    rc = np.sqrt(rx**2 + ry**2 + env.r0**2)

    expected = np.array([
        vx,
        vy,
        -env.mu * rx / rc**3 + env.n**2 * rx + ax,
        -env.mu * ry / rc**3 + ay,
        omegaz,
        alphaz
    ])

    assert np.allclose(state_dot, expected)


### Checks that the dynamics method correctly computes the state derivative when the action is zero and the velocity is zero, for psi=0.
def test_dynamics_zero_action_zero_velocity():
    env = Docking3DoFEnv()

    state = np.array([
        100.0,   # rx
        50.0,    # ry
        0.0,     # vx
        0.0,     # vy
        0.0,     # psi
        0.0      # omegaz
    ])

    action_body = np.array([
        0.0,
        0.0,
        0.0
    ])

    state_dot = env.dynamics(state, action_body)

    rx, ry, vx, vy, psi, omegaz = state

    rc = np.sqrt(rx**2 + ry**2 + env.r0**2)

    expected = np.array([
        0.0,
        0.0,
        -env.mu * rx / rc**3 + env.n**2 * rx,
        -env.mu * ry / rc**3,
        0.0,
        0.0
    ])

    assert np.allclose(state_dot, expected)


### Check that the integrate method returns a valid next state of the correct shape, and that all values are finite. 
@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_integrate_returns_valid_state(integrator):
    env = Docking3DoFEnv(
        dt=1.0,
        integrator=integrator
    )

    state = np.array([
        1000.0,   # rx
        10.0,     # ry
        -0.1,     # vx
        0.02,     # vy
        0.0,      # psi
        0.001     # omegaz
    ])

    action = np.array([
        0.0,
        0.0,
        0.0
    ])

    next_state = env.integrate(state, action)

    assert next_state.shape == (6,)
    assert np.all(np.isfinite(next_state))


### Check simple propagation of state when the net acceleration is zero, for psi=0.
@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_integrate_constant_velocity_when_net_acceleration_is_zero(integrator):
    env = Docking3DoFEnv(
        dt=1.0,
        integrator=integrator,
        ax_max=1.0,
        ay_max=1.0,
        alphaz_max=1.0
    )

    state = np.array([
        1000.0,   # rx
        0.0,      # ry
        -0.2,     # vx
        0.05,     # vy
        0.0,      # psi
        0.0       # omegaz
    ])

    rx, ry, vx, vy, psi, omegaz = state

    rc = np.sqrt(rx**2 + ry**2 + env.r0**2)

    natural_ax = -env.mu * rx / rc**3 + env.n**2 * rx
    natural_ay = -env.mu * ry / rc**3

    # for total acceleration to be zero:
    # vx_dot = natural_ax + ax_O = 0
    # vy_dot = natural_ay + ay_O = 0
    desired_ax_orbital = -natural_ax
    desired_ay_orbital = -natural_ay

    # Since psi = 0:
    # ax_O = -ax_Bc
    # ay_O = -ay_Bc
    action_body = np.array([
        -desired_ax_orbital,
        -desired_ay_orbital,
        0.0
    ])

    next_state = env.integrate(state, action_body)

    expected = np.array([
        rx + env.dt * vx,
        ry + env.dt * vy,
        vx,
        vy,
        psi,
        omegaz
    ])

    assert np.allclose(next_state, expected)


### Checks that dt is actually being used in the integrate method, for zero acceleration and psi=0.
@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_integrate_uses_timestep(integrator):
    env = Docking3DoFEnv(
        dt=0.5,
        integrator=integrator,
        ax_max=1.0,
        ay_max=1.0,
        alphaz_max=1.0
    )

    state = np.array([
        1000.0,   # rx
        0.0,      # ry
        -0.2,     # vx
        0.05,     # vy
        0.0,      # psi
        0.0
    ])

    rx, ry, vx, vy, psi, omegaz = state

    rc = np.sqrt(rx**2 + ry**2 + env.r0**2)

    natural_ax = -env.mu * rx / rc**3 + env.n**2 * rx
    natural_ay = -env.mu * ry / rc**3

    desired_ax_orbital = -natural_ax
    desired_ay_orbital = -natural_ay

    action_body = np.array([
        -desired_ax_orbital,
        -desired_ay_orbital,
        0.0
    ])

    next_state = env.integrate(state, action_body)

    expected = np.array([
        rx + 0.5 * vx,
        ry + 0.5 * vy,
        vx,
        vy,
        psi,
        omegaz
    ])

    assert np.allclose(next_state, expected)


### Check that constructor raises error for invalid integrator name.
def test_invalid_integrator_name_raises_error():
    with pytest.raises(ValueError, match="integrator must be either"):
        Docking3DoFEnv(integrator="invalid")





### Checks the format of the output from the step method.
def test_step_output_format():
    env = Docking3DoFEnv(seed=42)
    env.reset()

    action_body = np.zeros(3)

    state, reward, done, info = env.step(action_body)

    assert state.shape == (6,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert np.all(np.isfinite(state))