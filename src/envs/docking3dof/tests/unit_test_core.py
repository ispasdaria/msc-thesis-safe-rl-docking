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
Unit tests for the Docking3DoFEnv environment.
"""


def make_state_from_port(env, s, rho, vx=0.0, vy=0.0, psi=0.0, omegaz=0.0):
    r_p = np.array([s, rho], dtype=float)
    r_c_orbital = env.rotation_body_to_orbital_2d(psi) @ env.r_c_body
    r_com = r_p + env.r_t_orbital - r_c_orbital

    return np.array(
        [r_com[0], r_com[1], vx, vy, psi, omegaz],
        dtype=float,
    )


### Initialization and validation
def test_environment_requires_explicit_max_time():
    with pytest.raises(TypeError, match="max_time"):
        Docking3DoFEnv()


def test_environment_initialization():
    env = Docking3DoFEnv(max_time=12_000.0)
    assert env.dt > 0
    assert env.max_time == 12_000.0
    assert env.ax_max > 0
    assert env.ay_max > 0
    assert env.alphaz_max > 0
    assert np.allclose(env.r_c_body, np.array([4.1, 0.0]))
    assert np.allclose(env.r_t_orbital, np.array([20.0, 0.0]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0},
        {"dt": -1.0},
        {"max_time": 0.0},
        {"max_time": -1.0},
        {"mu": 0.0},
        {"mu": -1.0},
        {"r0": 0.0},
        {"r0": -1.0},
    ],
)
def test_init_rejects_invalid_scalars(kwargs):
    constructor_kwargs = {"max_time": 12_000.0, **kwargs}
    with pytest.raises(ValueError):
        Docking3DoFEnv(**constructor_kwargs)


@pytest.mark.parametrize("integrator", ["invalid", "", None])
def test_init_rejects_invalid_integrator(integrator):
    with pytest.raises(ValueError, match="integrator must be either"):
        Docking3DoFEnv(max_time=12_000.0, integrator=integrator)


@pytest.mark.parametrize("phase", ["invalid", "", None])
def test_init_rejects_invalid_phase(phase):
    with pytest.raises(ValueError, match="phase must be either"):
        Docking3DoFEnv(max_time=12_000.0 if phase == "closing" else 5_000.0, phase=phase)


def test_init_converts_negative_control_limits_to_positive():
    env = Docking3DoFEnv(max_time=12_000.0, ax_max=-0.01, ay_max=-0.02, alphaz_max=-0.003)
    assert env.ax_max == 0.01
    assert env.ay_max == 0.02
    assert env.alphaz_max == 0.003


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ax_max": np.inf},
        {"ay_max": np.nan},
        {"alphaz_max": np.inf},
    ],
)
def test_init_rejects_non_finite_control_limits(kwargs):
    with pytest.raises(ValueError, match="Control limits must be finite"):
        Docking3DoFEnv(max_time=12_000.0, **kwargs)


def test_init_rejects_invalid_geometry_shapes():
    with pytest.raises(ValueError, match="r_c_body must have shape"):
        Docking3DoFEnv(max_time=12_000.0, r_c_body=[1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="r_t_orbital must have shape"):
        Docking3DoFEnv(max_time=12_000.0, r_t_orbital=[1.0])


def test_init_rejects_invalid_p3_target_tolerance_shape():
    with pytest.raises(ValueError, match="p3_target_tolerance must have shape"):
        Docking3DoFEnv(max_time=12_000.0, p3_target_tolerance=np.ones(5))


### Angle utilities and action transforms
@pytest.mark.parametrize("angle", [-4.0 * np.pi, -3.5 * np.pi, 3.5 * np.pi, 4.0 * np.pi])
def test_wrap_to_pi_bounds(angle):
    env = Docking3DoFEnv(max_time=12_000.0, )
    wrapped = env.wrap_to_pi(angle)
    assert -np.pi <= wrapped < np.pi


def test_rotation_body_to_orbital_2d_basic_angles():
    env = Docking3DoFEnv(max_time=12_000.0, )

    r0 = env.rotation_body_to_orbital_2d(0.0)
    r90 = env.rotation_body_to_orbital_2d(np.pi / 2)

    assert np.allclose(r0 @ np.array([1.0, 0.0]), np.array([-1.0, 0.0]))
    assert np.allclose(r90 @ np.array([1.0, 0.0]), np.array([0.0, -1.0]))


def test_clip_action_rejects_invalid_shape():
    env = Docking3DoFEnv(max_time=12_000.0, )
    with pytest.raises(ValueError, match="Action must have shape"):
        env.clip_action([0.0, 0.0])


def test_transform_action_rejects_invalid_shape():
    env = Docking3DoFEnv(max_time=12_000.0, )
    with pytest.raises(ValueError, match="Body-frame action must have shape"):
        env.transform_action_to_orbital_frame([0.0, 0.0], psi=0.0)


def test_action_clipping():
    env = Docking3DoFEnv(
        max_time=12_000.0,
        ax_max=0.01,
        ay_max=0.02,
        alphaz_max=0.003,
    )

    action_body = np.array([1.0, -1.0, 1.0])
    clipped = env.clip_action(action_body)

    expected = np.array([0.01, -0.02, 0.003])

    assert np.allclose(clipped, expected)

    action_body_inside = np.array([0.005, -0.01, 0.001])
    clipped_inside = env.clip_action(action_body_inside)

    expected_inside = np.array([0.005, -0.01, 0.001])

    assert np.allclose(clipped_inside, expected_inside)


def test_action_transformation_zero_angle_and_pi_over_2():
    env = Docking3DoFEnv(max_time=12_000.0, )

    action_body = np.array([0.01, 0.02, 0.003])

    action_orbital_zero = env.transform_action_to_orbital_frame(
        action_body,
        psi=0.0,
    )

    expected_zero = np.array([-0.01, -0.02, 0.003])

    assert np.allclose(action_orbital_zero, expected_zero)

    action_orbital_pi_over_2 = env.transform_action_to_orbital_frame(
        action_body,
        psi=np.pi / 2,
    )

    expected_pi_over_2 = np.array([0.02, -0.01, 0.003])

    assert np.allclose(action_orbital_pi_over_2, expected_pi_over_2)


### Docking-port geometry
def test_port_relative_position_uses_offsets_and_rotation():
    env = Docking3DoFEnv(max_time=12_000.0, )

    state_zero = np.array([100.0, 50.0, 0.0, 0.0, 0.0, 0.0])
    r_p_zero = env.get_port_relative_position(state_zero)

    expected_zero = np.array([100.0 - 24.1, 50.0])

    assert np.allclose(r_p_zero, expected_zero)

    state_pi_over_2 = np.array([100.0, 50.0, 0.0, 0.0, np.pi / 2, 0.0])
    r_p_pi_over_2 = env.get_port_relative_position(state_pi_over_2)

    expected_pi_over_2 = np.array([100.0 - 20.0, 50.0 - 4.1])

    assert np.allclose(r_p_pi_over_2, expected_pi_over_2)


def test_port_relative_velocity_includes_rotation_term():
    env = Docking3DoFEnv(max_time=12_000.0, )

    state = np.array([100.0, 50.0, -0.2, 0.1, 0.0, 0.1])
    v_p = env.get_port_relative_velocity(state)

    expected = np.array([-0.2, 0.1 - 0.41])

    assert np.allclose(v_p, expected)


def test_port_relative_velocity_with_rotation_and_omega():
    env = Docking3DoFEnv(max_time=12_000.0, )

    psi = np.pi / 4
    omegaz = 0.2

    state = np.array([100.0, -50.0, 0.1, -0.2, psi, omegaz])
    v_p = env.get_port_relative_velocity(state)

    r_c_orbital = env.rotation_body_to_orbital_2d(psi) @ env.r_c_body
    omega_cross_r_c = np.array(
        [-omegaz * r_c_orbital[1], omegaz * r_c_orbital[0]]
    )

    expected = np.array([0.1, -0.2]) + omega_cross_r_c

    assert np.allclose(v_p, expected)


def test_is_in_X_dock_true_for_aligned_port_state():
    env = Docking3DoFEnv(max_time=5_000.0, phase="final")

    psi = 0.0
    r_c_orbital = env.rotation_body_to_orbital_2d(psi) @ env.r_c_body
    r_com = env.r_t_orbital - r_c_orbital

    state = np.array([
        r_com[0],
        r_com[1],
        0.0,
        0.0,
        psi,
        0.0,
    ])

    assert env.is_in_X_dock(state)


def test_is_in_X_dock_false_for_high_speed():
    env = Docking3DoFEnv(max_time=5_000.0, phase="final")

    psi = 0.0
    r_c_orbital = env.rotation_body_to_orbital_2d(psi) @ env.r_c_body
    r_com = env.r_t_orbital - r_c_orbital

    state = np.array([
        r_com[0],
        r_com[1],
        0.2,
        0.0,
        psi,
        0.0,
    ])

    assert not env.is_in_X_dock(state)


def test_geometry_membership_on_boundaries():
    env = Docking3DoFEnv(max_time=5_000.0, phase="final")
    env.reset(mode="nominal")

    s0_state = make_state_from_port(env, s=0.0, rho=0.0)
    q0 = env.get_derived_quantities(s0_state)
    assert env.is_in_C_dock(q0)

    s_end = env.cone_length
    rho_end = env.cone_slope * s_end
    s_end_state = make_state_from_port(env, s=s_end, rho=rho_end)
    q_end = env.get_derived_quantities(s_end_state)
    assert env.is_in_C_dock(q_end)

    corr_state = make_state_from_port(env, s=env.d_P3, rho=env.corridor_radius)
    q_corr = env.get_derived_quantities(corr_state)
    assert env.is_in_C_corr(q_corr)

    ellipse_state = np.array(
        [env.ae_x, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=float,
    )
    q_ellipse = env.get_derived_quantities(ellipse_state)
    assert env.is_in_E_AE(q_ellipse)

    kos_state = np.array(
        [env.r_kos, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=float,
    )
    q_kos = env.get_derived_quantities(kos_state)
    assert env.is_in_S_KOS(q_kos)


### Dynamics and integration
def test_dynamics_returns_correct_state_derivative():
    env = Docking3DoFEnv(
        max_time=12_000.0,
        ax_max=0.01,
        ay_max=0.02,
        alphaz_max=0.003,
    )

    state = np.array([
        100.0,
        50.0,
        -0.2,
        0.1,
        0.0,
        0.01,
    ])

    action_body = np.array([
        0.005,
        0.010,
        0.002,
    ])

    state_dot = env.dynamics(state, action_body)

    rx, ry, vx, vy, psi, omegaz = state

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
        alphaz,
    ])

    assert state_dot.shape == (6,)
    assert np.all(np.isfinite(state_dot))
    assert np.allclose(state_dot, expected)


def test_dynamics_clips_action_before_computing_derivative():
    env = Docking3DoFEnv(
        max_time=12_000.0,
        ax_max=0.01,
        ay_max=0.02,
        alphaz_max=0.003,
    )

    state = np.array([
        100.0,
        50.0,
        -0.2,
        0.1,
        0.0,
        0.01,
    ])

    action_body = np.array([
        1.0,
        -1.0,
        1.0,
    ])

    state_dot = env.dynamics(state, action_body)

    rx, ry, vx, vy, psi, omegaz = state

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
        alphaz,
    ])

    assert np.allclose(state_dot, expected)


def test_dynamics_zero_action_zero_velocity():
    env = Docking3DoFEnv(max_time=12_000.0, )

    state = np.array([
        100.0,
        50.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])

    action_body = np.array([
        0.0,
        0.0,
        0.0,
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
        0.0,
    ])

    assert np.allclose(state_dot, expected)


@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_integrate_returns_valid_state(integrator):
    env = Docking3DoFEnv(
        max_time=12_000.0,
        dt=1.0,
        integrator=integrator,
    )

    state = np.array([
        1000.0,
        10.0,
        -0.1,
        0.02,
        0.0,
        0.001,
    ])

    action = np.array([
        0.0,
        0.0,
        0.0,
    ])

    next_state = env.integrate(state, action)

    assert next_state.shape == (6,)
    assert np.all(np.isfinite(next_state))


@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_integrate_constant_velocity_when_net_acceleration_is_zero(integrator):
    env = Docking3DoFEnv(
        max_time=12_000.0,
        dt=1.0,
        integrator=integrator,
        ax_max=1.0,
        ay_max=1.0,
        alphaz_max=1.0,
    )

    state = np.array([
        1000.0,
        0.0,
        -0.2,
        0.05,
        0.0,
        0.0,
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
        0.0,
    ])

    next_state = env.integrate(state, action_body)

    expected = np.array([
        rx + env.dt * vx,
        ry + env.dt * vy,
        vx,
        vy,
        psi,
        omegaz,
    ])

    assert np.allclose(next_state, expected)


@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_integrate_uses_timestep(integrator):
    env = Docking3DoFEnv(
        max_time=12_000.0,
        dt=0.5,
        integrator=integrator,
        ax_max=1.0,
        ay_max=1.0,
        alphaz_max=1.0,
    )

    state = np.array([
        1000.0,
        0.0,
        -0.2,
        0.05,
        0.0,
        0.0,
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
        0.0,
    ])

    next_state = env.integrate(state, action_body)

    expected = np.array([
        rx + 0.5 * vx,
        ry + 0.5 * vy,
        vx,
        vy,
        psi,
        omegaz,
    ])

    assert np.allclose(next_state, expected)


def test_invalid_integrator_name_raises_error():
    with pytest.raises(ValueError, match="integrator must be either"):
        Docking3DoFEnv(max_time=12_000.0, integrator="invalid")


def test_euler_and_rk4_match_for_constant_velocity_case():
    dt = 0.5

    env_euler = Docking3DoFEnv(
        max_time=12_000.0,
        dt=dt,
        integrator="euler",
        ax_max=1.0,
        ay_max=1.0,
        alphaz_max=1.0,
    )

    env_rk4 = Docking3DoFEnv(
        max_time=12_000.0,
        dt=dt,
        integrator="rk4",
        ax_max=1.0,
        ay_max=1.0,
        alphaz_max=1.0,
    )

    state = np.array([
        1000.0,
        0.0,
        -0.2,
        0.05,
        0.0,
        0.0,
    ])

    rx, ry, vx, vy, psi, omegaz = state

    rc = np.sqrt(rx**2 + ry**2 + env_euler.r0**2)

    natural_ax = -env_euler.mu * rx / rc**3 + env_euler.n**2 * rx
    natural_ay = -env_euler.mu * ry / rc**3

    desired_ax_orbital = -natural_ax
    desired_ay_orbital = -natural_ay

    action_body = np.array([
        -desired_ax_orbital,
        -desired_ay_orbital,
        0.0,
    ])

    next_euler = env_euler.integrate(state, action_body)
    next_rk4 = env_rk4.integrate(state, action_body)

    assert np.allclose(next_euler, next_rk4)


### Reset, sampling, and reproducibility
def test_reset_rejects_invalid_initial_state_shape():
    env = Docking3DoFEnv(max_time=12_000.0, )
    with pytest.raises(ValueError, match="Initial state must have shape"):
        env.reset(initial_state=[0.0, 0.0, 0.0])


def test_reset_wraps_initial_psi():
    env = Docking3DoFEnv(max_time=12_000.0, )
    initial_state = np.array([100.0, 0.0, 0.0, 0.0, 3.0 * np.pi, 0.0])
    state = env.reset(initial_state=initial_state)
    assert state[4] == env.wrap_to_pi(initial_state[4])


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
                "rx": (280.0, 320.0),
                "ry": (-3.0, 3.0),
                "vx": (-0.05, -0.01),
                "vy": (-0.01, 0.01),
                "psi": (np.deg2rad(-1.0), np.deg2rad(1.0)),
                "omegaz": (np.deg2rad(-0.05), np.deg2rad(0.05)),
            },
            "training": {
                "rx": (250.0, 350.0),
                "ry": (-5.0, 5.0),
                "vx": (-0.06, -0.01),
                "vy": (-0.02, 0.02),
                "psi": (np.deg2rad(-2.0), np.deg2rad(2.0)),
                "omegaz": (np.deg2rad(-0.10), np.deg2rad(0.10)),
            },
        },
    }

    for phase, bounds_by_mode in bounds_by_phase.items():
        for mode, bounds in bounds_by_mode.items():
            env = Docking3DoFEnv(max_time=12_000.0 if phase == "closing" else 5_000.0, seed=42, phase=phase)
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


def test_is_at_P3_target_raises_without_sampling():
    env = Docking3DoFEnv(max_time=5_000.0, phase="final")
    env.reset(mode="nominal")
    with pytest.raises(RuntimeError, match="P3 target has not been sampled"):
        env.is_at_P3_target(env.state)


def test_is_in_X_P3_nominal_and_out_of_bounds():
    env = Docking3DoFEnv(max_time=12_000.0, )

    nominal_state = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    assert env.is_in_X_P3(nominal_state, mode="nominal")

    out_of_bounds_state = np.array([301.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    assert not env.is_in_X_P3(out_of_bounds_state, mode="nominal")


def test_reset_reproducible_with_seed():
    env_a = Docking3DoFEnv(max_time=12_000.0, phase="closing", seed=123)
    env_b = Docking3DoFEnv(max_time=12_000.0, phase="closing", seed=123)

    state_a = env_a.reset(mode="testing")
    state_b = env_b.reset(mode="testing")

    assert np.allclose(state_a, state_b)
    assert np.allclose(env_a.p3_target, env_b.p3_target)


def test_is_at_P3_target_tolerance_bounds():
    env = Docking3DoFEnv(max_time=12_000.0, phase="closing", seed=123)
    env.reset(mode="testing")

    inside = env.p3_target + 0.5 * env.p3_target_tolerance
    outside = env.p3_target + 1.1 * env.p3_target_tolerance

    assert env.is_at_P3_target(inside)
    assert not env.is_at_P3_target(outside)


### Termination and safety
def test_monitor_safety_flags_alignment_unsafe_in_final():
    env = Docking3DoFEnv(max_time=5_000.0, phase="final")
    env.reset(mode="nominal")

    psi = env.psi_unsafe * 1.1
    r_c_orbital = env.rotation_body_to_orbital_2d(psi) @ env.r_c_body
    r_com = env.r_t_orbital - r_c_orbital

    state = np.array([
        r_com[0],
        r_com[1],
        0.0,
        0.0,
        psi,
        0.0,
    ])

    monitor = env.monitor_safety(state)

    assert monitor["unsafe"]
    assert "alignment" in monitor["violated_constraints"]


def test_monitor_safety_geometry_safe_velocity_warn_closing():
    env = Docking3DoFEnv(max_time=12_000.0, phase="closing")
    env.reset(mode="nominal")

    s = 1000.0
    rho = 10.0
    vx = 0.2
    vy = np.sqrt(0.3**2 - 0.2**2)

    state = make_state_from_port(env, s=s, rho=rho, vx=vx, vy=vy)
    monitor = env.monitor_safety(state)

    assert bool(monitor["unsafe"]) is False
    assert bool(monitor["warning"]) is True


def test_monitor_safety_geometry_warn_velocity_safe_closing():
    env = Docking3DoFEnv(max_time=12_000.0, phase="closing")
    env.reset(mode="nominal")

    s = 1000.0
    rho = env.cone_slope * s + 50.0
    state = make_state_from_port(env, s=s, rho=rho, vx=0.05, vy=0.0)

    monitor = env.monitor_safety(state)

    assert bool(monitor["unsafe"]) is False
    assert bool(monitor["warning"]) is True


def test_monitor_safety_geometry_unsafe_overrides_velocity_safe():
    env = Docking3DoFEnv(max_time=12_000.0, phase="closing")
    env.reset(mode="nominal")

    state = make_state_from_port(env, s=3000.0, rho=0.0, vx=0.05, vy=0.0)
    monitor = env.monitor_safety(state)

    assert monitor["unsafe"] is True
    assert "geometry" in monitor["violated_constraints"]


def test_monitor_safety_velocity_unsafe_overrides_geometry_safe():
    env = Docking3DoFEnv(max_time=12_000.0, phase="closing")
    env.reset(mode="nominal")

    s = 1000.0
    rho = 10.0
    state = make_state_from_port(env, s=s, rho=rho, vx=0.4, vy=0.0)
    monitor = env.monitor_safety(state)

    assert bool(monitor["unsafe"]) is True
    assert "velocity" in monitor["violated_constraints"]


def test_check_done_success_in_final_dock_state():
    env = Docking3DoFEnv(max_time=5_000.0, phase="final")
    env.reset(mode="nominal")

    psi = 0.0
    r_c_orbital = env.rotation_body_to_orbital_2d(psi) @ env.r_c_body
    r_com = env.r_t_orbital - r_c_orbital

    state = np.array([
        r_com[0],
        r_com[1],
        0.0,
        0.0,
        psi,
        0.0,
    ])

    done, info = env.check_done(state)

    assert done
    assert info["success"]
    assert info["terminal_set"] == "X_dock"
    assert info["termination_reason"] == "success_docking"


def test_check_done_failure_on_alignment_violation_final():
    env = Docking3DoFEnv(max_time=5_000.0, phase="final")
    env.reset(mode="nominal")

    psi = env.psi_unsafe * 1.1
    r_c_orbital = env.rotation_body_to_orbital_2d(psi) @ env.r_c_body
    r_com = env.r_t_orbital - r_c_orbital

    state = np.array([
        r_com[0],
        r_com[1],
        0.0,
        0.0,
        psi,
        0.0,
    ])

    done, info = env.check_done(state)

    assert done
    assert info["failure"]
    assert info["termination_reason"] == "safety_violation"


def test_check_done_safety_overrides_timeout():
    env = Docking3DoFEnv(phase="final", max_time=1.0)
    env.reset(mode="nominal")

    env.time = env.max_time

    psi = env.psi_unsafe * 1.1
    r_c_orbital = env.rotation_body_to_orbital_2d(psi) @ env.r_c_body
    r_com = env.r_t_orbital - r_c_orbital

    state = np.array([r_com[0], r_com[1], 0.0, 0.0, psi, 0.0])

    done, info = env.check_done(state)

    assert done
    assert info["failure"]
    assert info["termination_reason"] == "safety_violation"


def test_check_done_success_overrides_timeout():
    env = Docking3DoFEnv(phase="closing", max_time=1.0, seed=7)
    env.reset(mode="nominal")

    env.time = env.max_time

    done, info = env.check_done(env.p3_target)

    assert done
    assert info["success"]
    assert info["termination_reason"] == "success_P3_target"


def test_check_done_success_at_p3_target_closing():
    env = Docking3DoFEnv(max_time=12_000.0, phase="closing", seed=7)
    env.reset(mode="nominal")

    done, info = env.check_done(env.p3_target)

    assert done
    assert info["success"]
    assert info["terminal_set"] == "P3_target"
    assert info["termination_reason"] == "success_P3_target"


### Step, logging, and history
def test_history_arrays_after_reset():
    env = Docking3DoFEnv(max_time=12_000.0, )
    env.reset(mode="nominal")

    times, states, actions_body_raw, actions_body_clipped, actions_orbital = (
        env.get_history_arrays()
    )

    assert times.shape == (1,)
    assert states.shape == (1, 6)
    assert actions_body_raw.shape == (1, 3)
    assert actions_body_clipped.shape == (1, 3)
    assert actions_orbital.shape == (1, 3)


def test_step_output_format():
    env = Docking3DoFEnv(max_time=12_000.0, seed=42)
    env.reset()

    action_body = np.zeros(3)

    state, done, info = env.step(action_body)

    assert state.shape == (6,)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert np.all(np.isfinite(state))


def test_step_time_limit_only_triggers_after_max_time():
    env = Docking3DoFEnv(
        dt=0.5,
        max_time=1.0,
        phase="closing",
        integrator="rk4",
    )

    initial_state = np.array([
        1000.0,
        0.0,
        -0.1,
        0.0,
        0.0,
        0.0,
    ])

    env.reset(initial_state=initial_state)

    action_body = np.zeros(3)

    state, done, info = env.step(action_body)

    assert env.time == 0.5
    assert done is False
    assert info["time_limit"] is False

    state, done, info = env.step(action_body)

    assert env.time == 1.0
    assert done is True
    assert info["time_limit"] is True
    assert info["success"] is False
    assert info["failure"] is False

    assert state.shape == (6,)
    assert np.all(np.isfinite(state))


def test_step_rejects_invalid_action_shape():
    env = Docking3DoFEnv(max_time=12_000.0, )
    env.reset()

    with pytest.raises(ValueError, match="Action must have shape"):
        env.step([0.0, 0.0])


def test_step_appends_history_entry_with_actions():
    env = Docking3DoFEnv(max_time=12_000.0, )
    env.reset(mode="nominal")

    action = np.array([0.0, 0.0, 0.0])
    env.step(action)

    assert len(env.history) == 2
    last = env.history[-1]

    assert np.allclose(last["action_body_raw"], action)
    assert np.allclose(last["action_body_clipped"], action)
    assert np.allclose(last["action_orbital"], action)
    assert "phase" in last["info"]


def test_step_logs_action_orbital_with_pre_step_psi():
    env = Docking3DoFEnv(max_time=12_000.0, )
    initial_state = np.array([1000.0, 0.0, 0.0, 0.0, np.pi / 6, 0.1])
    env.reset(initial_state=initial_state)

    action = np.array([0.005, -0.002, 0.001])
    env.step(action)

    expected = env.transform_action_to_orbital_frame(action, psi=initial_state[4])
    last = env.history[-1]

    assert np.allclose(last["action_orbital"], expected)


def test_step_non_finite_state_returns_failure():
    env = Docking3DoFEnv(max_time=12_000.0, )
    initial_state = np.array([np.inf, 0.0, 0.0, 0.0, np.inf, 0.0])
    env.reset(initial_state=initial_state)

    action = np.zeros(3)
    state, done, info = env.step(action)

    assert done is True
    assert info["failure"] is True
    assert info["failure_reason"] == "non_finite_state"


def test_history_arrays_after_multiple_steps():
    env = Docking3DoFEnv(max_time=12_000.0, )
    env.reset(mode="nominal")

    for _ in range(3):
        env.step(np.zeros(3))

    times, states, actions_body_raw, actions_body_clipped, actions_orbital = (
        env.get_history_arrays()
    )

    assert times.shape == (4,)
    assert states.shape == (4, 6)
    assert actions_body_raw.shape == (4, 3)
    assert actions_body_clipped.shape == (4, 3)
    assert actions_orbital.shape == (4, 3)
