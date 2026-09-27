import numpy as np

from envs.docking3dof.core import Docking3DoFEnv
from lqr_3dof_closing import (
    LQRClosingController,
    analyze_saturation,
    build_linearized_3dof_matrices,
    build_Q_closing,
    build_R_closing,
    compute_lqr_gain,
)


def _finite_difference_jacobian(function, value, step):
    value = np.asarray(value, dtype=float)
    output = np.asarray(function(value), dtype=float)
    jacobian = np.zeros((output.size, value.size), dtype=float)

    for index in range(value.size):
        perturbation = np.zeros_like(value)
        perturbation[index] = step
        forward = np.asarray(function(value + perturbation), dtype=float)
        backward = np.asarray(function(value - perturbation), dtype=float)
        jacobian[:, index] = (forward - backward) / (2.0 * step)

    return jacobian


def _build_controller(env):
    A, B = build_linearized_3dof_matrices(env)
    K, _ = compute_lqr_gain(A, B, build_Q_closing(), build_R_closing(env))
    return LQRClosingController(K)


def test_linear_model_matches_local_dynamics():
    env = Docking3DoFEnv(max_time=12_000.0, phase="closing")
    A, B = build_linearized_3dof_matrices(env)
    state = np.array([1e-3, -2e-3, 0.0, 0.0, 0.0, 0.0])
    action_body = np.zeros(3)

    A_finite_difference = _finite_difference_jacobian(
        lambda candidate: env.dynamics(candidate, action_body), state, 1e-4
    )
    B_finite_difference = _finite_difference_jacobian(
        lambda candidate: env.dynamics(state, candidate), action_body, 1e-4
    )
    body_to_orbital = np.diag([-1.0, -1.0, 1.0])

    assert np.allclose(A_finite_difference, A, atol=1e-5)
    assert np.allclose(B_finite_difference, B @ body_to_orbital, atol=1e-5)


def test_compute_lqr_gain_stable_and_shapes():
    env = Docking3DoFEnv(max_time=12_000.0)
    A, B = build_linearized_3dof_matrices(env)
    K, P = compute_lqr_gain(A, B, build_Q_closing(), build_R_closing(env))

    assert K.shape == (3, 6)
    assert P.shape == (6, 6)
    assert np.all(np.real(np.linalg.eigvals(A - B @ K)) < 0.0)


def test_linearized_pair_is_controllable():
    env = Docking3DoFEnv(max_time=12_000.0, phase="closing")
    A, B = build_linearized_3dof_matrices(env)
    dimension = A.shape[0]
    controllability = B

    for exponent in range(1, dimension):
        controllability = np.hstack(
            (controllability, np.linalg.matrix_power(A, exponent) @ B)
        )

    assert np.linalg.matrix_rank(controllability) == dimension


def test_linearized_pair_is_stabilizable_pbh():
    env = Docking3DoFEnv(max_time=12_000.0, phase="closing")
    A, B = build_linearized_3dof_matrices(env)
    dimension = A.shape[0]
    tolerance = 1e-9

    for eigenvalue in np.linalg.eigvals(A):
        if np.real(eigenvalue) >= -tolerance:
            pbh = np.hstack((A - eigenvalue * np.eye(dimension), B)).astype(complex)
            assert np.linalg.matrix_rank(pbh, tol=tolerance) == dimension


def test_linearized_pair_is_detectable_from_Q_pbh():
    env = Docking3DoFEnv(max_time=12_000.0, phase="closing")
    A, _ = build_linearized_3dof_matrices(env)
    Q = build_Q_closing()
    dimension = A.shape[0]
    output_matrix = np.diag(np.sqrt(np.diag(Q)))
    tolerance = 1e-9

    for eigenvalue in np.linalg.eigvals(A):
        if np.real(eigenvalue) >= -tolerance:
            pbh = np.vstack(
                (A - eigenvalue * np.eye(dimension), output_matrix)
            ).astype(complex)
            assert np.linalg.matrix_rank(pbh, tol=tolerance) == dimension


def test_controller_action_sanity():
    env = Docking3DoFEnv(max_time=12_000.0, phase="closing", seed=1)
    state = env.reset(mode="testing")
    action = _build_controller(env).compute_action(env, state)

    assert action.shape == (3,)
    assert np.all(np.isfinite(action))

    clipped = env.clip_action(action)
    assert np.all(clipped <= env.action_high + 1e-12)
    assert np.all(clipped >= env.action_low - 1e-12)


def test_saturation_analysis_returns_finite_values_after_rollout(capsys):
    env = Docking3DoFEnv(phase="closing", max_time=300.0, seed=0)
    state = env.reset(mode="testing")
    controller = _build_controller(env)

    for _ in range(10):
        action = controller.compute_action(env, state)
        state, done, _ = env.step(action)
        if done:
            break

    stats = analyze_saturation(env)
    capsys.readouterr()

    assert np.isfinite(stats["saturation_fraction_total"])
    assert 0.0 <= stats["saturation_fraction_total"] <= 1.0
