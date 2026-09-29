"""Docking-port kinematics and control-affine acceleration calculations."""

import numpy as np

from .cbf_types import PortAccelerationAffine, PortKinematics


_J = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)


def _validate_state(state):
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (6,):
        raise ValueError(f"Physical state must have shape (6,), got {state.shape}.")
    if not np.all(np.isfinite(state)):
        raise ValueError("Physical state must contain only finite values.")
    return state


def compute_port_kinematics(core_env, state):
    """Return safety-relevant port quantities using the simulator definition."""
    state = _validate_state(state)
    q = core_env.get_derived_quantities(state)
    v_p = np.asarray(q["v_p"], dtype=np.float64)

    return PortKinematics(
        s=float(q["s"]),
        y_p=float(q["r_p"][1]),
        s_dot=float(v_p[0]),
        y_p_dot=float(v_p[1]),
        v_p=v_p.copy(),
    )


def compute_port_acceleration_affine(core_env, state):
    """Return ``a_p = drift + control_matrix @ u`` for body-frame input ``u``.

    The drift includes the simulator's translational dynamics and the
    centripetal contribution of the rotating docking-port offset.  The third
    control column is the docking-port acceleration induced by angular
    acceleration.
    """
    state = _validate_state(state)
    psi = float(state[4])
    omega = float(state[5])

    rotation = np.asarray(
        core_env.rotation_body_to_orbital_2d(psi),
        dtype=np.float64,
    )
    r_c_orbital = rotation @ np.asarray(core_env.r_c_body, dtype=np.float64)

    zero_input_derivative = np.asarray(
        core_env.dynamics(state, np.zeros(3, dtype=np.float64)),
        dtype=np.float64,
    )
    com_acceleration_drift = zero_input_derivative[2:4]
    port_acceleration_drift = com_acceleration_drift - omega**2 * r_c_orbital

    control_matrix = np.zeros((2, 3), dtype=np.float64)
    control_matrix[:, :2] = rotation
    control_matrix[:, 2] = _J @ r_c_orbital

    return PortAccelerationAffine(
        drift=port_acceleration_drift,
        control_matrix=control_matrix,
    )


def evaluate_port_acceleration(affine, physical_action):
    """Evaluate an affine port-acceleration representation at an action."""
    action = np.asarray(physical_action, dtype=np.float64)
    if action.shape != (3,):
        raise ValueError(f"Physical action must have shape (3,), got {action.shape}.")
    return affine.drift + affine.control_matrix @ action
