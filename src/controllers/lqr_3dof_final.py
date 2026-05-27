import os
import sys
import numpy as np
from scipy.linalg import solve_continuous_are

# Allow imports from the repo's src/ directory when running from the project root.
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.core import Docking3DoFEnv


def build_port_error_mapping(env):
    """
    Builds linearized mapping from CoM state error to docking-port error.

    x = [rx, ry, vx, vy, psi, omega]
    z = [r_px, r_py, v_px, v_py, psi, omega]

    Linearized around psi = 0 and omega = 0.
    """

    x_c, y_c = env.r_c_body

    H = np.array(
        [
            # r_px ≈ rx + y_c * psi + constant
            [1.0, 0.0, 0.0, 0.0,  y_c, 0.0],

            # r_py ≈ ry - x_c * psi + constant
            [0.0, 1.0, 0.0, 0.0, -x_c, 0.0],

            # v_px ≈ vx + y_c * omega
            [0.0, 0.0, 1.0, 0.0, 0.0,  y_c],

            # v_py ≈ vy - x_c * omega
            [0.0, 0.0, 0.0, 1.0, 0.0, -x_c],

            # psi
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],

            # omega
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    return H


def build_Q_final():
    """
    Builds the state weighting matrix Q for the final-approach LQR.

    Error state:
        e = [r_px, r_py, v_px, v_py, psi_e, omega]

    The design scales are stricter than or comparable to the final docking
    requirements, because the controller should regulate the state before
    reaching the terminal boundary.
    """

    final_design_scale = np.array(
        [
            20.0,              # r_px scale [m]
            1.0,               # r_py scale [m]
            0.03,              # v_px scale [m/s]
            0.01,              # v_py scale [m/s]
            np.deg2rad(1.0),   # psi scale [rad]
            np.deg2rad(0.05),  # omega scale [rad/s]
        ],
        dtype=float,
    )

    importance = np.array(
        [
            4.0,   # r_px: very important
            8.0,   # r_py: very important, corridor/alignment sensitive
            2.0,   # v_px
            4.0,   # v_py
            4.0,   # psi
            2.0,   # omega
        ],
        dtype=float,
    )

    Q_final = np.diag(importance / final_design_scale**2)

    return Q_final


def build_R_final(env):
    """
    Builds the input weighting matrix R for the final-approach LQR.

    LQR input:
        u_O = [ax_O, ay_O, alphaz]

    The design scales are below the hard action limits, but not too low,
    because final approach requires precise regulation.
    """

    control_design_scale = np.array(
        [
            0.8 * env.ax_max,
            0.8 * env.ay_max,
            0.8 * env.alphaz_max,
        ],
        dtype=float,
    )

    control_importance = np.array(
        [
            1.0,
            1.0,
            1.0,
        ],
        dtype=float,
    )

    R_final = np.diag(control_importance / control_design_scale**2)

    return R_final


def project_vector_to_ball(x, radius):
    """
    Returns the closest point to x inside/on a ball of given radius.

    If ||x|| <= radius:
        projection = x

    If ||x|| > radius:
        projection = radius * x / ||x||
    """

    x = np.asarray(x, dtype=float)
    norm_x = np.linalg.norm(x)

    if norm_x <= radius or norm_x == 0.0:
        return x.copy()

    return radius * x / norm_x


def final_error(env, state):
    """
    Computes final-approach port-level set error.

    Position objective:
        enter ||r_p|| <= 0.1 m

    Velocity objective:
        enter ||v_p|| <= 0.1 m/s

    Attitude objective:
        stay within |psi_e| <= 4 deg

    Angular-rate objective:
        stay within |omega| <= 0.2 deg/s
    """

    state = np.asarray(state, dtype=float)

    if state.shape != (6,):
        raise ValueError("state must have shape (6,).")

    r_p = env.get_port_relative_position(state)
    v_p = env.get_port_relative_velocity(state)

    psi_e = env.wrap_to_pi(state[4])
    omega = state[5]

    # Position set error: zero once ||r_p|| <= 0.1
    r_p_projection = project_vector_to_ball(r_p, radius=0.1)
    r_error = r_p - r_p_projection

    # Velocity set error: zero once ||v_p|| <= 0.1
    v_p_projection = project_vector_to_ball(v_p, radius=0.1)
    v_error = v_p - v_p_projection

    # Attitude set error: zero once |psi_e| <= 4 deg
    psi_limit = np.deg2rad(4.0)
    if abs(psi_e) <= psi_limit:
        psi_error = 0.0
    else:
        psi_error = psi_e - np.sign(psi_e) * psi_limit

    # Angular-rate set error: zero once |omega| <= 0.2 deg/s
    omega_limit = np.deg2rad(0.2)
    if abs(omega) <= omega_limit:
        omega_error = 0.0
    else:
        omega_error = omega - np.sign(omega) * omega_limit

    e = np.array(
        [
            r_error[0],
            r_error[1],
            v_error[0],
            v_error[1],
            psi_error,
            omega_error,
        ],
        dtype=float,
    )

    return e

def build_linearized_3dof_matrices(env):
    """
    Builds the linearized 3-DoF model used for the final-approach phase LQR.

    The nonlinear simulator dynamics are:

        rc = sqrt(rx^2 + ry^2 + r0^2)

        rx_dot = vx
        ry_dot = vy

        vx_dot = -mu * rx / rc^3 + n^2 * rx + ax_O
        vy_dot = -mu * ry / rc^3 + ay_O

        psi_dot    = omegaz
        omegaz_dot = alphaz

    Linearization assumption:
        The relative distances rx and ry are small compared with the
        target orbital radius r0.

    Therefore:

        rc ≈ r0

    and since:

        n^2 = mu / r0^3

    the translational dynamics become:

        vx_dot ≈ -n^2 * rx + n^2 * rx + ax_O
               = ax_O

        vy_dot ≈ -n^2 * ry + ay_O

    Linearized state:
        x = [rx, ry, vx, vy, psi, omegaz]

    Linearized orbital-frame input:
        u_O = [ax_O, ay_O, alphaz]

    The LQR is designed in the orbital frame, while the controller later
    converts the command to the body frame before calling env.step().
    """

    n = env.n

    A = np.array(
        [
            # rx_dot = vx
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],

            # ry_dot = vy
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],

            # vx_dot = ax_O
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

            # vy_dot = -n^2 * ry + ay_O
            [0.0, -n**2, 0.0, 0.0, 0.0, 0.0],

            # psi_dot = omegaz
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],

            # omegaz_dot = alphaz
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    B = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    return A, B


def orbital_to_body_action(env, action_orbital, psi):
    """
    Converts an orbital-frame action to the chaser body frame.

    The environment applies body->orbital via rotation_body_to_orbital_2d(psi).
    This uses the inverse (transpose) to convert orbital->body.
    """

    action_orbital = np.asarray(action_orbital, dtype=float)

    if action_orbital.shape != (3,):
        raise ValueError(
            "Orbital-frame action must have shape (3,), "
            "i.e. [ax_O, ay_O, alphaz_O]."
        )

    R_Bc_to_O = env.rotation_body_to_orbital_2d(psi)

    a_orbital = action_orbital[:2]
    alphaz_orbital = action_orbital[2]

    a_body = R_Bc_to_O.T @ a_orbital

    return np.array(
        [
            a_body[0],
            a_body[1],
            alphaz_orbital,
        ],
        dtype=float,
    )


def compute_lqr_gain(A, B, Q, R, verbose=False):
    """
    Solves the continuous-time infinite-horizon LQR problem.

    Cost:
        J = integral( x.T Q x + u.T R u ) dt

    Control law:
        u = -K x
    """

    _check_psd(Q)
    _check_pd(R)
    _check_stabilizable_detectable(A, B, Q)

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)

    if verbose:
        print("Q:\n", Q)
        print("R:\n", R)
        print("K:\n", K)
        print("Closed-loop eigenvalues:\n", np.linalg.eigvals(A - B @ K))

    return K, P


def _sqrt_psd(M, tol=1e-12):
    M = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.clip(eigvals, 0.0, None)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def _check_psd(M, tol=1e-12):
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)
    eigvals = np.linalg.eigvalsh(M)
    if np.min(eigvals) < -tol:
        raise ValueError(
            "Q must be positive semidefinite (found eigenvalue below tolerance)."
        )


def _check_pd(M, tol=1e-12):
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)
    eigvals = np.linalg.eigvalsh(M)
    if np.min(eigvals) <= tol:
        raise ValueError(
            "R must be positive definite (found eigenvalue at/below tolerance)."
        )


def _check_stabilizable_detectable(A, B, Q, tol=1e-9):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)

    n = A.shape[0]
    eigvals = np.linalg.eigvals(A)
    unstable = [lam for lam in eigvals if np.real(lam) >= -tol]

    for lam in unstable:
        mat = np.hstack([A - lam * np.eye(n), B]).astype(complex)
        if np.linalg.matrix_rank(mat, tol=tol) < n:
            raise ValueError(
                f"(A, B) is not stabilizable: rank([A-λI, B]) < n at λ={lam}."
            )

    C = _sqrt_psd(Q, tol=tol)
    for lam in unstable:
        mat = np.vstack([A - lam * np.eye(n), C]).astype(complex)
        if np.linalg.matrix_rank(mat, tol=tol) < n:
            raise ValueError(
                f"(A, Q) is not detectable: rank([A-λI; C]) < n at λ={lam}."
            )


class LQRFinalController:
    """
    Final-approach LQR controller using port-error shaping.

    The LQR gain is designed on the CoM state, but Q is shaped using the
    linearized port-error mapping:

        Q_com = H.T @ Q_port @ H

    During feedback, the nonlinear port error is approximately mapped back
    into an equivalent CoM-state error using H^{-1}.
    """

    def __init__(self, K, H):
        self.K = np.asarray(K, dtype=float)
        self.H = np.asarray(H, dtype=float)

        if self.K.shape != (3, 6):
            raise ValueError("K must have shape (3, 6).")

        if self.H.shape != (6, 6):
            raise ValueError("H must have shape (6, 6).")

    def compute_action(self, env, state):
        if env.phase != "final":
            raise ValueError("LQRFinalController should only be used in final phase.")

        state = np.asarray(state, dtype=float)

        if state.shape != (6,):
            raise ValueError(
                "state must have shape (6,), i.e. [rx, ry, vx, vy, psi, omegaz]."
            )

        e_port = final_error(env, state)

        e_com_equiv = np.linalg.solve(self.H, e_port)

        action_orbital = -self.K @ e_com_equiv

        action_body = orbital_to_body_action(
            env,
            action_orbital,
            psi=state[4],
        )

        return action_body
    

def analyze_saturation(env, atol=1e-12):
    """
    Checks how often the controller command was clipped by the environment.

    Saturation occurs when the raw body-frame action requested by the controller
    differs from the clipped body-frame action applied by the environment.

    The environment logs both:
        action_body_raw
        action_body_clipped
    """

    _, _, actions_body_raw, actions_body_clipped, _, _ = env.get_history_arrays()

    if actions_body_raw.size == 0 or actions_body_clipped.size == 0:
        print("\nActuator saturation analysis")
        print("----------------------------")
        print("No action history available to analyze.")
        return {
            "saturation_fraction_total": 0.0,
            "saturation_fraction_channels": np.zeros(3, dtype=float),
            "max_raw": np.zeros(3, dtype=float),
            "max_clipped": np.zeros(3, dtype=float),
        }

    diff = actions_body_raw - actions_body_clipped

    saturated_per_channel = np.abs(diff) > atol
    saturated_any = np.any(saturated_per_channel, axis=1)

    saturation_fraction_total = np.mean(saturated_any)
    saturation_fraction_channels = np.mean(saturated_per_channel, axis=0)

    max_raw = np.max(np.abs(actions_body_raw), axis=0)
    max_clipped = np.max(np.abs(actions_body_clipped), axis=0)

    print("\nActuator saturation analysis")
    print("----------------------------")
    print(f"Fraction of timesteps with any saturation: {100.0 * saturation_fraction_total:.2f}%")

    print("\nSaturation fraction per channel:")
    print(f"  ax_Bc:     {100.0 * saturation_fraction_channels[0]:.2f}%")
    print(f"  ay_Bc:     {100.0 * saturation_fraction_channels[1]:.2f}%")
    print(f"  alphaz_Bc: {100.0 * saturation_fraction_channels[2]:.2f}%")

    print("\nMaximum absolute raw command:")
    print(f"  ax_Bc:     {max_raw[0]:.6f} m/s^2")
    print(f"  ay_Bc:     {max_raw[1]:.6f} m/s^2")
    print(f"  alphaz_Bc: {max_raw[2]:.6f} rad/s^2")

    print("\nMaximum absolute clipped command:")
    print(f"  ax_Bc:     {max_clipped[0]:.6f} m/s^2")
    print(f"  ay_Bc:     {max_clipped[1]:.6f} m/s^2")
    print(f"  alphaz_Bc: {max_clipped[2]:.6f} rad/s^2")

    return {
        "saturation_fraction_total": saturation_fraction_total,
        "saturation_fraction_channels": saturation_fraction_channels,
        "max_raw": max_raw,
        "max_clipped": max_clipped,
    }
    

def main():
    env = Docking3DoFEnv(
        phase="final",
        integrator="rk4",
        seed=42,
    )

    state = env.reset(mode="testing")

    Q_port = build_Q_final()
    R_final = build_R_final(env)

    A_final, B_final = build_linearized_3dof_matrices(env)

    H = build_port_error_mapping(env)
    Q_com = H.T @ Q_port @ H

    K_final, P_final = compute_lqr_gain(
        A=A_final,
        B=B_final,
        Q=Q_com,
        R=R_final,
        verbose=True,
    )

    controller = LQRFinalController(
        K=K_final,
        H=H,
    )

    done = False

    while not done:
        action = controller.compute_action(env, state)
        state, reward, done, info = env.step(action)

    print("Termination reason:", info["termination_reason"])
    print("Success:", info["success"])
    print("Failure:", info["failure"])
    print("Final state:", state)

    print("Final port relative position:", info["port_relative_position"])
    print("Final port relative velocity:", info["port_relative_velocity"])
    print("Final port distance:", info["port_distance"])
    print("Final port speed:", info["port_speed"])
    print("Final psi error [deg]:", np.rad2deg(info["psi_e"]))
    print("Final omega [deg/s]:", np.rad2deg(info["omega"]))

    saturation_stats = analyze_saturation(env)


if __name__ == "__main__":
    main()