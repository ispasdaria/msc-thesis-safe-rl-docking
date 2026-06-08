import os
import sys
import numpy as np
from scipy.linalg import solve_continuous_are

# Allow imports from the repo's src/ directory when running the script directly.
LQR_ROOT = os.path.dirname(__file__)
while not os.path.exists(os.path.join(LQR_ROOT, "lqr_3dof_closing.py")):
    parent_dir = os.path.dirname(LQR_ROOT)
    if parent_dir == LQR_ROOT:
        break
    LQR_ROOT = parent_dir

SRC_PATH = os.path.abspath(os.path.join(LQR_ROOT, "..", ".."))
if LQR_ROOT not in sys.path:
    sys.path.insert(0, LQR_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.core import Docking3DoFEnv


def build_Q_closing():
    """
    Builds the tuned state weighting matrix Q for the closing-phase LQR.

    The closing phase objective is:

        sampled P2 state  --->  sampled P3 target state

    The matrix is first based on Bryson-style normalization using physical
    design scales. These scales are not terminal tolerances; they represent
    the error magnitudes at which the controller should begin penalizing each
    state significantly.

    The final tuned matrix also includes the selected global Q scale and
    per-state multipliers obtained from the closing-phase Q/R sweep.
    """

    closing_design_scale = np.array(
        [
            100.0,              # rx error scale [m]
            5.0,                # ry error scale [m]
            0.02,               # vx error scale [m/s]
            0.01,               # vy error scale [m/s]
            np.deg2rad(2.0),    # psi error scale [rad]
            np.deg2rad(0.05),   # omegaz error scale [rad/s]
        ],
        dtype=float,
    )

    importance = np.array(
        [
            1.0,     # rx
            4.0,     # ry
            1.0,     # vx
            2.0,     # vy
            1.0,     # psi
            0.25,    # omegaz
        ],
        dtype=float,
    )

    # Selected tuning from the closing-phase sweep.
    q_scale = 6.5

    q_multipliers = np.array(
        [
            3.5,     # rx
            6.5,     # ry
            25.0,    # vx
            6.5,     # vy
            2.0,     # psi
            0.5,     # omegaz
        ],
        dtype=float,
    )

    Q_closing = np.diag(
        q_scale * q_multipliers * importance / closing_design_scale**2
    )

    return Q_closing


def build_R_closing(env):
    """
    Builds the tuned input weighting matrix R for the closing-phase LQR.

    The LQR is designed in the target-centered orbital frame.

    LQR input:
        u_O = [ax_O, ay_O, alphaz]

    The matrix is first based on Bryson-style control-effort scales, selected
    as fractions of the available acceleration limits. The final matrix also
    includes the selected global R scale and per-input multipliers from the
    closing-phase Q/R sweep.
    """

    control_design_scale = np.array(
        [
            0.75 * env.ax_max,
            0.75 * env.ay_max,
            (2.0 / 3.0) * env.alphaz_max,
        ],
        dtype=float,
    )

    control_importance = np.array(
        [
            1.0,  # ax_O
            1.0,  # ay_O
            1.0,  # alphaz
        ],
        dtype=float,
    )

    # Selected tuning from the closing-phase sweep.
    r_scale = 0.25

    r_multipliers = np.array(
        [
            1.0,  # ax_O
            1.0,  # ay_O
            5.0,  # alphaz
        ],
        dtype=float,
    )

    R_closing = np.diag(
        r_scale * r_multipliers * control_importance / control_design_scale**2
    )

    return R_closing


def closing_error(env, x, x_ref):
    """
    Computes the closing-phase tracking error.

    State order:
        x = [rx, ry, vx, vy, psi, omegaz]

    Reference:
        x_ref = sampled P3 target state

    The attitude error is wrapped to avoid discontinuities around +/- pi.
    """

    x = np.asarray(x, dtype=float)
    x_ref = np.asarray(x_ref, dtype=float)

    e = x - x_ref
    e[4] = env.wrap_to_pi(x[4] - x_ref[4])

    return e


def build_linearized_3dof_matrices(env):
    """
    Builds the linearized 3-DoF model used for the closing-phase LQR.

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


def _sqrt_psd(M, tol=1e-12):
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)

    eigvals, eigvecs = np.linalg.eigh(M)

    if np.min(eigvals) < -tol:
        raise ValueError("Matrix is not positive semidefinite.")

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

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix.")

    n = A.shape[0]

    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError("B must be a 2D matrix with the same number of rows as A.")

    if Q.shape != (n, n):
        raise ValueError("Q must have shape (n, n).")

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
        J = integral( e.T Q e + u.T R u ) dt

    Riccati equation:
        A.T P + P A - P B R^-1 B.T P + Q = 0

    Feedback gain:
        K = R^-1 B.T P

    Control law:
        u = -K e
    """

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    m = B.shape[1]

    if R.shape != (m, m):
        raise ValueError(f"R must have shape {(m, m)}, got {R.shape}.")

    _check_psd(Q)
    _check_pd(R)
    _check_stabilizable_detectable(A, B, Q)

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)

    cl_eigs = np.linalg.eigvals(A - B @ K)

    expected_shape = (B.shape[1], A.shape[0])

    if K.shape != expected_shape:
        raise ValueError(f"K has wrong shape: expected {expected_shape}, got {K.shape}.")

    if not np.all(np.isfinite(K)):
        raise ValueError("LQR gain K contains non-finite values.")

    if not np.all(np.real(cl_eigs) < 0.0):
        raise ValueError(f"Closed-loop system is not stable. Eigenvalues: {cl_eigs}")

    if verbose:
        print("Q_closing:\n", Q)
        print("R_closing:\n", R)
        print("K_closing:\n", K)
        print("Closed-loop eigenvalues:\n", cl_eigs)

    return K, P


class LQRClosingController:
    """
    Closing-phase LQR controller.

    The controller assumes a gain matrix K has already been computed from
    the linearized dynamics and the chosen Q/R matrices.

    Control law:

        u = -K (x - x_ref)

    where x_ref is the sampled P3 target.
    """

    def __init__(self, K, v_fast=-0.08, d_fast=100.0, d_slow=10.0):
        self.K = np.asarray(K, dtype=float)

        self.v_fast = float(v_fast)
        self.d_fast = float(d_fast)
        self.d_slow = float(d_slow)

        if self.K.shape != (3, 6):
            raise ValueError("K must have shape (3, 6).")

        if self.d_fast <= self.d_slow:
            raise ValueError("d_fast must be greater than d_slow.")

    def compute_action(self, env, state):
        if env.phase != "closing":
            raise ValueError("LQRClosingController should only be used in closing phase.")

        if env.p3_target is None:
            raise RuntimeError(
                "env.p3_target is None. Call env.reset() in closing phase before using the controller."
            )

        state = np.asarray(state, dtype=float)

        if state.shape != (6,):
            raise ValueError(
                "state must have shape (6,), i.e. [rx, ry, vx, vy, psi, omegaz]."
            )

        x_ref_closing = env.p3_target.copy()

        delta_rx = abs(state[0] - x_ref_closing[0])

        v_p3 = x_ref_closing[2]

        if delta_rx >= self.d_fast:
            vx_ref = self.v_fast
        elif delta_rx <= self.d_slow:
            vx_ref = v_p3
        else:
            alpha = (delta_rx - self.d_slow) / (self.d_fast - self.d_slow)
            vx_ref = (1.0 - alpha) * v_p3 + alpha * self.v_fast

        x_ref_closing[2] = vx_ref

        e = closing_error(env, state, x_ref_closing)

        action_orbital = -self.K @ e

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
        phase="closing",
        integrator="rk4",
        seed=42,
    )

    state = env.reset(mode="testing")

    Q_closing = build_Q_closing()
    R_closing = build_R_closing(env)

    A_closing, B_closing = build_linearized_3dof_matrices(env)

    K_closing, P_closing = compute_lqr_gain(
        A=A_closing,
        B=B_closing,
        Q=Q_closing,
        R=R_closing,
        verbose=True,
    )

    controller = LQRClosingController(K_closing)

    done = False

    while not done:
        action = controller.compute_action(env, state)
        state, reward, done, info = env.step(action)

    print("Termination reason:", info["termination_reason"])
    print("Success:", info["success"])
    print("Failure:", info["failure"])
    print("Final state:", state)
    print("P3 target:", env.p3_target)
    print("Final error:", closing_error(env, state, env.p3_target))

    saturation_stats = analyze_saturation(env)


if __name__ == "__main__":
    main()

