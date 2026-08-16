import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Allow imports from the repo's src/ directory when running pytest from the root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.core import Docking3DoFEnv


def plot_rollout(env, title_prefix):
    times, states, actions_raw, actions_clipped, actions_orbital = (
        env.get_history_arrays()
    )

    rx = states[:, 0]
    ry = states[:, 1]
    vx = states[:, 2]
    vy = states[:, 3]
    psi = np.rad2deg(states[:, 4])
    omegaz = np.rad2deg(states[:, 5])

    plt.close("all")

    plt.figure()
    plt.plot(rx, ry)
    plt.xlabel("rx [m]")
    plt.ylabel("ry [m]")
    plt.title(f"{title_prefix}: trajectory")
    plt.grid(True)

    plt.figure()
    plt.plot(times, rx)
    plt.xlabel("Time [s]")
    plt.ylabel("rx [m]")
    plt.title(f"{title_prefix}: rx over time")
    plt.grid(True)

    plt.figure()
    plt.plot(times, ry)
    plt.xlabel("Time [s]")
    plt.ylabel("ry [m]")
    plt.title(f"{title_prefix}: ry over time")
    plt.grid(True)

    plt.figure()
    plt.plot(times, vx)
    plt.xlabel("Time [s]")
    plt.ylabel("vx [m/s]")
    plt.title(f"{title_prefix}: vx over time")
    plt.grid(True)

    plt.figure()
    plt.plot(times, vy)
    plt.xlabel("Time [s]")
    plt.ylabel("vy [m/s]")
    plt.title(f"{title_prefix}: vy over time")
    plt.grid(True)

    plt.figure()
    plt.plot(times, psi)
    plt.xlabel("Time [s]")
    plt.ylabel("psi [deg]")
    plt.title(f"{title_prefix}: psi over time")
    plt.grid(True)

    plt.figure()
    plt.plot(times, omegaz)
    plt.xlabel("Time [s]")
    plt.ylabel("omegaz [deg/s]")
    plt.title(f"{title_prefix}: omegaz over time")
    plt.grid(True)

    plt.show()


def run_accelerate_toward_docking_point():
    """
    Test 1:
    Start from nominal closing condition.
    Apply a constant body-frame acceleration that should produce
    negative orbital x-acceleration, so rx should decrease faster.
    """

    env = Docking3DoFEnv(
        phase="closing",
        max_time=500.0,
        seed=42,
    )

    state = env.reset(mode="testing")

    # Because at psi = 0:
    # ax_O = -ax_Bc
    # Therefore positive ax_Bc accelerates toward smaller rx.
    action = np.array([0.001, 0.0, 0.0])

    done = False

    while not done:
        if env.time < 20.0:
            action = np.array([0.001, 0.0, 0.0])
        else:
            action = np.zeros(3)

        state, done, info = env.step(action)

    times, states, actions_raw, actions_clipped, actions_orbital = (
        env.get_history_arrays()
    )

    print("Initial state:", states[0])
    print("Final state:", states[-1])
    print("Initial rx:", states[0, 0])
    print("Final rx:", states[-1, 0])
    print("Initial vx:", states[0, 2])
    print("Final vx:", states[-1, 2])
    print("Termination reason:", info.get("termination_reason"))
    print("Failure reason:", info.get("failure_reason"))
    print("Violated constraints:", info.get("violated_constraints"))
    print("Success:", info.get("success"))
    print("Failure:", info.get("failure"))
    print("Final time:", times[-1])

    # Main check
    assert states[-1, 0] < states[0, 0], "rx did not decrease."

    plot_rollout(env, "Manual control test 1")

def run_test_2_lateral_correction():
    """
    Test 2:
    Start with positive ry.
    Apply body-frame ay_Bc > 0, which should create ay_O < 0
    at nominal alignment psi = 0.

    Expected:
        ry should decrease
        abs(ry) should decrease
        vy should become negative
    """

    env = Docking3DoFEnv(
        phase="closing",
        max_time=200.0,
        seed=42,
    )

    initial_state = np.array(
        [
            1700.0,          # rx [m]
            20.0,            # ry [m], positive lateral offset
            -0.10,           # vx [m/s]
            0.0,             # vy [m/s]
            0.0,             # psi [rad]
            0.0,             # omegaz [rad/s]
        ],
        dtype=float,
    )

    state = env.reset(initial_state=initial_state, mode="nominal")

    done = False

    while not done:
        if env.time < 20.0:
            # At psi = 0:
            # ay_O = -ay_Bc
            # positive ay_Bc should push ry downward.
            action = np.array([0.0, 0.001, 0.0])
        else:
            action = np.zeros(3)

        state, done, info = env.step(action)

    times, states, actions_raw, actions_clipped, actions_orbital = (
        env.get_history_arrays()
    )

    initial_ry = states[0, 1]
    final_ry = states[-1, 1]

    print("Initial state:", states[0])
    print("Final state:", states[-1])
    print("Initial ry:", initial_ry)
    print("Final ry:", final_ry)
    print("Initial |ry|:", abs(initial_ry))
    print("Final |ry|:", abs(final_ry))
    print("Initial vy:", states[0, 3])
    print("Final vy:", states[-1, 3])
    print("Termination reason:", info.get("termination_reason"))
    print("Failure reason:", info.get("failure_reason"))
    print("Violated constraints:", info.get("violated_constraints"))
    print("Success:", info.get("success"))
    print("Failure:", info.get("failure"))
    print("Final time:", times[-1])

    assert abs(final_ry) < abs(initial_ry), "abs(ry) did not decrease."

    plot_rollout(env, "Manual control test 2")

def run_test_3_attitude_correction():
    """
    Test 3:
    Start with positive psi.
    Apply negative angular acceleration.
    Check that abs(psi) decreases.
    """

    env = Docking3DoFEnv(
        phase="closing",
        max_time=100.0,
        seed=42,
    )

    initial_state = np.array(
        [
            1700.0,              # rx [m]
            0.0,                 # ry [m]
            -0.10,               # vx [m/s]
            0.0,                 # vy [m/s]
            np.deg2rad(5.0),     # psi [rad], positive attitude error
            0.0,                 # omegaz [rad/s]
        ],
        dtype=float,
    )

    state = env.reset(initial_state=initial_state, mode="nominal")

    done = False

    while not done:
        if env.time < 10.0:
            # Negative angular acceleration should reduce positive psi.
            action = np.array([0.0, 0.0, -0.0005])
        else:
            action = np.zeros(3)

        state, done, info = env.step(action)

    times, states, actions_raw, actions_clipped, actions_orbital = (
        env.get_history_arrays()
    )

    initial_psi = states[0, 4]
    final_psi = states[-1, 4]

    idx_20 = np.argmin(np.abs(times - 20.0))
    psi_20 = states[idx_20, 4]

    print("Initial state:", states[0])
    print("Final state:", states[-1])
    print("Initial psi [deg]:", np.rad2deg(initial_psi))
    print("Psi at 20 s [deg]:", np.rad2deg(psi_20))
    print("Final psi [deg]:", np.rad2deg(final_psi))
    print("Initial |psi| [deg]:", abs(np.rad2deg(initial_psi)))
    print("|psi| at 20 s [deg]:", abs(np.rad2deg(psi_20)))
    print("Final |psi| [deg]:", abs(np.rad2deg(final_psi)))
    print("Initial omegaz [deg/s]:", np.rad2deg(states[0, 5]))
    print("Final omegaz [deg/s]:", np.rad2deg(states[-1, 5]))
    print("Termination reason:", info.get("termination_reason"))
    print("Failure reason:", info.get("failure_reason"))
    print("Violated constraints:", info.get("violated_constraints"))
    print("Success:", info.get("success"))
    print("Failure:", info.get("failure"))
    print("Final time:", times[-1])

    assert abs(psi_20) < abs(initial_psi), \
        "abs(psi) did not decrease after attitude correction."

    plot_rollout(env, "Manual control test 3")

def run_test_4_axial_braking():
    """
    Test 4:
    If the chaser is moving toward the docking point too fast,
    apply acceleration that makes vx less negative.

    At psi = 0:
        ax_O = -ax_Bc

    To brake negative vx, we want ax_O > 0,
    so we command ax_Bc < 0.
    """

    env = Docking3DoFEnv(
        phase="closing",
        max_time=200.0,
        seed=42,
    )

    initial_state = np.array(
        [
            1700.0,          # rx [m]
            0.0,             # ry [m]
            -0.20,           # vx [m/s], fast closing speed
            0.0,             # vy [m/s]
            0.0,             # psi [rad]
            0.0,             # omegaz [rad/s]
        ],
        dtype=float,
    )

    state = env.reset(initial_state=initial_state, mode="nominal")

    done = False

    while not done:
        if env.time < 20.0:
            # Negative ax_Bc gives positive ax_O at psi = 0.
            action = np.array([-0.001, 0.0, 0.0])
        else:
            action = np.zeros(3)

        state, done, info = env.step(action)

    times, states, *_ = env.get_history_arrays()

    initial_vx = states[0, 2]
    final_vx = states[-1, 2]

    print("Initial state:", states[0])
    print("Final state:", states[-1])
    print("Initial vx:", initial_vx)
    print("Final vx:", final_vx)
    print("Termination reason:", info.get("termination_reason"))
    print("Violated constraints:", info.get("violated_constraints"))
    print("Success:", info.get("success"))
    print("Failure:", info.get("failure"))
    print("Final time:", times[-1])

    assert abs(final_vx) < abs(initial_vx), \
        "Axial braking failed: |vx| did not decrease."

    plot_rollout(env, "Manual control test 4: axial braking")

def run_test_5_negative_ry_lateral_correction():
    """
    Test 5:
    Start with negative ry.
    Apply acceleration that should move ry upward toward zero.

    At psi = 0:
        ay_O = -ay_Bc

    If ry < 0, we want ay_O > 0,
    so we command ay_Bc < 0.
    """

    env = Docking3DoFEnv(
        phase="closing",
        max_time=200.0,
        seed=42,
    )

    initial_state = np.array(
        [
            1700.0,          # rx [m]
            -20.0,           # ry [m], negative lateral offset
            -0.10,           # vx [m/s]
            0.0,             # vy [m/s]
            0.0,             # psi [rad]
            0.0,             # omegaz [rad/s]
        ],
        dtype=float,
    )

    state = env.reset(initial_state=initial_state, mode="nominal")

    done = False

    while not done:
        if env.time < 20.0:
            # Negative ay_Bc gives positive ay_O at psi = 0.
            action = np.array([0.0, -0.001, 0.0])
        else:
            action = np.zeros(3)

        state, done, info = env.step(action)

    times, states, *_ = env.get_history_arrays()

    initial_ry = states[0, 1]
    final_ry = states[-1, 1]

    idx_50 = np.argmin(np.abs(times - 50.0))
    ry_50 = states[idx_50, 1]

    print("Initial state:", states[0])
    print("Final state:", states[-1])
    print("Initial ry:", initial_ry)
    print("ry at 50 s:", ry_50)
    print("Final ry:", final_ry)
    print("Initial |ry|:", abs(initial_ry))
    print("|ry| at 50 s:", abs(ry_50))
    print("Final |ry|:", abs(final_ry))
    print("Initial vy:", states[0, 3])
    print("Final vy:", states[-1, 3])
    print("Termination reason:", info.get("termination_reason"))
    print("Violated constraints:", info.get("violated_constraints"))
    print("Success:", info.get("success"))
    print("Failure:", info.get("failure"))
    print("Final time:", times[-1])

    assert abs(ry_50) < abs(initial_ry), \
        "Negative-ry correction failed: |ry| did not decrease."

    plot_rollout(env, "Manual control test 5: negative-ry correction")

def run_test_6_negative_psi_attitude_correction():
    """
    Test 6:
    Start with negative psi.
    Apply positive angular acceleration.

    Since:
        omegaz_dot = alphaz

    If psi < 0, positive alphaz should make omegaz positive,
    and psi should move toward zero.
    """

    env = Docking3DoFEnv(
        phase="closing",
        max_time=20.0,
        seed=42,
    )

    initial_state = np.array(
        [
            1700.0,              # rx [m]
            0.0,                 # ry [m]
            -0.10,               # vx [m/s]
            0.0,                 # vy [m/s]
            np.deg2rad(-5.0),    # psi [rad]
            0.0,                 # omegaz [rad/s]
        ],
        dtype=float,
    )

    state = env.reset(initial_state=initial_state, mode="nominal")

    done = False

    while not done:
        if env.time < 10.0:
            action = np.array([0.0, 0.0, 0.0005])
        else:
            action = np.zeros(3)

        state, done, info = env.step(action)

    times, states, *_ = env.get_history_arrays()

    initial_psi = states[0, 4]

    idx_20 = np.argmin(np.abs(times - 20.0))
    psi_20 = states[idx_20, 4]

    final_psi = states[-1, 4]

    print("Initial state:", states[0])
    print("Final state:", states[-1])
    print("Initial psi [deg]:", np.rad2deg(initial_psi))
    print("Psi at 20 s [deg]:", np.rad2deg(psi_20))
    print("Final psi [deg]:", np.rad2deg(final_psi))
    print("Initial |psi| [deg]:", abs(np.rad2deg(initial_psi)))
    print("|psi| at 20 s [deg]:", abs(np.rad2deg(psi_20)))
    print("Final |psi| [deg]:", abs(np.rad2deg(final_psi)))
    print("Initial omegaz [deg/s]:", np.rad2deg(states[0, 5]))
    print("Final omegaz [deg/s]:", np.rad2deg(states[-1, 5]))
    print("Termination reason:", info.get("termination_reason"))
    print("Violated constraints:", info.get("violated_constraints"))
    print("Success:", info.get("success"))
    print("Failure:", info.get("failure"))
    print("Final time:", times[-1])

    assert abs(psi_20) < abs(initial_psi), \
        "Negative-psi correction failed: |psi| did not decrease."

    plot_rollout(env, "Manual control test 6: negative-psi correction")

def run_test_7_angular_rate_damping():
    """
    Test 7:
    Start with positive angular rate.
    Apply negative angular acceleration to reduce omegaz.

    This checks angular-rate damping, not just attitude correction.
    """

    env = Docking3DoFEnv(
        phase="closing",
        max_time=50.0,
        seed=42,
    )

    initial_state = np.array(
        [
            1700.0,                 # rx [m]
            0.0,                    # ry [m]
            -0.10,                  # vx [m/s]
            0.0,                    # vy [m/s]
            0.0,                    # psi [rad]
            np.deg2rad(0.05),       # omegaz [rad/s], positive angular rate
        ],
        dtype=float,
    )

    state = env.reset(initial_state=initial_state, mode="nominal")

    done = False

    while not done:
        if env.time < 10.0:
            action = np.array([0.0, 0.0, -0.00005])
        else:
            action = np.zeros(3)

        state, done, info = env.step(action)

    times, states, *_ = env.get_history_arrays()

    initial_omega = states[0, 5]

    idx_20 = np.argmin(np.abs(times - 20.0))
    omega_20 = states[idx_20, 5]

    final_omega = states[-1, 5]

    print("Initial state:", states[0])
    print("Final state:", states[-1])
    print("Initial omegaz [deg/s]:", np.rad2deg(initial_omega))
    print("Omegaz at 20 s [deg/s]:", np.rad2deg(omega_20))
    print("Final omegaz [deg/s]:", np.rad2deg(final_omega))
    print("Initial |omegaz| [deg/s]:", abs(np.rad2deg(initial_omega)))
    print("|omegaz| at 20 s [deg/s]:", abs(np.rad2deg(omega_20)))
    print("Final |omegaz| [deg/s]:", abs(np.rad2deg(final_omega)))
    print("Initial psi [deg]:", np.rad2deg(states[0, 4]))
    print("Final psi [deg]:", np.rad2deg(states[-1, 4]))
    print("Termination reason:", info.get("termination_reason"))
    print("Violated constraints:", info.get("violated_constraints"))
    print("Success:", info.get("success"))
    print("Failure:", info.get("failure"))
    print("Final time:", times[-1])

    assert abs(omega_20) < abs(initial_omega), \
        "Angular-rate damping failed: |omegaz| did not decrease."

    plot_rollout(env, "Manual control test 7: angular-rate damping")

def run_test_8_wrong_direction_lateral():
    """
    Test 8:
    Start with positive ry.
    Deliberately apply the wrong lateral acceleration.

    If ry > 0, the correct command is ay_Bc > 0.
    Here we command ay_Bc < 0, so |ry| should increase.
    """

    env = Docking3DoFEnv(
        phase="closing",
        max_time=100.0,
        seed=42,
    )

    initial_state = np.array(
        [
            1700.0,
            20.0,
            -0.10,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    state = env.reset(initial_state=initial_state, mode="nominal")

    done = False

    while not done:
        if env.time < 20.0:
            # Wrong sign for positive ry.
            action = np.array([0.0, -0.001, 0.0])
        else:
            action = np.zeros(3)

        state, done, info = env.step(action)

    times, states, *_ = env.get_history_arrays()

    initial_ry = states[0, 1]

    idx_50 = np.argmin(np.abs(times - 50.0))
    ry_50 = states[idx_50, 1]

    print("Initial state:", states[0])
    print("Final state:", states[-1])
    print("Initial ry:", initial_ry)
    print("ry at 50 s:", ry_50)
    print("Initial |ry|:", abs(initial_ry))
    print("|ry| at 50 s:", abs(ry_50))
    print("Termination reason:", info.get("termination_reason"))
    print("Violated constraints:", info.get("violated_constraints"))
    print("Success:", info.get("success"))
    print("Failure:", info.get("failure"))
    print("Final time:", times[-1])

    assert abs(ry_50) > abs(initial_ry), \
        "Wrong-direction lateral test failed: |ry| did not increase."

    plot_rollout(env, "Manual control test 8: wrong-direction lateral")

def run_test_9_wrong_direction_attitude():
    """
    Test 9:
    Start with positive psi.
    Deliberately apply positive angular acceleration.

    If psi > 0, the correct command is negative alphaz.
    Here we command positive alphaz, so |psi| should increase.
    """

    env = Docking3DoFEnv(
        phase="closing",
        max_time=50.0,
        seed=42,
    )

    initial_state = np.array(
        [
            1700.0,
            0.0,
            -0.10,
            0.0,
            np.deg2rad(5.0),
            0.0,
        ],
        dtype=float,
    )

    state = env.reset(initial_state=initial_state, mode="nominal")

    done = False

    while not done:
        if env.time < 10.0:
            # Wrong sign for positive psi.
            action = np.array([0.0, 0.0, 0.0005])
        else:
            action = np.zeros(3)

        state, done, info = env.step(action)

    times, states, *_ = env.get_history_arrays()

    initial_psi = states[0, 4]

    idx_20 = np.argmin(np.abs(times - 20.0))
    psi_20 = states[idx_20, 4]

    print("Initial state:", states[0])
    print("Final state:", states[-1])
    print("Initial psi [deg]:", np.rad2deg(initial_psi))
    print("Psi at 20 s [deg]:", np.rad2deg(psi_20))
    print("Initial |psi| [deg]:", abs(np.rad2deg(initial_psi)))
    print("|psi| at 20 s [deg]:", abs(np.rad2deg(psi_20)))
    print("Termination reason:", info.get("termination_reason"))
    print("Violated constraints:", info.get("violated_constraints"))
    print("Success:", info.get("success"))
    print("Failure:", info.get("failure"))
    print("Final time:", times[-1])

    assert abs(psi_20) > abs(initial_psi), \
        "Wrong-direction attitude test failed: |psi| did not increase."

    plot_rollout(env, "Manual control test 9: wrong-direction attitude")


if __name__ == "__main__":
    # run_test_5_negative_ry_lateral_correction()
    # run_test_6_negative_psi_attitude_correction()
    # run_test_7_angular_rate_damping()
    # run_test_8_wrong_direction_lateral()
    run_test_9_wrong_direction_attitude()
