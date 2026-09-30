import numpy as np
import matplotlib.pyplot as plt

if __package__:
    from ._path_setup import ensure_project_imports
else:
    from _path_setup import ensure_project_imports

ensure_project_imports()

from envs.docking3dof.core import Docking3DoFEnv


def run_zero_action_rollout():
    env = Docking3DoFEnv(
        phase="final",
        max_time=3600.0,
        seed=42,
    )

    state = env.reset(mode="training")

    done = False

    while not done:
        state, done, info = env.step(np.zeros(3))

    times, states, actions_raw, actions_clipped, actions_orbital = (
        env.get_history_arrays()
    )

    rx = states[:, 0]
    ry = states[:, 1]
    vx = states[:, 2]
    vy = states[:, 3]
    psi = states[:, 4]
    omegaz = states[:, 5]

    print("Final state:", state)
    print("Termination reason:", info.get("termination_reason"))
    print("Success:", info.get("success"))
    print("Failure:", info.get("failure"))
    print("Final time:", times[-1])

    plt.figure()
    plt.plot(rx, ry)
    plt.xlabel("rx [m]")
    plt.ylabel("ry [m]")
    plt.title("Zero-action rollout: trajectory")
    plt.grid(True)

    plt.figure()
    plt.plot(times, rx)
    plt.xlabel("Time [s]")
    plt.ylabel("rx [m]")
    plt.title("rx over time")
    plt.grid(True)

    plt.figure()
    plt.plot(times, ry)
    plt.xlabel("Time [s]")
    plt.ylabel("ry [m]")
    plt.title("ry over time")
    plt.grid(True)

    plt.figure()
    plt.plot(times, vx)
    plt.xlabel("Time [s]")
    plt.ylabel("vx [m/s]")
    plt.title("vx over time")
    plt.grid(True)

    plt.figure()
    plt.plot(times, vy)
    plt.xlabel("Time [s]")
    plt.ylabel("vy [m/s]")
    plt.title("vy over time")
    plt.grid(True)

    plt.figure()
    plt.plot(times, np.rad2deg(psi))
    plt.xlabel("Time [s]")
    plt.ylabel("psi [deg]")
    plt.title("psi over time")
    plt.grid(True)

    plt.figure()
    plt.plot(times, np.rad2deg(omegaz))
    plt.xlabel("Time [s]")
    plt.ylabel("omegaz [deg/s]")
    plt.title("omegaz over time")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    run_zero_action_rollout()
