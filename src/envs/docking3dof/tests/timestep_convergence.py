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

def run_zero_input_rollout(
    dt,
    integrator,
    initial_state,
    phase="closing",
    max_time=1000.0,
):
    """
    Runs one zero-input simulation for a fixed timestep and integrator.

    The same initial condition is used for all simulations so that only
    the timestep/integrator changes.
    """

    env = Docking3DoFEnv(
        dt=dt,
        max_time=max_time,
        integrator=integrator,
        phase=phase,
        seed=42
    )

    env.reset(
        initial_state=initial_state,
        mode="testing",
    )

    action_body = np.array([
        0.0002,
        0.0001,
        np.deg2rad(0.0001),
    ])

    done = False

    while not done:
        state, reward, done, info = env.step(action_body)

    times, states, actions_body_raw, actions_body_clipped, actions_orbital, rewards = (
        env.get_history_arrays()
    )

    return {
        "dt": dt,
        "integrator": integrator,
        "times": times,
        "states": states,
        "final_state": states[-1],
        "final_info": info,
    }


def interpolate_states_to_reference_time(times, states, reference_times):
    """
    Interpolates a trajectory onto the reference time grid.

    This allows full-trajectory comparison between simulations with different dt.
    """

    interpolated_states = np.zeros((len(reference_times), states.shape[1]))

    for i in range(states.shape[1]):
        interpolated_states[:, i] = np.interp(
            reference_times,
            times,
            states[:, i],
        )

    return interpolated_states


def compute_convergence_errors(results, reference_dt):
    """
    Computes final-state and trajectory errors relative to the smallest dt case.
    """

    reference = results[reference_dt]
    reference_times = reference["times"]
    reference_states = reference["states"]
    reference_final_state = reference["final_state"]

    errors = {}

    for dt, result in results.items():
        states_interp = interpolate_states_to_reference_time(
            times=result["times"],
            states=result["states"],
            reference_times=reference_times,
        )

        final_state = result["final_state"]

        final_position_error = np.linalg.norm(
            final_state[0:2] - reference_final_state[0:2]
        )

        final_velocity_error = np.linalg.norm(
            final_state[2:4] - reference_final_state[2:4]
        )

        final_attitude_error = abs(
            final_state[4] - reference_final_state[4]
        )

        final_angular_rate_error = abs(
            final_state[5] - reference_final_state[5]
        )

        trajectory_position_error = np.max(
            np.linalg.norm(
                states_interp[:, 0:2] - reference_states[:, 0:2],
                axis=1,
            )
        )

        trajectory_velocity_error = np.max(
            np.linalg.norm(
                states_interp[:, 2:4] - reference_states[:, 2:4],
                axis=1,
            )
        )

        errors[dt] = {
            "final_position_error": final_position_error,
            "final_velocity_error": final_velocity_error,
            "final_attitude_error": final_attitude_error,
            "final_angular_rate_error": final_angular_rate_error,
            "max_trajectory_position_error": trajectory_position_error,
            "max_trajectory_velocity_error": trajectory_velocity_error,
        }

    return errors


def print_convergence_table(errors):
    """
    Prints convergence errors in a readable format.
    """

    print("\nTimestep convergence results")
    print("-" * 90)
    print(
        f"{'dt [s]':>8} | "
        f"{'final pos err [m]':>18} | "
        f"{'final vel err [m/s]':>20} | "
        f"{'final psi err [rad]':>20} | "
        f"{'max traj pos err [m]':>22}"
    )
    print("-" * 90)

    for dt in sorted(errors.keys(), reverse=True):
        err = errors[dt]

        print(
            f"{dt:8.3f} | "
            f"{err['final_position_error']:18.6e} | "
            f"{err['final_velocity_error']:20.6e} | "
            f"{err['final_attitude_error']:20.6e} | "
            f"{err['max_trajectory_position_error']:22.6e}"
        )

    print("-" * 90)


def plot_position_trajectories(results, title):
    """
    Plots rx-ry trajectories for all tested timesteps.
    """

    plt.figure()

    for dt, result in results.items():
        states = result["states"]
        rx = states[:, 0]
        ry = states[:, 1]

        plt.plot(rx, ry, label=f"dt = {dt} s")

    plt.xlabel("rx [m]")
    plt.ylabel("ry [m]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_state_histories(results, title_prefix):
    """
    Plots individual state histories for comparison.
    """

    state_labels = [
        "rx [m]",
        "ry [m]",
        "vx [m/s]",
        "vy [m/s]",
        "psi [rad]",
        "omegaz [rad/s]",
    ]

    for state_index, label in enumerate(state_labels):
        plt.figure()

        for dt, result in results.items():
            times = result["times"]
            states = result["states"]

            plt.plot(
                times,
                states[:, state_index],
                label=f"dt = {dt} s",
            )

        plt.xlabel("time [s]")
        plt.ylabel(label)
        plt.title(f"{title_prefix}: {label}")
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    """
    Main convergence-study script.
    """

    # Fixed initial condition.
    # This should be the same for all dt values.
    #
    # State:
    # [rx, ry, vx, vy, psi, omegaz]
    initial_state = np.array([
        1700.0,
        20.0,
        -0.10,
        0.00,
        np.deg2rad(3.0),
        np.deg2rad(0.05),
    ])

    phase = "closing"
    integrator = "rk4"
    max_time = 200.0

    # Start coarse, then reduce dt.
    dts = [4.0, 2.0, 1.0, 0.5, 0.25]

    results = {}

    for dt in dts:
        result = run_zero_input_rollout(
            dt=dt,
            integrator=integrator,
            initial_state=initial_state,
            phase=phase,
            max_time=max_time,
        )

        results[dt] = result

        print(
            f"Finished simulation: integrator={integrator}, "
            f"dt={dt}, "
            f"final_state={result['final_state']}, "
            f"termination={result['final_info'].get('termination_reason')}"
        )

    # Use the smallest timestep as the numerical reference.
    reference_dt = min(dts)

    errors = compute_convergence_errors(
        results=results,
        reference_dt=reference_dt,
    )

    print_convergence_table(errors)

    plot_position_trajectories(
        results=results,
        title=f"Zero-input trajectory convergence using {integrator.upper()}",
    )

    plot_state_histories(
        results=results,
        title_prefix=f"Zero-input timestep convergence using {integrator.upper()}",
    )

    print(
        f"Finished simulation: integrator={integrator}, "
        f"dt={dt}, "
        f"final_time={result['times'][-1]}, "
        f"final_state={result['final_state']}, "
        f"termination={result['final_info'].get('termination_reason')}"
    )


if __name__ == "__main__":
    main()