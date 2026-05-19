# Allow imports from the repo's src/ directory when running pytest from the root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.core import Docking3DoFEnv

class DockingMissionRunner:
    """
    Utility class for running:

        1. closing-only episodes
        2. final-approach-only episodes
        3. full two-phase missions

    The environments remain phase-specific, so each phase can have its own
    controller, reward, history, and performance metrics.
    """

    def __init__(self, env_kwargs=None):
        if env_kwargs is None:
            env_kwargs = {}

        self.env_kwargs = env_kwargs

        self.closing_env = Docking3DoFEnv(
            phase="closing",
            **env_kwargs,
        )

        self.final_env = Docking3DoFEnv(
            phase="final",
            **env_kwargs,
        )

        self.last_results = None

    def run_phase(
        self,
        phase,
        controller,
        mode="training",
        initial_state=None,
        max_steps=None,
    ):
        """
        Runs one episode of one selected phase.

        Parameters
        ----------
        phase : str
            "closing" or "final".

        controller : callable
            Function that maps state -> action.

        mode : str
            "nominal", "testing", or "training".

        initial_state : np.ndarray or None
            If None:
                closing samples from X_P2 and samples a P3 target.
                final samples from X_P3.
            If provided:
                the phase starts from this state.

        max_steps : int or None
            Optional extra step cap for debugging.
        """

        if phase == "closing":
            env = self.closing_env
        elif phase == "final":
            env = self.final_env
        else:
            raise ValueError("phase must be either 'closing' or 'final'.")

        state = env.reset(
            initial_state=initial_state,
            mode=mode,
        )

        done = False
        step_count = 0
        total_reward = 0.0
        final_info = None

        while not done:
            action = controller(state)

            state, reward, done, info = env.step(action)

            total_reward += reward
            step_count += 1
            final_info = info

            if max_steps is not None and step_count >= max_steps:
                done = True
                final_info = info.copy()
                final_info["success"] = False
                final_info["failure"] = False
                final_info["time_limit"] = True
                final_info["termination_reason"] = "runner_max_steps"

        result = {
            "phase": phase,
            "success": final_info.get("success", False),
            "failure": final_info.get("failure", False),
            "time_limit": final_info.get("time_limit", False),
            "termination_reason": final_info.get("termination_reason", None),
            "terminal_state": state.copy(),
            "total_reward": total_reward,
            "num_steps": step_count,
            "final_info": final_info,
            "history": env.history.copy(),
        }

        return result

    def run_many_phase_episodes(
        self,
        phase,
        controller,
        num_episodes,
        mode="training",
        max_steps=None,
    ):
        """
        Runs many independent episodes for one phase only.

        This is useful for training/evaluating one controller at a time.

        For phase="closing":
            each episode samples a new P2 and a new P3 target.

        For phase="final":
            each episode samples a new P3 initial state,
            while the docking target remains X_dock.
        """

        results = []

        for episode_idx in range(num_episodes):
            result = self.run_phase(
                phase=phase,
                controller=controller,
                mode=mode,
                initial_state=None,
                max_steps=max_steps,
            )

            result["episode_idx"] = episode_idx
            results.append(result)

        summary = self.summarize_results(results)

        return results, summary

    def run_mission(
        self,
        closing_controller,
        final_controller,
        mode="testing",
        max_steps_closing=None,
        max_steps_final=None,
    ):
        """
        Runs the full mission as two connected sub-episodes:

            closing: sampled P2 -> sampled P3 target
            final: actual terminal state from closing -> docking

        The two histories and rewards remain separate.
        """

        closing_result = self.run_phase(
            phase="closing",
            controller=closing_controller,
            mode=mode,
            initial_state=None,
            max_steps=max_steps_closing,
        )

        if not closing_result["success"]:
            mission_result = {
                "mission_success": False,
                "failed_phase": "closing",
                "closing_result": closing_result,
                "final_result": None,
            }

            self.last_results = mission_result
            return mission_result

        final_result = self.run_phase(
            phase="final",
            controller=final_controller,
            mode=mode,
            initial_state=closing_result["terminal_state"],
            max_steps=max_steps_final,
        )

        mission_success = final_result["success"]

        mission_result = {
            "mission_success": mission_success,
            "failed_phase": None if mission_success else "final",
            "closing_result": closing_result,
            "final_result": final_result,
            "transition_state": closing_result["terminal_state"].copy(),
        }

        self.last_results = mission_result
        return mission_result

    def run_many_missions(
        self,
        closing_controller,
        final_controller,
        num_missions,
        mode="testing",
        max_steps_closing=None,
        max_steps_final=None,
    ):
        """
        Runs many full two-phase missions.
        """

        mission_results = []

        for mission_idx in range(num_missions):
            result = self.run_mission(
                closing_controller=closing_controller,
                final_controller=final_controller,
                mode=mode,
                max_steps_closing=max_steps_closing,
                max_steps_final=max_steps_final,
            )

            result["mission_idx"] = mission_idx
            mission_results.append(result)

        return mission_results

    def summarize_results(self, results):
        """
        Computes simple performance statistics for one-phase evaluations.
        """

        num_episodes = len(results)

        if num_episodes == 0:
            return {
                "num_episodes": 0,
                "success_rate": 0.0,
                "failure_rate": 0.0,
                "timeout_rate": 0.0,
                "mean_total_reward": 0.0,
                "mean_num_steps": 0.0,
            }

        successes = [r["success"] for r in results]
        failures = [r["failure"] for r in results]
        timeouts = [r["time_limit"] for r in results]

        total_rewards = [r["total_reward"] for r in results]
        num_steps = [r["num_steps"] for r in results]

        summary = {
            "num_episodes": num_episodes,
            "success_rate": sum(successes) / num_episodes,
            "failure_rate": sum(failures) / num_episodes,
            "timeout_rate": sum(timeouts) / num_episodes,
            "mean_total_reward": sum(total_rewards) / num_episodes,
            "mean_num_steps": sum(num_steps) / num_episodes,
        }

        return summary