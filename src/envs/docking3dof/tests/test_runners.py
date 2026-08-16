import os
import sys

import numpy as np
import pytest

# Allow imports from the repo's src/ directory when running pytest from the root.
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.runners import DockingMissionRunner


class ScriptedEnv:
    """Small deterministic environment used to test runner orchestration."""

    def __init__(self, transitions):
        self.transitions = list(transitions)
        self.history = []
        self.reset_calls = []
        self.actions = []
        self.state = None

    def reset(self, initial_state=None, mode="training"):
        self.reset_calls.append(
            {
                "initial_state": (
                    None if initial_state is None else np.asarray(initial_state).copy()
                ),
                "mode": mode,
            }
        )
        if initial_state is None:
            self.state = np.zeros(6, dtype=float)
        else:
            self.state = np.asarray(initial_state, dtype=float).copy()
        self.history = [{"state": self.state.copy()}]
        return self.state.copy()

    def step(self, action):
        if not self.transitions:
            raise AssertionError("Runner requested more scripted transitions than expected.")

        self.actions.append(np.asarray(action, dtype=float).copy())
        next_state, done, info = self.transitions.pop(0)
        self.state = np.asarray(next_state, dtype=float).copy()
        self.history.append(
            {
                "state": self.state.copy(),
                "action": self.actions[-1].copy(),
            }
        )
        return self.state.copy(), bool(done), dict(info)


class ObjectController:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=float)
        self.calls = []

    def compute_action(self, env, state):
        self.calls.append((env, np.asarray(state).copy()))
        return self.action.copy()


def make_runner(closing_env=None, final_env=None):
    runner = DockingMissionRunner()
    if closing_env is not None:
        runner.closing_env = closing_env
    if final_env is not None:
        runner.final_env = final_env
    return runner


def terminal_info(*, success=False, failure=False, time_limit=False, reason=None):
    return {
        "success": success,
        "failure": failure,
        "time_limit": time_limit,
        "termination_reason": reason,
    }


def test_initialization_creates_one_environment_for_each_phase():
    runner = DockingMissionRunner(
        closing_env_kwargs={"dt": 0.5, "seed": 1},
        final_env_kwargs={"dt": 0.25, "seed": 2},
    )

    assert runner.closing_env.phase == "closing"
    assert runner.final_env.phase == "final"
    assert runner.closing_env.dt == 0.5
    assert runner.final_env.dt == 0.25
    assert runner.controllers == {"closing": None, "final": None}
    assert runner.last_results is None


@pytest.mark.parametrize("phase", ["invalid", "", None])
def test_phase_selection_rejects_invalid_phase(phase):
    runner = DockingMissionRunner()

    with pytest.raises(ValueError, match="phase must be either"):
        runner._get_env(phase)

    with pytest.raises(ValueError, match="phase must be either"):
        runner.set_controller(phase, lambda state: np.zeros(3))


def test_set_controller_rejects_none():
    runner = DockingMissionRunner()

    with pytest.raises(ValueError, match="must not be None"):
        runner.set_controller("closing", None)


def test_registered_callable_controller_is_returned_unchanged():
    runner = DockingMissionRunner()
    controller = lambda state: np.ones(3)

    runner.set_controller("closing", controller)

    assert runner._policy_for_phase("closing") is controller


def test_registered_object_controller_receives_environment_and_state():
    env = ScriptedEnv([])
    runner = make_runner(closing_env=env)
    controller = ObjectController([0.1, 0.2, 0.3])
    runner.set_controller("closing", controller)

    policy = runner._policy_for_phase("closing")
    state = np.arange(6, dtype=float)
    action = policy(state)

    assert np.allclose(action, [0.1, 0.2, 0.3])
    assert controller.calls[0][0] is env
    assert np.allclose(controller.calls[0][1], state)


def test_registered_controller_errors_are_reported():
    runner = DockingMissionRunner()

    with pytest.raises(RuntimeError, match="No controller registered"):
        runner._policy_for_phase("closing")

    runner.set_controller("closing", object())
    with pytest.raises(TypeError, match="callable or implement compute_action"):
        runner._policy_for_phase("closing")


def test_run_phase_accumulates_steps_and_history():
    terminal_state = np.arange(6, dtype=float)
    env = ScriptedEnv(
        [
            (np.ones(6), False, terminal_info()),
            (
                terminal_state,
                True,
                terminal_info(success=True, reason="success_P3_target"),
            ),
        ]
    )
    runner = make_runner(closing_env=env)
    observed_states = []

    def controller(state):
        observed_states.append(state.copy())
        return np.array([0.01, 0.0, 0.0])

    result = runner.run_phase(
        phase="closing",
        controller=controller,
        mode="testing",
    )

    assert env.reset_calls[0]["mode"] == "testing"
    assert env.reset_calls[0]["initial_state"] is None
    assert len(observed_states) == 2
    assert result["phase"] == "closing"
    assert result["success"] is True
    assert result["failure"] is False
    assert result["time_limit"] is False
    assert result["termination_reason"] == "success_P3_target"
    assert np.allclose(result["terminal_state"], terminal_state)
    assert result["num_steps"] == 2
    assert len(result["history"]) == 3


def test_run_phase_forwards_an_explicit_initial_state():
    initial_state = np.arange(6, dtype=float)
    env = ScriptedEnv(
        [
            (
                initial_state + 1.0,
                True,
                terminal_info(success=True, reason="success_docking"),
            )
        ]
    )
    runner = make_runner(final_env=env)

    runner.run_phase(
        phase="final",
        controller=lambda state: np.zeros(3),
        mode="nominal",
        initial_state=initial_state,
    )

    assert env.reset_calls[0]["mode"] == "nominal"
    assert np.allclose(env.reset_calls[0]["initial_state"], initial_state)


def test_run_phase_max_steps_sets_runner_timeout_without_mutating_env_info():
    original_info = terminal_info()
    env = ScriptedEnv([(np.ones(6), False, original_info)])
    runner = make_runner(closing_env=env)

    result = runner.run_phase(
        phase="closing",
        controller=lambda state: np.zeros(3),
        max_steps=1,
    )

    assert result["success"] is False
    assert result["failure"] is False
    assert result["time_limit"] is True
    assert result["termination_reason"] == "runner_max_steps"
    assert result["num_steps"] == 1
    assert original_info["time_limit"] is False
    assert original_info["termination_reason"] is None


def test_run_phase_with_registered_uses_the_selected_controller():
    env = ScriptedEnv(
        [
            (
                np.ones(6),
                True,
                terminal_info(success=True, reason="success_P3_target"),
            )
        ]
    )
    runner = make_runner(closing_env=env)
    controller = ObjectController([0.01, 0.0, 0.0])
    runner.set_controller("closing", controller)

    result = runner.run_phase_with_registered("closing", mode="testing")

    assert result["success"] is True
    assert len(controller.calls) == 1
    assert np.allclose(env.actions[0], controller.action)


def test_run_many_phase_episodes_indexes_results_and_computes_summary(monkeypatch):
    runner = DockingMissionRunner()
    scripted_results = [
        {
            "success": True,
            "failure": False,
            "time_limit": False,
            "num_steps": 2,
        },
        {
            "success": False,
            "failure": True,
            "time_limit": False,
            "num_steps": 4,
        },
    ]
    calls = []

    def fake_run_phase(**kwargs):
        calls.append(kwargs)
        return scripted_results[len(calls) - 1].copy()

    monkeypatch.setattr(runner, "run_phase", fake_run_phase)

    results, summary = runner.run_many_phase_episodes(
        phase="closing",
        controller="controller",
        num_episodes=2,
        mode="testing",
        max_steps=7,
    )

    assert [result["episode_idx"] for result in results] == [0, 1]
    assert all(call["initial_state"] is None for call in calls)
    assert all(call["mode"] == "testing" for call in calls)
    assert all(call["max_steps"] == 7 for call in calls)
    assert summary == {
        "num_episodes": 2,
        "success_rate": 0.5,
        "failure_rate": 0.5,
        "timeout_rate": 0.0,
        "mean_num_steps": 3.0,
    }


def test_summarize_results_handles_empty_input():
    runner = DockingMissionRunner()

    assert runner.summarize_results([]) == {
        "num_episodes": 0,
        "success_rate": 0.0,
        "failure_rate": 0.0,
        "timeout_rate": 0.0,
        "mean_num_steps": 0.0,
    }


def test_run_mission_aborts_when_closing_phase_is_unsuccessful(monkeypatch):
    runner = DockingMissionRunner()
    closing_result = {
        "success": False,
        "terminal_state": np.ones(6),
    }
    calls = []

    def fake_run_phase(**kwargs):
        calls.append(kwargs)
        return closing_result

    monkeypatch.setattr(runner, "run_phase", fake_run_phase)

    result = runner.run_mission("closing_controller", "final_controller")

    assert len(calls) == 1
    assert calls[0]["phase"] == "closing"
    assert result["mission_success"] is False
    assert result["failed_phase"] == "closing"
    assert result["closing_result"] is closing_result
    assert result["final_result"] is None
    assert runner.last_results is result


@pytest.mark.parametrize("final_success", [True, False])
def test_run_mission_passes_closing_terminal_state_to_final_phase(
    monkeypatch, final_success
):
    runner = DockingMissionRunner()
    transition_state = np.arange(6, dtype=float)
    closing_result = {"success": True, "terminal_state": transition_state}
    final_result = {"success": final_success, "terminal_state": np.ones(6)}
    calls = []

    def fake_run_phase(**kwargs):
        calls.append(kwargs)
        return closing_result if kwargs["phase"] == "closing" else final_result

    monkeypatch.setattr(runner, "run_phase", fake_run_phase)

    result = runner.run_mission(
        "closing_controller",
        "final_controller",
        mode="testing",
        max_steps_closing=10,
        max_steps_final=20,
    )

    assert len(calls) == 2
    assert calls[0]["initial_state"] is None
    assert np.allclose(calls[1]["initial_state"], transition_state)
    assert calls[0]["max_steps"] == 10
    assert calls[1]["max_steps"] == 20
    assert result["mission_success"] is final_success
    assert result["failed_phase"] == (None if final_success else "final")
    assert np.allclose(result["transition_state"], transition_state)
    assert result["transition_state"] is not transition_state
    assert runner.last_results is result


def test_run_mission_with_registered_adapts_both_object_controllers(monkeypatch):
    runner = DockingMissionRunner()
    closing_controller = ObjectController([0.01, 0.0, 0.0])
    final_controller = ObjectController([0.0, 0.01, 0.0])
    runner.set_controller("closing", closing_controller)
    runner.set_controller("final", final_controller)
    captured = {}

    def fake_run_mission(**kwargs):
        captured.update(kwargs)
        return {"mission_success": True}

    monkeypatch.setattr(runner, "run_mission", fake_run_mission)

    result = runner.run_mission_with_registered(
        mode="nominal",
        max_steps_closing=3,
        max_steps_final=4,
    )

    assert result == {"mission_success": True}
    assert captured["mode"] == "nominal"
    assert captured["max_steps_closing"] == 3
    assert captured["max_steps_final"] == 4
    assert np.allclose(captured["closing_controller"](np.zeros(6)), [0.01, 0.0, 0.0])
    assert np.allclose(captured["final_controller"](np.zeros(6)), [0.0, 0.01, 0.0])


def test_run_many_missions_indexes_each_result(monkeypatch):
    runner = DockingMissionRunner()
    calls = []

    def fake_run_mission(**kwargs):
        calls.append(kwargs)
        return {"mission_success": len(calls) % 2 == 1}

    monkeypatch.setattr(runner, "run_mission", fake_run_mission)

    results = runner.run_many_missions(
        closing_controller="closing_controller",
        final_controller="final_controller",
        num_missions=3,
        mode="testing",
        max_steps_closing=5,
        max_steps_final=6,
    )

    assert [result["mission_idx"] for result in results] == [0, 1, 2]
    assert [result["mission_success"] for result in results] == [True, False, True]
    assert all(call["mode"] == "testing" for call in calls)
    assert all(call["max_steps_closing"] == 5 for call in calls)
    assert all(call["max_steps_final"] == 6 for call in calls)
