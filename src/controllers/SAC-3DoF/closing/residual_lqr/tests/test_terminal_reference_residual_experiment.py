from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SRC_PATH = Path(__file__).resolve().parents[5]
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
CLOSING_PATH = Path(__file__).resolve().parents[1]
if str(CLOSING_PATH) not in sys.path:
    sys.path.insert(0, str(CLOSING_PATH))

from analyze_terminal_reference_residual_oracle import (
    candidate_action_grid,
    select_paired_indices,
)
from lqr_reference_residual_env import (
    ReferenceResidualLQRGoalConditionedClosingEnv,
)
from terminal_reference_residual_env import (
    EmpiricalTerminalReferenceResidualWrapper,
    LateralTerminalCurriculumWrapper,
    load_terminal_reset_dataset,
)
from train_sac_closing_her_reference_residual_lqr import (
    collect_reference_residual_trajectories,
)
from train_sac_closing_reference_residual_lqr_terminal import (
    build_parser as build_terminal_parser,
    validate_args as validate_terminal_args,
)


def write_dataset(path):
    targets = np.asarray(
        [
            [300.0, 0.0, -0.03, 0.0, 0.0, 0.0],
            [310.0, 1.0, -0.04, 0.0, 0.0, 0.0],
        ]
    )
    states = targets.copy()
    states[:, 0] += [30.0, 40.0]
    np.savez_compressed(
        path,
        states=states,
        p3_targets=targets,
        max_target_ratios=np.asarray([1.5, 2.0]),
        source_seeds=np.asarray([10, 11]),
        source_steps=np.asarray([100, 200]),
        source_outcomes=np.asarray(["success", "time_limit"]),
        metadata_json=np.asarray("{}"),
    )


def write_lateral_curriculum_dataset(path):
    target = np.asarray([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    targets = np.repeat(target[None, :], 3, axis=0)
    states = targets.copy()
    lateral_ratios = np.asarray([1.10, 1.40, 1.80])
    states[:, 1] += 2.0 * lateral_ratios
    np.savez_compressed(
        path,
        states=states,
        p3_targets=targets,
        max_target_ratios=lateral_ratios,
        source_seeds=np.asarray([20, 21, 22]),
        source_steps=np.asarray([100, 200, 300]),
        source_outcomes=np.asarray(["success", "time_limit", "time_limit"]),
        metadata_json=np.asarray("{}"),
    )


def test_empirical_terminal_reset_uses_paired_state_and_target(tmp_path):
    path = tmp_path / "terminal.npz"
    write_dataset(path)
    base = ReferenceResidualLQRGoalConditionedClosingEnv(
        mode="training", seed=5, max_time=20.0, terminate_on_success=True
    )
    env = EmpiricalTerminalReferenceResidualWrapper(base, path)
    observation, info = env.reset(seed=5)
    index = info["terminal_reset_dataset_index"]
    dataset = load_terminal_reset_dataset(path)

    np.testing.assert_allclose(env.core_env.state, dataset["states"][index])
    np.testing.assert_allclose(
        env.core_env.p3_target, dataset["p3_targets"][index]
    )
    np.testing.assert_allclose(
        observation["achieved_goal"],
        dataset["states"][index] / base.state_scale,
    )
    assert info["terminal_reset_source"] == "empirical_lqr_trajectory"
    assert info["terminal_reset_source_outcome"] in {"success", "time_limit"}
    env.close()


def test_lateral_terminal_curriculum_expands_reset_difficulty(tmp_path):
    path = tmp_path / "lateral_terminal.npz"
    write_lateral_curriculum_dataset(path)
    base = ReferenceResidualLQRGoalConditionedClosingEnv(
        mode="training",
        seed=6,
        dt=0.1,
        max_time=20.0,
        terminate_on_success=True,
    )
    env = LateralTerminalCurriculumWrapper(
        base,
        path,
        minimum_lateral_ratio=1.05,
        stage_maximum_ratios=(1.25, 1.5, 2.0),
        stage_boundaries=(0.25, 0.60),
        total_curriculum_steps=100,
    )

    assert [len(indices) for indices in env.stage_eligible_indices] == [1, 2, 3]
    _, easy_info = env.reset(seed=6)
    assert easy_info["terminal_reset_curriculum_stage"] == 0
    assert np.isclose(easy_info["terminal_reset_initial_lateral_ratio"], 1.10)

    env.curriculum_steps_elapsed = 25
    _, medium_info = env.reset(seed=7)
    assert medium_info["terminal_reset_curriculum_stage"] == 1
    assert medium_info["terminal_reset_initial_lateral_ratio"] <= 1.5

    env.curriculum_steps_elapsed = 60
    _, hard_info = env.reset(seed=8)
    assert hard_info["terminal_reset_curriculum_stage"] == 2
    assert hard_info["terminal_reset_initial_lateral_ratio"] <= 2.0
    env.close()


def test_terminal_curriculum_can_supply_replay_prefill(tmp_path):
    path = tmp_path / "lateral_prefill.npz"
    write_lateral_curriculum_dataset(path)

    def env_factory(seed):
        base = ReferenceResidualLQRGoalConditionedClosingEnv(
            mode="training",
            seed=seed,
            dt=0.1,
            max_time=0.3,
            progress_gamma=0.9999**0.1,
            terminate_on_success=True,
        )
        return LateralTerminalCurriculumWrapper(
            base,
            path,
            total_curriculum_steps=3,
        )

    transitions, rows = collect_reference_residual_trajectories(
        minimum_transitions=3,
        seed_start=30,
        max_episodes=2,
        max_steps=0.3,
        env_kwargs={"dt": 0.1},
        residual_std=0.0,
        residual_clip=0.5,
        zero_episode_fraction=1.0,
        random_seed=30,
        retain_failure_episodes=True,
        env_factory=env_factory,
    )

    assert len(transitions) == 3
    assert rows[0]["steps"] == 3
    assert transitions[0]["info"]["p3_max_target_ratio"] <= 2.0


def test_oracle_grid_contains_zero_and_selects_one_state_per_episode(tmp_path):
    path = tmp_path / "terminal.npz"
    write_dataset(path)
    dataset = load_terminal_reset_dataset(path)
    grid = candidate_action_grid([-1.0, 0.0, 1.0])
    indices = select_paired_indices(
        dataset, n_states=2, seed=12, nonsuccess_fraction=0.5
    )

    assert grid.shape == (27, 3)
    assert np.any(np.all(grid == 0.0, axis=1))
    assert len(np.unique(dataset["source_seeds"][indices])) == 2
    assert set(dataset["source_outcomes"][indices]) == {"success", "time_limit"}
