from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SRC_PATH = Path(__file__).resolve().parents[4]
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
    load_terminal_reset_dataset,
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
