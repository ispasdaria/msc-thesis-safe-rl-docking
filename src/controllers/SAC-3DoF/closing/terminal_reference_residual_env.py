"""Empirical terminal-reset support for the LQR-reference-residual task.

The reset bank is collected from ordinary full closing trajectories.  Resetting
to a stored state/target pair therefore changes the training start-state
distribution without changing the dynamics, controller, observation, reward,
or terminal conditions.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np


REQUIRED_DATASET_FIELDS = (
    "states",
    "p3_targets",
    "max_target_ratios",
    "source_seeds",
    "source_steps",
    "source_outcomes",
)


def load_terminal_reset_dataset(path):
    """Load and validate a terminal reset bank into ordinary NumPy arrays."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Terminal reset dataset not found: {dataset_path}")
    with np.load(dataset_path, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_DATASET_FIELDS if name not in archive]
        if missing:
            raise ValueError(
                f"Terminal reset dataset is missing fields: {', '.join(missing)}"
            )
        dataset = {name: np.asarray(archive[name]).copy() for name in archive.files}

    count = len(dataset["states"])
    if dataset["states"].shape != (count, 6):
        raise ValueError("states must have shape (N, 6).")
    if dataset["p3_targets"].shape != (count, 6):
        raise ValueError("p3_targets must have shape (N, 6).")
    for name in REQUIRED_DATASET_FIELDS[2:]:
        if len(dataset[name]) != count:
            raise ValueError(f"{name} must contain N entries.")
    if count == 0:
        raise ValueError("Terminal reset dataset is empty.")
    if not np.all(np.isfinite(dataset["states"])):
        raise ValueError("Terminal states must be finite.")
    if not np.all(np.isfinite(dataset["p3_targets"])):
        raise ValueError("Terminal P3 targets must be finite.")
    return dataset


class EmpiricalTerminalReferenceResidualWrapper(gym.Wrapper):
    """Reset a reference-residual closing environment from empirical states."""

    def __init__(
        self,
        env,
        dataset_path,
        minimum_max_target_ratio=1.05,
        maximum_max_target_ratio=5.0,
        include_source_outcomes=("success", "failure", "time_limit"),
    ):
        super().__init__(env)
        self.dataset_path = Path(dataset_path).resolve()
        self.dataset = load_terminal_reset_dataset(self.dataset_path)
        self.minimum_max_target_ratio = float(minimum_max_target_ratio)
        self.maximum_max_target_ratio = float(maximum_max_target_ratio)
        self.include_source_outcomes = tuple(str(x) for x in include_source_outcomes)
        if not 1.0 < self.minimum_max_target_ratio < self.maximum_max_target_ratio:
            raise ValueError(
                "Terminal ratio limits must satisfy 1 < minimum < maximum."
            )

        ratios = np.asarray(self.dataset["max_target_ratios"], dtype=np.float64)
        outcomes = np.asarray(self.dataset["source_outcomes"]).astype(str)
        eligible = (
            (ratios >= self.minimum_max_target_ratio)
            & (ratios <= self.maximum_max_target_ratio)
            & np.isin(outcomes, self.include_source_outcomes)
        )
        self.eligible_indices = np.flatnonzero(eligible)
        if self.eligible_indices.size == 0:
            raise ValueError("No terminal reset states satisfy the requested filters.")
        self.current_dataset_index = None

    @property
    def core_env(self):
        return self.env.unwrapped.core_env

    def _annotate(self, info, index, source):
        result = dict(info)
        result.update(
            {
                "terminal_reset_source": source,
                "terminal_reset_dataset": str(self.dataset_path),
                "terminal_reset_dataset_index": index,
            }
        )
        if index is not None:
            result.update(
                {
                    "terminal_reset_source_seed": int(
                        self.dataset["source_seeds"][index]
                    ),
                    "terminal_reset_source_step": int(
                        self.dataset["source_steps"][index]
                    ),
                    "terminal_reset_source_outcome": str(
                        self.dataset["source_outcomes"][index]
                    ),
                    "terminal_reset_initial_max_target_ratio": float(
                        self.dataset["max_target_ratios"][index]
                    ),
                }
            )
        return result

    def reset(self, seed=None, options=None):
        # Explicit state/target pairs are required by the paired oracle study
        # and always take precedence over dataset sampling.
        if options and options.get("initial_state") is not None:
            observation, info = self.env.reset(seed=seed, options=options)
            self.current_dataset_index = None
            return observation, self._annotate(info, None, "explicit")

        # First seed the ordinary environment and its core RNG.  The resulting
        # P2 state is discarded and replaced by one empirical state/P3 pair.
        self.env.reset(seed=seed, options=options)
        rng = self.core_env.rng
        index = int(rng.choice(self.eligible_indices))
        mode = (options or {}).get("mode", self.env.unwrapped.mode)
        observation, info = self.env.reset(
            options={
                "initial_state": np.asarray(
                    self.dataset["states"][index], dtype=np.float64
                ),
                "p3_target": np.asarray(
                    self.dataset["p3_targets"][index], dtype=np.float64
                ),
                "mode": mode,
            }
        )
        if bool(info.get("unsafe", False)):
            raise RuntimeError(
                f"Stored terminal reset {index} is unsafe under the current model."
            )
        self.current_dataset_index = index
        return observation, self._annotate(info, index, "empirical_lqr_trajectory")

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        source = (
            "explicit"
            if self.current_dataset_index is None
            else "empirical_lqr_trajectory"
        )
        return (
            observation,
            reward,
            terminated,
            truncated,
            self._annotate(info, self.current_dataset_index, source),
        )
