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

    def _candidate_indices(self):
        """Return reset candidates; subclasses may implement a curriculum."""
        return self.eligible_indices

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
        index = int(rng.choice(self._candidate_indices()))
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


class LateralTerminalCurriculumWrapper(
    EmpiricalTerminalReferenceResidualWrapper
):
    """Easy-to-hard reset curriculum for lateral-limited near-P3 states."""

    def __init__(
        self,
        env,
        dataset_path,
        *,
        minimum_lateral_ratio=1.05,
        stage_maximum_ratios=(1.25, 1.5, 2.0),
        stage_boundaries=(0.25, 0.60),
        total_curriculum_steps=500_000,
        include_source_outcomes=("success", "time_limit"),
    ):
        stage_maximum_ratios = tuple(
            float(value) for value in stage_maximum_ratios
        )
        stage_boundaries = tuple(float(value) for value in stage_boundaries)
        if not stage_maximum_ratios:
            raise ValueError("At least one curriculum stage is required.")
        if len(stage_boundaries) != len(stage_maximum_ratios) - 1:
            raise ValueError(
                "Curriculum boundaries must contain one fewer value than "
                "stage maximum ratios."
            )
        if not 1.0 < float(minimum_lateral_ratio) < stage_maximum_ratios[0]:
            raise ValueError(
                "minimum_lateral_ratio must lie in (1, first stage maximum)."
            )
        if any(
            current <= previous
            for previous, current in zip(
                stage_maximum_ratios,
                stage_maximum_ratios[1:],
            )
        ):
            raise ValueError("Curriculum stage maximum ratios must increase.")
        if any(not 0.0 < value < 1.0 for value in stage_boundaries) or any(
            current <= previous
            for previous, current in zip(
                stage_boundaries,
                stage_boundaries[1:],
            )
        ):
            raise ValueError(
                "Curriculum stage boundaries must increase within (0, 1)."
            )
        if int(total_curriculum_steps) <= 0:
            raise ValueError("total_curriculum_steps must be positive.")

        super().__init__(
            env,
            dataset_path,
            minimum_max_target_ratio=minimum_lateral_ratio,
            maximum_max_target_ratio=stage_maximum_ratios[-1],
            include_source_outcomes=include_source_outcomes,
        )
        self.minimum_lateral_ratio = float(minimum_lateral_ratio)
        self.stage_maximum_ratios = stage_maximum_ratios
        self.stage_boundaries = stage_boundaries
        self.total_curriculum_steps = int(total_curriculum_steps)
        self.curriculum_steps_elapsed = 0
        self.current_curriculum_stage = 0

        states = np.asarray(self.dataset["states"], dtype=np.float64)
        targets = np.asarray(self.dataset["p3_targets"], dtype=np.float64)
        errors = states - targets
        errors[:, 4] = np.arctan2(
            np.sin(errors[:, 4]),
            np.cos(errors[:, 4]),
        )
        tolerance = np.asarray(
            self.core_env.p3_target_tolerance,
            dtype=np.float64,
        )
        component_ratios = np.abs(errors) / np.maximum(tolerance, 1e-12)
        lateral_ratios = component_ratios[:, 1]
        lateral_is_limiting = np.argmax(component_ratios, axis=1) == 1
        outcomes = np.asarray(self.dataset["source_outcomes"]).astype(str)
        common = (
            lateral_is_limiting
            & (lateral_ratios >= self.minimum_lateral_ratio)
            & np.isin(outcomes, self.include_source_outcomes)
        )
        self.lateral_ratios = lateral_ratios
        self.stage_eligible_indices = tuple(
            np.flatnonzero(common & (lateral_ratios <= maximum))
            for maximum in self.stage_maximum_ratios
        )
        if any(indices.size == 0 for indices in self.stage_eligible_indices):
            raise ValueError(
                "At least one lateral curriculum stage has no eligible resets."
            )
        self.eligible_indices = self.stage_eligible_indices[-1]

    def _stage_for_elapsed_steps(self):
        progress = min(
            self.curriculum_steps_elapsed / self.total_curriculum_steps,
            1.0,
        )
        return int(np.searchsorted(self.stage_boundaries, progress, side="right"))

    def _candidate_indices(self):
        self.current_curriculum_stage = self._stage_for_elapsed_steps()
        return self.stage_eligible_indices[self.current_curriculum_stage]

    def _annotate(self, info, index, source):
        result = super()._annotate(info, index, source)
        result.update(
            {
                "terminal_reset_lateral_curriculum": True,
                "terminal_reset_curriculum_stage": int(
                    self.current_curriculum_stage
                ),
                "terminal_reset_curriculum_stage_maximum_ratio": float(
                    self.stage_maximum_ratios[self.current_curriculum_stage]
                ),
                "terminal_reset_curriculum_steps_elapsed": int(
                    self.curriculum_steps_elapsed
                ),
            }
        )
        if index is not None:
            result["terminal_reset_initial_lateral_ratio"] = float(
                self.lateral_ratios[index]
            )
        return result

    def step(self, action):
        result = super().step(action)
        self.curriculum_steps_elapsed += 1
        return result
