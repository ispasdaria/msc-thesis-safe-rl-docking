"""Empirical terminal-reset support for the LQR-reference-residual task.

The reset bank is collected from ordinary full closing trajectories.  Resetting
to a stored state/target pair therefore changes the training start-state
distribution without changing the dynamics, controller, observation, reward,
or terminal conditions.
"""

from __future__ import annotations

import hashlib
import json
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

PROVENANCE_SCHEMA_VERSION = 2


def sha256_file(path):
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def terminal_reset_metadata(dataset):
    """Parse the scalar JSON metadata stored with a terminal reset bank."""
    if "metadata_json" not in dataset:
        raise ValueError("Terminal reset dataset has no metadata_json field.")
    encoded = np.asarray(dataset["metadata_json"])
    if encoded.shape != ():
        raise ValueError("metadata_json must be a scalar JSON string.")
    try:
        metadata = json.loads(str(encoded.item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("metadata_json is not valid JSON.") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must encode a JSON object.")
    return metadata


def validate_terminal_reset_provenance(
    dataset,
    *,
    dataset_path,
    baseline_manifest,
    baseline_manifest_path,
    required_outcomes=("success", "time_limit"),
):
    """Require a bank produced by the fingerprinted final 10 Hz LQR.

    ``validate_saved_baseline_manifest`` separately proves that the manifest
    still matches the current controller sources and numerical gain.  This
    function binds a reset bank to that already-validated manifest.
    """
    metadata = terminal_reset_metadata(dataset)
    if int(metadata.get("schema_version", -1)) < PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            "Terminal reset dataset predates the provenance-locked dt=0.1 "
            "schema and cannot be used for final-LQR training."
        )

    configuration = baseline_manifest.get("configuration", {})
    controller = baseline_manifest.get("controller", {})
    expected_dt = float(configuration.get("dt_seconds", np.nan))
    recorded_dt = float(metadata.get("dt_seconds", np.nan))
    if not np.isclose(recorded_dt, expected_dt, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Terminal bank dt mismatch: {recorded_dt!r} != {expected_dt!r}."
        )
    if not np.isclose(recorded_dt, 0.1, rtol=0.0, atol=1e-12):
        raise ValueError("Final-LQR terminal banks must use dt=0.1 s.")

    recorded_max_time = float(metadata.get("max_time_seconds", np.nan))
    recorded_max_steps = int(metadata.get("max_steps", -1))
    if not np.isfinite(recorded_max_time) or recorded_max_time <= 0.0:
        raise ValueError("Terminal bank must record a positive max_time_seconds.")
    expected_max_steps = int(round(recorded_max_time / recorded_dt))
    if recorded_max_steps != expected_max_steps:
        raise ValueError(
            "Terminal bank horizon is inconsistent with its timestep: "
            f"max_steps={recorded_max_steps} but max_time_seconds / dt="
            f"{expected_max_steps}."
        )

    expected_fingerprint = controller.get("numeric_fingerprint_sha256")
    recorded_fingerprint = metadata.get("controller_numeric_fingerprint_sha256")
    if not expected_fingerprint or recorded_fingerprint != expected_fingerprint:
        raise ValueError(
            "Terminal bank controller fingerprint does not match the final "
            "dt=0.1 LQR baseline."
        )

    expected_integrator = str(controller.get("integrator", ""))
    if str(metadata.get("integrator", "")) != expected_integrator:
        raise ValueError("Terminal bank integrator does not match the baseline.")

    manifest_path = Path(baseline_manifest_path).resolve()
    recorded_manifest_hash = metadata.get("baseline_manifest_sha256")
    if recorded_manifest_hash != sha256_file(manifest_path):
        raise ValueError(
            "Terminal bank baseline-manifest hash does not match the supplied "
            "manifest."
        )

    outcomes = np.asarray(dataset["source_outcomes"]).astype(str)
    episode_seeds = np.asarray(dataset["source_seeds"], dtype=np.int64)
    outcome_episode_counts = {
        outcome: int(len(np.unique(episode_seeds[outcomes == outcome])))
        for outcome in sorted(set(outcomes.tolist()))
    }
    missing = [
        outcome
        for outcome in required_outcomes
        if outcome_episode_counts.get(str(outcome), 0) == 0
    ]
    if missing:
        raise ValueError(
            "Terminal bank lacks required source outcomes: " + ", ".join(missing)
        )

    return {
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "schema_version": int(metadata["schema_version"]),
        "dt_seconds": recorded_dt,
        "max_time_seconds": recorded_max_time,
        "max_steps": recorded_max_steps,
        "integrator": expected_integrator,
        "controller_numeric_fingerprint_sha256": expected_fingerprint,
        "baseline_manifest_path": str(manifest_path),
        "baseline_manifest_sha256": recorded_manifest_hash,
        "source_outcome_episode_counts": outcome_episode_counts,
    }


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
        balance_source_outcomes=False,
    ):
        super().__init__(env)
        self.dataset_path = Path(dataset_path).resolve()
        self.dataset = load_terminal_reset_dataset(self.dataset_path)
        self.minimum_max_target_ratio = float(minimum_max_target_ratio)
        self.maximum_max_target_ratio = float(maximum_max_target_ratio)
        self.include_source_outcomes = tuple(str(x) for x in include_source_outcomes)
        self.balance_source_outcomes = bool(balance_source_outcomes)
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
        if self.balance_source_outcomes:
            available = set(outcomes[self.eligible_indices].tolist())
            missing = [
                outcome
                for outcome in self.include_source_outcomes
                if outcome not in available
            ]
            if missing:
                raise ValueError(
                    "Cannot balance absent terminal source outcomes: "
                    + ", ".join(missing)
                )
        self.current_dataset_index = None

    def _candidate_indices(self):
        """Return reset candidates; subclasses may implement a curriculum."""
        return self.eligible_indices

    def _choose_candidate_index(self, rng):
        candidates = np.asarray(self._candidate_indices(), dtype=np.int64)
        if not self.balance_source_outcomes:
            return int(rng.choice(candidates))
        outcomes = np.asarray(self.dataset["source_outcomes"]).astype(str)
        available_outcomes = tuple(
            outcome
            for outcome in self.include_source_outcomes
            if np.any(outcomes[candidates] == outcome)
        )
        chosen_outcome = str(rng.choice(available_outcomes))
        matching = candidates[outcomes[candidates] == chosen_outcome]
        return int(rng.choice(matching))

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
        index = self._choose_candidate_index(rng)
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
        balance_source_outcomes=False,
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
            balance_source_outcomes=balance_source_outcomes,
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
