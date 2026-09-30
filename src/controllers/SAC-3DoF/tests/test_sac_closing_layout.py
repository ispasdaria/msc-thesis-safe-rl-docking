"""Structural checks for the SAC closing experiment archive."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


CLOSING_DIR = Path(__file__).resolve().parents[1] / "closing"
EXPECTED_FAMILIES = {
    "vanilla_sac",
    "curriculum_and_p3_reach",
    "phase_balanced_replay",
    "lqr_demonstration_learning",
    "residual_lqr",
    "timestep_transfer",
    "two_stage_capture",
}
EXECUTABLE_PREFIXES = (
    "analyze_",
    "build_",
    "curate_",
    "evaluate_",
    "generate_",
    "plot_",
    "train_",
    "tune_",
    "validate_",
)
PURPOSE_BASED_NAMES = ("article", "publication", "thesis")
SHARED_ROOT_MODULES = {
    "behavior_cloned_warmup_sac.py",
    "near_p3_reset_wrapper.py",
    "train_sac_closing_multiseed.py",
}
UNNORMALIZED_MODEL_FAMILIES = (
    "her",
    "her_lqr_reference_residual",
    "her_lqr_residual",
    "lqr_reference_residual",
)


def _has_main_guard(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            return True
    return False


def test_approved_experiment_families_have_readmes():
    assert (CLOSING_DIR / "README.md").is_file()
    assert {
        path.name
        for path in CLOSING_DIR.iterdir()
        if path.is_dir() and path.name in EXPECTED_FAMILIES
    } == EXPECTED_FAMILIES
    assert all((CLOSING_DIR / family / "README.md").is_file() for family in EXPECTED_FAMILIES)


def test_closing_root_contains_only_shared_support_modules():
    assert {path.name for path in CLOSING_DIR.glob("*.py")} == SHARED_ROOT_MODULES
    assert not any(
        (CLOSING_DIR / name).exists()
        for name in ("demonstrations", "figures", "logs", "models", "results", "tests_lqr_replay")
    )


def test_active_paths_use_content_based_names():
    offenders = [
        path.relative_to(CLOSING_DIR).as_posix()
        for family in EXPECTED_FAMILIES
        for path in (CLOSING_DIR / family).rglob("*")
        if any(term in path.name.lower() for term in PURPOSE_BASED_NAMES)
    ]

    assert offenders == []


def test_executable_scripts_have_main_guards():
    offenders = [
        path.relative_to(CLOSING_DIR).as_posix()
        for path in CLOSING_DIR.rglob("*.py")
        if path.name.startswith(EXECUTABLE_PREFIXES) and not _has_main_guard(path)
    ]

    assert offenders == []


def test_non_her_model_directories_include_normalization_state():
    offenders = []
    for directory in (CLOSING_DIR / "models").rglob("*"):
        if not directory.is_dir() or not list(directory.glob("*.zip")):
            continue
        relative = directory.relative_to(CLOSING_DIR / "models").as_posix()
        if relative.startswith(UNNORMALIZED_MODEL_FAMILIES):
            continue
        if not list(directory.glob("*.pkl")):
            offenders.append(relative)

    assert offenders == []


def test_vanilla_sac_family_contains_entry_points_and_artifacts():
    family = CLOSING_DIR / "vanilla_sac"

    assert (family / "train_sac_closing.py").is_file()
    assert (family / "train_sac_closing_v15dt_full_budget_dt01.py").is_file()
    assert (family / "evaluate_sac_closing.py").is_file()
    assert (family / "evaluate_sac_closing_v15_baseline.py").is_file()
    assert (family / "tune_sac_closing_optuna.py").is_file()
    assert (family / "plotters" / "plot_sac_closing_rollouts.py").is_file()
    assert (family / "closing_port_observation_env.py").is_file()
    assert (family / "tests" / "test_train_sac_closing_dt01.py").is_file()
    assert (family / "results" / "multiseed").is_dir()
    assert (family / "models" / "reward_shaping_trials").is_dir()
    assert (family / "results" / "reward_shaping_trials").is_dir()
    assert (
        family
        / "results"
        / "terminal_state_convergence"
        / "closing_vanilla_sac_state_convergence_seed15.png"
    ).is_file()
    assert not (family / "train_sac_closing_best_standard_dt01.py").exists()
    assert not (family / "evaluate_rewardv15_baseline.py").exists()


def test_vanilla_sac_scripts_import_from_repository_root():
    family = CLOSING_DIR / "vanilla_sac"
    failures = []
    for script in sorted(family.rglob("*.py")):
        command = (
            "import runpy; "
            f"runpy.run_path({str(script)!r}, run_name='__import_test__')"
        )
        completed = subprocess.run(
            [sys.executable, "-c", command],
            cwd=CLOSING_DIR.parents[3],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            failures.append(f"{script.name}: {completed.stderr}")

    assert failures == []


def test_curriculum_and_p3_reach_family_contains_entry_points_and_artifacts():
    family = CLOSING_DIR / "curriculum_and_p3_reach"

    assert (family / "train_p3_reach_reset_distribution_curriculum.py").is_file()
    assert (family / "train_p3_reach_reward_from_scratch.py").is_file()
    assert (family / "evaluate_p3_reach_checkpoints.py").is_file()
    assert (family / "tune_p3_reach_v12_from_scratch_optuna.py").is_file()
    assert (
        family / "plotters" / "plot_vanilla_v17_p3_relative_trajectory.py"
    ).is_file()
    assert (family / "tests" / "test_evaluate_p3_reach_checkpoints.py").is_file()
    for artifact_dir in (
        "ten_hz_adaptive_terminal_continuation",
        "reset_distribution_curriculum",
        "nonterminal_safety_continuation",
        "from_scratch_reward_progression",
        "safety_terminal_braking_experiments",
    ):
        assert (family / "models" / artifact_dir).is_dir()
        assert (family / "logs" / artifact_dir).is_dir()
        assert (family / "results" / artifact_dir).is_dir()
        assert (family / "results" / artifact_dir / "README.md").is_file()

    assert (family / "logs" / "p3_reach_v12_optuna").is_dir()
    assert (family / "results" / "p3_reach_v12_optuna" / "README.md").is_file()

    old_entry_points = {
        "evaluate_sac_closing_p3reach.py",
        "train_sac_closing_adaptive_dt01_terminal.py",
        "train_sac_closing_braking_envelope.py",
        "train_sac_closing_braking_terminal_v20.py",
        "train_sac_closing_curriculum.py",
        "train_sac_closing_no_safety_termination.py",
        "train_sac_closing_p3reach_from_scratch.py",
        "train_sac_closing_port_lqr_envelope_v21.py",
        "tune_sac_closing_p3reach_v12_optuna.py",
    }
    assert not any((family / name).exists() for name in old_entry_points)


def test_curriculum_and_p3_reach_scripts_import_from_repository_root():
    family = CLOSING_DIR / "curriculum_and_p3_reach"
    failures = []
    for script in sorted(family.rglob("*.py")):
        if "tests" in script.parts:
            continue
        command = (
            "import runpy; "
            f"runpy.run_path({str(script)!r}, run_name='__import_test__')"
        )
        completed = subprocess.run(
            [sys.executable, "-c", command],
            cwd=CLOSING_DIR.parents[3],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            failures.append(f"{script.name}: {completed.stderr}")

    assert failures == []


def test_phase_balanced_replay_family_contains_code_and_artifacts():
    family = CLOSING_DIR / "phase_balanced_replay"

    assert (family / "phase_balanced_replay_buffer_implementation.py").is_file()
    assert (family / "guarded_simultaneous_target_replay_buffer.py").is_file()
    assert (family / "train_sac_closing_v15_phase_balanced_replay.py").is_file()
    assert (
        family / "train_sac_closing_v15_guarded_simultaneous_target_replay.py"
    ).is_file()
    assert (
        family / "train_sac_closing_v15_deterministic_near_p3_replay.py"
    ).is_file()
    assert (
        family / "train_sac_closing_v15_deterministic_near_p3_replay_cbf.py"
    ).is_file()
    assert (family / "tests" / "test_phase_balanced_replay_buffer.py").is_file()
    assert (
        family / "tests" / "test_guarded_simultaneous_target_replay.py"
    ).is_file()
    assert (family / "models" / "multiseed").is_dir()
    assert (family / "logs" / "multiseed").is_dir()
    assert (family / "results" / "multiseed").is_dir()
    assert (family / "results" / "retained_figures").is_dir()

    old_names = {
        "phase_balanced_replay_buffer_v2.py",
        "train_sac_closing_v15_phase_balanced_replay_v2.py",
        "train_sac_closing_v15_phase_balanced_replay_v21.py",
        "train_sac_closing_v15_phase_balanced_replay_v21_cbf.py",
    }
    assert not any((family / name).exists() for name in old_names)


def test_retained_phase_balanced_checkpoint_loads_from_repository_root():
    """Catch removal of the legacy module path embedded in the checkpoint."""

    model_path = (
        CLOSING_DIR
        / "phase_balanced_replay"
        / "models"
        / "multiseed"
        / "sac_closing_v15_trial16_terminal_phase_balanced_guarded600k_seed0"
        / "seed_0"
        / "model_0100000_steps.zip"
    )
    command = (
        "from stable_baselines3 import SAC; "
        f"model = SAC.load({str(model_path)!r}, device='cpu'); "
        "assert model.observation_space.shape == (12,); "
        "assert model.action_space.shape == (3,)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=CLOSING_DIR.parents[3],
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_lqr_demonstration_family_contains_code_data_and_artifacts():
    family = CLOSING_DIR / "lqr_demonstration_learning"

    assert (family / "generate_lqr_replay_dataset.py").is_file()
    assert (family / "train_sac_closing_lqr_replay.py").is_file()
    assert (family / "train_sac_closing_lqr_bc_warmstart.py").is_file()
    assert (family / "plotters" / "plot_behavior_cloning_sac_figures.py").is_file()
    assert (family / "tests" / "test_lqr_replay_dataset.py").is_file()
    assert (family / "demonstrations" / "lqr_closing_training_rewardv15_28successes.npz").is_file()
    for artifact_dir in (
        "lqr_replay",
        "lqr_mixed_replay",
        "lqr_bc_warmstart",
        "lqr_bc_constrained",
    ):
        assert (family / "models" / artifact_dir).is_dir()
        assert (family / "logs" / artifact_dir).is_dir()
        assert (family / "results" / artifact_dir).is_dir()


def test_residual_lqr_family_contains_code_data_and_artifacts():
    family = CLOSING_DIR / "residual_lqr"

    assert (family / "lqr_reference_residual_env.py").is_file()
    assert (family / "terminal_reference_residual_env.py").is_file()
    assert (family / "train_sac_closing_reference_residual_final_lqr_dt01.py").is_file()
    assert (family / "validate_residual_lqr_dt01_baseline.py").is_file()
    assert (family / "tests" / "test_lqr_residual_env.py").is_file()
    assert (family / "demonstrations" / "terminal_reference_residual_final_lqr_dt0p1_bank.npz").is_file()
    assert (family / "models" / "lqr_reference_residual_final_dt01").is_dir()
    assert (family / "results" / "residual_lqr_baseline").is_dir()

    assert (family / "train_sac_closing_v15_source_warmstarted_her.py").is_file()
    assert (family / "train_sac_closing_v15_baseline_residual_her.py").is_file()
    assert (family / "train_sac_closing_exploratory_residual_her.py").is_file()
    assert (
        family
        / "train_sac_closing_reference_residual_lqr_annealed_actor_prior_1hz.py"
    ).is_file()
    assert (
        family
        / "train_sac_closing_reference_residual_lqr_annealed_actor_prior_dt01.py"
    ).is_file()
    assert (
        family
        / "train_sac_closing_reference_residual_lqr_lateral_reward_v2_dt01.py"
    ).is_file()
    assert (
        family / "train_sac_closing_her_lateral_balanced_residual_200k.py"
    ).is_file()
    assert (
        family / "train_sac_closing_her_terminal_fade_residual_200k.py"
    ).is_file()
    assert not any(
        (family / name).exists()
        for name in {
            "train_sac_closing_her_v3_source_warmstart.py",
            "train_sac_closing_her_v4_residual_v15.py",
            "train_sac_closing_her_v5_exploratory_residual.py",
            "train_sac_closing_reference_residual_lqr_pilot_e.py",
            "train_sac_closing_reference_residual_lqr_pilot_e_dt01.py",
            "train_sac_closing_reference_residual_lqr_pilot_g_dt01_reward_v2.py",
            "train_sac_closing_her_residual_lqr_pilot_lateral_balanced_200k.py",
            "train_sac_closing_her_residual_lqr_pilot_terminal_fade_200k.py",
        }
    )


def test_timestep_transfer_family_contains_code_tests_and_artifacts():
    family = CLOSING_DIR / "timestep_transfer"

    assert (family / "evaluate_sac_closing_timestep_transfer.py").is_file()
    assert (family / "train_sac_closing_uniform_dt01_actor_transfer.py").is_file()
    assert (family / "episode_stratified_replay_buffer.py").is_file()
    assert (family / "plotters" / "plot_v15_horizon_comparison.py").is_file()
    assert (family / "tests" / "test_evaluate_sac_closing_timestep_transfer.py").is_file()
    assert (family / "models" / "uniform_dt01_actor_transfer").is_dir()
    assert (family / "logs" / "uniform_dt01_actor_transfer").is_dir()
    assert (family / "results" / "uniform_dt01_actor_transfer").is_dir()
    assert (family / "results" / "timestep_transfer").is_dir()


def test_two_stage_capture_family_replaces_trial_layout():
    family = CLOSING_DIR / "two_stage_capture"

    assert not (CLOSING_DIR / "two_stage_capture_trial").exists()
    assert (family / "train_capture_sac.py").is_file()
    assert (family / "train_systematic_capture_sac.py").is_file()
    assert (family / "evaluate_capture_sweep.py").is_file()
    assert (family / "artifacts").is_dir()
    assert (family / "tests" / "test_capture_experiments.py").is_file()
    assert (family / "tests" / "test_systematic_capture.py").is_file()
