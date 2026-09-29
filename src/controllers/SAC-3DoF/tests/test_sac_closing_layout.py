"""Structural checks for the SAC closing experiment archive."""

from __future__ import annotations

import ast
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
    assert (family / "evaluate_sac_closing.py").is_file()
    assert (family / "tune_sac_closing_optuna.py").is_file()
    assert (family / "plotters" / "plot_sac_closing_rollouts.py").is_file()
    assert (family / "models" / "reward_shaping_trials").is_dir()
    assert (family / "results" / "reward_shaping_trials").is_dir()
    assert (
        family
        / "results"
        / "terminal_state_convergence"
        / "closing_vanilla_sac_state_convergence_seed15.png"
    ).is_file()


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

    assert (family / "train_sac_closing_curriculum.py").is_file()
    assert (family / "train_sac_closing_p3reach_from_scratch.py").is_file()
    assert (family / "evaluate_sac_closing_p3reach.py").is_file()
    assert (family / "tune_sac_closing_p3reach_v12_optuna.py").is_file()
    assert (family / "plotters" / "plot_p3_relative_trajectory.py").is_file()
    assert (family / "tests" / "test_p3reach_evaluator_diagnostics.py").is_file()
    for artifact_dir in (
        "adaptive_dt01_terminal",
        "curriculum",
        "no_safety_termination",
        "p3reach_from_scratch",
    ):
        assert (family / "models" / artifact_dir).is_dir()
        assert (family / "logs" / artifact_dir).is_dir()
        assert (family / "results" / artifact_dir).is_dir()


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

    assert (family / "phase_balanced_replay_buffer.py").is_file()
    assert (family / "phase_balanced_replay_buffer_v2.py").is_file()
    assert (family / "train_sac_closing_v15_phase_balanced_replay.py").is_file()
    assert (family / "tests" / "test_phase_balanced_replay_buffer.py").is_file()
    assert (family / "models" / "multiseed").is_dir()
    assert (family / "logs" / "multiseed").is_dir()
    assert (family / "results" / "multiseed").is_dir()
    assert (family / "results" / "retained_figures").is_dir()


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


def test_timestep_transfer_family_contains_code_tests_and_artifacts():
    family = CLOSING_DIR / "timestep_transfer"

    assert (family / "evaluate_sac_closing_timestep_transfer.py").is_file()
    assert (family / "train_sac_closing_uniform_dt01_actor_transfer.py").is_file()
    assert (family / "plotters" / "plot_v15_horizon_comparison.py").is_file()
    assert (family / "tests" / "test_evaluate_sac_closing_timestep_transfer.py").is_file()
    assert (family / "models" / "uniform_dt01_actor_transfer").is_dir()
    assert (family / "logs" / "uniform_dt01_actor_transfer").is_dir()
    assert (family / "results" / "uniform_dt01_actor_transfer").is_dir()
    assert (family / "results" / "timestep_transfer").is_dir()
