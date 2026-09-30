import configparser
import importlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
DOCKING_PACKAGE = REPO_ROOT / "src" / "envs" / "docking3dof"


SAFETY_EXPLORATION_EXPORTS = (
    "ClosingSafetyExplorationCoreEnv",
    "ClosingSafetyExplorationGymEnv",
    *(f"ClosingSafetyExplorationV{version}GymEnv" for version in range(2, 20)),
    "ClosingSafetyTerminalV19GymEnv",
    "ClosingSafetyTerminalV20GymEnv",
    "ClosingSafetyTerminalV21GymEnv",
)


def test_pytest_configuration_discovers_environment_tests_from_src():
    config = configparser.ConfigParser()
    config.read(REPO_ROOT / "pytest.ini")

    assert config["pytest"]["pythonpath"].split() == [
        "src",
        "src/controllers/LQR-3DoF",
        "src/controllers/CBF-3DoF",
    ]
    assert config["pytest"]["testpaths"].split() == [
        "src/envs/docking3dof/tests"
    ]


def test_production_modules_do_not_bootstrap_python_import_paths():
    production_modules = DOCKING_PACKAGE.glob("*.py")

    offenders = [
        path.name
        for path in production_modules
        if "sys.path" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_diagnostics_and_results_have_dedicated_directories():
    tests_dir = DOCKING_PACKAGE / "tests"

    assert (tests_dir / "diagnostics").is_dir()
    assert (
        tests_dir / "results" / "integrator_timestep_verification"
    ).is_dir()
    assert not (DOCKING_PACKAGE / "tests_gym").exists()
    assert not (tests_dir / "integrator_timestep_verification_results").exists()


@pytest.mark.parametrize(
    "module_name",
    (
        "envs.docking3dof.core",
        "envs.docking3dof.gym_env",
        "envs.docking3dof.runners",
        "envs.docking3dof.closing_cbf_shield_env",
        "envs.docking3dof.goal_conditioned_closing_env",
    ),
)
def test_public_environment_modules_remain_importable(module_name):
    assert importlib.import_module(module_name).__name__ == module_name


@pytest.mark.parametrize("class_name", SAFETY_EXPLORATION_EXPORTS)
def test_safety_exploration_classes_remain_available_from_legacy_module(class_name):
    module = importlib.import_module(
        "envs.docking3dof.closing_safety_exploration_env"
    )

    exported_class = getattr(module, class_name)

    assert exported_class.__name__ == class_name
