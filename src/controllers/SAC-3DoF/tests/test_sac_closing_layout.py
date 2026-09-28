"""Structural checks for the SAC closing experiment archive."""

from __future__ import annotations

import ast
from pathlib import Path


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
