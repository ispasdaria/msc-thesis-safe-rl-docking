"""Import-safety checks for executable final-approach scripts."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys


FINAL_APPROACH_DIR = Path(__file__).resolve().parents[1]


def _executables() -> list[Path]:
    scripts = [
        path
        for path in FINAL_APPROACH_DIR.glob("*.py")
        if path.name.startswith(("evaluate_", "train_", "tune_"))
    ]
    scripts.extend((FINAL_APPROACH_DIR / "plotters").glob("plot_*.py"))
    return sorted(scripts)


def _is_main_guard(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


def test_executable_scripts_have_main_guards():
    missing = []
    for path in _executables():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not any(_is_main_guard(node) for node in tree.body):
            missing.append(path.name)

    assert missing == []


def test_imports_do_not_create_directories():
    offenders = []
    for path in _executables():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            function = node.value.func
            if (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "os"
                and function.attr == "makedirs"
            ):
                offenders.append(path.name)

    assert offenders == []


def test_final_geometry_plotter_help_starts_from_repository_root():
    script = FINAL_APPROACH_DIR / "plotters" / "plot_final_geometry_trajectory.py"

    completed = subprocess.run(
        [sys.executable, str(script), "-h"],
        cwd=FINAL_APPROACH_DIR.parents[3],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_near_target_evaluator_imports_from_repository_root():
    script = FINAL_APPROACH_DIR / "evaluate_sac_final_near_target.py"
    command = (
        "import runpy; "
        f"runpy.run_path({str(script)!r}, run_name='__import_test__')"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=FINAL_APPROACH_DIR.parents[3],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
