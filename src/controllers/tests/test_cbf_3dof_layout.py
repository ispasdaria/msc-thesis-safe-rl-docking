"""Structural checks for the organized CBF-3DoF controller archive."""

from pathlib import Path


CONTROLLERS_DIR = Path(__file__).resolve().parents[1]
CBF_DIR = CONTROLLERS_DIR / "CBF-3DoF"


def test_cbf_controller_uses_outer_archive_and_internal_package():
    assert CBF_DIR.is_dir()
    assert not (CONTROLLERS_DIR / "cbf_3DoF").exists()
    assert (CBF_DIR / "cbf_3dof" / "__init__.py").is_file()
    assert (CBF_DIR / "evaluations" / "evaluate_closing_cbf.py").is_file()
    assert (CBF_DIR / "evaluations" / "evaluate_final_approach_cbf.py").is_file()
    assert (CBF_DIR / "evaluations" / "evaluate_lqr_cbf.py").is_file()
    assert (CBF_DIR / "tuning" / "tune_closing_cbf.py").is_file()
    assert (CBF_DIR / "tuning" / "tune_final_approach_cbf.py").is_file()
    assert (CBF_DIR / "plotters" / "plot_constraint_and_control_diagnostics.py").is_file()
    assert (CBF_DIR / "plotters" / "plot_lqr_cbf_evaluation.py").is_file()


def test_cbf_result_reports_have_content_based_names():
    results = CBF_DIR / "results"
    assert (results / "final_approach_alignment_cbf_initial_evaluation.md").is_file()
    assert (results / "final_approach_multicontroller_cbf_evaluation.md").is_file()
    assert not (CBF_DIR / "ALIGNMENT_CBF_RESULTS.md").exists()
    assert not (CBF_DIR / "MULTICONTROLLER_RESULTS.md").exists()


def test_approved_incomplete_cbf_results_are_absent():
    closing = CBF_DIR / "results" / "closing" / "evaluations"
    final = CBF_DIR / "results" / "final_approach" / "evaluations"
    assert not (closing / "campaign_phase_balanced_best_full_cbf_seeds_0_49_h20000").exists()
    assert not (closing / "campaign_vanilla_final_full_cbf_seeds_0_49_h20000").exists()
    assert not any(final.rglob("*_incomplete_previous_run"))
