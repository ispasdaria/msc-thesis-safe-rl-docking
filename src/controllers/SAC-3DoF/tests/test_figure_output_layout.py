from pathlib import Path


SAC_DIR = Path(__file__).resolve().parents[1]


def test_full_mission_plotter_uses_retained_figure_directory_and_formats() -> None:
    plotter = (
        SAC_DIR
        / "plotters"
        / "plot_representative_mixed_full_mission_trajectories.py"
    )
    source = plotter.read_text(encoding="utf-8")

    assert (
        'SAC_ROOT / "figures" / "representative_full_mission_figures"'
        in source
    )
    assert 'with_suffix(".pdf")' not in source
    assert "full_mission_state_evolution_seed0803_lqr_" not in source
    assert "def make_state_evolution_figure(" not in source
    assert (
        "representative_lqr_closing_sac_final_trajectories_seeds_306_403_803"
        in source
    )
    assert "final_sac_policy_comparison_after_lqr_handover_seed_803" in source


def test_closing_plotter_uses_closing_figure_directory() -> None:
    source = (
        SAC_DIR
        / "closing"
        / "vanilla_sac"
        / "plotters"
        / "plot_v15_state_convergence_seed15.py"
    ).read_text(encoding="utf-8")

    assert 'Path(__file__).resolve().parents[1]' in source
    assert '/ "results"' in source
    assert '/ "terminal_state_convergence"' in source


def test_evaluation_summary_uses_headless_backend_before_pyplot() -> None:
    source = (
        SAC_DIR / "plotters" / "plot_sac_evaluation_summary.py"
    ).read_text(encoding="utf-8")

    backend_selection = source.index('matplotlib.use("Agg")')
    pyplot_import = source.index("import matplotlib.pyplot as plt")
    assert backend_selection < pyplot_import
