from pathlib import Path


SAC_DIR = Path(__file__).resolve().parent


def test_full_mission_plotter_uses_retained_figure_directory_and_formats() -> None:
    source = (SAC_DIR / "plot_representative_mixed_full_mission_trajectories.py").read_text(
        encoding="utf-8"
    )

    assert 'HERE / "figures" / "full_mission"' in source
    assert 'with_suffix(".pdf")' not in source
    assert "full_mission_state_evolution_seed0803_lqr_" not in source
    assert "def make_state_evolution_figure(" not in source


def test_closing_plotter_uses_closing_figure_directory() -> None:
    source = (SAC_DIR / "closing" / "plot_v15_state_convergence_seed15.py").read_text(
        encoding="utf-8"
    )

    assert 'Path(__file__).resolve().parent / "figures"' in source
    assert 'ROOT / "figures"' not in source
