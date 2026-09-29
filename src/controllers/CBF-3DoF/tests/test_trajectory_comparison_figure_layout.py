import importlib.util
from pathlib import Path
import sys
import unittest

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "plotters"
    / "plot_sac_cbf_trajectory_comparisons.py"
)
DIAGNOSTIC_SCRIPT = SCRIPT.with_name("plot_constraint_and_control_diagnostics.py")


def test_diagnostic_plotter_help_has_no_output_side_effects(monkeypatch):
    spec = importlib.util.spec_from_file_location("cbf_diagnostic_plotter", DIAGNOSTIC_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    called = []
    monkeypatch.setattr(module, "closing_physical_figure", lambda: called.append("closing"))
    monkeypatch.setattr(module, "final_control_figure", lambda: called.append("final"))
    monkeypatch.setattr(sys, "argv", [str(DIAGNOSTIC_SCRIPT), "--help"])

    with pytest.raises(SystemExit) as exit_info:
        module.main()

    assert exit_info.value.code == 0
    assert called == []


class TrajectoryComparisonFigureLayoutTest(unittest.TestCase):
    def test_trajectory_comparison_uses_requested_titles_and_panel_order(self):
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('"round1": "Round 1, seed 12, final 100k (eval. seed 0)"', source)
        self.assertIn('"combined": (', source)
        self.assertIn('"phase-balanced", "round1", "round3", "round5_seed10"', source)
        self.assertIn(
            'fig3.suptitle("Representative nominal vs. CBF-filtered trajectories", fontsize=13, fontweight="bold", x=0.075, y=0.935, ha="left")',
            source,
        )
        self.assertIn('add_shared_labels_and_legend(fig3, axes, label_fontsize=11.5, legend_fontsize=8.2)', source)
        self.assertIn(
            '"round1": "Round 1, seed 12, final 100k,\\ninitial 100% success rate (eval. seed 0)"',
            source,
        )
        self.assertIn(
            '"round3": "Round 3, seed 18, best 160k,\\ninitial 42% success rate (eval. seed 39)"',
            source,
        )
        self.assertIn(
            '"round5_seed10": "Round 5, seed 10, best 490k,\\ninitial 66% success rate (eval. seed 7)"',
            source,
        )
        self.assertIn('save_figure(fig3, "closing_and_final_approach_representative_trajectories_2x2")', source)


if __name__ == "__main__":
    unittest.main()
