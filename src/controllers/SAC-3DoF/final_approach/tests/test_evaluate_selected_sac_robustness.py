import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "evaluate_selected_sac_robustness_expanded_bounds.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_selected_sac_robustness_expanded_bounds", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EnlargedRangeEvaluationTests(unittest.TestCase):
    def test_default_outputs_use_controller_local_representative_names(self):
        evaluator = load_module()

        self.assertEqual(
            evaluator.DEFAULT_OUTPUT_ROOT,
            SCRIPT.parent / "results" / "robustness_evaluations"
            / "robustness_expanded_test_bounds",
        )
        self.assertEqual(
            evaluator.POLICIES["round1_seed12_final100k"]["output_directory"],
            "round1_seed12_100k_steps",
        )
        self.assertEqual(
            evaluator.POLICIES["round5_seed11_final600k"]["output_directory"],
            "round5_seed11_600k_steps_buffer600k",
        )

    def test_evaluator_does_not_write_redundant_combined_rollouts(self):
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertNotIn('"all_rollouts.csv"', source)
        self.assertIn('"rollout_results.csv"', source)
        self.assertIn('"evaluation_summary.json"', source)
        self.assertIn('"policy_comparison_summary.json"', source)

    def test_repository_model_paths_are_stored_relative_to_repository(self):
        evaluator = load_module()
        model_path = (
            evaluator.REPO_ROOT
            / "src"
            / "controllers"
            / "SAC-3DoF"
            / "final_approach"
            / "models"
            / "retained_policies"
            / "policy.zip"
        )

        self.assertEqual(
            evaluator.portable_path(model_path),
            "src/controllers/SAC-3DoF/final_approach/models/retained_policies/policy.zip",
        )

    def test_seeded_initial_states_are_repeatable_and_inside_enlarged_bounds(self):
        evaluator = load_module()

        first = evaluator.sample_initial_state(17)
        repeated = evaluator.sample_initial_state(17)

        np.testing.assert_array_equal(first, repeated)
        self.assertEqual(first.shape, (6,))
        bounds = (
            (260.0, 340.0),
            (-4.0, 4.0),
            (-0.055, -0.01),
            (-0.015, 0.015),
            (np.deg2rad(-1.5), np.deg2rad(1.5)),
            (np.deg2rad(-0.075), np.deg2rad(0.075)),
        )
        self.assertTrue(all(low <= value <= high for value, (low, high) in zip(first, bounds)))


    def test_summary_counts_outcomes_and_reports_requested_terminal_metrics(self):
        evaluator = load_module()
        rows = [
        {
            "success": True,
            "termination_reason": "success_docking",
            "duration_s": 10.0,
            "terminal_port_distance_m": 0.08,
            "terminal_port_speed_mps": 0.04,
            "terminal_attitude_error_deg": 0.5,
            "terminal_angular_rate_error_deg_s": 0.02,
            "action_saturation_fraction": 0.1,
            "docking_cone_violation_count": 0,
        },
        {
            "success": False,
            "termination_reason": "timeout",
            "duration_s": 20.0,
            "terminal_port_distance_m": 2.0,
            "terminal_port_speed_mps": 0.06,
            "terminal_attitude_error_deg": 1.5,
            "terminal_angular_rate_error_deg_s": 0.04,
            "action_saturation_fraction": 0.3,
            "docking_cone_violation_count": 2,
        },
        {
            "success": False,
            "termination_reason": "safety_violation",
            "duration_s": 30.0,
            "terminal_port_distance_m": 3.0,
            "terminal_port_speed_mps": 0.08,
            "terminal_attitude_error_deg": 2.5,
            "terminal_angular_rate_error_deg_s": 0.06,
            "action_saturation_fraction": 0.5,
            "docking_cone_violation_count": 1,
        },
        ]

        summary = evaluator.summarize("example", rows)

        self.assertEqual(summary["successful_rollouts"], 1)
        self.assertEqual(summary["success_rate"], 1 / 3)
        self.assertEqual(summary["timeouts"], 1)
        self.assertEqual(summary["hard_safety_terminations"], 1)
        self.assertEqual(summary["termination_reason_breakdown"], {
            "safety_violation": 1,
            "success_docking": 1,
            "timeout": 1,
        })
        self.assertEqual(summary["mean_duration_s"], 20.0)
        self.assertEqual(summary["std_duration_s"], np.std([10.0, 20.0, 30.0]))
        self.assertEqual(summary["mean_terminal_port_distance_m"], np.mean([0.08, 2.0, 3.0]))
        self.assertEqual(summary["max_terminal_port_distance_m"], 3.0)
        self.assertEqual(summary["docking_cone_violation_count"], 3)

    def test_custom_initial_ranges_control_all_six_sampled_components(self):
        evaluator = load_module()
        ranges = {
            "x_o": (245.0, 245.0),
            "y_o": (-5.5, -5.5),
            "xdot_o": (-0.065, -0.065),
            "ydot_o": (-0.022, -0.022),
            "psi_deg": (-2.2, -2.2),
            "omega_z_deg_s": (-0.11, -0.11),
        }

        state = evaluator.sample_initial_state(0, ranges)

        np.testing.assert_allclose(
            state,
            [245.0, -5.5, -0.065, -0.022, np.deg2rad(-2.2), np.deg2rad(-0.11)],
        )

    def test_cli_accepts_paths_seed_count_and_all_range_overrides(self):
        evaluator = load_module()
        args = evaluator.parse_args(
            [
                "--policy", "round1_seed12_final100k",
                "--model-path", "model.zip",
                "--vecnormalize-path", "stats.pkl",
                "--seed-start", "0",
                "--num-seeds", "50",
                "--x-o-min", "245", "--x-o-max", "355",
                "--y-o-min", "-5.5", "--y-o-max", "5.5",
                "--xdot-o-min", "-0.065", "--xdot-o-max", "-0.005",
                "--ydot-o-min", "-0.022", "--ydot-o-max", "0.022",
                "--psi-min-deg", "-2.2", "--psi-max-deg", "2.2",
                "--omega-z-min-deg-s", "-0.11",
                "--omega-z-max-deg-s", "0.11",
            ]
        )

        self.assertEqual(args.model_path, Path("model.zip"))
        self.assertEqual(args.vecnormalize_path, Path("stats.pkl"))
        self.assertEqual(args.seed_end, 49)
        self.assertEqual(args.initial_ranges["x_o"], (245.0, 355.0))
        self.assertEqual(args.initial_ranges["omega_z_deg_s"], (-0.11, 0.11))


if __name__ == "__main__":
    unittest.main()
