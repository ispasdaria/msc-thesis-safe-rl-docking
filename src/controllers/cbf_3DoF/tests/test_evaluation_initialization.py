import argparse

import numpy as np

from controllers.evaluation_initialization import (
    add_initial_range_arguments,
    initial_ranges_from_args,
    sample_initial_state,
)
from controllers.cbf_3DoF.evaluate_lqr_cbf import resolve_initial_ranges, run_rollout


def test_explicit_evaluation_ranges_generate_repeatable_six_component_state():
    parser = argparse.ArgumentParser()
    add_initial_range_arguments(parser)
    args = parser.parse_args(
        [
            "--x-o-min", "235", "--x-o-max", "365",
            "--y-o-min", "-6.5", "--y-o-max", "6.5",
            "--xdot-o-min", "-0.075", "--xdot-o-max", "0.005",
            "--ydot-o-min", "-0.028", "--ydot-o-max", "0.028",
            "--psi-min-deg", "-2.75", "--psi-max-deg", "2.75",
            "--omega-z-min-deg-s", "-0.14",
            "--omega-z-max-deg-s", "0.14",
        ]
    )
    ranges = initial_ranges_from_args(args)

    first = sample_initial_state(12, ranges)
    repeated = sample_initial_state(12, ranges)

    np.testing.assert_array_equal(first, repeated)
    assert 235 <= first[0] <= 365
    assert -6.5 <= first[1] <= 6.5
    assert -0.075 <= first[2] <= 0.005
    assert -0.028 <= first[3] <= 0.028
    assert np.deg2rad(-2.75) <= first[4] <= np.deg2rad(2.75)
    assert np.deg2rad(-0.14) <= first[5] <= np.deg2rad(0.14)


def test_plain_final_lqr_rollout_accepts_explicit_initial_ranges():
    ranges = {
        "x_o": (300.0, 300.0),
        "y_o": (0.0, 0.0),
        "xdot_o": (-0.03, -0.03),
        "ydot_o": (0.0, 0.0),
        "psi_deg": (0.0, 0.0),
        "omega_z_deg_s": (0.0, 0.0),
    }

    row = run_rollout(
        "final", 0, "testing", 1.0, initial_ranges=ranges, use_cbf=False
    )

    assert row["phase"] == "final"
    assert row["intervention_count"] == 0
    assert row["qp_failure_count"] == 0


def test_native_initialization_bypasses_explicit_evaluation_ranges():
    args = argparse.Namespace(native_initialization=True)

    assert resolve_initial_ranges(args) is None
