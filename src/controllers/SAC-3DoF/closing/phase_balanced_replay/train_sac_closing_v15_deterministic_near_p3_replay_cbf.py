"""Train guarded V15 SAC with deterministic near-P3 replay and deployment CBF.

The default pilot protects the complete closing cone plus the total and axial
port-speed limits (the evaluator's historical ``all`` variant).  The nominal
SAC action remains in replay, the simulator executes the filtered action, and
the reward includes a small squared normalized correction penalty.
"""

from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_sac_closing_v15_guarded_simultaneous_target_replay import (
    build_parser,
    main as run_v2,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SELECTED_SOURCE_DIR = (
    SCRIPT_DIR
    / "models"
    / "multiseed"
    / "sac_closing_v15_trial16_terminal_phase_balanced_guarded600k_seed0"
    / "seed_0"
)
SELECTED_SOURCE_MODEL = SELECTED_SOURCE_DIR / "model_0100000_steps.zip"
SELECTED_SOURCE_VECNORMALIZE = (
    SELECTED_SOURCE_DIR / "model_0100000_steps_vecnormalize.pkl"
)


def parse_args():
    parser = build_parser()
    parser.set_defaults(
        method_version="v2.1-cbf-all-from-guarded100k",
        experiment_name=(
            "sac_closing_v15_guarded100k_terminal_phase_balanced_"
            "v21_cbf_all_50k_seed0"
        ),
        source_model=str(SELECTED_SOURCE_MODEL),
        source_vecnormalize=str(SELECTED_SOURCE_VECNORMALIZE),
        training_timesteps=50_000,
        checkpoint_steps=[10_000, 25_000, 50_000],
        minimum_anchor_transitions=60_000,
        minimum_anchor_episodes=5,
        priority_capture_fraction=0.0,
        warmup_near_p3_reset_fraction=0.50,
        near_p3_reset_fraction=0.20,
        near_p3_min_ratio=1.05,
        near_p3_max_ratio=1.50,
        near_p3_other_max_ratio=0.50,
        deterministic_reset_mixture=True,
        early_stop_after_steps=25_000,
        early_stop_failure_evals=1,
        early_stop_regression_evals=2,
        training_cbf_variant="all",
        cbf_intervention_penalty_weight=0.1,
        cbf_solver="osqp",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_v2(parse_args())
