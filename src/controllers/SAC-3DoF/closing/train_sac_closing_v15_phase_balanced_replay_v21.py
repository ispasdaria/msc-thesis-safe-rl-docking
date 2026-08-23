"""Run the guarded V2.1 no-priority closing SAC pilot.

V2.1 is a controlled correction of V2: it disables sparse event-marker
sampling, requires five complete standard-P2 anchor episodes, and uses a
deterministic 20% mixture of easy one-component near-P3 resets.
"""

from __future__ import annotations

from train_sac_closing_v15_phase_balanced_replay_v2 import (
    build_parser,
    main as run_v2,
)


def parse_args():
    parser = build_parser()
    parser.set_defaults(
        method_version="v2.1",
        experiment_name=(
            "sac_closing_v15_trial16_terminal_phase_balanced_v21_50k_seed0"
        ),
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
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_v2(parse_args())
