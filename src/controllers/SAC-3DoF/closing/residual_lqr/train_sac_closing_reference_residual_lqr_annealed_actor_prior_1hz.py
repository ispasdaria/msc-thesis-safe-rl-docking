"""Train the 1 Hz reference-residual configuration with an annealed actor prior."""

from __future__ import annotations

import sys

from train_sac_closing_reference_residual_lqr_stratified import main


DEFAULT_ARGUMENTS = [
    "--experiment-name",
    (
        "sac_closing_lqr_reference_residual_"
        "pilot_e_actor_prior_annealed_1hz_50k_seed0"
    ),
    "--pilot-label",
    "E",
    "--zero-residual-actor-weight",
    "1.0",
    "--zero-residual-actor-final-weight",
    "0.1",
    "--zero-residual-actor-anneal-end-step",
    "30000",
]


if __name__ == "__main__":
    main([*DEFAULT_ARGUMENTS, *sys.argv[1:]])
