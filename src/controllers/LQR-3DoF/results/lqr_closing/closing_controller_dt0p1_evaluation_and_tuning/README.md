# LQR Closing Controller Evaluation and Tuning at 0.1 s

This directory contains the evaluation, diagnosis, and tuning records for the
LQR closing controller with a simulation time step of `0.1 s`. Unless stated
otherwise, the development evaluations use seeds 0-49, a `12,000 s` episode
limit, and the selected `Q` and `R` scales of 1.

## Baseline files

- `closing_phase_0_49_dt0p1.csv` contains the per-rollout results from the
  original 50-seed evaluation.
- `closing_phase_0_49_dt0p1_summary.json` contains its aggregate metrics and
  configuration.
- `closing_phase_seeds_0_49_dt0p1_errors.png` shows terminal errors across the
  evaluated seeds.
- `closing_phase_seeds_0_49_dt0p1_error_histories.npz` stores the histories
  used by the terminal-error plots.

The original stored evaluation reports 23 successes, 27 timeouts, and no hard
safety violations. It was generated with an earlier repository state. A later
paired run with the current implementation produced 26 successes at the same
`12,000 s` cutoff. The difference is associated with code or configuration
drift, not with the episode-duration change. The 300-seed held-out evaluation
provides the stronger estimate of current controller performance.

## Directory structure

- `heldout_validation_300_seeds` contains the independent validation campaign.
- `lateral_steady_state_error_diagnostics` explains the dominant lateral
  terminal-position failures.
- `broad_lateral_position_velocity_weight_sweep` and
  `fine_lateral_velocity_weight_sweep` contain the lateral `Q`-weight studies.
- `episode_limit_13000s_evaluation`, `episode_limit_14000s_evaluation`, and
  `episode_duration_comparison` assess sensitivity to the episode limit.
- `representative_rollout_histories` contains selected trajectories used for
  behavior plots.
- `baseline_evaluation_partial_runs` contains the partial result files combined
  into the original 50-seed baseline.

Each subdirectory contains a README describing its purpose and recorded result.
