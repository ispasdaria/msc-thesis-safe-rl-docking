# Final-approach SAC + CBF evaluations

This directory contains grouped final-approach evaluation artifacts:

- `controller_campaigns/` contains selected controller campaigns.
- `alignment_screen_100k/` contains the 100k-policy alignment screen.
- `alignment_screen_300k/` contains the 300k-policy alignment screen and schema-2 diagnostics.
- `cbf_development_and_holdout/` contains gain development, filter ablations, and held-out studies.

Each complete evaluation directory normally contains:

- `run_configuration.json`
- `per_episode_metrics.csv`
- `aggregate_metrics_by_filter_variant.csv`
- per-step trajectory files named `step_timeseries_*.npz`

The run names identify the SAC training configuration and seed, the CBF
comparison, and the evaluation-seed range. The explicitly incomplete previous
run was removed.

Representative figures derived from selected evaluations are stored separately
in `../plots/`.
