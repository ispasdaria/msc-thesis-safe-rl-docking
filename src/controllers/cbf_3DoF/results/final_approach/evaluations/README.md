# Final-approach SAC + CBF evaluations

This directory contains the raw final-approach evaluation artifacts. Each
evaluation directory preserves its original descriptive name and normally
contains:

- `run_configuration.json`
- `per_episode_metrics.csv`
- `aggregate_metrics_by_filter_variant.csv`
- per-step trajectory files named `step_timeseries_*.npz`

The directory names identify the SAC training configuration and seed, the CBF
comparison, and the evaluation-seed range. The intentionally retained
`incomplete_previous_run` directory is labelled explicitly and must not be
included in aggregate claims.

Thesis-ready figures derived from selected evaluations are stored separately
in `../plots/`.
