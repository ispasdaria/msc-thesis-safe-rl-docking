# CBF result archive

This directory contains the committed tuning and evaluation evidence for the
3-DoF final-approach CBF-QP safety filter.

## Naming convention

Evaluation-directory names identify the SAC training seed, CBF gain status,
hard or soft constraint handling, evaluated filter variant, and evaluation
seed range. For example:

```text
sac_seed10_tuned_hard_combined_cbf_holdout_eval_seeds0200_0299
```

means SAC training seed 10, tuned gains, no slack, both CBF families enabled,
and held-out evaluation seeds 200 through 299.

Each evaluation directory contains:

- `run_configuration.json`: exact checkpoint, seed range, simulator mode, and
  CBF parameters.
- `per_episode_metrics.csv`: one row per filter variant and evaluation seed.
- `aggregate_metrics_by_filter_variant.csv`: aggregate outcome and solver
  metrics.
- `step_timeseries_<filter>_eval_seedXXXX.npz`: complete step-level state,
  action, barrier, residual, slack, and solver diagnostics.

The tuning directory contains the Optuna database, complete gain-search trial
table, search plan, and selected gain configuration. The database is retained
so that the study can be audited or resumed.

## Gain labels

- `untuned`: `cone_k1 = cone_k2 = 0.1`, `speed_alpha = 0.5`.
- `tuned`: `cone_k1 = cone_k2 = 0.3`, `speed_alpha = 0.5`.

The cone slope `kappa = 0.1` and port-speed limit `0.15 m/s` are fixed safety
specifications in both cases.
