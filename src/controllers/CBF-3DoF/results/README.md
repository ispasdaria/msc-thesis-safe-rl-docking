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

means SAC training seed 10, tuned gains, no slack, the historical cone and
speed CBF families enabled, and held-out evaluation seeds 200 through 299.

Each evaluation directory contains:

- `run_configuration.json`: exact checkpoint, seed range, simulator mode, and
  CBF parameters.
- `per_episode_metrics.csv`: one row per filter variant and evaluation seed.
- `aggregate_metrics_by_filter_variant.csv`: aggregate outcome and solver
  metrics.
- `step_timeseries_<filter>_eval_seedXXXX.npz`: complete step-level state,
  action, barrier, residual, slack, and solver diagnostics. Runs made with the
  current evaluator additionally record post-step barriers and explicit
  protected-set violation flags; older archived runs contain the pre-step
  barrier samples only.

The tuning directory contains the Optuna database, complete gain-search trial
table, search plan, and selected gain configuration. The database is retained
so that the study can be audited or resumed.

## Gain labels

- `untuned`: `cone_k1 = cone_k2 = 0.1`, `speed_alpha = 0.5`.
- `tuned`: the historical cone/speed selection `cone_k1 = cone_k2 = 0.3`,
  `speed_alpha = 0.5`.
- `cone_tuned_alignment_initial`: the historical tuned cone/speed gains plus
  initial, not yet jointly tuned, alignment gains `attitude_k1 =
  attitude_k2 = 0.3` and `angular_rate_alpha = 0.5`.

The cone slope `kappa = 0.1` and port-speed limit `0.15 m/s` are fixed safety
specifications for every gain label.

## Historical scope

Directories whose names contain `combined_cbf` were generated before the
alignment CBF was added, so that label means cone plus port speed. New
evaluations use `all_cbf` when cone, port speed, attitude, and angular-rate
protection are enabled together. See
`final_approach_alignment_cbf_initial_evaluation.md` for the
scope and interpretation of the initial alignment-filter runs.
