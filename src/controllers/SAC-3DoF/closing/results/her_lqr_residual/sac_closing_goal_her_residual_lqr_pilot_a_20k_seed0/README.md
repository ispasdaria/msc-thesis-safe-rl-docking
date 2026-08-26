# Aborted Pilot A configuration

This run was intentionally stopped during replay collection on 2026-08-26.
It proposed holding each SAC residual action for ten simulator seconds. That
design was rejected because the residual controller is intended to respond to
potentially unsafe behavior at the one-second control cadence.

No SAC model or validation checkpoint was obtained from this directory. The
remaining `experiment_config.json` is retained only as an audit record and
must not be treated as a completed experiment.

The replacement experiment is
`sac_closing_goal_her_residual_lqr_pilot_a_1hz_20k_seed0`, which keeps SAC and
LQR feedback decisions at 1 Hz while retaining the increased critic update
ratio and critic warm-up.
