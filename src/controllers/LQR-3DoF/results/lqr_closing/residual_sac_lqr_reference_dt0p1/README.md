# Residual-SAC LQR reference at `dt = 0.1 s`

This directory contains the pure-LQR reference dataset used to validate the
closing-phase residual-SAC environments.

Residual SAC augments the closing LQR controller with a learned correction.
The direct-residual formulation adds the SAC output to the LQR action. The
reference-residual formulation modifies the reference tracked by LQR. A zero
residual must reproduce the pure-LQR trajectory and terminal outcome.

`lqr_dt0p1_validation_seeds100_109.csv` records pure-LQR rollouts for testing
seeds 100 through 109 with the following configuration:

- controller timestep: `0.1 s` (`10 Hz`)
- integration method: RK4
- maximum rollout duration: `12,000 s`
- evaluation mode: testing

The dataset contains four successful handovers and six timeouts. No safety
violations were recorded.

`src/controllers/SAC-3DoF/closing/validate_residual_lqr_dt01_baseline.py`
compares zero-residual direct and reference wrappers against this dataset. The
retained validation manifest reports identical wrapper states and actions and
matching pure-LQR terminal outcomes.

The CSV is a validation reference, not the output of a learned SAC policy.
