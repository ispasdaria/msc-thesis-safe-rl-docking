# Final-Approach LQR Out-of-Distribution Robustness

This experiment evaluates the plain final-approach LQR controller from initial
conditions outside the standard evaluation range. The purpose is to measure how
success and safety change as the initial-condition distribution becomes wider.

## Shared configuration

- Controller: `LQRFinalController`
- Phase: final approach
- Control interval: `1 s`
- Maximum episode duration: `5,000 s`
- Integrator: RK4
- Mode: `testing`
- Evaluation seeds: 0 to 49
- CBF safety filter: disabled
- SAC policy: not used
- VecNormalize statistics: not used

## Standard evaluation range

- Axial position: 280 to 320 m
- Lateral position: -3 to 3 m
- Axial velocity: -0.05 to -0.01 m/s
- Lateral velocity: -0.01 to 0.01 m/s
- Attitude: -1 to 1 degrees
- Angular rate: -0.05 to 0.05 degrees/s

## Moderately expanded initial conditions

The `moderately_expanded_initial_conditions` directory uses the following
ranges:

- Axial position: 235 to 365 m
- Lateral position: -6.5 to 6.5 m
- Axial velocity: -0.075 to 0.005 m/s
- Lateral velocity: -0.028 to 0.028 m/s
- Attitude: -2.75 to 2.75 degrees
- Angular rate: -0.14 to 0.14 degrees/s

Results:

- 49 successful episodes from 50
- 98 percent success rate
- One hard angular-rate safety violation
- No docking-cone, translational-velocity, or attitude violations

## Widely expanded initial conditions

The `widely_expanded_initial_conditions` directory uses the following ranges:

- Axial position: 220 to 380 m
- Lateral position: -8 to 8 m
- Axial velocity: -0.085 to 0.015 m/s
- Lateral velocity: -0.035 to 0.035 m/s
- Attitude: -3.5 to 3.5 degrees
- Angular rate: -0.18 to 0.18 degrees/s

Results:

- 38 successful episodes from 50
- 76 percent success rate
- 12 hard angular-rate safety violations
- No docking-cone, translational-velocity, or attitude violations

## File contents

Each experiment directory contains:

- `run_configuration.json`, which records the controller, simulation settings,
  seeds, and initial-state ranges.
- `per_episode_metrics.csv`, which contains the outcome, safety indicators,
  duration, and terminal measurements for every seed.
- `aggregate_metrics.json`, which contains success rates, violation counts,
  mean duration, and mean terminal-state measurements.

## Interpretation

The plain LQR controller remains reliable under the moderately expanded range.
Performance decreases under the widest range because larger initial angular
rates cause hard angular-rate safety violations. No CBF intervention is present
in either experiment, so these results describe the nominal LQR controller
without an external safety filter.
