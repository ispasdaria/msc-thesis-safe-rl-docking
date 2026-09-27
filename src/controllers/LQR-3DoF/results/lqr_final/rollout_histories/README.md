# Representative LQR Final-Approach Rollout Histories

This directory contains representative trajectories selected from the LQR
final-approach evaluation over seeds 0 to 49. The source evaluation is recorded
in `../final_approach_phase_0_49.csv`.

## Evaluation configuration

- Phase: final approach
- Controller: LQR final-approach controller
- Simulation time step: `1 s`
- Maximum rollout duration: `5,000 s`
- Mode: `testing`
- Q scale: `1`
- R scale: `1`
- Additional Q multipliers: none
- Additional R multipliers: none

All 50 source rollouts completed successfully.

## Selected histories

- `q1_r1_seed0000_first_success.npz` contains the first successful rollout.
- `q1_r1_seed0011_median_success.npz` contains the successful rollout with the
  median final-error norm.
- `q1_r1_seed0041_worst_final_error.npz` contains the rollout with the largest
  final-error norm. The rollout still completed successfully.
- `q1_r1_seed0004_highest_saturation.npz` contains the rollout with the largest
  actuator-saturation fraction. The rollout still completed successfully.

Each compressed NumPy file stores the time, state, requested and applied
command, final-error, controller-matrix, and rollout-metadata histories needed
for diagnostic plotting. The histories are selected examples rather than the
complete 50-rollout dataset.
