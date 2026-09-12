# Final-Approach Control-Effort Comparison Design

## Objective

Add a reproducible, paired comparison of applied physical control effort for the final-approach LQR, retained SAC Round 1 seed 12 final-100k policy, and SAC+CBF Round 5 seed 10 best-490k policy over evaluation seeds 0--49.

The analysis reports equivalent translational accumulated Delta-v, integrated rotational effort, maneuver duration, physical actuator saturation, and CBF intervention context. It does not interpret equivalent Delta-v as propellant consumption.

## Existing Infrastructure and Exact Artifacts

The implementation reuses these repository components:

- `src/controllers/SAC-3DoF/final_approach/evaluate_retained_sac_lqr50.py` for the retained-policy evaluation convention.
- `src/controllers/SAC-3DoF/final_approach/evaluate_selected_policies_50.py` for selected-policy path conventions.
- `src/controllers/cbf_3DoF/evaluate_final_approach_cbf.py` for full CBF construction, filtering, diagnostics, and configuration conventions.
- `src/controllers/LQR-3DoF/tuning_lqr_final_approach/lqr_sweep_final.py` for final-approach LQR rollout conventions.
- `src/controllers/LQR-3DoF/lqr_3dof_final.py` for the selected final-approach LQR weights, gain construction, port-error mapping, and velocity schedule.
- `src/envs/docking3dof/gym_env.py` for normalized/physical action conversion.
- `src/envs/docking3dof/core.py` for test-state generation, body-frame actuator clipping, dynamics, termination, elapsed time, and command history.

The retained SAC artifacts are:

- `src/controllers/SAC-3DoF/final_approach/models/optuna12_100ksteps_buffer100k/sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12.zip`
- `src/controllers/SAC-3DoF/final_approach/models/optuna12_100ksteps_buffer100k/sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12_vecnormalize.pkl`

The SAC+CBF artifacts are:

- `src/controllers/SAC-3DoF/final_approach/models/optuna12_600ksteps_buffer600k/sac_final_approach_portobs_rewardv9_600ksteps_seed10_optuna12_buffer600k_best.zip`
- `src/controllers/SAC-3DoF/final_approach/models/optuna12_600ksteps_buffer600k/sac_final_approach_portobs_rewardv9_600ksteps_seed10_optuna12_buffer600k_best_vecnormalize.pkl`

Although the artifact basename identifies its 600k training campaign, the existing campaign directory and publication metadata identify the retained best artifact as the 490k checkpoint. The exact full-CBF settings come from `src/controllers/cbf_3DoF/results/final_approach/evaluations/campaign_round5_seed10_best490k_seeds_0_49/run_configuration.json`: full cone, speed, and alignment filtering; hard constraints; no slack; and the recorded gains, bounds, penalties, solver tolerances, and intervention tolerance.

## Environment and Command Semantics

Every controller uses final-approach testing initialization with evaluation seed passed both at environment construction and reset. The evaluation set must equal the integer sequence 0 through 49 for every controller. SAC inference is deterministic and each policy uses its matching frozen VecNormalize statistics with training and reward normalization disabled.

The source-of-truth final-approach timestep is 1.0 s. The source-of-truth physical body-frame limits are `[0.01, 0.01, 0.003]` for `[a_x, a_y, alpha_z]`, in m/s2, m/s2, and rad/s2 respectively.

Effort is always computed from the final clipped body-frame command stored as `action_body_clipped` by the core environment:

- For SAC, the normalized action is converted by the Gym environment and clipped by the core environment.
- For SAC+CBF, the nominal SAC action is converted to physical units, filtered by the full CBF, converted back only to pass through the normalized Gym interface, and then recovered from the core environment's final `action_body_clipped` record after the step.
- For LQR, the requested physical body-frame action is passed directly to the core environment and recovered after environment clipping.

The initial zero-action reset history entry is not an applied rollout timestep and is excluded from effort, duration, saturation, and intervention calculations.

## Metrics

For each rollout with applied commands `u_k = [a_x,k, a_y,k, alpha_z,k]` and timestep `dt`:

- `delta_v_eq_mps = sum(sqrt(a_x,k**2 + a_y,k**2) * dt)`.
- `J_alpha_radps = sum(abs(alpha_z,k) * dt)`.
- `duration_s` is the environment's final elapsed time and must equal `num_steps * dt` within floating-point tolerance.
- `saturation_fraction` is the fraction of applied timesteps on which at least one final applied component is at its corresponding physical bound within a documented absolute tolerance. This describes operation at the actuator bound, not CBF intervention.
- `cbf_intervention_fraction` is the fraction of SAC+CBF steps flagged as interventions by the existing CBF evaluator's intervention tolerance. It is contextual and excluded from both integrals.
- `path_length_m = sum(||r_p,k+1 - r_p,k||_2)` uses signed docking-port positions including the initial and terminal samples.
- `path_efficiency = ||r_p,0 - r_p,f||_2 / path_length_m` is dimensionless; values closer to one indicate a more direct geometric path.
- `J_delta_a_mps2 = sum(||a_k+1 - a_k||_2)` measures translational command variation.
- `J_delta_alpha_radps2 = sum(|alpha_k+1 - alpha_k|)` measures rotational command variation.

Translational and rotational control variation remain separate because their physical units are incompatible. These variation sums are not multiplied by the timestep: they measure total successive command change, as defined above.

All per-rollout effort values must be finite and nonnegative.

## Evaluation and Data Flow

A new integrated analysis script loads the exact artifacts and constructs all three controllers. For each controller and seed, it creates a fresh final/testing environment, executes the established deterministic rollout path, captures the final applied command after every plant step, and records success and termination without discarding failures.

Each controller/seed history is saved as a compressed NPZ containing at least:

- `time`
- `a_x_Bc_applied`
- `a_y_Bc_applied`
- `alpha_z_Bc_applied`
- signed `r_p_x` and `r_p_y` state samples, including initial and terminal positions

CBF histories also contain nominal and filtered command context and intervention flags when available.

The per-rollout CSV contains one row per controller and seed with the requested identifiers, outcomes, duration, effort metrics, saturation, step count, and CBF intervention fraction.

## Statistical Conventions

Primary thesis statistics use the intersection of seeds successful for all three controllers. If all controllers succeed on all 50 seeds, the primary set is seeds 0--49. Results for every rollout remain available regardless of outcome, and the output metadata states both the complete and primary populations.

Distributions report mean, sample standard deviation (`ddof=1`), median, Q1, Q3, IQR, minimum, and maximum. Quartiles use NumPy's default linear percentile method. Overall saturation percentage is total saturated applied timesteps divided by total applied timesteps in the selected population, rather than an unweighted mean of per-rollout fractions.

Paired differences are method minus LQR. Lower/higher counts exclude exact numerical ties, with ties reported separately in detailed output. Percentage change in means is `100 * (mean_method - mean_LQR) / mean_LQR`; negative means lower than LQR.

## Outputs

Create `src/controllers/analyze_final_approach_control_effort.py` and focused tests under `src/controllers/tests/`. Generated results go into `src/controllers/results/final_approach_control_effort/` and include:

- `per_rollout_control_effort.csv`
- `summary_statistics.csv`
- `paired_comparisons.csv`
- `run_configuration.json`
- `validation_report.json`
- `histories/<controller>/seed_XXXX.npz`
- `equivalent_translational_delta_v_paired.png` and `.pdf`
- `integrated_rotational_effort_paired.png` and `.pdf`

The figures combine compact controller distributions with lightly drawn seed-wise paired lines so correspondence is retained without making individual trajectories the primary visual weight. Translational and rotational metrics remain in separate figures and use the requested thesis labels.

The console output prints the two requested tables and a result-derived factual interpretation without fuel or propellant claims.

## Validation and Failure Handling

Before accepting results, the script validates exact paths, matching VecNormalize files, evaluation seeds, testing initialization, deterministic inference, timestep, action limits, full-CBF configuration, and applied-action provenance. It checks success counts and explicitly reports discrepancies.

One saved rollout is independently reloaded and both integrals are recomputed from its timestep history, then compared with the aggregate CSV. The script also checks history lengths, time spacing, duration consistency, applied-command bounds, finiteness, nonnegative efforts, and completeness of the controller/seed Cartesian product.

Any structural or provenance validation failure raises an error and prevents a misleading completed report. Controller rollout failures do not abort analysis: they are saved and reported, and primary statistics switch to the common successful set.

## Scope and Repository Safety

No controller behavior, reward, CBF setting, model, VecNormalize statistic, initial-condition distribution, actuator limit, timestep, success criterion, or termination rule is changed. Existing unrelated modified plot files and generated figures remain untouched.
