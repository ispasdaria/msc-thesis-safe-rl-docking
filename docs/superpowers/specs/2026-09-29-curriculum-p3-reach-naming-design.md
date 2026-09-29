# Curriculum and P3-Reach Naming Design

## Objective

Reorganize `src/controllers/SAC-3DoF/closing/curriculum_and_p3_reach` so that script and artifact names state the experiment that was implemented. Historical reward-version identifiers remain present, while opaque trial labels and unnecessary timestamps are replaced by descriptions of the relevant reset strategy, reward formulation, safety semantics, training source, timestep, or evaluation purpose.

All changes remain uncommitted unless a later user request explicitly authorizes a commit.

## Scope

The reorganization covers:

- training, evaluation, tuning, and plotting script names;
- model, log, and result group names;
- individual run-directory names mirrored across artifact types;
- tests and all live path references;
- explanatory READMEs for each experiment group;
- the family-level README and repository documentation that links to these artifacts.

No model, normalization state, log, dataset, result table, configuration, or figure may be deleted or regenerated as part of this work.

## Naming rules

- Preserve reward labels such as `p3_reach_v12`, `v19`, and `v21`.
- Include a reward version in a script filename only when that script is hard-coded to the corresponding reward or environment implementation. Keep reusable trainers and evaluators version-neutral.
- Use snake case and content-based technical descriptions.
- Retain seed, training budget, gamma, retry, checkpoint, and Optuna trial identifiers when they distinguish experiments.
- Remove timestamps from directory names when the remaining fields uniquely identify the run. Preserve original timestamps in stored configuration files and document them when relevant.
- Use the same run name under `models`, `logs`, and `results` whenever those directories describe the same execution.
- Do not rewrite binary artifacts or historical result contents solely to replace embedded old paths.
- Update executable defaults, plotter inputs, tests, and live documentation to the new paths.

## Experiment groups

### Reset-distribution curriculum

Directory name: `reset_distribution_curriculum`

This group contains staged-reset experiments that begin near P3 and expand toward the complete P2 closing distribution. It includes warm-started and from-scratch variants, gamma comparisons, bootstrap comparisons, and the V19/V20 safety-terminal curriculum runs.

Representative names:

- `sac_closing_rewardv15_trial16_curriculum_seed0` becomes `p3_reach_v15_trial16_staged_reset_curriculum_seed0`.
- `sac_closing_rewardv15_trial16_curriculum_easy_s0_seed0` becomes `p3_reach_v15_trial16_near_p3_stage0_seed0`.
- `sac_closing_rewardv15_trial16_curriculum_easy_s0_retry1_seed0` becomes `p3_reach_v15_trial16_near_p3_stage0_retry1_seed0`.
- `sac_closing_rewardv15_trial16_curriculum_from_scratch_seed0` becomes `p3_reach_v15_staged_reset_curriculum_from_scratch_seed0`.
- `gamma_09995_seed0` becomes `p3_reach_v15_curriculum_gamma0p9995_seed0`.
- `bootstrap_gamma_09995_seed0` becomes `p3_reach_v15_bootstrap_curriculum_gamma0p9995_seed0`.
- `curriculum_v19_seed0` becomes `p3_reach_v19_safety_terminal_staged_curriculum_seed0`.
- `curriculum_v20_seed0` becomes `p3_reach_v20_safety_terminal_staged_curriculum_seed0`.
- Retry and full-distribution variants retain `retry1` and `full_distribution` in their names.

### Nonterminal-safety continuation

Directory name: `nonterminal_safety_continuation`

This group contains continuation experiments initialized from the retained reward-V15 Trial-16 policy. Hard safety violations remain penalized but do not terminate the episode.

Run names:

- `p3_reach_v15_nonterminal_safety_continuation_from_trial16_400k_plus300k_seed0`
- `p3_reach_v1_nonterminal_safety_continuation_from_v15_trial16_400k_plus300k_seed0`

### From-scratch reward progression

Directory name: `from_scratch_reward_progression`

This group contains newly initialized SAC policies trained on successive P3-reaching reward formulations with nonterminal safety violations. Each run name includes the reward version and its defining change:

- V1: `p3_reach_v1_terminal_error_shaping`
- V2: `p3_reach_v2_physically_scaled_terminal_capture`
- V3: `p3_reach_v3_global_goal_progress`
- V4: `p3_reach_v4_log_scaled_component_balance`
- V5: `p3_reach_v5_gated_terminal_velocity_capture`
- V6: `p3_reach_v6_tolerance_normalized_smooth_worst_component`
- V7: `p3_reach_v7_balanced_mean_and_worst_component`
- V8: `p3_reach_v8_symmetric_progress_and_unsafe_occupancy`
- V9: `p3_reach_v9_distance_gated_terminal_velocity`
- V10: `p3_reach_v10_dense_terminal_capture_and_departure_deterrence`
- V11: `p3_reach_v11_bounded_capture_progress`
- V12: `p3_reach_v12_balanced_capture_progress_failed_approach_reset`
- V13: `p3_reach_v13_worst_unsatisfied_component_progress`
- V14: `p3_reach_v14_alignment_soft_constraint`
- V15: `p3_reach_v15_lateral_cone_occupancy_constraint`
- V16: `p3_reach_v16_lateral_cone_progress`
- V17: `p3_reach_v17_cone_feasibility_gated_progress`
- V18: `p3_reach_v18_position_velocity_weighted_capture`
- V19: `p3_reach_v19_lqr_axial_braking_envelope`

The training budget, gamma override, seed, and other differentiators follow the formulation label. For example, `p3_reach_v12_balanced_capture_progress_failed_approach_reset_gamma0p999_600k_seed0`.

The retained V12 phase-balanced replay run belongs to `phase_balanced_replay`, not this group. If moving it would create a cross-family artifact change outside the approved directory, it remains here under a clearly marked `cross_family_phase_balanced_replay_reference` subgroup with a README explaining ownership.

### Safety-terminal braking experiments

Directory name: `safety_terminal_braking_experiments`

This group contains from-scratch safety-terminal reward experiments:

- V19: LQR-inspired axial braking envelope with hard safety termination.
- V20: V19 braking guidance with calibrated terminal success/failure rewards and removed persistent goal/time costs.
- V21: V20 terminal calibration with a coupled docking-port velocity and alignment envelope.

Run names preserve version, formulation, budget, and seed.

### Ten-hertz adaptive terminal continuation

Directory name: `ten_hz_adaptive_terminal_continuation`

This group contains the V15-dt continuation at `dt = 0.1 s`, initialized from the selected phase-balanced checkpoint. The run uses distance-dependent action cadence, deterministic replay warm-up, and terminal-focused reset coverage.

The retained run name is `p3_reach_v15dt_ten_hz_adaptive_cadence_continuation_200k_seed0_run2`.

### P3-reach V12 Optuna tuning

Directory name: `p3_reach_v12_optuna`

This group contains the 600,000-step V12 hyperparameter study. Optuna trial numbers remain unchanged. The study and trial names explicitly state `p3_reach_v12`, the 600k budget, and that policies are trained from scratch.

## Script names

- `train_sac_closing_curriculum.py` becomes `train_p3_reach_reset_distribution_curriculum.py`.
- `train_sac_closing_no_safety_termination.py` becomes `train_p3_reach_nonterminal_safety_continuation.py`.
- `train_sac_closing_p3reach_from_scratch.py` becomes `train_p3_reach_reward_from_scratch.py`.
- `train_sac_closing_braking_envelope.py` becomes `train_p3_reach_v19_lqr_axial_braking_envelope.py`.
- `train_sac_closing_braking_terminal_v20.py` becomes `train_p3_reach_v20_calibrated_safety_terminal.py`.
- `train_sac_closing_port_lqr_envelope_v21.py` becomes `train_p3_reach_v21_port_velocity_alignment_envelope.py`.
- `train_sac_closing_adaptive_dt01_terminal.py` becomes `train_p3_reach_v15dt_ten_hz_adaptive_cadence.py`.
- `tune_sac_closing_p3reach_v12_optuna.py` becomes `tune_p3_reach_v12_from_scratch_optuna.py`.
- `evaluate_sac_closing_p3reach.py` becomes `evaluate_p3_reach_checkpoints.py`.
- Plotters are renamed to identify their input experiment and plotted diagnostic.
- Tests are renamed to match the production module or behavior they verify.

## Documentation

The family README provides:

- an experiment-progression overview;
- a table mapping V1-V21 to the implemented reward or termination change;
- the relationship between training entry points and artifact groups;
- instructions for reproducing evaluations and interpreting stored outputs.

Each experiment group receives a README that states:

- the hypothesis or failure mode being tested;
- the implemented environment, reward, reset, termination, and timestep behavior;
- initialization and source checkpoints;
- variants and run-name fields;
- the artifacts stored under models, logs, and results;
- the observed outcome supported by committed result files;
- whether the experiment is retained as a selected method, diagnostic comparison, negative result, or historical progression step.

## Correctness and preservation

- Record a pre-move inventory of every `.zip`, `.pkl`, `.npz`, `.csv`, `.json`, `.log`, and `.png` artifact by SHA-256 hash.
- After the moves, require the same artifact count and hash multiset.
- Check that model ZIP files remain paired with their required `VecNormalize` pickle files.
- Compile every Python file.
- Import executable modules without invoking their `main()` functions.
- Run the family tests, SAC closing structural tests, and complete repository test suite.
- Search active Python and Markdown files for old paths and old script names.
- Remove only generated `__pycache__` and `.pytest_cache` directories.

## Non-goals

- No training or evaluation is rerun.
- No numerical experiment result is altered.
- No reward function, environment behavior, hyperparameter, checkpoint-selection rule, or model architecture is changed.
- No artifact is deleted because its result appears unsuccessful.
- No Git commit, merge, push, or branch operation is performed.
