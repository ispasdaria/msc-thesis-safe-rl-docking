# SAC Closing Reorganization Design

## Objective

Reorganize `src/controllers/SAC-3DoF/closing` by experimental purpose so that each experiment family can be understood and reproduced without searching across central model, log, result, test, and plotting directories.

The reorganization must preserve experiment artifacts, executable behavior, and links from the rest of the repository. Comments and documentation will describe implemented behavior objectively and will not use conversational or thesis-specific wording.

## Preservation policy

All experiment families and stored artifacts will be preserved. The cleanup will not remove a script solely because a newer or similarly named script exists.

The current root scripts contain no exact duplicates. Short training scripts are retained as reproducibility presets when they provide experiment-specific arguments to a shared trainer. A script may be removed only if review proves that it has no unique behavior, configuration, artifact provenance, or external consumer.

Before moving files, an inventory of important artifacts will be recorded in memory for post-move comparison. After reorganization, tracked models, normalization data, numerical results, logs, demonstrations, and figures will be checked against the original inventory.

## Target structure

The `closing` root will contain:

- `README.md`, providing the experiment-family index and reproduction overview.
- Shared Python modules used by multiple experiment families.
- Seven experiment-family directories.

The experiment-family directories are:

1. `vanilla_sac`
   - Reward-shaping trials.
   - Optuna studies.
   - Standard single-run and multiseed SAC.
   - Observation-space comparison.
   - Action-hold and standard timestep variants that do not depend on another experimental method.

2. `curriculum_and_p3_reach`
   - Curriculum training and comparison.
   - P3-reach training and tuning.
   - Near-P3 reset studies.
   - No-safety-termination and braking-envelope investigations.

3. `phase_balanced_replay`
   - Phase-balanced replay implementations and experiments.
   - V1, V2, V2.1, guarded, and CBF variants.
   - Retained diagnostics, figures, and closest-approach analyses.

4. `lqr_demonstration_learning`
   - LQR demonstration generation and curation.
   - Replay-prefill SAC.
   - Behavior-cloning warm start and constrained behavior cloning.
   - Mixed replay and component ablations.

5. `residual_lqr`
   - Direct residual and reference-residual environments.
   - HER and non-HER residual training.
   - Stratified and terminal replay experiments.
   - Pilot E, G, and J configurations.
   - Final-LQR terminal-bank generation and residual baseline validation.

6. `timestep_transfer`
   - Evaluation of policies transferred between 1 Hz and 10 Hz execution.
   - Uniform `dt=0.1` actor-transfer training.
   - Native, action-hold, and recomputed-action comparisons.

7. `two_stage_capture`
   - Existing systematic and learned capture experiments.
   - Capture-specific support code, tests, models, logs, results, and datasets.

## Shared modules

Code used by multiple families remains directly under `closing`. This includes general environment wrappers, replay-buffer implementations, HER utilities, behavior-cloning implementations, and shared configuration modules.

A module remains at the root only when at least two experiment families import it or when moving it would create an artificial dependency on one family. Experiment-specific support modules move with their family.

## Family layout

Each experiment family uses the following directories when relevant:

- `models/` - trained policies, checkpoints, and normalization state.
- `logs/` - TensorBoard logs and retained execution logs.
- `results/` - numerical evaluations and generated figures associated with the family.
- `demonstrations/` - generated training or reference datasets.
- `plotters/` - scripts whose primary purpose is generating figures.
- `tests/` - tests specific to the family.

Training, evaluation, tuning, analysis, validation, and data-generation scripts remain at the family root. Their filenames begin with the corresponding role when practical: `train_`, `evaluate_`, `tune_`, `analyze_`, `validate_`, `build_`, `generate_`, or `curate_`.

Empty standard directories will not be created. Each family README will explain:

- the experimental question;
- the methods and major variants;
- which runs are retained or superseded;
- the relationship between scripts and artifact directories;
- required model and normalization pairs;
- representative commands using paths relative to the repository root.

## Naming policy

Names will describe technical content rather than document purpose. Terms such as `article`, `publication`, and `thesis` will be replaced in active filenames and directory names with descriptions of the experiment, metric, policy, or comparison.

Historical run identifiers embedded in stored result data will remain unchanged when altering them would modify experiment evidence. Ambiguous active script names will be expanded only when the new name can be derived from implemented behavior.

## Code review and cleanup

Every moved Python file will be reviewed for:

- imports and paths affected by the new location;
- module-level execution and missing main guards;
- output paths that depend on the process working directory;
- missing model, normalization, dataset, or result inputs;
- stale references to old directories;
- Windows-specific or unused imports;
- conversational, speculative, or AI-like comments;
- comments that restate code rather than explain a constraint;
- obvious correctness defects exposed by tests or import checks.

Behavioral changes will be limited to verified defects. Experiment algorithms and stored results will not be altered as part of the structural cleanup.

## Dependency updates

All imports and path constants will be updated after moves. Direct execution from the repository root is the required invocation context. Family scripts may add their family directory and the `closing` root to `sys.path` where the existing non-package layout requires it.

References in repository tests, READMEs, plotting scripts, and controller integrations will be updated. Stored absolute paths in historical CSV or JSON data will be documented as provenance rather than treated as live runtime paths unless code consumes them.

## Verification

Verification will include:

1. Syntax compilation for all closing Python files.
2. Import checks for executable scripts without running their main functions.
3. Help or argument-parser checks for representative entry points from the repository root.
4. Family-specific tests after each family migration.
5. The complete repository test suite after all migrations.
6. Repository-wide scans for stale paths and old content-purpose names.
7. Post-move artifact inventory comparison.
8. Removal of generated `__pycache__` and `.pytest_cache` directories.

Known third-party warnings will be reported separately from failures.
