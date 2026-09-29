# CBF-3DoF Reorganization Design

## Objective

Rename `src/controllers/cbf_3DoF` to `src/controllers/CBF-3DoF` and reorganize
its implementation, executable scripts, tests, documentation, and results so
that controller code and recorded experiments can be understood independently.

The reorganization must preserve executable behavior and every complete,
nonredundant result. Confirmed byte-identical duplicates and clearly incomplete
runs are deleted. All changes remain unstaged and uncommitted unless a later
user request explicitly authorizes a commit.

## Package boundary

`CBF-3DoF` cannot itself be imported as a Python package because hyphens are
invalid in dotted module names. The renamed controller directory therefore
contains an internal importable package named `cbf_3dof`:

```text
src/controllers/CBF-3DoF/
├── cbf_3dof/
├── evaluations/
├── tuning/
├── plotters/
├── tests/
├── results/
└── README.md
```

Executables and tests add `src/controllers/CBF-3DoF` to the module search path
and import implementation modules through `cbf_3dof`. Repository code must no
longer import `controllers.cbf_3DoF` after the migration.

## Implementation package

The following modules move into `CBF-3DoF/cbf_3dof`:

- `__init__.py`
- `barriers.py`
- `closing_barriers.py`
- `cbf_types.py`
- `controller.py`
- `diagnostics.py`
- `port_dynamics.py`
- `qp_filter.py`

These modules retain their current responsibilities and public behavior. The
reorganization may update imports and objective comments but does not alter
barrier equations, QP construction, solver settings, fallback behavior,
controller limits, or diagnostic calculations.

## Executable groups

### Evaluations

`evaluations/` contains:

- `evaluate_closing_cbf.py`
- `evaluate_final_approach_cbf.py`
- `evaluate_lqr_cbf.py`

Each evaluator retains its command-line interface and resolves default result
paths relative to the renamed controller root.

### Tuning

`tuning/` contains:

- `tune_closing_cbf.py`
- `tune_final_approach_cbf.py`

Tuning scripts import evaluator functions through an explicit executable-module
path or a package-safe local interface. Existing Optuna study locations and
resume behavior remain supported after result paths are migrated.

### Plotters

`plotters/` contains:

- `plot_final_approach_cbf_trajectories.py`
- `plot_constraint_and_control_diagnostics.py`, renamed from
  `plot_logged_constraint_and_control_figures.py`
- `plot_lqr_cbf_evaluation.py`, renamed from
  `plot_lqr_cbf_publication_figures.py`
- `plot_sac_cbf_trajectory_comparisons.py`

Plotter names describe their content rather than a thesis or publication use.
Plotters continue to read retained result artifacts without rerunning
evaluations.

## Tests

All CBF tests remain under `CBF-3DoF/tests`. Imports change from
`controllers.cbf_3DoF` to `cbf_3dof`. Test names remain unchanged unless a test
is coupled to a renamed executable.

The test suite gains structural checks for:

- the outer `CBF-3DoF` directory and internal `cbf_3dof` package;
- absence of the old `cbf_3DoF` directory and old import path;
- separation of implementation, evaluations, tuning, plotters, tests, and
  results;
- existence of the renamed result reports;
- absence of specifically approved duplicate and incomplete result folders.

## Results structure

The retained result categories remain phase-oriented:

```text
results/
├── closing/
│   ├── evaluations/
│   ├── tuning/
│   └── README.md
├── final_approach/
│   ├── evaluations/
│   ├── plots/
│   ├── tuning/
│   └── README.md
├── lqr/
│   └── README.md
├── trajectory_figures/
│   └── README.md
├── final_approach_alignment_cbf_initial_evaluation.md
└── final_approach_multicontroller_cbf_evaluation.md
```

The reports move and receive content-based names:

- `ALIGNMENT_CBF_RESULTS.md` becomes
  `results/final_approach_alignment_cbf_initial_evaluation.md`.
- `MULTICONTROLLER_RESULTS.md` becomes
  `results/final_approach_multicontroller_cbf_evaluation.md`.

Existing Markdown links and embedded example paths are updated to the renamed
controller and result locations.

## Result classification

Complete evaluation directories retain their identifying fields: controller or
training seed, checkpoint, filter variant, hard or soft constraint handling,
scenario-seed range, horizon, and schema version when relevant.

Timestamp-only and exploratory closing evaluations are retained under a
documented `development_evaluations` subgroup. Selected campaign and held-out
evaluations remain separate from development runs. Moving an evaluation into a
subgroup must not rewrite its CSV, JSON, NPZ, database, or figure contents.

Each results README states:

- the purpose of the contained experiments;
- the controller family and phase;
- the evaluation seed and horizon conventions;
- which runs are development, tuning, selected, held-out, or historical;
- the interpretation of schema versions and incomplete-run deletion;
- the relationship between plots and their source evaluations.

## Approved deletions

Deletion is limited to these verified classes:

1. Exact directory duplicates, confirmed by matching relative paths, sizes, and
   SHA-256 hashes for every file. The retained directory receives the clearer
   experiment name when necessary.
2. Evaluation directories containing only partial trajectory files and no run
   configuration, per-episode table, or aggregate summary.
3. Directories explicitly labelled as an incomplete previous run.
4. Generated `__pycache__` and `.pytest_cache` directories.

Initial confirmed candidates are:

- one of `manual_closing_dt0p1_no_cbf_seeds_0_49` and
  `rerun_20260913_closing_dt0p1_no_cbf_native_init_seeds_0_49`, which are
  byte-identical;
- `campaign_phase_balanced_best_full_cbf_seeds_0_49_h20000`, which contains
  three trajectory files without summary artifacts;
- `campaign_vanilla_final_full_cbf_seeds_0_49_h20000`, which contains two
  trajectory files without summary artifacts;
- `sac_100k_seed12_final_alignment_screen_hard_unfiltered_cone_speed_full_cbf_eval_seeds0120_0129_incomplete_previous_run`.

Directories that share only a configuration or subset of metrics are not
deleted until full-directory equivalence or complete supersession is proved.
In particular, enriched LQR constraint-family breakdowns are retained while
their reduced predecessors are reviewed field by field.

Before each material deletion, the exact path must be resolved inside
`CBF-3DoF/results`, its classification must be rechecked, and a deletion record
must be written to the execution ledger.

## Script and documentation review

The review covers:

- stale and fragile imports;
- paths dependent on the current working directory;
- import-time execution in executable scripts;
- unused imports and unreachable commented-out code;
- conversational, speculative, or publication-purpose comments;
- duplicated evaluator or plotter logic that can be shared without changing
  behavior;
- scripts that no longer consume or produce retained artifacts.

Redundant scripts are deleted only when all functions and outputs are fully
superseded and no live import, documentation link, or retained reproduction
workflow depends on them. Otherwise they are kept and documented.

## Verification and preservation

Before moving files, record relative paths, sizes, and SHA-256 hashes for all
non-cache files. Record a second content-hash inventory that excludes the
approved deletion candidates and treats path changes independently from file
content.

After reorganization:

- require the expected outer-directory rename and internal package structure;
- require zero unexpected missing or changed content hashes;
- verify approved deletions against the recorded deletion list;
- search Python and Markdown files for `controllers.cbf_3DoF`,
  `src/controllers/cbf_3DoF`, and renamed script names;
- compile every Python file under `CBF-3DoF`;
- import the implementation package and executable modules without running
  evaluations, tuning, or plotting;
- run the CBF test suite;
- run the complete repository test suite;
- validate Markdown relative links;
- run `git diff --check`;
- confirm that no file is staged and no commit is created.

## Non-goals

- No barrier, dynamics, controller, QP, solver, tuning-objective, or evaluation
  behavior changes.
- No evaluation, tuning study, or plot is regenerated.
- No complete unique result is deleted.
- No historical artifact content is rewritten solely to replace embedded old
  absolute paths.
- No PDF or other figure format is removed without a separate explicit request.
- No Git commit, staging operation, merge, push, or branch change is performed.
