# CBF-3DoF Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename and reorganize the 3-DoF CBF controller into a navigable `CBF-3DoF` archive with an importable internal package, separated executable groups, documented results, and no redundant or incomplete runs.

**Architecture:** The outer controller directory follows the repository's human-facing `*-3DoF` naming convention, while implementation modules live in the valid Python package `cbf_3dof`. Evaluators, tuners, plotters, tests, and result artifacts remain separate consumers of that package. Result moves preserve artifact contents; deletions require an inventory-backed duplicate, supersession, or incomplete-run proof.

**Tech Stack:** Python, pytest, NumPy, SciPy, OSQP, Optuna, Stable-Baselines3, PowerShell, Markdown

**Spec:** `docs/superpowers/specs/2026-09-29-cbf-3dof-reorganization-design.md`

## Global Constraints

- Rename `src/controllers/cbf_3DoF` to `src/controllers/CBF-3DoF`.
- Keep the importable implementation package at `src/controllers/CBF-3DoF/cbf_3dof`.
- Preserve barrier equations, dynamics, QP behavior, solver settings, fallback behavior, tuning objectives, evaluation metrics, and command-line behavior.
- Do not regenerate evaluations, tuning studies, plots, or historical artifacts.
- Delete only inventory-proven duplicates, fully superseded reduced outputs, explicitly incomplete runs, and generated caches.
- Do not rewrite historical artifact contents solely to replace embedded old paths.
- Keep all changes unstaged and uncommitted.

## Review Focus

- Imports must work from repository-root commands even though the outer directory contains a hyphen.
- Direct script execution must resolve `cbf_3dof` without relying on the caller's working directory.
- Tuning scripts must continue to import evaluator functions and resume the same Optuna databases.
- Plotters must resolve moved evaluation folders and retain source-to-figure traceability.
- Result deletion must not remove a unique configuration, metric field, trajectory, or selected figure.

---

### Task 1: Add migration and preservation guards

**Files:**
- Create: `src/controllers/tests/test_cbf_3dof_layout.py`
- Create ignored records under `.superpowers/sdd/2026-09-29-cbf-3dof-reorganization/`

**Interfaces:**
- Consumes: the approved directory structure and deletion list.
- Produces: failing structural checks, a complete pre-move inventory, and deletion-candidate proofs used by later tasks.

- [ ] Record every non-cache file under `src/controllers/cbf_3DoF` with relative path, byte size, and SHA-256.
- [ ] Record a content-hash multiset excluding approved deletion candidates so path-only moves can be verified independently.
- [ ] Add structural assertions for `CBF-3DoF/cbf_3dof`, `evaluations`, `tuning`, `plotters`, `tests`, `results`, and the renamed result reports.
- [ ] Assert that `src/controllers/cbf_3DoF`, old imports, old plotter names, and approved deletion paths are absent.
- [ ] Run the structural test and confirm that it fails on the old layout.

### Task 2: Rename the outer directory and establish the internal package

**Files:**
- Rename: `src/controllers/cbf_3DoF` to `src/controllers/CBF-3DoF`
- Create: `src/controllers/CBF-3DoF/cbf_3dof/`
- Move: implementation modules listed in the specification into `cbf_3dof/`

**Interfaces:**
- Consumes: existing public objects exported by `__init__.py` and the seven implementation modules.
- Produces: import path `cbf_3dof` with unchanged public classes, functions, and configuration dataclasses.

- [ ] Move `__init__.py`, `barriers.py`, `closing_barriers.py`, `cbf_types.py`, `controller.py`, `diagnostics.py`, `port_dynamics.py`, and `qp_filter.py` into `cbf_3dof`.
- [ ] Convert internal imports to package-relative imports and preserve the existing `cbf_3dof.__init__` exports.
- [ ] Add the outer `CBF-3DoF` path only at executable and test boundaries, not inside implementation modules.
- [ ] Compile and import `cbf_3dof`, then run barrier, controller, port-dynamics, and QP-filter tests.

### Task 3: Separate evaluation entry points

**Files:**
- Create: `src/controllers/CBF-3DoF/evaluations/`
- Move: the three `evaluate_*_cbf.py` scripts
- Create: `src/controllers/CBF-3DoF/evaluations/README.md`

**Interfaces:**
- Consumes: `cbf_3dof` implementation APIs and external controller/environment modules.
- Produces: unchanged evaluator CLI entry points and result schemas.

- [ ] Move closing, final-approach, and LQR evaluators into `evaluations/`.
- [ ] Update package imports and derive repository, controller, and result roots from `__file__`.
- [ ] Preserve every parser option, default CBF parameter, variant name, output schema, and failure diagnosis.
- [ ] Add a README mapping each evaluator to its controller, phase, inputs, and output directory.
- [ ] Update evaluator tests and run them from the repository root.

### Task 4: Separate tuning entry points

**Files:**
- Create: `src/controllers/CBF-3DoF/tuning/`
- Move: `tune_closing_cbf.py`, `tune_final_approach_cbf.py`
- Create: `src/controllers/CBF-3DoF/tuning/README.md`

**Interfaces:**
- Consumes: the moved evaluator functions and `cbf_3dof` implementation APIs.
- Produces: unchanged Optuna study creation, resumption, trial tables, plans, and selected configurations.

- [ ] Move both tuning scripts and update their evaluator imports.
- [ ] Resolve default study paths below the migrated `results/<phase>/tuning` directories.
- [ ] Document fixed safety limits, tuned gains, validation seeds, and resumable files.
- [ ] Run tuning unit tests and one parser/import smoke check without executing a study.

### Task 5: Separate and rename plotters

**Files:**
- Create: `src/controllers/CBF-3DoF/plotters/`
- Move: all four plotting scripts
- Rename two scripts according to the specification
- Create: `src/controllers/CBF-3DoF/plotters/README.md`

**Interfaces:**
- Consumes: retained evaluation CSV, JSON, and NPZ schemas.
- Produces: the same figures and figure-summary metadata at migrated paths.

- [ ] Move `plot_final_approach_cbf_trajectories.py` and `plot_sac_cbf_trajectory_comparisons.py` unchanged in name.
- [ ] Rename `plot_logged_constraint_and_control_figures.py` to `plot_constraint_and_control_diagnostics.py`.
- [ ] Rename `plot_lqr_cbf_publication_figures.py` to `plot_lqr_cbf_evaluation.py`.
- [ ] Update imports and derive result roots from the controller directory.
- [ ] Remove publication-purpose wording from comments without changing plotted content or file formats.
- [ ] Update and run figure-layout tests without regenerating retained figures.

### Task 6: Migrate tests and repository references

**Files:**
- Move: existing CBF tests to `src/controllers/CBF-3DoF/tests/`
- Modify: Python and Markdown consumers throughout the repository

**Interfaces:**
- Consumes: the finalized package and executable paths.
- Produces: no live `controllers.cbf_3DoF` or `src/controllers/cbf_3DoF` references.

- [ ] Update tests to add `CBF-3DoF` to `sys.path` and import through `cbf_3dof`.
- [ ] Update executable-module imports to use the new `evaluations`, `tuning`, and `plotters` locations.
- [ ] Update repository documentation and command examples.
- [ ] Search active Python and Markdown files for every old path and old script name.
- [ ] Run the migrated CBF test suite.

### Task 7: Reorganize closing and final-approach results

**Files:**
- Reorganize directories under `results/closing/evaluations/`
- Reorganize directories under `results/final_approach/evaluations/`
- Create or update phase READMEs

**Interfaces:**
- Consumes: evaluation directory names and stored run configurations.
- Produces: phase-local groups that separate campaigns, retained evaluations, development ablations, and historical runs.

- [ ] Under closing evaluations, create `controller_campaigns`, `reward_progression_evaluations`, and `cbf_development_ablations`.
- [ ] Move `campaign_*` directories into `controller_campaigns`.
- [ ] Move `P3reach-*` and `P3-recovery-*` directories into `reward_progression_evaluations`.
- [ ] Move timestamp-only, `trial16_*`, and `v12_*` directories into `cbf_development_ablations`.
- [ ] Under final-approach evaluations, create `controller_campaigns`, `alignment_screen_100k`, `alignment_screen_300k`, and `cbf_development_and_holdout`.
- [ ] Move `campaign_*`, `sac_100k_*`, `sac_300k_*`, and remaining `sac_seed*` directories into their corresponding groups.
- [ ] Update plotter defaults, report links, and READMEs without rewriting stored artifact contents.
- [ ] Verify that every pre-move retained evaluation directory appears exactly once after grouping.

### Task 8: Consolidate LQR results and delete verified redundancy

**Files:**
- Reorganize: `src/controllers/CBF-3DoF/results/lqr/`
- Create: `src/controllers/CBF-3DoF/results/lqr/README.md`
- Delete only verified redundant directories

**Interfaces:**
- Consumes: hash comparisons and common-field comparisons from Task 1.
- Produces: one canonical result per LQR configuration and documented retained distinctions.

- [ ] Prove that common fields in the reduced closing/full-CBF and final/full-CBF results equal the enriched family-breakdown results for all seeds.
- [ ] Keep `closing_dt0p1_full_cbf_family_breakdown_seeds_0_49` and delete the reduced `closing_dt0p1_full_cbf_seeds_0_49` after supersession proof.
- [ ] Keep `final_dt1p0_full_cbf_family_breakdown_seeds_0_49` and delete the reduced `final_dt1p0_full_cbf_seeds_0_49` after supersession proof.
- [ ] Delete `manual_closing_dt0p1_full_cbf_seeds_0_49` only after confirming its numerical tables equal the enriched closing result and its configuration adds no unique executed setting.
- [ ] Retain one exact no-CBF native-initialization run as `closing_dt0p1_no_cbf_native_initialization_seeds_0_49`; delete its byte-identical duplicate.
- [ ] Rename the distinct P3-initialized no-CBF run to `closing_dt0p1_no_cbf_p3_initialization_seeds_0_49`.
- [ ] Document the final and closing phases, timesteps, initialization distributions, filters, and retained figures.

### Task 9: Delete incomplete runs and move result reports

**Files:**
- Delete the three explicitly approved incomplete directories
- Move and rename the two root result reports
- Update report links and paths

**Interfaces:**
- Consumes: Task 1 inventory and the finalized result grouping.
- Produces: no unlabelled partial evaluations and two content-named reports under `results`.

- [ ] Reconfirm that each `_h20000` candidate lacks configuration, per-episode, and aggregate files before deletion.
- [ ] Delete both incomplete closing `_h20000` directories and record their exact files in the ledger.
- [ ] Delete the explicitly labelled final-approach `incomplete_previous_run` directory and record its exact files.
- [ ] Move `ALIGNMENT_CBF_RESULTS.md` to `results/final_approach_alignment_cbf_initial_evaluation.md`.
- [ ] Move `MULTICONTROLLER_RESULTS.md` to `results/final_approach_multicontroller_cbf_evaluation.md`.
- [ ] Update paths in both reports to the grouped result structure.

### Task 10: Rewrite the controller index and result documentation

**Files:**
- Modify: `src/controllers/CBF-3DoF/README.md`
- Create or modify READMEs under each code and result group

**Interfaces:**
- Consumes: all finalized code and result paths.
- Produces: a navigable index explaining implementation, reproduction, result provenance, and known limitations.

- [ ] Replace the monolithic README with an overview, package map, evaluator/tuner/plotter entry-point table, result map, and reproduction commands.
- [ ] Preserve the technical CBF definitions and sampled-data limitations, moving detailed result interpretation into the two result reports.
- [ ] Document the QP fallback as non-certified and distinguish continuous-time CBF conditions from discrete-time rollout evidence.
- [ ] Validate every relative Markdown link.

### Task 11: Verify preservation and repository integration

**Files:**
- Modify tests only if verification reveals a stale structural assumption.
- Remove generated cache directories after testing.

**Interfaces:**
- Consumes: inventories, deletion records, and final paths.
- Produces: evidence that all unexpected content is preserved and the repository remains executable.

- [ ] Compare post-move hashes with the pre-move inventory after subtracting approved deletions; require zero unexpected missing or changed hashes.
- [ ] Verify every approved deletion against its ledger record.
- [ ] Search Python and Markdown files for old package, folder, script, and report names.
- [ ] Compile every Python file under `CBF-3DoF`.
- [ ] Import implementation and executable modules without running evaluations, tuning, or plotting.
- [ ] Run the complete CBF test suite.
- [ ] Run the complete repository test suite.
- [ ] Run `git diff --check` and inspect working-tree status.
- [ ] Remove generated repository-contained `__pycache__` and `.pytest_cache` directories.
- [ ] Confirm that no file is staged and no commit was created.
