# SAC Closing Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the SAC closing work into self-contained experiment families without losing artifacts or breaking training, evaluation, plotting, or tests.

**Architecture:** Seven experiment-family directories own their scripts and artifacts. Python modules imported by multiple families remain directly under `closing`; family-specific helpers move with their family. Each migration updates imports and paths immediately and ends with focused tests before the next family moves.

**Tech Stack:** Python 3.12, Gymnasium, Stable-Baselines3, PyTorch, NumPy, pandas, Matplotlib, pytest, Optuna

**Spec:** `docs/superpowers/specs/2026-09-29-sac-closing-reorganization-design.md`

## Global Constraints

- Preserve all experiment families and stored artifacts.
- Do not remove a script unless it has no unique behavior, configuration, artifact provenance, or external consumer.
- Retain short experiment presets that supply distinct arguments to shared trainers.
- Keep modules used by multiple experiment families directly under `closing`.
- Use content-based names; active filenames and directories must not use `article`, `publication`, or `thesis`.
- Do not alter experiment algorithms or stored result values during structural cleanup.
- Use paths derived from `__file__`, not the process working directory.
- Use objective comments that explain constraints or implemented behavior.
- Run scripts from the repository root.

## Review Focus

- Imports between a family script and a shared root module must work both during direct execution and path-based import; Task 2 adds and Task 10 reruns import checks.
- Preset trainers must retain the exact default argument lists used by existing runs; each family migration runs its preset tests before and after the move.
- Model files and their `VecNormalize` state must remain paired after moves; Task 1 records pairs and Task 10 validates them.
- Stored result metadata may contain historical absolute paths, but executable code must not consume obsolete paths; Tasks 2-9 scan live code separately from stored evidence.
- Files with identical basenames in different experiments must not overwrite one another; Task 1 records relative paths and Task 10 compares the full artifact inventory.

---

### Task 1: Establish preservation and layout guards

**Files:**
- Create: `src/controllers/SAC-3DoF/tests/test_sac_closing_layout.py`
- Create: `src/controllers/SAC-3DoF/closing/README.md`
- Create: one initial `README.md` in each approved family directory
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Current `closing` tree and tracked artifacts.
- Produces: `EXPECTED_FAMILIES`, executable import checks, model-pair checks, and the artifact inventory used by later tasks.

- [ ] **Step 1: Record the pre-move artifact inventory outside the repository**

Create a temporary JSON inventory containing relative path, byte size, and SHA-256 for `.zip`, `.pkl`, `.npz`, `.csv`, `.json`, `.log`, and `.png` files. Do not commit the inventory.

- [ ] **Step 2: Write failing structural tests**

Add tests asserting that the seven approved family directories exist, active purpose-based names are absent, all executable scripts have main guards, and each `.zip` model has its documented normalization pairing or documented exception.

- [ ] **Step 3: Run the structural tests and verify they fail on the current flat layout**

Run: `.venv/Scripts/python.exe -m pytest src/controllers/SAC-3DoF/tests/test_sac_closing_layout.py -q`

Expected: FAIL because the approved family directories and READMEs do not yet exist.

- [ ] **Step 4: Add the root and family README skeletons and ignore generated caches**

Create each family directory with a README that states its scope, document the seven families in the root README, and state that shared modules remain at the root. Ensure `.gitignore` covers `__pycache__/` and `.pytest_cache/` without adding artifact directories. Rerun the structural tests and expect PASS.

- [ ] **Step 5: Commit the guards and overview**

```powershell
git add .gitignore src/controllers/SAC-3DoF/closing/README.md src/controllers/SAC-3DoF/tests/test_sac_closing_layout.py
git commit -m "test: guard SAC closing reorganization"
```

### Task 2: Migrate the vanilla SAC family

**Files:**
- Create: `src/controllers/SAC-3DoF/closing/vanilla_sac/README.md`
- Move: vanilla training, evaluation, tuning, and plotting scripts from `closing/`
- Move: vanilla run directories from `models/multiseed` plus `models/reward_shaping_trials` to `vanilla_sac/models/`
- Move: vanilla run directories from `logs/multiseed` plus `logs/{optuna,reward_shaping_trials}` to `vanilla_sac/logs/`
- Move: vanilla run directories from `results/multiseed_evaluation` plus `results/{observation_comparison,optuna,reward_shaping_trials}` to `vanilla_sac/results/`
- Move and rename: `results/publication_figures` to a content-based directory under `vanilla_sac/results/`
- Move: `figures/closing_vanilla_sac_state_convergence_seed15.png` to `vanilla_sac/results/terminal_state_convergence/`
- Test: `src/controllers/SAC-3DoF/tests/test_sac_closing_layout.py`

**Interfaces:**
- Consumes: Shared root modules and `envs.docking3dof`.
- Produces: Reproducible vanilla SAC baselines and artifact paths used by later comparison families.

- [ ] **Step 1: Extend the layout test with vanilla script and artifact destinations**

Assert representative model, normalization, result, and figure files resolve under `vanilla_sac` and old live directories do not remain.

- [ ] **Step 2: Verify the new assertions fail**

Run the vanilla-specific layout tests and confirm failure on old paths.

- [ ] **Step 3: Move vanilla scripts and artifacts**

Include standard trainers and evaluators, multiseed scripts, reward-V15 baseline analysis, observation comparison, action-hold presets, `dt=0.1` standard training, Optuna tuning, and generic vanilla rollout/learning-curve plotters.

- [ ] **Step 4: Update imports, `__file__`-relative paths, and external consumers**

Preserve direct execution from the repository root. Rename purpose-based directories and update `src/controllers/SAC-3DoF/tests/test_figure_output_layout.py`.

- [ ] **Step 5: Review comments and write the family README**

Explain reward-shaping, Optuna, multiseed, observation, action-hold, and timestep variants; identify retained baselines without describing them as thesis/publication artifacts.

- [ ] **Step 6: Run vanilla imports, focused tests, and smoke help commands**

Expected: all moved scripts compile and import; representative `-h` commands exit 0.

- [ ] **Step 7: Commit the vanilla migration**

```powershell
git add src/controllers/SAC-3DoF/closing/vanilla_sac src/controllers/SAC-3DoF/tests
git commit -m "refactor: organize vanilla SAC closing experiments"
```

### Task 3: Migrate curriculum and P3-reach experiments

**Files:**
- Create: `src/controllers/SAC-3DoF/closing/curriculum_and_p3_reach/README.md`
- Move: curriculum, P3-reach, braking, no-safety-termination, and adaptive-terminal scripts
- Move: `models/{adaptive_dt01_terminal,curriculum,no_safety_termination,p3reach_from_scratch}`
- Move: matching directories from `logs/` and `results/`
- Move: curriculum, P3 trajectory, and no-safety-termination plotters
- Move: applicable tests from `tests_lqr_replay/` into `curriculum_and_p3_reach/tests/`

**Interfaces:**
- Consumes: Shared reset wrappers and closing environment contracts.
- Produces: Self-contained curriculum, P3-reach, and terminal-training experiments.

- [ ] **Step 1: Add failing destination and import tests for the family**
- [ ] **Step 2: Move scripts, tests, models, logs, and results without changing artifact contents**
- [ ] **Step 3: Update imports and paths, keeping cross-family dependencies pointed at shared root modules**
- [ ] **Step 4: Review code comments and document the experimental progression and retained runs**
- [ ] **Step 5: Run the moved tests and import checks**
- [ ] **Step 6: Commit**

```powershell
git add src/controllers/SAC-3DoF/closing/curriculum_and_p3_reach src/controllers/SAC-3DoF/closing/tests_lqr_replay src/controllers/SAC-3DoF/tests
git commit -m "refactor: organize closing curriculum experiments"
```

### Task 4: Migrate phase-balanced replay experiments

**Files:**
- Create: `src/controllers/SAC-3DoF/closing/phase_balanced_replay/README.md`
- Move: phase-balanced training presets and implementations from `closing/`
- Move: phase-balanced plotters and V15 analyses
- Move: phase-balanced tests from `tests_lqr_replay/`
- Split: phase-balanced run directories currently nested under central `models/multiseed`, `logs/multiseed`, and related results into family-local artifact directories

**Interfaces:**
- Consumes: Shared closing environment, reset wrapper, and replay-buffer contracts.
- Produces: V1, V2, V2.1, guarded, and CBF experiment entry points with retained diagnostics.

- [ ] **Step 1: Add failing tests for all phase-balanced variants and their selected source checkpoints**
- [ ] **Step 2: Move the implementation, presets, tests, and run-specific artifacts**
- [ ] **Step 3: Update source-checkpoint constants and plotter inputs**
- [ ] **Step 4: Replace conversational version comments with objective descriptions of controlled changes**
- [ ] **Step 5: Document variant differences, early-stopping rules, and retained checkpoints**
- [ ] **Step 6: Run phase-balanced tests, imports, and selected checkpoint-resolution tests**
- [ ] **Step 7: Commit**

```powershell
git add src/controllers/SAC-3DoF/closing/phase_balanced_replay src/controllers/SAC-3DoF/closing/vanilla_sac src/controllers/SAC-3DoF/tests
git commit -m "refactor: organize phase-balanced closing experiments"
```

### Task 5: Migrate LQR demonstration learning

**Files:**
- Create: `src/controllers/SAC-3DoF/closing/lqr_demonstration_learning/README.md`
- Move: LQR dataset generation and curation scripts
- Move: replay, mixed-replay, BC warm-start, and BC-constrained trainers and evaluators
- Move: relevant behavior-cloning and replay helpers when they have no consumer outside this family
- Move: `models/{lqr_replay,lqr_mixed_replay,lqr_bc_warmstart,lqr_bc_constrained,lqr_bc_constrained_dt01}`
- Move: matching `logs/` and `results/` directories
- Move: applicable demonstration `.npz`, progress, stdout, and stderr files
- Move: LQR replay and BC tests into `lqr_demonstration_learning/tests/`

**Interfaces:**
- Consumes: LQR controller outputs and shared SAC environment contracts.
- Produces: Curated demonstration datasets and behavior-cloning/replay experiment entry points.

- [ ] **Step 1: Add failing dataset, preset, and artifact-path tests**
- [ ] **Step 2: Move scripts, helpers, tests, artifacts, and demonstrations**
- [ ] **Step 3: Update dataset defaults and ensure both 1 Hz and 10 Hz presets resolve**
- [ ] **Step 4: Review comments and document dataset provenance, stride, episode counts, and method differences**
- [ ] **Step 5: Run LQR dataset, replay, BC, and mixed-buffer tests**
- [ ] **Step 6: Commit**

```powershell
git add src/controllers/SAC-3DoF/closing/lqr_demonstration_learning src/controllers/SAC-3DoF/closing/demonstrations src/controllers/SAC-3DoF/tests
git commit -m "refactor: organize LQR demonstration learning"
```

### Task 6: Migrate residual-LQR experiments

**Files:**
- Create: `src/controllers/SAC-3DoF/closing/residual_lqr/README.md`
- Move: direct residual, reference-residual, HER, stratified, terminal, Pilot E/G/J, oracle-analysis, bank-generation, and validation scripts
- Move: residual-specific environments, configuration, replay, and constrained-SAC helpers
- Move: all `models/`, `logs/`, and `results/` directories whose names contain `her_lqr_residual`, `her_lqr_reference_residual`, `lqr_reference_residual`, or `residual_lqr_baseline`
- Move: terminal-reference demonstration bank files
- Move: residual and HER tests from `tests_lqr_replay/`

**Interfaces:**
- Consumes: Shared closing environment contracts and LQR controller implementation.
- Produces: Direct and reference-residual SAC experiment families, including Pilot J and baseline validation.

- [ ] **Step 1: Add failing tests for residual environment imports, Pilot presets, and terminal-bank paths**
- [ ] **Step 2: Move residual scripts, helpers, tests, models, logs, results, and bank data**
- [ ] **Step 3: Update intra-family imports and references from LQR documentation outside `closing`**
- [ ] **Step 4: Review code logic and comments without changing algorithm definitions or stored results**
- [ ] **Step 5: Write READMEs for major residual subfamilies if the family overview would otherwise require run-by-run detail**
- [ ] **Step 6: Run all residual, HER, terminal-bank, and baseline-validation tests**
- [ ] **Step 7: Commit**

```powershell
git add src/controllers/SAC-3DoF/closing/residual_lqr src/controllers/LQR-3DoF src/controllers/SAC-3DoF/tests
git commit -m "refactor: organize residual LQR closing experiments"
```

### Task 7: Migrate timestep-transfer experiments

**Files:**
- Create: `src/controllers/SAC-3DoF/closing/timestep_transfer/README.md`
- Move: `evaluate_sac_closing_timestep_transfer.py`
- Move: `train_sac_closing_uniform_dt01_actor_transfer.py`
- Move: timestep-transfer comparison plotters
- Move: `models/uniform_dt01_actor_transfer`, matching logs, and results
- Move: `results/timestep_transfer`
- Move: transfer-specific replay helper if it has no other family consumer
- Move: transfer tests from `tests_lqr_replay/`

**Interfaces:**
- Consumes: Retained vanilla policies through explicit model paths.
- Produces: Native 1 Hz, held-action 10 Hz, recomputed-action 10 Hz, and actor-transfer comparisons.

- [ ] **Step 1: Add failing tests for transfer modes and source-policy resolution**
- [ ] **Step 2: Move scripts, tests, models, logs, and results**
- [ ] **Step 3: Rename active `article_*` directories to describe policy, horizon, seed range, and execution mode**
- [ ] **Step 4: Update all live references while preserving historical identifiers stored inside result files**
- [ ] **Step 5: Document timing semantics and each comparison group**
- [ ] **Step 6: Run timestep-transfer and actor-transfer tests**
- [ ] **Step 7: Commit**

```powershell
git add src/controllers/SAC-3DoF/closing/timestep_transfer src/controllers/SAC-3DoF/tests
git commit -m "refactor: organize closing timestep transfer experiments"
```

### Task 8: Normalize the two-stage capture family

**Files:**
- Modify: `src/controllers/SAC-3DoF/closing/two_stage_capture/README.md`
- Reorganize: `two_stage_capture_trial/` as `two_stage_capture/`
- Move: capture scripts into training, evaluation, data-generation, and family-support roles
- Move: capture plotters into `plotters/` when present
- Keep: capture-specific `artifacts/{models,logs,results}` content under explicit family directories
- Move: capture tests into `two_stage_capture/tests/`

**Interfaces:**
- Consumes: Shared docking environment contracts where applicable.
- Produces: Self-contained systematic and learned two-stage capture experiments.

- [ ] **Step 1: Add failing layout and import tests for the renamed family**
- [ ] **Step 2: Rename and normalize the family without changing stored artifacts**
- [ ] **Step 3: Update repository references and path constants**
- [ ] **Step 4: Review comments and expand the README with experiment and artifact descriptions**
- [ ] **Step 5: Run both capture test modules and import checks**
- [ ] **Step 6: Commit**

```powershell
git add src/controllers/SAC-3DoF/closing/two_stage_capture src/controllers/SAC-3DoF/tests
git commit -m "refactor: organize two-stage capture experiments"
```

### Task 9: Review remaining shared root modules and redundancy

**Files:**
- Modify: shared `.py` modules remaining directly under `src/controllers/SAC-3DoF/closing/`
- Modify: `src/controllers/SAC-3DoF/closing/README.md`
- Move: remaining family-specific scripts discovered by the import graph
- Delete: only scripts proven to have no unique behavior, configuration, provenance, or consumer

**Interfaces:**
- Consumes: Completed family migrations and repository-wide import graph.
- Produces: A root containing only shared modules, the overview README, and seven experiment families.

- [ ] **Step 1: Generate an import graph for every remaining root module**
- [ ] **Step 2: Assert each remaining module is imported by at least two families or document why it must remain shared**
- [ ] **Step 3: Move any misclassified family-specific module and update imports**
- [ ] **Step 4: Review shared code for correctness, unused imports, module-level execution, and non-objective comments**
- [ ] **Step 5: Record the redundancy decision for every short preset in the appropriate family README**
- [ ] **Step 6: Run all closing tests and commit**

```powershell
git add src/controllers/SAC-3DoF/closing src/controllers/SAC-3DoF/tests
git commit -m "refactor: finalize shared closing support code"
```

### Task 10: Verify preservation and repository integration

**Files:**
- Modify: repository READMEs and tests containing old closing paths
- Modify: family READMEs when verification exposes missing provenance

**Interfaces:**
- Consumes: All migrated families and the Task 1 artifact inventory.
- Produces: A verified, reproducible closing tree with no stale live paths.

- [ ] **Step 1: Compile every closing Python file**

Run: `.venv/Scripts/python.exe -m compileall -q src/controllers/SAC-3DoF/closing`

Expected: exit 0.

- [ ] **Step 2: Import every executable without invoking its main function**

Expected: zero import failures from the repository root.

- [ ] **Step 3: Run all family and shared closing tests**

Run: `.venv/Scripts/python.exe -m pytest src/controllers/SAC-3DoF/closing src/controllers/SAC-3DoF/tests -q`

Expected: all tests pass; third-party warnings are reported separately.

- [ ] **Step 4: Compare the post-move artifact inventory with Task 1**

Assert every original artifact hash occurs once at an approved destination. Report intentional filename changes separately and confirm file content is unchanged.

- [ ] **Step 5: Scan the repository for stale live paths and content-purpose names**

Exclude stored historical CSV/JSON path fields from live-code failures, but document them as provenance.

- [ ] **Step 6: Run the complete repository suite**

Run: `.venv/Scripts/python.exe -m pytest -q`

Expected: all tests pass.

- [ ] **Step 7: Remove generated cache directories and run `git diff --check`**

- [ ] **Step 8: Commit integration updates**

```powershell
git add README.md docs src
git commit -m "docs: finalize SAC closing experiment layout"
```
