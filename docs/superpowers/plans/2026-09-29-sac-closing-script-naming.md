# SAC Closing Script Naming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize script names, references, comments, tests, and READMEs across the remaining SAC closing families without changing or renaming recorded artifacts.

**Architecture:** Treat each closing experiment family as an independently verifiable unit. Add a structural naming guard and immutable-file inventory first, classify each executable from its actual imports and fixed configuration, then rename only scripts whose current names obscure their behavior or attach an unnecessary version. Keep shared modules at the closing root and update all live consumers atomically with each family.

**Tech Stack:** Python, pytest, Stable-Baselines3, Gymnasium, PowerShell, Markdown

**Spec:** `docs/superpowers/specs/2026-09-29-sac-closing-script-naming-design.md`

## Global Constraints

- Reusable scripts remain reward-version neutral.
- A reward or environment version appears in a filename only when the implementation or fixed input is specific to that version.
- `behavior_cloned_warmup_sac.py`, `near_p3_reset_wrapper.py`, and cross-family multiseed support remain directly under `closing` while they have consumers in more than one family.
- Logs, models, results, figures, datasets, saved configurations, and recorded artifact paths are not renamed, moved, deleted, or regenerated.
- Reward calculations, environment behavior, hyperparameters, checkpoint selection, evaluation metrics, and controller behavior do not change.
- Comments and docstrings use objective technical wording.
- All changes remain unstaged and uncommitted.

## Review Focus

- Direct execution from the repository root and from the family directory must resolve renamed sibling modules.
- Hard-coded version-specific launchers must retain the version needed to identify their fixed reward, environment, or checkpoint.
- Reusable modules must not inherit a version solely because a currently retained artifact uses that version.
- Historical JSON, CSV, log, TensorBoard, model, and dataset contents must remain byte-identical.
- Importing a renamed executable must not start training, evaluation, plotting, or dataset generation.

---

### Task 1: Add naming and artifact-preservation guards

**Files:**
- Modify: `src/controllers/SAC-3DoF/tests/test_sac_closing_layout.py`
- Create ignored record: `.superpowers/sdd/2026-09-29-sac-closing-script-naming/noncode-inventory-before.json`

**Interfaces:**
- Consumes: the family boundaries and shared-root decisions from the specification.
- Produces: structural assertions and a SHA-256 baseline used by every later task.

- [ ] Add assertions that the three shared modules remain at the closing root and are absent from family subfolders.
- [ ] Add a table-driven assertion that every production filename uses an action prefix where applicable and contains no publication-purpose wording.
- [ ] Add an explicit allowlist for version-bearing scripts, with a reason category of fixed reward, fixed environment, fixed checkpoint, or fixed result input.
- [ ] Record relative paths, sizes, and SHA-256 hashes for all files below `logs`, `models`, `results`, `figures`, and dataset directories.
- [ ] Run the structural test before renaming and record every failure as the rename checklist.

### Task 2: Clean LQR demonstration-learning scripts

**Files:**
- Modify or rename Python files under `src/controllers/SAC-3DoF/closing/lqr_demonstration_learning/`
- Modify: `src/controllers/SAC-3DoF/closing/lqr_demonstration_learning/README.md`
- Modify matching tests under `lqr_demonstration_learning/tests/`

**Interfaces:**
- Consumes: `behavior_cloned_warmup_sac.BehaviorClonedWarmupSAC` from the closing root.
- Produces: version-neutral names for reusable behavior-cloning, replay-prefill, dataset-generation, and evaluation code.

- [ ] Classify every root and plotter script from its imports, parser defaults, and fixed artifact paths; add each version-bearing exception to the Task 1 allowlist.
- [ ] Rename scripts only where the existing filename is opaque or unnecessarily versioned, preserving `dt01` where the implementation is fixed to `dt = 0.1 s`.
- [ ] Update imports, test module loading, help text, and README script tables.
- [ ] Replace conversational or speculative comments with objective descriptions without changing executable statements.
- [ ] Run the family tests and direct-import checks.

### Task 3: Clean phase-balanced replay scripts

**Files:**
- Modify or rename Python files under `src/controllers/SAC-3DoF/closing/phase_balanced_replay/`
- Modify: `src/controllers/SAC-3DoF/closing/phase_balanced_replay/README.md`
- Modify matching tests under `phase_balanced_replay/tests/`

**Interfaces:**
- Consumes: `near_p3_reset_wrapper.FixedNearP3ResetWrapper` from the closing root.
- Produces: descriptive replay-buffer, training, analysis, and plotting entry points.

- [ ] Determine whether each `v2`, `v12`, `v15`, and `v21` token identifies code behavior, a fixed reward/environment, or only historical chronology.
- [ ] Keep fixed reward/environment labels; replace chronology-only labels with the implemented replay, reset, terminal, or CBF distinction.
- [ ] Update sibling imports, tests, documentation, and cross-family consumers.
- [ ] Review comments and docstrings for objective wording.
- [ ] Run the family tests and direct-import checks.

### Task 4: Clean residual-LQR scripts

**Files:**
- Modify or rename Python files under `src/controllers/SAC-3DoF/closing/residual_lqr/`
- Modify: `src/controllers/SAC-3DoF/closing/residual_lqr/README.md`
- Modify matching tests under `residual_lqr/tests/`

**Interfaces:**
- Consumes: shared behavior-cloned warm-up support and existing residual environment modules.
- Produces: names that distinguish HER, reference residuals, lateral residuals, terminal fade, stratified replay, source warm-start, and constrained residuals.

- [ ] Resolve `pilot_e`, `pilot_g`, `v2` through `v5`, `reward_v2`, and budget suffixes against code and stored configurations.
- [ ] Replace opaque pilot or chronology labels only when the implemented distinction is provable; retain hard-coded reward/environment labels.
- [ ] Keep reusable environment, replay-buffer, configuration, and evaluator modules version-neutral.
- [ ] Update imports, test loaders, README tables, and cross-family references.
- [ ] Review comments and docstrings without changing algorithms or parameters.
- [ ] Run the residual-LQR tests and direct-import checks.

### Task 5: Clean timestep-transfer scripts

**Files:**
- Modify or rename Python files under `src/controllers/SAC-3DoF/closing/timestep_transfer/`
- Modify: `src/controllers/SAC-3DoF/closing/timestep_transfer/README.md`
- Modify matching tests under `timestep_transfer/tests/`

**Interfaces:**
- Consumes: shared near-P3 reset support and fixed source checkpoints where configured.
- Produces: source-to-target timestep, actor-transfer, replay-warm-up, evaluation, and horizon-comparison names.

- [ ] Keep the evaluator version-neutral if model paths are supplied at runtime.
- [ ] Retain version or timestep tokens in fixed-input trainers and plotters only when required to identify their inputs.
- [ ] Update imports, tests, help text, comments, and README script tables.
- [ ] Run the timestep-transfer tests and direct-import checks.

### Task 6: Clean two-stage capture scripts

**Files:**
- Modify or rename Python files under `src/controllers/SAC-3DoF/closing/two_stage_capture/`
- Modify: `src/controllers/SAC-3DoF/closing/two_stage_capture/README.md`
- Modify matching tests under `two_stage_capture/tests/`

**Interfaces:**
- Consumes: the existing handover, capture-reset, safety-terminal, and systematic-action interfaces.
- Produces: descriptive long-range, capture-stage, dataset-collection, baseline, and sweep entry points.

- [ ] Classify generic support modules separately from executable entry points.
- [ ] Rename only ambiguous modules; keep already descriptive version-neutral names unchanged.
- [ ] Update imports, tests, README tables, comments, and docstrings.
- [ ] Confirm all executable modules use a safe `main()` guard without changing runtime behavior.
- [ ] Run the two-stage capture tests and direct-import checks.

### Task 7: Clean vanilla-SAC scripts

**Files:**
- Modify or rename Python files under `src/controllers/SAC-3DoF/closing/vanilla_sac/`
- Modify: `src/controllers/SAC-3DoF/closing/vanilla_sac/README.md`
- Modify matching tests under `vanilla_sac/tests/`

**Interfaces:**
- Consumes: baseline closing environments and existing model/result paths.
- Produces: generic baseline, timestep, action-hold, port-observation, multiseed, tuning, and fixed-reward evaluation entry points.

- [ ] Keep generic training, evaluation, and tuning scripts version-neutral.
- [ ] Retain V15 only for evaluators or plotters with fixed V15 inputs.
- [ ] Replace unclear `best_standard` wording with the implemented configuration when the script is renamed.
- [ ] Update imports, tests, help text, README tables, comments, and docstrings.
- [ ] Run the vanilla-SAC tests and direct-import checks.

### Task 8: Validate shared-root ownership and closing documentation

**Files:**
- Modify: `src/controllers/SAC-3DoF/closing/README.md`
- Modify when required: `src/controllers/SAC-3DoF/closing/behavior_cloned_warmup_sac.py`
- Modify when required: `src/controllers/SAC-3DoF/closing/near_p3_reset_wrapper.py`
- Modify when required: `src/controllers/SAC-3DoF/closing/train_sac_closing_multiseed.py`

**Interfaces:**
- Consumes: finalized family script names and shared-module consumers.
- Produces: one closing index with accurate ownership and navigation.

- [ ] Confirm each shared root module has consumers in multiple families and document those consumers.
- [ ] Keep shared filenames version-neutral and review comments/docstrings for objective wording.
- [ ] Update the closing README family index and script ownership descriptions.
- [ ] Search all active Python and Markdown files for old filenames and correct live references.
- [ ] Run the structural layout test.

### Task 9: Verify preservation and repository integration

**Files:**
- Modify tests only when verification exposes a stale structural assumption.
- Remove generated cache directories after testing.

**Interfaces:**
- Consumes: the Task 1 inventory and all finalized names.
- Produces: evidence that naming and documentation changed without changing recorded artifacts or behavior.

- [ ] Compare artifact paths, sizes, and SHA-256 hashes with the Task 1 baseline and require zero additions, removals, or changes.
- [ ] Compile every Python file under `closing`.
- [ ] Import executable modules without invoking runtime work.
- [ ] Run all closing-family tests and `src/controllers/SAC-3DoF/tests/test_sac_closing_layout.py`.
- [ ] Run the complete repository test suite.
- [ ] Run `git diff --check` and inspect working-tree status.
- [ ] Remove repository-contained `__pycache__` and `.pytest_cache` directories created by verification.
- [ ] Confirm that no files are staged and no commit was created.
