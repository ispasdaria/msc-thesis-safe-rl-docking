# Curriculum and P3-Reach Naming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not create commits.

**Goal:** Give every curriculum and P3-reaching script, experiment group, and retained artifact run a content-based name while preserving reward-version labels and all recorded artifacts.

**Architecture:** Keep the existing family boundary and its standard `models`, `logs`, `results`, `plotters`, and `tests` layout. Rename mirrored artifact groups and runs in lockstep, update path consumers after each group, and add group-level documentation derived from committed configuration and result files.

**Tech Stack:** Python, pytest, Stable-Baselines3, Gymnasium, PowerShell, Git rename detection, Markdown

**Spec:** `docs/superpowers/specs/2026-09-29-curriculum-p3-reach-naming-design.md`

## Global Constraints

- Preserve all reward-version labels.
- Do not change training, evaluation, environment, reward, or checkpoint-selection behavior.
- Do not delete or regenerate recorded artifacts.
- Preserve artifact count and SHA-256 hash multiset for `.zip`, `.pkl`, `.npz`, `.csv`, `.json`, `.log`, and `.png` files.
- Keep related model, log, and result runs under matching names.
- Do not rewrite historical artifact contents solely to replace old embedded paths.
- Use objective technical wording without conversational commentary.
- Do not create Git commits, merge, push, or change branches.

## Review Focus

- A renamed model path must still resolve from its evaluator, plotter, and continuation trainer.
- Every model that requires `VecNormalize` must remain paired with its `.pkl` state.
- Run directories that exist in only two artifact categories must not be mistaken for missing data.
- Timestamp removal must not cause two historical runs to collide.
- Historical configuration paths may remain stale inside immutable artifacts, while live Python and Markdown references must use current paths.

---

### Task 1: Add structural and preservation guards

**Files:**
- Modify: `src/controllers/SAC-3DoF/tests/test_sac_closing_layout.py`
- Create ignored working record: `.superpowers/sdd/2026-09-29-curriculum-p3-reach-naming/artifact-inventory-before.json`

**Interfaces:**
- Consumes: the existing curriculum family path and approved group names.
- Produces: failing layout assertions for the future names and a SHA-256 artifact baseline.

- [ ] Add assertions for the six approved experiment-group names under `models`, `logs`, and `results` where applicable.
- [ ] Add assertions for the renamed training, evaluation, tuning, plotting, and test entry points.
- [ ] Assert that old script names and old top-level artifact-group names are absent.
- [ ] Generate the pre-move artifact inventory, including relative path, byte size, and SHA-256.
- [ ] Run the structural test and confirm it fails only because the approved names do not exist yet.

### Task 2: Rename executable scripts and family-specific tests

**Files:**
- Rename the nine root entry points listed in the specification.
- Rename plotters under `plotters/` to identify the input experiment and diagnostic.
- Rename tests under `tests/` to match the relevant production behavior.
- Modify imports and module references within the family and SAC structural tests.

**Interfaces:**
- Consumes: current script behavior and public functions used by tests.
- Produces: content-based executable filenames with unchanged functions and command-line interfaces.

- [ ] Rename production files without editing behavior.
- [ ] Rename tests and update imports.
- [ ] Update sibling imports in V19, V20, and V21 preset launchers.
- [ ] Update family and repository references to the new filenames.
- [ ] Compile the renamed scripts and run the family tests.

### Task 3: Rename reset-distribution curriculum artifacts

**Files:**
- Rename: `models/curriculum` to `models/reset_distribution_curriculum`
- Rename: `logs/curriculum` to `logs/reset_distribution_curriculum`
- Rename: `results/curriculum` to `results/reset_distribution_curriculum`
- Create: `results/reset_distribution_curriculum/README.md`
- Modify: curriculum trainer and curriculum comparison plotter paths.

**Interfaces:**
- Consumes: mirrored run identities and curriculum stage configuration.
- Produces: matching descriptive run paths across available artifact categories.

- [ ] Build a collision-free mapping for every curriculum, gamma, bootstrap, retry, and full-distribution run.
- [ ] Rename matching model, log, and result runs in lockstep.
- [ ] Update trainer defaults and plotter inputs.
- [ ] Document the reset progression, initialization, gamma variants, V19/V20 variants, and supported results.
- [ ] Run curriculum-focused tests and import checks.

### Task 4: Rename nonterminal-safety continuation artifacts

**Files:**
- Rename the `no_safety_termination` group across models, logs, and results to `nonterminal_safety_continuation`.
- Create: `results/nonterminal_safety_continuation/README.md`
- Modify: continuation trainer and trajectory plotter paths.

**Interfaces:**
- Consumes: the retained reward-V15 Trial-16 source checkpoint.
- Produces: two explicitly named continuation runs for reward V15 and P3-reach V1.

- [ ] Rename both runs consistently while removing nonessential timestamps.
- [ ] Verify the remaining fields uniquely identify each run.
- [ ] Update defaults, plotting paths, and documentation links.
- [ ] Document nonterminal safety semantics, initialization, reward difference, and observed results.
- [ ] Run continuation-focused import and test checks.

### Task 5: Rename the V1-V19 from-scratch reward progression

**Files:**
- Rename `p3reach_from_scratch` across models, logs, and results to `from_scratch_reward_progression`.
- Create: `results/from_scratch_reward_progression/README.md`
- Modify: from-scratch trainer, evaluator, and relevant plotter paths.

**Interfaces:**
- Consumes: the V1-V19 reward implementations in `closing_safety_exploration_env.py`.
- Produces: one content-based run name for every retained training execution, including budget and gamma variants.

- [ ] Apply the approved V1-V19 formulation labels.
- [ ] Preserve `gamma0p999`, 300k/600k, seed, and other true differentiators.
- [ ] Place the phase-balanced replay reference in a clearly documented cross-family reference subgroup without moving it outside the approved family.
- [ ] Separate the safety-terminal V19-V21 runs into the group owned by Task 6.
- [ ] Update evaluator model/result discovery and plotter inputs.
- [ ] Document the reward progression and summarize outcomes only from recorded metrics.
- [ ] Run evaluator diagnostics, path checks, and import checks.

### Task 6: Group safety-terminal braking experiments

**Files:**
- Create matching `safety_terminal_braking_experiments` groups under models, logs, and results.
- Create: `results/safety_terminal_braking_experiments/README.md`
- Modify: the renamed V19, V20, and V21 launcher paths.

**Interfaces:**
- Consumes: V19 axial braking, V20 terminal calibration, and V21 port-level envelope configurations.
- Produces: three clearly separated safety-terminal experiment runs.

- [ ] Move the V19, V20, and V21 artifacts from the legacy from-scratch group.
- [ ] Rename each run with version, formulation, budget, and seed.
- [ ] Update the preset launchers and any result references.
- [ ] Document the implemented difference and recorded outcome of each version.
- [ ] Compile and import all three launchers.

### Task 7: Rename ten-hertz adaptive continuation artifacts

**Files:**
- Rename `adaptive_dt01_terminal` across models, logs, and results to `ten_hz_adaptive_terminal_continuation`.
- Rename its retained run to `p3_reach_v15dt_ten_hz_adaptive_cadence_continuation_200k_seed0_run2`.
- Create: `results/ten_hz_adaptive_terminal_continuation/README.md`
- Modify: the renamed trainer and its test.

**Interfaces:**
- Consumes: the selected phase-balanced source checkpoint and shared near-P3 reset wrapper.
- Produces: paths that state timestep, adaptive cadence, continuation budget, and seed.

- [ ] Rename the mirrored run and update path constants.
- [ ] Update test imports and expected paths.
- [ ] Document source initialization, deterministic warm-up, reset coverage, action cadence, and recorded evaluation outcome.
- [ ] Run the adaptive-continuation tests.

### Task 8: Rename V12 Optuna tuning artifacts

**Files:**
- Rename `optuna/p3reach_v12` to `p3_reach_v12_optuna` under logs and results.
- Rename the study directory to state V12, from-scratch training, and the 600k budget.
- Create: `results/p3_reach_v12_optuna/README.md`
- Modify: the renamed tuning script.

**Interfaces:**
- Consumes: retained trial numbers and Optuna study outputs.
- Produces: a descriptive study path while leaving trial identities unchanged.

- [ ] Rename study containers without altering individual trial numbers.
- [ ] Update tuning output defaults and study references.
- [ ] Document search space, validation protocol, retained trials, and best recorded configuration.
- [ ] Compile and import the tuning script.

### Task 9: Rewrite family documentation and validate live references

**Files:**
- Modify: `src/controllers/SAC-3DoF/closing/curriculum_and_p3_reach/README.md`
- Modify: repository Markdown or Python files that reference renamed live paths.

**Interfaces:**
- Consumes: all finalized names and group READMEs.
- Produces: a navigable family index and no stale executable references.

- [ ] Add the V1-V21 implementation table.
- [ ] Add a script-to-experiment-group table.
- [ ] Explain which experiments are continuations, from-scratch runs, safety-terminal variants, tuning runs, and cross-family references.
- [ ] Search active Python and Markdown files for every old script and group name.
- [ ] Correct live references while leaving immutable historical artifact contents unchanged.
- [ ] Run Markdown path checks and the SAC structural test.

### Task 10: Verify preservation and repository integration

**Files:**
- Modify tests only if verification exposes a real stale assumption.
- Remove generated cache directories after testing.

**Interfaces:**
- Consumes: the pre-move inventory and completed rename mapping.
- Produces: evidence that names improved without losing artifacts or changing behavior.

- [ ] Compare the post-move artifact count and SHA-256 multiset with the baseline; require zero missing or changed hash instances.
- [ ] Check model/VecNormalize pairings.
- [ ] Compile every family Python file.
- [ ] Import each non-test module with direct-execution path semantics and without invoking `main()`.
- [ ] Run `pytest src/controllers/SAC-3DoF/closing/curriculum_and_p3_reach src/controllers/SAC-3DoF/tests/test_sac_closing_layout.py -q`.
- [ ] Run the complete repository test suite.
- [ ] Run `git diff --check` and inspect `git status --short`.
- [ ] Remove generated `__pycache__` and `.pytest_cache` directories using validated repository-contained paths.
- [ ] Confirm that all changes remain uncommitted.
