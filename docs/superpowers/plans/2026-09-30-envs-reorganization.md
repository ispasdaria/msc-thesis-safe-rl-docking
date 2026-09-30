# Environment Package Reorganization Implementation Plan

**Goal:** Reorganize `src/envs/docking3dof` without changing environment behavior, public imports, reward values, or termination semantics.

**Architecture:** Consolidate automated tests under one directory, isolate executable diagnostics and their generated results, centralize test import configuration, and replace oversized reward modules with compatibility facades over focused internal modules. Existing tests and new characterization tests define the behavioral boundary.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, pytest.

## File Map

- Modify `pytest.ini`: configure the repository `src` import root and standard test discovery.
- Rename `src/envs/docking3dof/tests/unit_test_core.py` to `src/envs/docking3dof/tests/test_core.py`.
- Move `src/envs/docking3dof/tests_gym/*.py` into `src/envs/docking3dof/tests/`.
- Move manual programs into `src/envs/docking3dof/validation/` with non-test names.
- Move generated files into `src/envs/docking3dof/tests/results/integrator_timestep_verification/`.
- Modify `src/envs/docking3dof/runners.py`: remove path mutation and clean formatting.
- Create `src/envs/docking3dof/rewards/closing.py`: standard closing reward implementation.
- Create `src/envs/docking3dof/rewards/safety_exploration.py`: complete safety-exploration hierarchy.
- Create `src/envs/docking3dof/rewards/goal_conditioned.py`: goal-conditioned wrappers.
- Create `src/envs/docking3dof/shields/`: the closing CBF-shield implementation.
- Modify `src/envs/docking3dof/gym_env.py`: delegate closing reward computation while preserving its public class.
- Modify `src/envs/docking3dof/closing_safety_exploration_env.py`: compatibility facade for extracted classes.
- Modify affected tests and repository callers only where paths, not public APIs, change.

## Interfaces

- Existing environment imports remain valid, including every class currently exported from `closing_safety_exploration_env.py`.
- `Docking3DoFGymEnv._compute_sac_reward(...)` retains its current signature and return value.
- Validation programs remain runnable from the repository root with `.venv/Scripts/python.exe <path>`.
- Integrator verification accepts the existing explicit output argument and defaults to `tests/results/integrator_timestep_verification`.

## Review Focus

- A plain pytest run must collect `test_core.py`; verify its 73 tests are included.
- Reward component dictionaries and totals must remain identical for all existing characterized states.
- Every safety-exploration version and terminal variant must remain importable from its old module path.
- Diagnostic imports must not open plots or execute simulations as a side effect.
- No production module may modify `sys.path` after the cleanup.

### Task 1: Establish the structural safety net

**Files:**
- Create: `src/envs/docking3dof/tests/test_package_structure.py`
- Modify: `pytest.ini`

- [ ] **Step 1: Write failing structure tests**

Add tests that assert:

- `pytest.ini` exposes `src` through `pythonpath`;
- production modules do not contain `sys.path` mutation;
- the intended diagnostics and results directories exist;
- legacy `tests_gym` and `integrator_timestep_verification_results` paths do not exist;
- legacy imports for core, Gym, runner, CBF, goal-conditioned, and all safety-exploration classes succeed.

- [ ] **Step 2: Run the structure tests and confirm expected failures**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src/envs/docking3dof/tests/test_package_structure.py -q
```

Expected: failures for the current layout and production `sys.path` mutation.

- [ ] **Step 3: Configure central test imports**

Add `pythonpath = src` and `testpaths = src/envs/docking3dof/tests` to `pytest.ini`, retaining the existing `slow` marker.

- [ ] **Step 4: Verify import configuration independently**

Run pytest collection without setting `PYTHONPATH`; imports must resolve.

### Task 2: Consolidate tests, diagnostics, and results

**Files:**
- Rename: `src/envs/docking3dof/tests/unit_test_core.py` → `src/envs/docking3dof/tests/test_core.py`
- Move: `src/envs/docking3dof/tests_gym/test_gym_wrapper.py` → `src/envs/docking3dof/tests/test_gym_wrapper.py`
- Move: `src/envs/docking3dof/tests_gym/test_gym_rollout.py` → `src/envs/docking3dof/tests/test_gym_rollout.py`
- Move: `src/envs/docking3dof/tests_gym/test_closing_safety_exploration_env.py` → `src/envs/docking3dof/tests/test_closing_safety_exploration_env.py`
- Move/rename: `src/envs/docking3dof/tests/manual_control_tests.py` → `src/envs/docking3dof/tests/diagnostics/manual_control.py`
- Move/rename: `src/envs/docking3dof/tests/test_action_response.py` → `src/envs/docking3dof/tests/diagnostics/action_response.py`
- Move: `src/envs/docking3dof/tests/timestep_convergence.py` → `src/envs/docking3dof/tests/diagnostics/timestep_convergence.py`
- Move: `src/envs/docking3dof/tests/integrator_timestep_verification.py` → `src/envs/docking3dof/tests/diagnostics/integrator_timestep_verification.py`
- Move directory contents: `src/envs/docking3dof/tests/integrator_timestep_verification_results/` → `src/envs/docking3dof/tests/results/integrator_timestep_verification/`

- [ ] **Step 1: Record the current collection baseline**

Run collection and save the names/counts used to confirm that the relocated suite contains the existing 120 tests plus the 73 core tests and new structure tests.

- [ ] **Step 2: Perform file moves without altering behavior**

Use filesystem moves so Git recognizes renames; preserve file content initially.

- [ ] **Step 3: Remove test-local path mutations and normalize imports**

Delete repeated `REPO_ROOT`, `SRC_PATH`, and `sys.path` blocks. Replace `from core import Docking3DoFEnv` with `from envs.docking3dof.core import Docking3DoFEnv`.

- [ ] **Step 4: Set the diagnostic output default**

In `diagnostics/integrator_timestep_verification.py`, set the default output directory to:

```text
src/envs/docking3dof/tests/results/integrator_timestep_verification
```

An explicit caller-provided output path continues to override the default.

- [ ] **Step 5: Rewrite conversational comments objectively**

Replace first-person constructions in `diagnostics/manual_control.py`, such as “we want,” with statements of the required acceleration or expected response. Preserve the physical explanation.

- [ ] **Step 6: Run collection, automated tests, and diagnostic smoke checks**

Verify all core and Gym tests are collected from one hierarchy. Import each diagnostic without side effects and run `integrator_timestep_verification.py --help`.

### Task 3: Clean production imports and formatting

**Files:**
- Modify: `src/envs/docking3dof/runners.py`
- Modify: `src/envs/docking3dof/gym_env.py`
- Test: `src/envs/docking3dof/tests/test_package_structure.py`

- [ ] **Step 1: Run the production-path structure test and confirm failure**

Expected: `runners.py` is reported as mutating `sys.path`.

- [ ] **Step 2: Remove path bootstrapping from `runners.py`**

Remove `os`, `sys`, `REPO_ROOT`, and `SRC_PATH`; retain the absolute package import. Normalize whitespace and class spacing.

- [ ] **Step 3: Make empty Gym hooks explicit**

Change `render()` and `close()` to return `None` explicitly. Do not add rendering or cleanup behavior.

- [ ] **Step 4: Run runner, Gym, and structure tests**

Run the focused tests, followed by the entire reorganized environment suite.

### Task 4: Extract the standard closing reward

**Files:**
- Create: `src/envs/docking3dof/rewards/closing.py`
- Modify: `src/envs/docking3dof/gym_env.py`
- Modify: `src/envs/docking3dof/tests/test_gym_wrapper.py`

- [ ] **Step 1: Add characterization tests before extraction**

For every supported closing reward version, capture representative totals and component dictionaries for nominal progress, regression, velocity warning, geometry warning, success, safety failure, and timeout cases already available through test fixtures. Assert exact keys and numerically equivalent values.

- [ ] **Step 2: Run new characterization tests against existing code**

Expected: pass. Temporarily perturb one asserted expected value to confirm the characterization detects a reward change, then restore it.

- [ ] **Step 3: Extract closing reward helpers**

Move only closing-reward calculations into `rewards/closing.py`. Expose one internal dispatcher consumed by `Docking3DoFGymEnv._compute_sac_reward`; keep final-approach behavior and the public method signature in `gym_env.py`.

- [ ] **Step 4: Run characterization and complete Gym tests**

Expected: reward totals, component dictionaries, and termination behavior remain unchanged.

### Task 5: Organize specialized reward and shield implementations

**Files:**
- Rename: `src/envs/docking3dof/rewards/closing_safety.py` → `src/envs/docking3dof/rewards/safety_exploration.py`
- Create: `src/envs/docking3dof/rewards/goal_conditioned.py`
- Create: `src/envs/docking3dof/shields/__init__.py`
- Create: `src/envs/docking3dof/shields/closing_cbf.py`
- Modify: `src/envs/docking3dof/closing_safety_exploration_env.py`
- Modify: `src/envs/docking3dof/goal_conditioned_closing_env.py`
- Modify: `src/envs/docking3dof/closing_cbf_shield_env.py`
- Modify: `src/envs/docking3dof/tests/test_closing_safety_exploration_env.py`
- Modify: repository callers only if they import internal implementation locations

- [ ] **Step 1: Add a complete legacy-export test**

Enumerate every current public environment class and assert that importing it from `envs.docking3dof.closing_safety_exploration_env` returns a class with the expected name and usable constructor.

- [ ] **Step 2: Run the export test against the existing module**

Expected: pass. Temporarily include a nonexistent expected export to demonstrate failure, then restore the valid list.

- [ ] **Step 3: Rename the complete safety-exploration implementation**

Keep the complete inheritance hierarchy together in
`rewards/safety_exploration.py`. Re-export every public class from the
original top-level module and run all safety-exploration tests.

- [ ] **Step 4: Move the goal-conditioned implementation**

Move all goal-conditioned classes to `rewards/goal_conditioned.py`. Replace
the original module with explicit compatibility exports and run its tests and
repository import checks.

- [ ] **Step 5: Move the closing CBF shield implementation**

Move `build_closing_cbf_config`, `CBF_TRAINING_VARIANTS`, and
`ClosingCBFShieldedGymEnv` to `shields/closing_cbf.py`. Replace the original
module with explicit compatibility exports and run CBF and repository import
tests.

- [ ] **Step 6: Verify all compatibility modules**

Each old top-level module contains only a module docstring, explicit imports,
and `__all__`. No reward or shield calculations remain in the facades.

### Task 5A: Move diagnostics to top-level validation

**Files:**
- Create: `src/envs/docking3dof/validation/__init__.py`
- Move: `src/envs/docking3dof/tests/diagnostics/manual_control.py` → `src/envs/docking3dof/validation/manual_control.py`
- Move: `src/envs/docking3dof/tests/diagnostics/action_response.py` → `src/envs/docking3dof/validation/action_response.py`
- Move: `src/envs/docking3dof/tests/diagnostics/timestep_convergence.py` → `src/envs/docking3dof/validation/timestep_convergence.py`
- Move: `src/envs/docking3dof/tests/diagnostics/integrator_timestep_verification.py` → `src/envs/docking3dof/validation/integrator_timestep_verification.py`
- Move: `src/envs/docking3dof/tests/diagnostics/_path_setup.py` → `src/envs/docking3dof/validation/_path_setup.py`
- Modify: `src/envs/docking3dof/tests/test_package_structure.py`

- [ ] **Step 1: Change the structure test to require top-level validation**

Assert that `validation/` and its four programs exist and that
`tests/diagnostics/` no longer exists. Run the test and observe failure before
moving files.

- [ ] **Step 2: Move validation programs and adjust result paths**

Move files without changing their behavior. Update the integrator program's
default output path so it still resolves to
`tests/results/integrator_timestep_verification/` from its new location.

- [ ] **Step 3: Run direct-execution and import smoke checks**

Run the integrator program with `--help` and import every validation module
without executing simulations or opening plots.

- [ ] **Step 7: Run repository-wide import and test checks**

Search all Python callers for the old module and confirm each import remains valid without modification unless it used a private symbol.

### Task 6: Final verification and hygiene

**Files:** All changed files.

- [ ] **Step 1: Verify structure requirements**

Run:

```powershell
rg -n "sys\.path|REPO_ROOT|SRC_PATH" src/envs/docking3dof -g "*.py"
rg -ni "\b(we|you|your|let's|simply|obviously|clearly|basically)\b" src/envs/docking3dof -g "*.py"
```

Review any remaining matches individually; mathematical use of “simple” or quoted API text is not automatically conversational.

- [ ] **Step 2: Compile all environment modules**

```powershell
.\.venv\Scripts\python.exe -m compileall -q src/envs
```

- [ ] **Step 3: Verify default collection**

```powershell
.\.venv\Scripts\python.exe -m pytest --collect-only -q
```

Confirm that `test_core.py` contributes its 73 tests and the old directories contribute none.

- [ ] **Step 4: Run the full configured test suite**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all collected tests pass without warnings introduced by the reorganization.

- [ ] **Step 5: Verify public imports and diagnostic entry points**

Run a clean interpreter import smoke test for all public environment classes and `--help` for diagnostics that expose a CLI.

- [ ] **Step 6: Inspect the final diff**

Run `git diff --check`, review `git status --short`, and confirm that generated results were renamed rather than duplicated and no unrelated files changed.
