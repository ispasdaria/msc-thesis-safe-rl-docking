# Environment Package Reorganization Design

## Purpose

Reorganize `src/envs/docking3dof` so production environments, reward
experiments, automated tests, diagnostic programs, and generated verification
results have clear locations. Preserve environment behavior and existing
public imports while making every automated test discoverable by the default
pytest configuration.

## Scope

The reorganization covers:

- automated test discovery and test-directory consolidation;
- diagnostic and manual simulation scripts;
- generated integrator-verification results;
- repeated import-path setup;
- objective wording in comments and docstrings;
- decomposition of oversized reward implementations; and
- compatibility exports for existing training and evaluation scripts.

The change does not alter dynamics, safety boundaries, observations, reward
formulas, termination rules, reset distributions, or public class names.

## Target Structure

```text
src/envs/docking3dof/
  core.py
  gym_env.py
  runners.py
  closing_cbf_shield_env.py
  goal_conditioned_closing_env.py
  closing_safety_exploration_env.py
  rewards/
    __init__.py
    closing.py
    safety_exploration.py
    goal_conditioned.py
  shields/
    __init__.py
    closing_cbf.py
  tests/
    test_core.py
    test_runners.py
    test_runner_validation.py
    test_gym_wrapper.py
    test_gym_rollout.py
    test_closing_safety_exploration_env.py
    results/
      integrator_timestep_verification/
        detailed_convergence.csv
        worst_case_summary.csv
        terminal_event_resolution.csv
        study_metadata.json
        rk4_dt_0p1_summary.json
  validation/
    manual_control.py
    action_response.py
    timestep_convergence.py
    integrator_timestep_verification.py
```

## Test Organization

`unit_test_core.py` becomes `test_core.py`, allowing pytest's standard
`test_*.py` pattern to collect its 73 tests. The current `tests_gym` files move
into the main `tests` directory. This produces one automated-test hierarchy.

Files whose entry points run simulations, create plots, or write reports are
validation programs rather than pytest modules. They move to the top-level
`validation` package and receive names that do not imply automatic collection.
Their direct-execution entry points remain available.

Generated verification data remains under `tests`, as requested, at
`tests/results/integrator_timestep_verification`. The directory name identifies
the diagnostic that produced the files. The integrator verification program
will document this path and use it as its default output location unless the
caller supplies an explicit output directory.

## Import and Compatibility Strategy

Repository-level pytest configuration will add `src` to Python's import path.
Automated tests will import package modules normally and will not mutate
`sys.path` independently. Diagnostic scripts will use the same package imports
and remain runnable from the repository root with the project interpreter.

Production modules must not modify `sys.path`. In particular, `runners.py`
will contain only package imports.

Existing imports such as:

```python
from envs.docking3dof.closing_safety_exploration_env import (
    ClosingSafetyTerminalV21GymEnv,
)
```

will continue to work. The current module becomes a small compatibility
facade that re-exports implementations from focused reward modules. Existing
class names, inheritance relationships required by callers, and constructor
interfaces remain available.

## Reward Decomposition

The closing reward logic will move behind focused helpers or mixins grouped by
reward family. `Docking3DoFGymEnv` continues to own Gymnasium interaction,
action scaling, step orchestration, and reward dispatch. Extracted reward code
receives the same state and diagnostic inputs and returns the same total and
component dictionary.

The standard closing reward implementation moves to `rewards/closing.py` and
is invoked by `Docking3DoFGymEnv._compute_sac_reward` through a focused
dispatcher. The dispatcher owns closing-phase reward calculations; the Gym
environment retains observation, action, transition, and final-approach
orchestration.

The complete safety-exploration hierarchy moves to
`rewards/safety_exploration.py`. Its inheritance chain remains together
because later historical variants reuse earlier private calculations.

The goal-conditioned implementation moves to `rewards/goal_conditioned.py`,
and the CBF-shielded implementation moves to `shields/closing_cbf.py`.
Original top-level modules remain explicit compatibility facades. This is a
structural refactor, not an opportunity to redesign reward formulas.

Before moving each reward family, characterization tests will record public
outputs for representative nominal, warning, unsafe, success, and timeout
states. The existing detailed reward tests remain authoritative.

## Comment and Style Cleanup

First-person or conversational explanations will be rewritten as objective
technical statements. Comments that explain intent, invariants, physical
units, or non-obvious safety behavior will remain. Formatting cleanup will
include trailing whitespace, import grouping, and standard blank-line spacing.

Empty Gymnasium `render` and `close` hooks may explicitly return `None`; their
observable behavior will not change.

## Verification

The completed reorganization must satisfy all of the following:

1. `python -m compileall -q src/envs` succeeds.
2. Imports of core, Gym, runner, CBF, goal-conditioned, and every
   safety-exploration class succeed through their existing module paths.
3. Default pytest collection includes the 73 core tests previously hidden by
   `unit_test_core.py`.
4. The complete environment suite passes after all moves and extractions.
5. Diagnostic scripts display usable `--help` output or pass an equivalent
   import/smoke check without opening plots during collection.
6. No production module under `src/envs/docking3dof` mutates `sys.path`.
7. Existing generated verification files are present under
   `tests/results/integrator_timestep_verification` and no generated-result
   directory remains mixed with test source files.

## Risks and Controls

The largest risk is accidentally changing reward behavior while extracting
large methods and inheritance chains. Characterization tests and incremental
family-by-family moves control this risk. Compatibility facades prevent
downstream import breakage. File moves occur separately from behavioral
extraction so failures can be attributed to one kind of change at a time.

The repository may contain callers outside `src/envs`; repository-wide import
searches will therefore precede each public-module change. Compatibility
exports remain in place even after internal callers migrate.
