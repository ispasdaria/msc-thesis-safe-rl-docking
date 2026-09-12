# Final-Approach Control-Effort Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one command that reuses complete saved SAC+CBF histories, generates only missing LQR and retained-SAC histories, and produces the requested paired control-effort and trajectory-efficiency analysis.

**Architecture:** A standalone analysis module owns pure metric/statistics functions, saved-artifact validation, the two missing rollout collectors, output tables, and figures. Existing controller/environment modules remain unchanged; the CBF campaign is read-only input.

**Tech Stack:** Python 3.12, NumPy, pandas, Matplotlib, Stable-Baselines3, Gymnasium, pytest.

**Spec:** `docs/superpowers/specs/2026-09-12-final-approach-control-effort-design.md`

## Global Constraints

- Use evaluation seeds exactly 0--49 with final/testing initialization and deterministic SAC inference.
- Integrate only final clipped physical body-frame commands.
- Reuse existing SAC+CBF histories and do not rerun the CBF campaign.
- Keep translational and rotational quantities separate because their units differ.
- Do not describe equivalent Delta-v as fuel or propellant consumption.

---

### Task 1: Pure rollout metrics and statistics

**Files:**
- Create: `src/controllers/analyze_final_approach_control_effort.py`
- Create: `src/controllers/tests/test_final_approach_control_effort.py`

**Interfaces:**
- Produces: `compute_rollout_metrics(times, port_positions, applied_actions, action_limits)` and `distribution_stats(values)`.

- [ ] Write literal-fixture tests for Euclidean Delta-v, rotational effort, duration, saturation-at-bound, path length, path efficiency, and separated control variation.
- [ ] Run the focused tests and confirm failure because the module does not exist.
- [ ] Implement the pure functions with finite/shape/nonnegative validation.
- [ ] Run focused tests and confirm they pass.

### Task 2: Existing-artifact loaders and provenance checks

**Files:**
- Modify: `src/controllers/analyze_final_approach_control_effort.py`
- Modify: `src/controllers/tests/test_final_approach_control_effort.py`

**Interfaces:**
- Produces: CBF history loader, saved campaign validation, and canonical rollout-history schema.

- [ ] Add failing tests using temporary NPZ/CSV/JSON fixtures for filtered-command selection, seed completeness, and mismatch rejection.
- [ ] Implement loaders that require `action_filtered`, intervention flags, exact seeds/configuration, and physical bounds.
- [ ] Confirm focused tests pass.

### Task 3: Incremental LQR and retained-SAC collection

**Files:**
- Modify: `src/controllers/analyze_final_approach_control_effort.py`

**Interfaces:**
- Consumes: existing LQR constructors, environment history, exact SAC model and VecNormalize paths.
- Produces: canonical per-seed histories with state-sample times, signed docking-port positions, and applied clipped commands.

- [ ] Add failing tests for canonical history alignment and initial/terminal position count.
- [ ] Implement LQR and SAC collectors without changing controller behavior.
- [ ] Add CLI modes `--audit-only`, `--generate-missing`, and `--analyze` with CBF reuse mandatory.
- [ ] Confirm focused tests pass.

### Task 4: Tables, validation, interpretation, and figures

**Files:**
- Modify: `src/controllers/analyze_final_approach_control_effort.py`
- Modify: `src/controllers/tests/test_final_approach_control_effort.py`

**Interfaces:**
- Produces: per-rollout CSV, summary CSV, paired CSV, configuration/validation JSON, PNG/PDF figures, and console interpretation.

- [ ] Add failing tests for common-success selection, sample standard deviation, paired signs/counts, and manual history recomputation.
- [ ] Implement output generation and separate paired figures for Delta-v and rotational effort.
- [ ] Include path length, path efficiency, and both control-variation metrics in detailed summaries and paired results.
- [ ] Confirm focused and relevant repository tests pass.

### Task 5: Command smoke test and handoff

**Files:**
- Modify only if a defect is exposed by verification.

- [ ] Run `--audit-only` against repository artifacts.
- [ ] Run CLI help and a one-seed temporary-output smoke test for both missing-history collectors.
- [ ] Verify no CBF rollout executes in `--generate-missing` mode.
- [ ] Report the exact full command for the user to run seeds 0--49.
