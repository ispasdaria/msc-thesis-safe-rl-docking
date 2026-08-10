# Final-approach CBF-QP safety filter

This package filters the final-approach SAC action before it is passed to the
3-DoF docking simulator. It implements two safety properties as three affine
QP rows:

1. Exact docking-cone containment (upper and lower HOCBF rows)
2. Docking-port total-speed envelope (one relative-degree-one CBF row)

The filter uses the unnormalized simulator state and physical body-frame
action

```text
state = [rx, ry, vx, vy, psi, omega]
u     = [ax_Bc, ay_Bc, alpha_Bc]
```

The SAC action is converted to physical units before the QP and converted back
to the normalized Gym action afterward. Actuator limits are included inside
the QP, so no post-solve clipping is needed to make the command admissible.

## Barriers

For cone slope `kappa = 0.1`, the exact cone is represented by

```text
h_upper = kappa*s - y_p >= 0
h_lower = kappa*s + y_p >= 0
```

Each side uses the relative-degree-two condition

```text
h_ddot + (k1 + k2)*h_dot + k1*k2*h >= 0.
```

The port-speed barrier is

```text
h_speed = port_speed_max**2 - dot(v_p, v_p) >= 0
```

with `port_speed_max = 0.15 m/s`. This is a final-approach speed envelope. It
does not by itself guarantee the stricter terminal contact-speed condition or
a distance-dependent braking profile.

All port quantities and their affine acceleration model reuse the simulator's
port offset, target-frame geometry, and dynamics. Rotational acceleration is
therefore represented in the translational port constraints.

## Installation and tests

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest src/controllers/cbf_3DoF/tests -q
```

The tests check port kinematics, the affine acceleration model, barrier signs,
finite-difference derivatives, HOCBF reset-set admissibility, exact and
simultaneous boundary cases, actuator-bounded hard feasibility, one-step RK4
invariance, randomized near-boundary states, individual QP behavior, and both
SciPy and OSQP solver paths. `diagnostics.py` also provides an LP-based hard
feasibility-margin calculation for tuning and stress-test scripts.

## Paired SAC evaluation

The evaluator defaults to the seed-10 best checkpoint and compares the
unfiltered controller with both CBF properties:

```powershell
.\.venv\Scripts\python.exe src/controllers/cbf_3DoF/evaluate_final_approach_cbf.py `
  --variants baseline both `
  --num-seeds 10 `
  --solver osqp
```

Useful ablations are available through `--variants baseline cone speed both`.
Every variant receives the same initial-state seeds. The output contains:

- `per_episode_metrics.csv`: one row per variant and evaluation seed
- `aggregate_metrics_by_filter_variant.csv`: aggregate success, intervention,
  slack, and QP data
- `step_timeseries_<filter>_eval_seedXXXX.npz`: states, actions, barriers, QP
  residuals, and slack at every step
- `run_configuration.json`: the exact checkpoint and frozen filter parameters

Slack is enabled by default to make infeasibility visible during development.
Every slack value is logged. Use `--hard-constraints` only after feasibility
has been checked under the actuator limits; if slack is retained, the filter
must be reported as a soft safety filter rather than hard enforcement.

These are continuous-time CBF conditions applied to a sampled controller with
a finite integration step. The analytic CBF construction plus sampled rollout
tests support the implementation, but should not be presented as a formal
discrete-time safety proof.

See `MULTICONTROLLER_RESULTS.md` for the initial hard-CBF comparison across six
SAC training seeds and its limitations.

## Gain tuning

`tune_final_approach_cbf.py` performs a resumable Optuna grid search over the
HOCBF gains and speed-CBF response gain. It rejects configurations that fail
the admissible boundary precheck, hard QP solve, QP residual, or sampled
post-step barrier check. The physical cone slope and speed limit are never
tuned. The default study uses validation scenario seeds 100--119 and keeps the
previous development seeds and future final-test seeds separate.

```powershell
.\.venv\Scripts\python.exe src/controllers/cbf_3DoF/tune_final_approach_cbf.py
```

For a quick integration check before the full study:

```powershell
.\.venv\Scripts\python.exe src/controllers/cbf_3DoF/tune_final_approach_cbf.py `
  --controller-seeds 10 `
  --num-eval-seeds 1 `
  --max-steps 10 `
  --cone-k-values 0.1 `
  --speed-alpha-values 0.5 `
  --n-trials 1
```

The output directory contains a resumable `optuna_gain_search.sqlite3`, the
complete `gain_search_trial_results.csv`, `gain_search_plan.json`, and—if at
least one safe trial completes—`selected_cbf_gain_configuration.json`.
