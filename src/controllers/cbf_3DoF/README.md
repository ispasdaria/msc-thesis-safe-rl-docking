# Final-approach CBF-QP safety filter

This package filters the final-approach SAC action before it is passed to the
3-DoF docking simulator. It implements three safety properties as seven affine
QP rows:

1. Exact docking-cone containment (upper and lower HOCBF rows)
2. Docking-port total-speed envelope (one relative-degree-one CBF row)
3. Attitude and angular-rate alignment envelopes (two attitude HOCBF rows and
   two angular-rate CBF rows)

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

The protected alignment box is

```text
|psi_error| <= 3 deg
|omega|     <= 0.15 deg/s
```

Attitude uses the relative-degree-two condition

```text
h_ddot + (attitude_k1 + attitude_k2)*h_dot
       + attitude_k1*attitude_k2*h >= 0.
```

Angular rate uses the relative-degree-one condition

```text
h_dot + angular_rate_alpha*h >= 0.
```

These limits match the simulator's conservative safe thresholds and remain
inside its failure thresholds of 4 deg and 0.2 deg/s.

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
unfiltered controller with all three protected safety properties:

```powershell
.\.venv\Scripts\python.exe src/controllers/cbf_3DoF/evaluate_final_approach_cbf.py `
  --variants baseline all `
  --num-seeds 10 `
  --solver osqp
```

Useful ablations are available through
`--variants baseline cone speed alignment both all`. Here, `both` retains its
historical meaning of cone plus speed, while `all` also enables alignment.
Every variant receives the same initial-state seeds. The output contains:

- `per_episode_metrics.csv`: one row per variant and evaluation seed
- `aggregate_metrics_by_filter_variant.csv`: aggregate success, intervention,
  slack, and QP data
- `step_timeseries_<filter>_eval_seedXXXX.npz`: states, actions, pre/post-step
  barriers, QP residuals, strict and tolerance-aware protected-barrier
  violations, QP failure diagnoses, and slack at every step
- `run_configuration.json`: the result-schema version, exact checkpoint, frozen
  filter parameters, protected-barrier reporting tolerance, and QP fallback
  behavior

Result schema version 2 reports post-step barrier crossings in two separate
fields. `strict_protected_barrier_violation*` uses the mathematical condition
`h < 0`. `protected_barrier_violation_beyond_tolerance*` uses
`h < -1e-6` to distinguish crossings larger than numerical reporting noise.
The tolerance affects diagnostics only; simulator termination and success
remain strict.

Every failed QP step records the original solver status, an independently
computed actuator-bounded hard-feasibility margin, and the minimal infeasible
constraint family or family combination. For example,
`hard_constraints_infeasible[cone]` means that the cone rows are infeasible
even without speed or alignment constraints. A feasible independent LP paired
with a failed QP is instead reported as
`solver_failed_despite_hard_feasibility[...]`. Episode and variant CSVs contain
counts of these reasons as well as the per-step NPZ records.

Slack is enabled by default to make infeasibility visible during development.
Every slack value is logged. Use `--hard-constraints` only after feasibility
has been checked under the actuator limits; if slack is retained, the filter
must be reported as a soft safety filter rather than hard enforcement.

The existing failed-QP behavior is intentionally unchanged: the filter returns
the clipped nominal SAC action and sets `qp_failed=True`. This action is not
CBF-certified. The evaluator documents why the failure occurred on every such
step so that nominal-fallback outcomes cannot be interpreted as uninterrupted
hard-CBF enforcement.

These are continuous-time CBF conditions applied to a sampled controller with
a finite integration step. The analytic CBF construction plus sampled rollout
tests support the implementation, but should not be presented as a formal
discrete-time safety proof.

## Trajectory plots

`plot_final_approach_cbf_trajectories.py` creates figures directly from a
completed evaluator result folder; it does not rerun SAC or the simulator. A
single command produces spatial geometry, all protected-state envelopes,
filter action corrections, QP-failure diagnostics, and a symmetric-log zoom
of the sampled cone boundary. PNG and vector PDF are written by default under
the central
`results/final_approach/plots/<result-folder>/eval_seedXXXX/` directory.

```powershell
.\.venv\Scripts\python.exe src/controllers/cbf_3DoF/plot_final_approach_cbf_trajectories.py `
  --result-dir "src/controllers/cbf_3DoF/results/final_approach/evaluations/RESULT_FOLDER" `
  --eval-seeds 120 `
  --variants baseline both all `
  --output-dir "src/controllers/cbf_3DoF/results/final_approach/plots/REPRESENTATIVE_CASE_NAME"
```

Use `--formats png` for faster inspection or keep the default PNG and PDF for
thesis use. Multiple trajectories from the same result folder can be rendered
in one call, for example `--eval-seeds 120 127`. The outcome shown in each
legend is read from `per_episode_metrics.csv`, and all plotted samples are read
from the exact NPZ file named by that row.

See `MULTICONTROLLER_RESULTS.md` for the initial hard-CBF comparison across six
SAC training seeds and its limitations.

## Gain tuning

`tune_final_approach_cbf.py` performs a resumable Optuna grid search over the
cone, speed, attitude, and angular-rate response gains. It rejects
configurations that fail the admissible boundary precheck, hard QP solve, QP
residual, or sampled post-step barrier check. The physical cone slope, speed
limit, attitude limit, and angular-rate limit are never tuned. The default
study uses validation scenario seeds 100--119 and keeps the previous
development seeds and future final-test seeds separate.

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
  --attitude-k-values 0.3 `
  --angular-rate-alpha-values 0.5 `
  --n-trials 1
```

By default, studies are stored under `results/final_approach/tuning/`, beside
the `evaluations/` and `plots/` collections.

The output directory contains a resumable `optuna_gain_search.sqlite3`, the
complete `gain_search_trial_results.csv`, `gain_search_plan.json`, and—if at
least one safe trial completes—`selected_cbf_gain_configuration.json`.
