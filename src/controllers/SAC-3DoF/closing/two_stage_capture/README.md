# Two-Stage Capture Experiments

This directory implements the experiment without modifying existing docking
environments, controllers, checkpoints, or training scripts.  Generated data,
models, logs, and evaluations stay under `artifacts/` in this directory.

## Systematic three-controller experiment (current)

The current interface is explicit and shared by both new controllers:

```
sampled P2 -> X_H -> validated X_P3(testing) -> docking
```

`systematic_interface.py` is the single source of truth. For an episode P3
reference sampled from the final controller's validated testing box, the
candidate X_H center is 150 m farther out, has 0.08 m/s additional closing
speed, and retains the P3 lateral velocity, attitude, and angular rate. The
six X_H tolerances are `[20 m, 5 m, 0.015 m/s, 0.01 m/s, 1 deg,
0.05 deg/s]`.

Capture training no longer needs a long-range rollout dataset. It samples
complete, safe X_H/P3 pairs directly at the full 150 m handover distance from
the first episode; no curriculum learning is used. Success means direct
membership in the exact P3 testing box:

- `rx`: 280-320 m;
- `ry`: -3--3 m;
- `vx`: -0.05---0.01 m/s;
- `vy`: -0.01--0.01 m/s;
- `psi`: -1--1 deg;
- `omega`: -0.05--0.05 deg/s.

Geometry, velocity, attitude, and angular-rate unsafe states terminate. The
The v7 reward persistently penalizes P3-target errors in `vx`, `ry`, `vy`,
attitude, and angular rate using pseudo-Huber costs with linear tails. There is no
distance-dependent `vx` reference: `vx` is always compared with the sampled P3
target. Only its penalty weight increases near P3, so the controller is not
driven to brake to terminal speed immediately at X_H. Positive axial progress
is reduced only beyond maneuver-level lateral/alignment bounds and retains a
0.05 floor; outward motion retains its full penalty. Explicit lateral-position
and lateral-velocity improvement rewards teach recovery direction. Geometry
warnings use a stronger continuous violation depth, and velocity/alignment
warnings remain separate.

v7 restores the v5 approach weights outside the terminal zone. Only within 30
m of the upper P3 face does it strengthen `vx` tracking and reward potential
reduction toward the actual global six-component P3 box. Inward motion is
penalized while any non-axial terminal component is invalid, using a reduced
0.20 maximum weight. Lateral and attitude weights are not increased near P3.
The inherited sampled-target
`passed_P3_region` warning is not charged between 280 and 320 m, aligning
geometry shaping with the systematic success contract.
Crossing inward below `rx = 280 m` without valid P3 capture terminates
immediately. Valid capture gives +1000, every safety/missed/flyaway failure
gives -1500, and timeout gives -500. Per-step costs are deliberately scaled so
terminal outcomes dominate the return. The SAC entropy coefficient remains
fixed at 0.01.

v7 retains the v5 replacement of failure-dominated adaptive observation normalization with
fixed physical component scales. Lateral divergence beyond 50 m position error
or 0.15 m/s velocity error terminates early as a failed capture. These changes
apply only to new v5 models; the evaluator detects legacy v3/v4 normalization
files and preserves their original observation path.
The replay-buffer size always equals `--total-timesteps`.

Dry-run the new trainer:

```powershell
& ".venv\Scripts\python.exe" `
  src/controllers/SAC-3DoF/closing/two_stage_capture/train_systematic_capture_sac.py `
  --total-timesteps 150000 `
  --checkpoint-frequency 50000 `
  --max-time 2000 `
  --experiment-name systematic_capture_v7_h150_pilot_seed0 `
  --dry-run
```

Remove `--dry-run` and add `--progress-bar` to train. New outputs are written
under `artifacts/systematic/`. By default, an existing experiment directory is
protected; pass `--overwrite` to replace only the matching isolated model,
result, and log directories. Evaluate the finished model with
`evaluate_systematic_capture.py` at 30, 60, 100, and 150 m. Train
`train_systematic_long_range_sac.py` only after the 150 m capture result is
reliable; that trainer samples P2 and targets the exact same six-state X_H set.

## Earlier radius-entry experiment (preserved)

The scripts and artifacts below are the historical first-entry dataset trial.
They remain reproducible, but they are not inputs to the systematic trainer.

## 1. Collect long-range first-entry states

Pilot collection on 20 training-distribution P2/P3 pairs:

```powershell
python src/controllers/SAC-3DoF/closing/two_stage_capture/collect_handover_dataset.py `
  --seed-start 10000 `
  --num-seeds 20 `
  --radii 50 75 100 150
```

For the real training dataset, increase `--num-seeds` to at least 500.  The
collector uses the frozen guarded 100k model and its matching VecNormalize by
default.  Each deterministic rollout records the first valid entry for every
radius, including the complete state, exact P3 target, error vector, target
ratios, radial rate, seed, time, and step.

Use a disjoint seed range for the later unbiased capture evaluation, for
example a second dataset beginning at seed 20000.

## 2. Inspect the entry rates

The collector writes three sibling files:

- `guarded100k_handover_sweep.npz`: machine-readable reset dataset;
- `guarded100k_handover_sweep.csv`: compact entry diagnostics;
- `guarded100k_handover_sweep_summary.json`: hit/miss results per source seed.

Do not train at a radius that the source controller almost never reaches.

## 3. Train a new capture SAC

First verify configuration without training:

```powershell
python src/controllers/SAC-3DoF/closing/two_stage_capture/train_capture_sac.py `
  --capture-radius 100 `
  --dry-run
```

Then run the isolated trial:

```powershell
python src/controllers/SAC-3DoF/closing/two_stage_capture/train_capture_sac.py `
  --capture-radius 100 `
  --empirical-fraction 0.8 `
  --total-timesteps 300000 `
  --progress-bar
```

Eighty percent of resets reproduce real long-range first-entry states.  The
remaining resets are generated using component-specific position, lateral,
velocity, attitude, and rate bounds with safety and approach-direction
rejection. Capture uses the isolated `capture_progress_safety_terminal_v1`
reward and the existing exact six-state P3 success condition. The reward uses
bounded progress in a tolerance-normalized six-component P3 cost rather than a
persistent distance cost. Hard closing-safety violations terminate immediately
with a -1500 reward; valid P3 capture terminates with +1000.

For this capture-only trial, alignment is also safety-critical. Attitude and
angular-rate values up to 3 deg and 0.15 deg/s are safe; values between 3--4 deg
or 0.15--0.2 deg/s are warnings and receive the continuous warning-depth
penalty. Exceeding 4 deg or 0.2 deg/s is a hard alignment safety violation and
terminates immediately. This extension is local to the capture trial.

The capture maneuver has a 2000 s default time limit (`--max-time 2000`). This
is 60% shorter than the final-approach environment's 5000 s default. The limit
can be changed per isolated run without modifying the shared environments.

## 4. Evaluate the capture region of attraction

Collect a second handover dataset with disjoint source seeds, then evaluate a
saved capture model on all four radii:

```powershell
python src/controllers/SAC-3DoF/closing/two_stage_capture/evaluate_capture_sweep.py `
  --model <capture-model.zip> `
  --vecnormalize <matching-vecnormalize.pkl> `
  --dataset <disjoint-evaluation-handover-dataset.npz> `
  --radii 50 75 100 150
```

The principal result is
`valid_p3_success_given_entry`.  A valid P3 success must satisfy the closing
P3 target tolerance and must not be unsafe when classified as a final-approach
initial state.  Predeclare a reliability threshold, such as 95%, and choose
the largest radius meeting it.  Report the long-range entry rate separately;
the evaluator also reports their product as a diagnostic, not as a substitute
for a later end-to-end mission evaluation.
