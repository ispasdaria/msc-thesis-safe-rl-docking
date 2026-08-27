# Terminal residual-action controllability diagnostic

This is a **pre-training feasibility diagnostic**, not a learned controller and
not a mathematical controllability proof.

## Setup

- Diagnostic source trajectories: ordinary full-closing zero-residual LQR,
  testing seeds 400--449.
- Terminal bank: 2,719 empirical state/P3 pairs in the worst-tolerance-ratio
  band 1.05--5.0.
- Paired sample: 20 states, one per selected source episode. All ten available
  timeout-origin episodes were included, plus ten success-origin controls.
- Rollout horizon: 1,500 s.
- Baseline: zero residual, which exactly reproduces closing LQR.
- Search controller: at every second, test all 27 combinations of
  `{-1, 0, +1}` for the three bounded reference residuals in an exact simulator,
  score the next state, execute the best action for one second, and replan.
- Reference bounds: 0.02 m/s axial velocity, 0.01 m/s lateral velocity, and
  1 degree attitude.
- No HER or CBF is involved.

## Results

| Controller | Success | Mean best worst ratio |
|---|---:|---:|
| Zero-residual LQR | 10/20 (50%) | 1.0807 |
| One-step grid search | 9/20 (45%) | 1.0766 |

Paired outcome changes:

- 2 LQR timeouts were rescued;
- 3 LQR successes were lost;
- 5 pairs were discordant in total.

## Interpretation

The two rescues demonstrate that the bounded reference-residual interface has
real terminal action authority. The search controller is nevertheless worse
overall because its one-step objective is myopic: locally reducing the worst
P3 ratio can damage a later simultaneous tolerance crossing.

A search failure does not prove uncontrollability. The diagnostic searches a
finite action grid and uses only one-step lookahead. A three-step pilot was
abandoned because its simulator cost was too high for a statistically useful
sample.

The machine-readable results are in [summary.json](summary.json),
[config.json](config.json), and [paired_rollouts.csv](paired_rollouts.csv).

## Reproduction

```powershell
.\.venv\Scripts\python.exe `
  "src/controllers/SAC-3DoF/closing/analyze_terminal_reference_residual_oracle.py" `
  --dataset "src/controllers/SAC-3DoF/closing/demonstrations/terminal_reference_residual_oracle_seeds400_449.npz" `
  --output-dir "src/controllers/SAC-3DoF/closing/results/lqr_reference_residual_terminal/oracle_study_n20_h1_rerun" `
  --n-states 20 --nonsuccess-fraction 0.5 `
  --rollout-horizon 1500 --lookahead-steps 1
```
