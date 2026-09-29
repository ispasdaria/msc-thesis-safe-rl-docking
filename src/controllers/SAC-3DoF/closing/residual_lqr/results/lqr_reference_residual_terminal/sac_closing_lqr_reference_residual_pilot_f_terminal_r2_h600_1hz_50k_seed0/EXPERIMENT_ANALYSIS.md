# Pilot F terminal-reset SAC experiment analysis

## Conclusion

Pilot F is **not recommended over the zero-residual LQR**. Terminal-reset
training produced a small validation improvement, but it did not generalize to
the 100 held-out full-closing seeds.

## What changed

Only the online start-state distribution and corresponding online horizon were
changed relative to Pilot D:

- online episodes reset to empirical state/P3 pairs from real LQR trajectories;
- initial worst P3 tolerance ratio was restricted to 1.05--2.0;
- online episode horizon was 600 s;
- the fixed reset bank contained 1,885 eligible snapshots from 100 independent
  source episodes (seeds 1000--1099): 1,110 success-origin and 775
  timeout-origin snapshots.

The controller interface, reward, 1 Hz cadence, zero actor initialization,
constant zero-residual actor penalty, 60k full-closing prefill, 50/25/25 replay,
and all SAC hyperparameters were retained. HER and CBF remained absent.

This is fixed terminal-distribution training, not curriculum learning.

## Oracle study

The paired one-step simulator search scored 9/20 versus 10/20 for LQR, with two
rescues and three losses. It established that residual actions can change
terminal outcomes, but it did not supply a superior oracle controller. See
[the oracle README](../oracle_study_n20_h1/README.md).

## Training and validation

| Checkpoint | Full-closing validation success | Hard failures | Mean best worst ratio |
|---:|---:|---:|---:|
| 0 (LQR) | 8/10 | 0 | 1.0047 |
| 10k | 8/10 | 0 | 1.0047 |
| 25k | **9/10** | 0 | **0.9977** |
| 50k | 9/10 | 0 | 1.0037 |

Checkpoint 25k was selected before held-out testing. The 10k actor is exactly
the zero-initialized policy because the first 10k interactions are critic-only
warm-up.

## Held-out full closing: seeds 300--399

| Controller | Success | Hard failure | Timeout | Mean best worst ratio | Mean minimum position error |
|---|---:|---:|---:|---:|---:|
| Zero-residual LQR | **75/100** | 0 | 25 | 1.04631 | 14.137 m |
| Pilot F 25k | 72/100 | 0 | 28 | 1.04529 | 13.715 m |

Paired changes were one rescue (seed 379) and four losses (seeds 307, 356,
363, and 364). The two-sided exact paired p-value is 0.375. Therefore the
observed -3 percentage-point change is not statistically significant, but its
direction and rescue/loss balance provide no reason to replace LQR.

The learned controller slightly improved continuous approach metrics:

- mean best worst ratio: -0.00102;
- mean minimum position error: -0.422 m;
- mean longest dwell within 2x tolerances: +31.95 s.

However, exact success requires all six ratios to cross 1 simultaneously. Four
baseline successes close to that boundary were perturbed into timeouts. Three
of those learned-policy cases were limited by lateral error at their best joint
state; the fourth was attitude-limited. The policy improved closeness on
average without improving reliable simultaneous capture.

The baseline and SAC halves were saved in separate folders because the original
combined evaluator was interrupted after safely writing the baseline. Both use
the identical seeds, deterministic policies, environment, and horizon. The SAC
half used one CPU inference thread; this affects scheduling only.

## Held-out terminal resets

The two policies were also compared on 50 paired empirical states, one from
each oracle-bank source episode, using the same 600 s horizon as training.

| Controller | Success | Hard failure | Timeout | Mean best worst ratio | Mean final worst ratio |
|---|---:|---:|---:|---:|---:|
| Zero-residual LQR | 18/50 | 0 | 32 | 1.35155 | **1.42318** |
| Pilot F 25k | 19/50 | 0 | 31 | **1.34946** | 1.43975 |

There were two rescues and one loss; the two-sided exact paired p-value is 1.0.
The learned normalized action-vector RMS was only 0.0225. Thus Pilot F learned
at most a very small terminal-task change, not a strong capture policy that
failed only because of long-range distribution shift.

## Final interpretation

The experiment tested the terminal-reset hypothesis cleanly and found no robust
benefit. The most likely explanation is that the remaining LQR timeouts sit on
a narrow, coupled six-tolerance boundary. The unchanged scalar reward and
small residual policy produce marginal local improvements, but those
improvements are not reliably coordinated into simultaneous P3 membership.

This negative result shows that increasing the number of terminal encounters
alone was insufficient. It also rules out the
simple explanation that prior SAC variants failed only because full closing
episodes contained too few near-P3 transitions.

## Reproduction

Training (use a new experiment name because the recorded output is protected):

```powershell
.\.venv\Scripts\python.exe `
  "src/controllers/SAC-3DoF/closing/train_sac_closing_reference_residual_lqr_terminal.py" `
  --experiment-name "pilot_f_terminal_rerun_seed0" --device cpu
```

Full-closing held-out evaluation:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe `
  "src/controllers/SAC-3DoF/closing/evaluate_sac_closing_reference_residual_no_her.py" `
  --artifact-family terminal --seed-start 300 --n-seeds 100 --device cpu `
  --output-name "heldout_seeds300_399_n100_rerun"
```

Held-out terminal evaluation:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe `
  "src/controllers/SAC-3DoF/closing/evaluate_terminal_reference_residual_models.py" `
  --device cpu --output-name "heldout_terminal_n50_h600_rerun"
```

Raw configuration is in [experiment_config.json](experiment_config.json),
checkpoint metrics are in [checkpoint_metrics.csv](checkpoint_metrics.csv),
the full-closing paired summary is in
[heldout_paired_comparison.json](evaluations/heldout_paired_comparison.json), and
the terminal summary is in
[summary.json](evaluations/heldout_terminal_n50_h600/summary.json).
