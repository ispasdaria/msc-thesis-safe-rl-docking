# Rollout-count convergence protocol

This protocol supports the rollout counts used for the LQR results. It separates
retrospective evidence for the historical 50-rollout tuning runs from the
independent final evaluation, which can be extended with held-out seeds.

## What can be claimed

The original 50-rollout evaluations were conducted separately for each phase:
50 closing-phase rollouts for the selected closing controller and a separate 50
final-approach rollouts for the selected final-approach controller. They were
part of controller tuning. The full list of candidate controllers and their
intermediate rankings is not available, so candidate-ranking convergence cannot
be reconstructed. The supporting analysis should therefore be described as a
retrospective stability check of the selected controller's phase-level metrics,
with one analysis for each phase, not as proof that all candidate rankings had
converged.

The selected controllers should be frozen before the final evaluation. Use a
non-overlapping seed range for the final claim, for example seeds 300--599.
The existing seeds 0--299 may remain as historical results, but they overlap
with tuning seeds 0--49 and should not be called independent validation.

## Run the evaluations

From the repository root, using the project environment:

```powershell
.\.venv\Scripts\python.exe src\controllers\LQR-3DoF\run_lqr_full_mission.py `
  --num-missions 50 `
  --seed-start 0 `
  --skip-closing-only `
  --skip-one-mission `
  --results-csv src\controllers\LQR-3DoF\results\lqr_mission\historical_0_49.csv
```

This rerun is optional if the original per-rollout records are available. It
reproduces the selected controllers on the historical tuning seed range, but it
does not replace the fact that the original closing and final-approach samples
were separate phase-level evaluations.

If phase-level per-rollout records were not saved, rerun the selected
controllers separately with the phase sweep scripts, using the same settings
as the historical tuning run. The scripts now support `--results-csv`; run one
candidate at a time so that the exported file belongs to one frozen controller.
Save one record file for the closing controller and one for the final-approach
controller. Do not combine the two phase samples into a single 100-rollout
sample.

For example, with the selected controller represented by the default tuned
scales:

```powershell
.\.venv\Scripts\python.exe src\controllers\LQR-3DoF\tuning_lqr_closing\lqr_sweep_closing.py `
  --q-scales 1.0 --r-scales 1.0 --rollouts 50 --seed-start 0 `
  --results-csv src\controllers\LQR-3DoF\results\lqr_closing\closing_phase_0_49.csv

.\.venv\Scripts\python.exe src\controllers\LQR-3DoF\tuning_lqr_final_approach\lqr_sweep_final.py `
  --q-scales 1.0 --r-scales 1.0 --rollouts 50 --seed-start 0 `
  --results-csv src\controllers\LQR-3DoF\results\lqr_final\final_approach_phase_0_49.csv
```

For the two historical phase checks, use phase-appropriate primary metrics:

- closing controller: normalized terminal error relative to the sampled
  $P_3$ handover target, together with closing success rate;
- final-approach controller: docking-port distance or completion time,
  together with final-approach success rate.

For independent final evaluation, run a fresh range after freezing both
controllers:

```powershell
.\.venv\Scripts\python.exe src\controllers\LQR-3DoF\run_lqr_full_mission.py `
  --num-missions 300 `
  --seed-start 300 `
  --skip-closing-only `
  --skip-one-mission `
  --results-csv src\controllers\LQR-3DoF\results\lqr_mission\validation_300_599.csv
```

The command can be repeated with `--num-missions 600 --seed-start 300` if a
larger final sample is desired. Do not combine results from different
controller versions or simulator settings.

## Analyze cumulative convergence

For the retrospective closing-phase check, use the closing-controller CSV:

```powershell
.\.venv\Scripts\python.exe src\controllers\LQR-3DoF\analyze_rollout_convergence.py `
  --input src\controllers\LQR-3DoF\results\lqr_closing\closing_phase_0_49.csv `
  --output src\controllers\LQR-3DoF\results\lqr_closing\closing_phase_0_49_convergence.csv `
  --metric final_error `
  --sample-sizes 10 20 30 40 50
```

Run the same analysis independently for the final-approach controller:

```powershell
.\.venv\Scripts\python.exe src\controllers\LQR-3DoF\analyze_rollout_convergence.py `
  --input src\controllers\LQR-3DoF\results\lqr_final\final_approach_phase_0_49.csv `
  --output src\controllers\LQR-3DoF\results\lqr_final\final_approach_phase_0_49_convergence.csv `
  --metric port_distance `
  --sample-sizes 10 20 30 40 50
```

The complete-mission CSV may additionally be used for a 50-run historical
mission-level diagnostic, but that is a different analysis from the two
phase-level checks above. The historical phase samples remain the evidence for
the per-candidate screening budget; the held-out complete-mission sample is
the evidence for final mission-level precision.

For the final evaluation, combine non-overlapping held-out batches only after
verifying that the controller and simulator settings are identical:

```powershell
$first = "src\controllers\LQR-3DoF\results\lqr_mission\validation_300_599.csv"
$second = "src\controllers\LQR-3DoF\results\lqr_mission\validation_600_899.csv"
$combined = "src\controllers\LQR-3DoF\results\lqr_mission\validation_300_899.csv"
$header = Get-Content $first -TotalCount 1
$header | Set-Content $combined
Get-Content $first | Select-Object -Skip 1 | Add-Content $combined
Get-Content $second | Select-Object -Skip 1 | Add-Content $combined
```

Then run:

```powershell
.\.venv\Scripts\python.exe src\controllers\LQR-3DoF\analyze_rollout_convergence.py `
  --input src\controllers\LQR-3DoF\results\lqr_mission\validation_300_599.csv `
  --output src\controllers\LQR-3DoF\results\lqr_mission\validation_300_599_convergence.csv `
  --metric closing_final_error_norm
```

The script reports cumulative success and safety rates, Wilson intervals,
means, sample variances, and 5th--95th percentile ranges over 1,000 random
seed permutations. This addresses the dependence of cumulative curves on seed
ordering.

## Decision rules

Define these rules before inspecting the convergence output:

- historical tuning check at 50: success-rate change at most 5 percentage
  points, mean change at most 5%, and variance change at most 10% over the last
  three available increments;
- final evaluation at 300: success-rate change at most 2 percentage points,
  mean change at most 2%, and variance change at most 5% over the last three
  increments;
- final success-rate Wilson half-width below 5 percentage points;
- use engineering absolute tolerances when a mean is close to zero.

For each historical phase-specific 50-run sample, report whether that selected
controller's metrics met the criteria, but do not claim unavailable
candidate-ranking stability. For the final complete-mission runs, report both
convergence and precision. If a rare event causes a continuous-metric variance
to change substantially, report that fact rather than discarding the event;
the rollout count can still be justified for binary-rate precision. A zero
observed safety-violation rate is not a guarantee of safety; it should be
reported with its one-sided binomial upper confidence bound. The same applies
to a nonzero observed safety rate, which should be reported directly with a
binomial confidence interval.

## Article-sized statement

After the analysis has been run, the article can state:

> Rollout counts were selected using a cumulative convergence analysis. The
> Historical screening used two separate 50-rollout phase evaluations: one for
> the closing controller and one for the final-approach controller. For each
> phase, a retrospective analysis of the selected controller checked
> stabilization of the phase success rate, a phase-appropriate terminal metric,
> and its sample variance. Because the complete set of tuning candidates was
> not retained, this analysis is not used to claim candidate-ranking
> convergence. After freezing both controllers, an independent evaluation was
> performed on held-out seeds 300--599. At each cumulative sample size, mission
> success and safety rates, Wilson confidence intervals, and continuous-metric
> means and variances were calculated. Three hundred missions were retained when
> the predefined convergence criteria were satisfied and the success-rate
> confidence interval had a half-width below approximately five percentage
> points.

Replace the final sentence with the observed result, or increase the sample
size if the criteria are not met.
