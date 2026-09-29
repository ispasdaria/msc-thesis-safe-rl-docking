# Pilot D training-seed variance study

## Purpose

Pilot D was trained with SAC/training-environment seeds 0, 1, and 2 to measure
optimization variance without hyperparameter tuning. All three runs use the
same reward, controller interface, 50/25/25 replay distribution, 60,000-source
prefill, source seeds, prefill RNG, validation seeds, and SAC hyperparameters.
No HER or CBF is used. Only the training seed changes.

## Checkpoint selection

| Training seed | Selected checkpoint | Validation success | Interpretation |
|---:|---:|---:|---|
| 0 | 25k | 8/10 | Learned checkpoint won continuous tie-breakers |
| 1 | 0 | 8/10 | No learned checkpoint beat zero residual |
| 2 | 50k | 9/10 | Learned checkpoint gained one validation success |

Checkpoint selection itself is unstable: the same method selects 0, 25k, and
50k across the three training seeds.

## Held-out results

| Training seed | Seeds 200--219 | Seeds 300--399 | Combined | Hard failures |
|---:|---:|---:|---:|---:|
| 0 | 15/20 | 75/100 | 90/120 (75.0%) | 0 |
| 1 | 14/20 | 75/100 | 89/120 (74.2%) | 0 |
| 2 | 14/20 | 68/100 | 82/120 (68.3%) | 0 |
| Exact LQR reference | 14/20 | 75/100 | 89/120 (74.2%) | 0 |

Across selected Pilot D policies, combined success averages 87/120 (72.5%),
with sample standard deviation 4.36 episodes (3.63 percentage points) and a
range of 82--90 successes. With only three training seeds these descriptive
statistics should not be treated as a precise population confidence interval.

## Paired findings

- Seed 0 versus LQR: two rescues and one loss across 120 seeds; 90 versus 89,
  exact paired `p = 1.0`.
- Seed 1 selected checkpoint 0 and is exactly LQR.
- Seed 2 versus LQR: one rescue and eight losses across 120 seeds; 82 versus
  89, exact paired `p = 0.0391`.
- On seeds 300--399 alone, seed 2 rescued none and lost seven LQR successes;
  exact paired `p = 0.015625`.

Seed 2 demonstrates why the 9/10 validation result cannot be interpreted as
evidence of improvement: its selected checkpoint generalizes substantially
worse than both LQR and the seed-0 policy.

## Conclusion

Pilot D is sensitive to its training seed. One replication falls back to the
baseline, one is statistically worse on the 100-seed set, and the best
replication gains only one non-significant success over LQR. The three-seed
mean is below LQR (72.5% versus 74.2%). These results do not support claiming
that the residual SAC reliably improves the closing controller, and they do
not justify hyperparameter tuning on the current formulation.

For deployment or the thesis reference controller, retain the exact LQR. If a
residual-SAC artifact must be shown as the best observed candidate, use Pilot D
training seed 0 checkpoint 25k, clearly labeled as an exploratory result rather
than a statistically validated improvement.

Detailed numeric values are in `training_seed_summary.csv`; raw rollout files
remain inside each experiment's `evaluations/` directory.
