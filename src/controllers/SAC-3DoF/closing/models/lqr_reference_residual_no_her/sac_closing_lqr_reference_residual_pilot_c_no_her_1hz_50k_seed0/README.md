# Pilot C fixed-goal reference-residual SAC model card

Run: `sac_closing_lqr_reference_residual_pilot_c_no_her_1hz_50k_seed0`  
Status: **complete**  
Selected best checkpoint: **25,000 SAC decisions**

## What changed from Pilot B

Pilot C retains the same 1 Hz SAC reference residual and unchanged closing
LQR, but removes HER completely. Every replay transition keeps the real fixed
P3 target. Exact P3 success terminates an episode. The task reward was divided
by 100 so critic values and the actor residual prior operate at comparable
scales.

No CBF is instantiated, called, observed, or used in the reward.

## Controller

```text
delta_reference = [
    0.020 * a[0] m/s,
    0.010 * a[1] m/s,
    1.000 * a[2] deg,
]
u_nominal = LQR(state, scheduled_P3_reference + delta_reference)
```

SAC and LQR both update every second. There is no action hold or distance
gate, and zero SAC action exactly reproduces the established LQR.

## Reward

```text
reward = 1.000 * (J(previous) - gamma * J(next))
         + 1.000 * exact_P3_success
         - 10.000 * hard_failure
         - 1.000 * unsuccessful_timeout
         - 0.010 * ||a||^2
         - 0.010 * ||a - a_previous||^2
```

`J` equally combines the mean and worst log-transformed P3 tolerance ratios.

## SAC configuration

| Parameter | Value |
|---|---:|
| Online interactions | 50,000 |
| Replay prefill | 60,000 |
| Replay method | Fixed-goal `DictReplayBuffer`; no HER |
| Critic-only warm-up | 10,000 interactions |
| Learning rate | 0.0001 |
| Effective actor learning rate | 2.5e-05 |
| Actor zero-reference weight | 1 |
| Train frequency | every 4 interactions |
| Gradient steps | 1 |
| Update-to-data ratio | 0.250 |
| Batch size | 256 |
| Discount factor | 0.9999 |
| Target smoothing | 0.0050 |
| Entropy coefficient | `auto_0.01` |
| Target entropy | -3.0 |

## Validation results

Validation uses seeds 100--109.

| Checkpoint | Success | Hard failure | Timeout | Mean minimum position error [m] | Mean best worst ratio | Mean seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 80.0% | 0.0% | 20.0% | 16.531 | 1.005 | 11361.6 |
| 10,000 | 80.0% | 0.0% | 20.0% | 16.531 | 1.005 | 11361.6 |
| 25,000 | 90.0% | 0.0% | 10.0% | 17.313 | 1.000 | 11291.6 |
| 50,000 | 90.0% | 0.0% | 10.0% | 17.995 | 1.004 | 11272.3 |

## Interpretation

Checkpoint 25,000 outranked the baseline and is stored as `best_model.zip`.
It converted validation seed 109 from a 12,000-second LQR timeout into an
exact P3 success at 11,109 seconds. Seed 101 remains the only timeout.

The learned correction remains genuinely residual. On 1,137 states sampled
every 10 seconds from the seed-100 zero-residual LQR trajectory, checkpoint
25,000 had mean normalized action
`[-0.00117, -0.00410, +0.00313]` and RMS action
`[0.00397, 0.00410, 0.00321]`. It never placed any action component above
0.9 in magnitude. The mean physical axial-reference correction is only about
`-0.0000233 m/s`, compared with Pilot B's pathological approximately
`+0.01995 m/s` offset. Measurements are stored in
`policy_action_diagnostics.csv` in the results directory.

Checkpoint 50,000 also achieved 9/10 but was not selected because its mean
best simultaneous target ratio was 1.0040, compared with 1.0003 at checkpoint
25,000. `final_model.zip` therefore differs from the recommended
`best_model.zip`.

## Independent held-out tests

After checkpoint selection, checkpoint 0 and the selected checkpoint 25,000
were evaluated on the previously unused seeds 200--219. These seeds did not
influence training or model selection.

| Controller | Success | Hard failure | Timeout | Mean best worst ratio | Mean seconds |
|---|---:|---:|---:|---:|---:|
| Zero-residual LQR | 14/20 (70%) | 0/20 | 6/20 | 1.0392 | 11500.3 |
| Pilot C checkpoint 25,000 | 15/20 (75%) | 0/20 | 5/20 | 1.0350 | 11475.6 |

In this first 20-seed set, Pilot C converted seed 219 from an LQR timeout into
success at 11,382 seconds without losing a baseline success.

A subsequent, larger evaluation used the previously unseen seeds 300--399:

| Controller | Success | Hard failure | Timeout | Mean best worst ratio | Mean seconds |
|---|---:|---:|---:|---:|---:|
| Zero-residual LQR | 75/100 (75%) | 0/100 | 25/100 | 1.0463 | 11300.3 |
| Pilot C checkpoint 25,000 | 74/100 (74%) | 0/100 | 26/100 | 1.0466 | 11302.6 |

Pilot C rescued no LQR timeout in the 100-seed set and converted seed 356 from
an LQR success into a timeout. Across both independent sets, LQR and Pilot C
are exactly tied at 89/120 successes with zero hard failures. Pilot C rescues
seed 219 and loses seed 356; all other seeds retain the same success category.
The exact paired test gives `p = 1.0`, so the held-out evidence does not
support a success-rate improvement over LQR.

The learned policy is stable, but its small reference corrections redistribute
error rather than robustly improve closing success. The 20-seed records are in
`heldout_test_summary.json` and `heldout_test_rollout_metrics.csv`; the
100-seed records are under `evaluations/heldout_seeds300_399_n100/`.

## Artifacts

- `checkpoint_0_model.zip`: exact zero-reference LQR baseline;
- `checkpoint_<N>_model.zip`: evaluated SAC checkpoint;
- `best_model.zip`: validation-ranked checkpoint;
- `final_model.zip`: end-of-training policy.

Metrics and configuration are in
`../../../results/lqr_reference_residual_no_her/sac_closing_lqr_reference_residual_pilot_c_no_her_1hz_50k_seed0/`.
