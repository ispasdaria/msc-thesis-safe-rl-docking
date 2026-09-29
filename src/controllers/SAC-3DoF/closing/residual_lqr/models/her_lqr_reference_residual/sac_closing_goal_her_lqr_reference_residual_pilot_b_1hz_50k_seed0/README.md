# Pilot B LQR-reference-residual SAC model card

Run: `sac_closing_goal_her_lqr_reference_residual_pilot_b_1hz_50k_seed0`  
Status: **complete**  
Selected best checkpoint: **0 SAC decisions**

## Controller architecture

SAC produces a fresh normalized reference correction every simulator second.
There is no action hold and no distance gate. The correction changes only the
reference tracked by the existing closing LQR:

```text
delta_reference = [
    0.020 * a[0] m/s,
    0.010 * a[1] m/s,
    1.000 * a[2] deg,
]
u_nominal = LQR(state, scheduled_P3_reference + delta_reference)
```

The zero action exactly reproduces the established closing LQR. CBF is not
instantiated, called, observed, or used in the reward during this experiment.
Hard failure termination remains part of the unfiltered simulator.

## Reward

Let `J(g, x)` be an equally weighted mean/worst log transform of the six
absolute P3 tolerance ratios. For the SAC discount `0.9999`:

```text
reward = 100.0 * (J(previous) - gamma * J(next))
         + 100.0 * exact_P3_success
         - 1000.0 * hard_failure
         - 100.0 * unsuccessful_timeout
         - 0.010 * ||a||^2
         - 0.010 * ||a - a_previous||^2
```

The previous achieved goal and both actions are stored in every replay info
dictionary, allowing HER to recompute all terms after goal relabeling.

## SAC configuration

| Parameter | Value |
|---|---:|
| Online interactions / decisions | 50,000 |
| SAC and LQR decision interval | 1 s |
| Replay prefill transitions | 60,000 |
| Critic-only warm-up | 10,000 interactions |
| Learning rate | 0.0001 |
| Effective actor learning rate | 2.5e-05 |
| Actor zero-reference weight | 1 |
| Train frequency | every 4 interactions |
| Gradient steps per update | 1 |
| Update-to-data ratio | 0.250 |
| Batch size | 256 |
| Discount factor | 0.9999 |
| Target smoothing coefficient | 0.0050 |
| Entropy coefficient | `auto_0.01` |
| Target entropy | -3.0 |
| HER future goals per transition | 4 |

## Validation results

Validation uses seeds 100--109.

| Checkpoint | Success | Hard failure | Timeout | Mean minimum position error [m] | Mean best worst ratio | Mean seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 80.0% | 0.0% | 20.0% | 16.531 | 1.005 | 11361.6 |
| 10,000 | 80.0% | 0.0% | 20.0% | 16.531 | 1.005 | 11361.6 |
| 25,000 | 0.0% | 0.0% | 100.0% | 82.073 | 4.137 | 12000.0 |
| 50,000 | 0.0% | 0.0% | 100.0% | 81.673 | 4.117 | 12000.0 |

## Interpretation

No learned checkpoint outranked the zero-reference LQR. Success changed from
80% at checkpoint 0 to 0% at checkpoints 25,000 and 50,000. Both learned
policies avoided hard failure but prevented convergence and timed out on all
ten validation seeds. `best_model.zip` therefore intentionally remains the
exact zero-reference baseline. `final_model.zip` is retained for diagnosis
and is not the recommended controller.

The actor collapsed toward the reference-action boundary. On 1,137 states
sampled every 10 seconds from the seed-100 zero-residual LQR trajectory,
checkpoint 25,000 produced a mean normalized action of approximately
`[+0.9990, +0.9765, +0.7357]`. Checkpoint 50,000 still produced approximately
`+0.9977` on the axial channel, corresponding to a nearly constant
`+0.01995 m/s` axial-velocity reference offset. This makes the desired closing
velocity less negative and explains the approximately 82 m terminal position
error. The associated measurements are stored in
`policy_action_diagnostics.csv` in the results directory.

The training log also reported a zero-reference prior loss of 2.58 (the
three-action maximum is 3) while the unregularized SAC actor loss had magnitude
about 915. Consequently, the weight-1 actor prior was too weak relative to the
critic scale. With dense task progress already available, future work should
also ablate HER: relabeled future goals plus repeated success bonuses can
inflate critic values and teach behavior that is valuable for relabeled goals
but not for the fixed physical P3 target.

## Artifacts

- `checkpoint_0_model.zip`: zero-reference-residual LQR baseline;
- `checkpoint_<N>_model.zip`: evaluated SAC checkpoint;
- `best_model.zip`: validation-ranked checkpoint;
- `final_model.zip`: final policy, whether or not it ranked best.

Metrics and the machine-readable configuration are in
`../../../results/her_lqr_reference_residual/sac_closing_goal_her_lqr_reference_residual_pilot_b_1hz_50k_seed0/`.
