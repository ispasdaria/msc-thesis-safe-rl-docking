# Pilot A residual SAC model card

Run: `sac_closing_goal_her_residual_lqr_dt0p1_pilot20k_seed0`  
Status: **complete**  
Selected best checkpoint: **0 residual decisions**

## Purpose

This experiment leaves the closing LQR law unchanged. SAC selects one
normalized residual action every 0.1 s (1
simulator step(s)), while the LQR action is recomputed from the current physical
state every 0.1 s. With one hold step, both feedback decisions
operate at 10 Hz.

```text
u_executed = clip(u_LQR_clipped + 0.05 * gate(distance) * a_residual, -1, 1)
```

The residual gate is zero at or below
100 m, uses a
smoothstep ramp, and reaches full authority at or above
200 m.

## SAC configuration

| Parameter | Value |
|---|---:|
| Online residual decisions | 20,000 |
| Residual decision interval | 0.1 s |
| LQR update interval | 0.1 s |
| Replay prefill transitions | 60,000 |
| Critic warm-up | 10,000 residual decisions/updates |
| Critic learning rate | 0.0001 |
| Actor learning-rate scale | 0.1 |
| Effective actor learning rate | 1e-05 |
| Train frequency | 1 |
| Gradient steps | 1 |
| Update-to-data ratio | 1 |
| Discount factor | 0.999900 per residual decision |
| Target smoothing coefficient | 0.0050 |
| Entropy coefficient | `auto_0.001` |
| Target entropy | -6.0 |
| Zero-residual actor weight | 10.0 |
| Batch size | 512 |
| HER future goals per transition | 4 |

The first 10,000 online residual decisions use
the deterministic zero actor and update only the critic. With `train_freq=1`
and `gradient_steps=1`, this produces one critic update per collected residual
transition during warm-up.

## Reward

The environment reward remains `goal_conditioned_p3_residual_lqr_v1`, whose
numerical calculation is inherited from `GoalConditionedClosingV3Env`:

```text
reward = 0.1 * (-0.01 * goal_cost
                           -velocity_warning_depth
                           -geometry_warning_depth
                           -alignment_warning_depth)
         +100 * exact_P3_success
         -1000 * hard_failure
         -100 * unsuccessful_timeout
```

The persistent shaping terms are scaled by the 0.1 s / 1.0 s reference
interval. Terminal events are not time-scaled.

The constrained actor additionally optimizes
`10.0 * ||a_residual||^2`. This is an
actor-only regularizer and is not part of the environment reward or critic
target. The reward returned for a held residual decision is the final
0.1-second substep reward so HER can recompute it consistently; the sum across
substeps is retained only as an `info` diagnostic.

## Validation results

Validation uses seeds 100--109.
`Mean decisions` counts SAC residual decisions; `Approx. seconds` multiplies
that count by 0.1 s and may overestimate an episode whose final held
action ended early on success or termination.

| Checkpoint | Success | Hard failure | Timeout | Mean minimum position error [m] | Mean best worst ratio | Mean decisions | Approx. seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 40.0% | 0.0% | 60.0% | 15.006 | 1.133 | 117819.7 | 11782.0 |
| 20,000 | 40.0% | 0.0% | 60.0% | 15.010 | 1.133 | 117820.8 | 11782.1 |

## Artifacts

- `checkpoint_0_model.zip`: exact zero-residual LQR baseline;
- `checkpoint_<N>_model.zip`: evaluated model at residual decision `N`;
- `best_model.zip`: best validation-ranked checkpoint;
- `final_model.zip`: policy at the end of training.

The matching metrics and exact configuration are under
`../../../results/her_lqr_residual/sac_closing_goal_her_residual_lqr_dt0p1_pilot20k_seed0/`.
