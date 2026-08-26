# Residual SAC around the closing LQR: experiment results

This directory records three residual-SAC pilots evaluated on closing
validation seeds 100 through 109.  Each episode was allowed at most 12,000
one-second control steps.  These ten-episode evaluations are diagnostic pilot
comparisons, not final statistically powered thesis results.

The corresponding saved models are documented in
[`models/her_lqr_residual/README.md`](../../models/her_lqr_residual/README.md).

## Reward function

The residual experiments use the HER-compatible reward implemented by
`GoalConditionedClosingV3Env` in
[`goal_conditioned_closing_env.py`](../../../../../envs/docking3dof/goal_conditioned_closing_env.py).
They do not use the LQR quadratic cost as an RL reward.  Although the wrapped
base Gym environment computes the older V15 closing reward internally, that
value is deliberately discarded by the goal-conditioned wrapper and is
reported only as the diagnostic field `inherited_v15_reward_ignored`.

### Goal-error ratios

For physical achieved state `x`, desired P3 state `g`, and P3 tolerance vector
`t`, define six component ratios

```text
r_i = abs(x_i - g_i) / t_i
```

with the attitude error wrapped to `[-pi, pi]`.  The tolerance vector is

```text
[20 m, 2 m, 0.01 m/s, 0.01 m/s, 0.5 deg, 0.05 deg/s].
```

Exact P3 success requires every ratio to satisfy `r_i <= 1` simultaneously.

The ratios are transformed without clipping:

```text
n_i = log(1 + r_i) / log(101)
```

and combined as

```text
goal_cost = 0.5 * mean(n_i) + 0.5 * max(n_i).
```

This gives equal weight to average six-component quality and the currently
worst normalized component.

### Complete per-step reward

The V3 reward used by all three residual experiments is

```text
reward = -0.01 * goal_cost
         +100 * success
         -1000 * hard_failure
         -1 * velocity_warning_depth
         -1 * geometry_warning_depth
         -1 * alignment_warning_depth
         -100 * timeout * (1 - success)
```

where each warning depth lies in `[0, 1]`:

- velocity depth increases between the safe and unsafe docking-port or axial
  closing-speed thresholds;
- geometry depth measures proximity to or violation of the approach ellipse
  and approach-cone boundaries;
- alignment depth increases between the safe and unsafe attitude/angular-rate
  thresholds;
- `hard_failure` is goal-independent and comes from the simulator's closing
  safety termination;
- timeout is penalized only if the transition is not successful.

HER relabels `desired_goal` with future achieved states and recomputes this
same reward.  Goal-independent failure and warning data are retained through
the copied transition `info` dictionary.

There is no residual magnitude, action effort, or action-rate penalty in this
environment reward.

### Actor-only regularization in experiments 2 and 3

The constrained SAC runs modify the actor loss after a 10,000-step critic-only
warm-up:

```text
actor_loss = mean(alpha * log_pi(a|s) - Q(s,a))
             + 20 * mean(||a_residual||^2).
```

This zero-residual prior should not be described as part of the reward.  It is
used only when optimizing the actor; the critic continues to learn the reward
defined above.

## What the reported outcomes mean

- **Success:** all six P3 error ratios were at most one at the same step.
- **Hard failure:** the simulator terminated on a closing safety constraint.
- **Timeout:** no exact P3 success and no hard failure within 12,000 seconds.
- **Mean minimum position error:** for each episode, the smallest Euclidean P3
  position error reached at any step, then averaged across episodes.  It is not
  necessarily the terminal error.
- **Best maximum target ratio:** for each episode, the smallest value achieved
  by the worst of the six tolerance ratios.  Lower is better; a value at most
  one indicates that the complete target set was reached in that episode.

## Common zero-residual LQR baseline

Checkpoint 0 initializes the deterministic SAC residual to exactly zero.
Therefore, checkpoint 0 is an exact actuator-realizable LQR rollout, not an
independently learned SAC policy.

On validation seeds 100--109 it achieved:

| Metric | Result |
|---|---:|
| Successes | 8/10 |
| Hard failures | 0/10 |
| Timeouts | 2/10 |
| Mean minimum position error | 16.531 m |
| Mean best maximum target ratio | 1.005 |

This ten-seed 80% result should not be confused with the separate 50-seed LQR
baseline study, which used a different seed set.

## Experiment comparison

The table uses each run's 50,000-step learned checkpoint, with checkpoint 0
shown as the common reference.

| Experiment/checkpoint | Success | Hard failure | Timeout | Mean minimum position error | Mean best maximum target ratio |
|---|---:|---:|---:|---:|---:|
| Exact LQR, checkpoint 0 | 8/10 | 0/10 | 2/10 | 16.531 m | 1.005 |
| Plain residual, 50k | 6/10 | 0/10 | 4/10 | 13.546 m | 1.230 |
| Constrained residual, 50k | 5/10 | 0/10 | 5/10 | 12.599 m | 1.344 |
| Gated constrained residual, 50k | 7/10 | 0/10 | 3/10 | 14.894 m | 1.016 |

The learned policies sometimes reduced position error while worsening exact
six-component success.  Position alone is therefore not a sufficient closing
handoff metric.  Velocity, lateral position, attitude, and angular rate must
all enter their respective P3 tolerances at the same time.

## Experiment directories

### Plain residual SAC

[`sac_closing_goal_her_residual_lqr_pilot50k_seed0`](sac_closing_goal_her_residual_lqr_pilot50k_seed0)

- ordinary SAC;
- no critic warm-up;
- no actor zero-residual prior;
- no distance gate;
- learned actor collapsed near an action-space corner;
- best selected checkpoint: 0.

### Constrained residual SAC

[`sac_closing_goal_her_residual_lqr_constrained50k_seed0`](sac_closing_goal_her_residual_lqr_constrained50k_seed0)

- 10,000-step critic-only warm-up;
- actor zero-residual weight: 20;
- actor learning-rate scale: 0.1;
- no distance gate;
- action-corner collapse eliminated;
- best selected checkpoint: 0.

### Gated constrained residual SAC

[`sac_closing_goal_her_residual_lqr_gated_constrained50k_seed0`](sac_closing_goal_her_residual_lqr_gated_constrained50k_seed0)

- same critic warm-up and actor prior as the constrained run;
- residual authority zero at or below 100 m;
- smoothstep authority from 100 to 200 m;
- full residual authority at or above 200 m;
- best selected checkpoint: 0.

Each experiment directory contains:

- `experiment_config.json`: exact run configuration;
- `checkpoint_metrics.csv`: aggregate validation metrics by checkpoint;
- `validation_rollout_metrics.csv`: per-seed checkpoint metrics;
- `zero_residual_validation_rollouts.csv`: checkpoint-0/LQR rollouts;
- `source_episode_collection.csv`: replay-prefill episode summary;
- `source_initialization.json`: replay-prefill and initial validation summary;
- `best_checkpoint.json`: selected checkpoint and its summary.

## Interpretation

All evaluated policies remained free of observed hard-safety failures on the
ten validation seeds.  None of the learned checkpoints exceeded the exact LQR
success rate, so model selection correctly retained checkpoint 0 in every
experiment.

The results indicate that direct additive action residuals have limited room
to improve this nominal controller because the closing LQR already requests
saturated actuator commands during most of the long-range maneuver.  A
residual can reduce or redirect an actuator command but cannot create thrust
beyond the physical limit.  Learning corrections to the LQR reference
schedule is therefore a more promising follow-up than simply increasing the
action-residual authority.
