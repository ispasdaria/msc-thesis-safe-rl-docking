# Residual SAC around the closing LQR: saved models

This directory contains three 50,000-step pilot experiments in which SAC
learns a bounded normalized action correction around the existing closing
LQR controller.  The LQR remains active at every simulator step.

The executed normalized body-frame command is

```text
u_executed = clip(u_LQR_clipped + 0.05 * gate(distance) * a_residual, -1, 1)
```

where `a_residual` is the SAC action in `[-1, 1]^3`.  The nominal LQR command
is converted to the body frame, clipped to the physical actuator limits, and
normalized before the residual is added.  A zero residual therefore exactly
reproduces the actuator-realizable LQR controller.

The environment and action composition are implemented in
[`lqr_residual_env.py`](../../lqr_residual_env.py).  The constrained SAC
algorithm is implemented in
[`zero_residual_constrained_sac.py`](../../zero_residual_constrained_sac.py).

## Common artifact names

Every experiment directory contains:

| Artifact | Meaning |
|---|---|
| `checkpoint_0_model.zip` | Zero-initialized residual actor. This is exact LQR behavior. |
| `checkpoint_10000_model.zip` | Model evaluated after 10,000 online SAC steps. |
| `checkpoint_25000_model.zip` | Model evaluated after 25,000 online SAC steps. |
| `checkpoint_50000_model.zip` | Model evaluated after 50,000 online SAC steps. |
| `final_model.zip` | Model saved at the end of the 50,000-step training call. |
| `best_model.zip` | Best checkpoint under the validation ranking, not necessarily the last checkpoint. |

For all three pilots, validation selected checkpoint 0.  Consequently,
`best_model.zip` is the protected zero-residual LQR model; it is not the final
learned residual policy.  Use `checkpoint_25000_model.zip`,
`checkpoint_50000_model.zip`, or `final_model.zip` when intentionally studying
the learned residual policies.

Checkpoint ranking prioritizes, in order: higher exact P3 success rate, lower
hard-failure rate, lower best worst-component target ratio, longer dwell near
P3, lower final worst-component ratio, and higher return.

## Experiment 1: plain residual SAC

Directory:

[`sac_closing_goal_her_residual_lqr_pilot50k_seed0`](sac_closing_goal_her_residual_lqr_pilot50k_seed0)

This experiment used ordinary Stable-Baselines3 SAC with a zero-initialized
actor but no critic-only warm-up and no explicit zero-residual actor prior.

Key configuration:

- residual scale: `0.05`;
- entropy coefficient: `auto_0.01`;
- actor/critic learning rate: `2.5e-5`;
- distance-dependent residual gate: none;
- replay prefill: 60,000 transitions;
- online training: 50,000 transitions.

The actor rapidly collapsed to an almost constant action-space corner near
`[+1, -1, -1]`.  This was diagnosed as critic extrapolation outside the
well-supported residual actions.  The final validation result was 6/10 P3
successes, 0/10 hard failures, and 4/10 timeouts.  Validation therefore kept
checkpoint 0, which achieved 8/10 successes.

Metrics:

[`checkpoint_metrics.csv`](../../results/her_lqr_residual/sac_closing_goal_her_residual_lqr_pilot50k_seed0/checkpoint_metrics.csv)

## Experiment 2: constrained residual SAC

Directory:

[`sac_closing_goal_her_residual_lqr_constrained50k_seed0`](sac_closing_goal_her_residual_lqr_constrained50k_seed0)

This experiment added:

- 10,000 critic-only warm-up steps with the residual actor frozen at zero;
- a `20 * ||a_residual||^2` term in the actor objective;
- actor learning-rate scale `0.1`, giving an actor rate of `2.5e-6`;
- entropy coefficient `auto_0.001`.

The actor-corner collapse was eliminated.  Checkpoint 10,000 was exactly
identical to LQR because the actor was frozen during warm-up.  The learned
policy reached smaller position errors, but the exact six-component P3
success rate declined to 7/10 at 25,000 steps and 5/10 at 50,000 steps.  All
episodes remained free of hard-safety failures.

Metrics:

[`checkpoint_metrics.csv`](../../results/her_lqr_residual/sac_closing_goal_her_residual_lqr_constrained50k_seed0/checkpoint_metrics.csv)

## Experiment 3: gated constrained residual SAC

Directory:

[`sac_closing_goal_her_residual_lqr_gated_constrained50k_seed0`](sac_closing_goal_her_residual_lqr_gated_constrained50k_seed0)

This experiment retained the critic warm-up and actor constraint and added a
smooth axial-distance authority gate:

```text
distance <= 100 m: gate = 0, exact LQR
100 m < distance < 200 m: smoothstep interpolation
distance >= 200 m: gate = 1, full residual authority
```

The intention was to let SAC influence long-range closing while returning
terminal braking and P3 regulation entirely to LQR.  The final result was
7/10 successes, 0/10 hard failures, and 3/10 timeouts.  This was better than
the ungated constrained final policy but still below the 8/10 zero-residual
LQR baseline.

Metrics:

[`checkpoint_metrics.csv`](../../results/her_lqr_residual/sac_closing_goal_her_residual_lqr_gated_constrained50k_seed0/checkpoint_metrics.csv)

## Reward used by all three experiments

All three experiments used the same HER-compatible numerical reward inherited
from `GoalConditionedClosingV3Env`.  The residual environment reports the
scheme name `goal_conditioned_p3_residual_lqr_v1`, but it does not replace the
V3 reward mathematics.

The reward is based on the six P3 tolerance-normalized errors, an exact P3
success bonus, hard-safety failure penalty, continuous velocity/geometry/
alignment warning depths, and a timeout penalty.  There is no action-effort
or residual-magnitude term in the environment reward.

The constrained experiments' `20 * ||a_residual||^2` term belongs only to the
actor optimization objective.  It is not stored as reward, is not recomputed
by HER, and is not part of the critic target.

See the detailed formula in the results
[`README.md`](../../results/her_lqr_residual/README.md).

## Loading a model

Models saved with `HerReplayBuffer` must be loaded with a compatible
environment attached.  Use ordinary `SAC` for the first experiment and
`ZeroResidualConstrainedSAC` for the constrained experiments.  For example:

```python
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from her_closing_utils import make_her_env
from zero_residual_constrained_sac import ZeroResidualConstrainedSAC

env_kwargs = {
    "residual_scale": 0.05,
    "residual_zero_distance": 100.0,
    "residual_full_distance": 200.0,
}
env = DummyVecEnv(
    [
        lambda: Monitor(
            make_her_env(
                seed=999,
                mode="testing",
                max_time=12_000.0,
                reward_version="lqr_residual",
                **env_kwargs,
            )
        )
    ]
)
model = ZeroResidualConstrainedSAC.load(
    "sac_closing_goal_her_residual_lqr_gated_constrained50k_seed0/"
    "checkpoint_50000_model.zip",
    env=env,
    device="cpu",
)
```

Use the same residual scale and distance-gate parameters that were used during
training.  They are recorded in each corresponding result directory's
`experiment_config.json`.
