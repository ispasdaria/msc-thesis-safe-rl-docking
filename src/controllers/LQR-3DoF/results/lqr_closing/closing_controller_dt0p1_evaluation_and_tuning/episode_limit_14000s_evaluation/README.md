# Evaluation with a 14,000 s Episode Limit

This directory contains the 50-rollout evaluation for seeds 0-49 with
`dt=0.1 s` and a `14,000 s` episode limit.

The stored CSV reports 28 successes, 22 timeouts, and no additional success
relative to the `13,000 s` evaluation. Extending the limit by another
`1,000 s` therefore did not improve the outcome for this seed set. The result
supports treating the remaining timeouts as convergence failures rather than
episodes that only require slightly more time.

No separate aggregate JSON summary was retained for this run; the CSV is the
authoritative record.
