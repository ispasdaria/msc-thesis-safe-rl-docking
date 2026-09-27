# Evaluation with a 13,000 s Episode Limit

This evaluation uses seeds 0-49, `dt=0.1 s`, and a `13,000 s` episode limit.
The same generated trajectories are also classified at a `12,000 s` cutoff,
which isolates the effect of the longer duration.

The paired current-code comparison reports:

- 26 successes and 24 timeouts at `12,000 s`;
- 28 successes and 22 timeouts at `13,000 s`;
- no hard safety violations at either cutoff.

Seeds 4 and 21 completed shortly after `12,000 s`. The longer limit therefore
added two successes, or four percentage points, in this 50-seed pilot. The
result is a modest horizon benefit rather than a resolution of the lateral
convergence problem.

The historical root baseline reports only 23 successes because it was generated
with an earlier repository state. `comparison_summary.json` records this
provenance warning explicitly.
