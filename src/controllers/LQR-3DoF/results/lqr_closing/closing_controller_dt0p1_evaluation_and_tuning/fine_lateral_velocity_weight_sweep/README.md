# Fine Lateral Velocity Weight Sweep

This follow-up study holds the lateral position multiplier at `my=1` and tests
six lateral velocity multipliers from `mvy=0.75` through `mvy=1.0`. The sweep
uses seeds 0-49, `dt=0.1 s`, and a `12,000 s` episode limit.

All candidates passed the gain and sampled-system stability checks. The best
safety-preserving candidates were `mvy=0.85` and `mvy=0.80`, each with 26
successes, 24 timeouts, and no hard safety violations. The original `mvy=1.0`
configuration recorded 23 successes in this historical sweep. The
`mvy=0.75` candidate reached 27 successes but caused two hard velocity
violations.

The results show a limited performance improvement near `mvy=0.80--0.85`, but
not a complete correction of the lateral steady-state error. Candidate
subdirectories contain per-rollout CSV files and summaries. The aggregate files
record the comparison and safety-first ranking.
