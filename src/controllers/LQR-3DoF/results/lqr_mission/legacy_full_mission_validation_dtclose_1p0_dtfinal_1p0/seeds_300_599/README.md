# Legacy Full-Mission Validation for Seeds 300 to 599

This batch contains 300 full LQR missions using `dt = 1 s` for both closing and
final approach.

Results:

- 239 successful missions
- 61 closing timeouts
- No hard safety violations
- 100 percent final-approach success among the 239 missions that reached it

`validation_300_599.csv` contains one record per mission. The log preserves the
per-rollout console output and aggregate summary. The convergence CSV evaluates
sample sizes from 10 to 300 using 1,000 deterministic permutations. Its JSON
file records the analysis configuration and data paths.
