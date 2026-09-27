# Legacy Full-Mission Validation for Seeds 600 to 899

This batch contains 300 full LQR missions using `dt = 1 s` for both closing and
final approach.

Results:

- 251 successful missions
- 48 closing timeouts
- One closing safety violation
- 100 percent final-approach success among the 251 missions that reached it

`validation_600_899.csv` contains one record per mission. The corresponding log
preserves the per-rollout console output and aggregate summary. No separate
convergence analysis was retained for this batch because the combined 600-seed
evaluation provides the larger convergence study.
