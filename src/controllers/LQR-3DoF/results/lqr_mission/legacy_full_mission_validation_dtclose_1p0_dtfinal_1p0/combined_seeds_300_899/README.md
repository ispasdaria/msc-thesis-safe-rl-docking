# Combined Legacy Validation for Seeds 300 to 899

`validation_300_899.csv` is the exact row-wise concatenation of the seed 300 to
599 and seed 600 to 899 result tables. It contains 600 full missions using a
`1 s` control interval in both phases.

Results:

- 490 successful missions, or 81.67 percent
- 109 closing timeouts
- One closing safety violation
- 100 percent final-approach success among missions that reached it

The convergence analysis evaluates sample sizes from 25 to 600. For each sample
size, 1,000 deterministic permutations estimate the sensitivity of success
rate, safety rate, closing terminal-error mean, and variance to seed ordering.
The JSON file records the analysis configuration and current data paths.
