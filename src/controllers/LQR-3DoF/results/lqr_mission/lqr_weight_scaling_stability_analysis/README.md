# LQR Weight-Scaling Stability Analysis

This analysis checks closed-loop stability after independently scaling the LQR
Q and R matrices by `0.8`, `1.0`, and `1.2` for both mission phases.

Configuration:

- Closing control interval: `0.1 s`
- Final-approach control interval: `1 s`
- Controller synthesis: continuous-time CARE
- Sampled-system model: exact zero-order hold

The 18 evaluated cases cover nine Q and R combinations for closing and nine for
final approach. Every case passed both criteria:

- All continuous closed-loop eigenvalues have negative real parts.
- Every sampled closed-loop spectral radius is below one.

The CSV contains the compact result table. The JSON additionally stores all
continuous and sampled closed-loop eigenvalues. This is a linear stability
sensitivity analysis, not an empirical rollout robustness evaluation.
