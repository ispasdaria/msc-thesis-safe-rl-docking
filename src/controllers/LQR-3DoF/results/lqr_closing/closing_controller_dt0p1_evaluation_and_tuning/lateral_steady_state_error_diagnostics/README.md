# Lateral Steady-State Error Diagnostics

This analysis investigates the lateral-position failures observed after the
closing controller was evaluated with `dt=0.1 s`. It includes representative
trajectory histories, late-trajectory plots, a comparison with `dt=1.0 s`, and
an aggregate diagnostic summary.

The `0.1 s` closed-loop response is stable, but the closing reference combines
a fixed lateral-position target with a nonzero target lateral velocity. The LQR
feedback balances the position and velocity errors and can converge to a
nonzero lateral-position offset. The predicted lateral equilibrium classified
all 50 baseline outcomes, and the stored terminal positions match the predicted
equilibria to numerical precision.

The historical `dt=1.0 s` behavior sometimes reached the target through strong
actuator switching and saturation. Its higher apparent success rate does not
represent smoother or more accurate closed-loop convergence.

- `lateral_diagnostic_summary.json` contains the aggregate findings.
- `representative_summary.csv` lists the selected diagnostic cases.
- `histories` contains the corresponding trajectory data.
- `plots` contains the generated diagnostic figures.
