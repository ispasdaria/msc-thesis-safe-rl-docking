# LQR 3DoF controllers

This directory contains the LQR controllers and the scripts used to tune, evaluate, and analyze them.

The controller implementations remain at the directory root:

- `lqr_3dof_closing.py` implements the selected closing-phase controller.
- `lqr_3dof_final.py` implements the final-approach controller using docking-port error coordinates.
- `lqr_3dof_closing_error_sat.py` implements the experimental closing controller with state-error saturation.

Supporting code is grouped by purpose:

- `analysis` contains numerical analyses of saved evaluation results.
- `evaluation` contains held-out and full-mission evaluation runners.
- `plotters` contains LQR-specific result visualization scripts.
- `tuning` contains controller weight sweeps and tuning diagnostics.
- `tests` contains controller and plotting tests.
- `results` contains retained experiment outputs and their documentation.

All supporting scripts resolve inputs and outputs relative to this directory, so their behavior does not depend on the current working directory.
