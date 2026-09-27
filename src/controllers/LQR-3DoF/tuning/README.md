# LQR tuning

This directory contains the scripts used to tune and diagnose the LQR controllers. The scripts are grouped by mission phase because the closing and final-approach controllers use different state objectives and terminal conditions.

- `closing` contains closing-phase weight sweeps, convergence diagnostics, and rollout plotting.
- `final_approach` contains final-approach weight sweeps and rollout plotting.

Generated histories, tables, and figures are stored under `results/lqr_closing` or `results/lqr_final`. They are not stored beside the tuning scripts.

The scripts retain their original numerical settings and command-line interfaces. Execute a script directly to view its available options.
