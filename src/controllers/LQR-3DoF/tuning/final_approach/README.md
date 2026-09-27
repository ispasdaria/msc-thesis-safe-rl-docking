# Final-approach LQR tuning

This directory contains tuning and plotting scripts for the final-approach LQR controller.

- `sweep_final_approach_lqr_weights.py` evaluates Q and R scale combinations, records rollout metrics, and can save selected rollout histories.
- `plot_final_approach_rollout_histories.py` creates state-error, command, and phase-plane figures from saved histories without rerunning the controller.

Generated outputs are stored under `results/lqr_final`. The retained terminal-state convergence result is stored in `results/lqr_final/terminal_state_convergence`.
