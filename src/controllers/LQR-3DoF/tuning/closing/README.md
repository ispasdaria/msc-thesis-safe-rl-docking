# Closing-phase LQR tuning

This directory contains tuning and diagnostic scripts for the closing-phase LQR controller.

- `sweep_closing_lqr_weights.py` evaluates combinations of global Q and R scales and supports per-state and per-input diagonal multipliers. It also provides rollout functions reused by the held-out evaluation and closing result plotters.
- `sweep_closing_lateral_weights.py` evaluates targeted multipliers for lateral position and velocity weights while keeping the selected controller configuration unchanged.
- `diagnose_closing_lateral_convergence.py` compares representative lateral-convergence histories and includes the historical time-step comparison.
- `plot_closing_episode_duration_comparison.py` compares matched rollouts under several closing-phase duration limits.
- `plot_closing_rollout_histories.py` plots saved closing rollout histories without rerunning the controller.
- `experimental_error_saturation` contains the sweep for the alternate error-saturation controller.

The retained tuning and evaluation outputs are located under `results/lqr_closing/closing_controller_dt0p1_evaluation_and_tuning`.
