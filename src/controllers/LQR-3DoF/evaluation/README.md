# LQR evaluation

- `evaluate_lqr_closing_heldout.py` evaluates the selected closing controller on held-out seeds without retuning the controller. It records success, timeout, safety, terminal-error, and stability statistics.
- `run_lqr_full_mission.py` executes the closing and final-approach controllers as a two-phase mission. It supports individual missions, multi-seed campaigns, summary generation, and selected history retention.

Default outputs are written under `../results`. Existing held-out outputs are protected against accidental overwriting unless the script is explicitly configured otherwise.
