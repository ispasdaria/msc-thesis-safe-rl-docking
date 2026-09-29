# Final-approach CBF gain tuning

This directory contains the resumable Optuna studies used to select the
final-approach CBF response gains.

- `optuna_gain_search_validation_controller_seeds00_01_10_16_eval_seeds0100_0119`
  searches the cone and speed response gains on controller seeds 0, 1, 10,
  and 16 with evaluation seeds 100--119.
- `alignment_cbf_competent_controllers_grid` keeps the selected cone and speed
  gains fixed and searches the attitude and angular-rate response gains on
  controller seeds 1, 10, 14, 15, and 16 with evaluation seeds 100--119.

Each study preserves its search plan, complete trial table, resumable SQLite
database, and selected configuration. The physical cone, speed, attitude, and
angular-rate limits were not tuning variables.

Raw controller evaluations and representative figures are stored beside this
directory under `../evaluations/` and `../plots/`, respectively.
