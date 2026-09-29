# CBF gain tuning

- `tune_closing_cbf.py` searches closing cone, speed, and alignment response gains.
- `tune_final_approach_cbf.py` searches final-approach cone, speed, attitude, and angular-rate response gains.

The physical safety limits are fixed. Only response gains are tuned. Each study stores its search plan, trial table, resumable Optuna database, and selected configuration under the corresponding `../results/<phase>/tuning/` directory.

Run either entry point from the repository root; `--help` lists its controller,
seed, solver, and search-space arguments. Representative retained studies are
`../results/final_approach/tuning/alignment_cbf_competent_controllers_grid/`
and
`../results/final_approach/tuning/optuna_gain_search_validation_controller_seeds00_01_10_16_eval_seeds0100_0119/`.
