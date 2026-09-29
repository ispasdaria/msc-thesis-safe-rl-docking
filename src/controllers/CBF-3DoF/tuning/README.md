# CBF gain tuning

- `tune_closing_cbf.py` searches closing cone, speed, and alignment response gains.
- `tune_final_approach_cbf.py` searches final-approach cone, speed, attitude, and angular-rate response gains.

The physical safety limits are fixed. Only response gains are tuned. Each study stores its search plan, trial table, resumable Optuna database, and selected configuration under the corresponding `../results/<phase>/tuning/` directory.
