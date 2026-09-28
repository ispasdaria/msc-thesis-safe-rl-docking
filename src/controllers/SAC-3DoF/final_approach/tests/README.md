# Final-Approach Tests

- `test_artifact_layout.py` verifies script classification, artifact categories, retained model pairs, and documentation.
- `test_evaluate_selected_sac_robustness.py` verifies enlarged-range sampling, summaries, output naming, and command-line arguments.
- `test_external_consumers.py` verifies paths used by full-mission, comparison, and CBF workflows.
- `test_sac_verification.py` verifies the environment contract, normalization, deterministic evaluation, checkpoint restoration, and explicit horizons.
- `test_script_imports.py` verifies main guards and the absence of module-level directory creation.

The tests use short environments and small neural networks. They do not rerun full training, Optuna studies, or large evaluation campaigns.
