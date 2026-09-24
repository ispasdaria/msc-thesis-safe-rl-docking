# Engineering Conventions

This document defines the engineering conventions used by the project. Their
purpose is to maintain consistent source code, reproducible experiments, and
traceable results. Validated controller behavior remains unchanged unless a
change explicitly targets that behavior.

## Environment

- The supported Python version is Python 3.12.
- Project dependencies are defined in `requirements.txt`.
- Large demonstration datasets are managed through Git LFS as described in
  `README.md`.
- Virtual environments and local tool configuration are excluded from version
  control.

## Repository organization

- `benchmark_problems/` contains the Mountain Car and Pendulum SAC examples.
- `docs/` contains implementation and scenario documentation.
- `src/envs/docking3dof/` contains the planar 3-DoF docking environment.
- `src/controllers/LQR-3DoF/` contains the LQR controllers and evaluations.
- `src/controllers/SAC-3DoF/` contains the SAC controllers, training scripts,
  evaluations, retained models, results, and figures.
- `src/controllers/cbf_3DoF/` contains the CBF safety filter and associated
  evaluations.

Tests and experiment artifacts are generally stored with the subsystem that
produces or consumes them. Retained models, numerical results, and figures are
versioned when they support reported results or reproduction.

## Source-code conventions

- Python source uses four spaces for indentation.
- Variables, functions, modules, and new file names use `snake_case`.
- Classes use `PascalCase`.
- Module-level constants use `UPPER_CASE`.
- Names are descriptive rather than abbreviated. Units are included when
  the quantity would otherwise be ambiguous, such as `duration_s` or
  `terminal_speed_mps`.
- Imports are grouped into standard-library, third-party, and local-project
  groups.
- Modules and functions or classes whose purpose is not evident from their
  name and signature include concise docstrings.
- Comments describe assumptions, formulas, units, non-obvious decisions, or
  experiment provenance in objective language. Comments do not address the
  reader.
- Functions remain focused on a single responsibility where this can be
  achieved without risking validated behavior.

Established directory names such as `LQR-3DoF`, `SAC-3DoF`, and `cbf_3DoF`
are retained because scripts and saved experiment records refer to them. New
paths follow the surrounding subsystem's established structure.

## Reproducibility conventions

- Training and evaluation use explicit seeds when stochastic behavior is
  involved.
- Simulation horizons are provided explicitly rather than through implicit
  core defaults.
- Repository paths are derived from `Path(__file__).resolve()` so execution
  does not depend on the current working directory.
- Model checkpoints remain paired with the normalization statistics required
  to load them.
- Run and artifact names record the controller, mission phase, distinguishing
  configuration, training seed, and training budget when those details are
  relevant.
- New result and figure file names use lowercase words separated by
  underscores.
- Vague names such as `final_run`, `test_new`, or `output1` are avoided.

Representative names include:

- `round1_seed12_100k_steps`
- `round5_seed11_600k_steps_buffer600k`
- `closing_vanilla_sac_state_convergence_seed15.png`
- `policy_comparison_summary.json`

## Testing

The complete test suite runs from the repository root with `src` on
`PYTHONPATH`:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

An equivalent environment-variable syntax applies in non-PowerShell shells.
Behavior changes include focused tests, which run before the complete suite.

## Change management

- Changes remain limited to one task or one logically related set of tasks.
- Commit messages are clear, concise, and action-oriented.
- Caches, local environments, editor settings, and temporary files covered by
  `.gitignore` are not versioned.
- Replacement experimental artifacts include a record of how they were
  generated.
