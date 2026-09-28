# Safe Reinforcement Learning for Autonomous Close-Proximity Spacecraft Docking

This repository contains the planar spacecraft rendezvous-and-docking
simulation, controllers, experiments, and retained artifacts developed for an
MSc thesis on safe reinforcement learning.

The implemented mission is divided into two phases:

1. **Closing:** transfer from the P2 initial region to a sampled P3 handover
   state.
2. **Final approach:** transfer from P3 to an admissible docking state.

The repository evaluates classical Linear Quadratic Regulator (LQR)
controllers, Soft Actor-Critic (SAC) policies, and Control Barrier Function
quadratic-program (CBF-QP) safety filters. Full-mission evaluations combine a
closing controller with a final-approach controller at the P3 handover.

## Implemented system

### Planar 3-DoF environment

The target-relative state is

```text
[rx, ry, vx, vy, psi, omega_z]
```

and the control input is a body-frame acceleration command

```text
[a_x_Bc, a_y_Bc, alpha_z].
```

The implementation includes:

- closing and final-approach mission phases;
- center-of-mass and docking-port geometry;
- docking-cone, keep-out, corridor, velocity, attitude, and angular-rate
  conditions;
- fourth-order Runge-Kutta and forward-Euler integration;
- Gymnasium-compatible observations and normalized actions;
- nominal, testing, and training initialization distributions;
- seeded resets and configurable timesteps;
- explicit simulation horizons; and
- optional action holding and phase-specific wrappers used by the SAC
  experiments.

The complete model definition, frames, equations, limits, initialization
ranges, rewards, and termination conditions are documented in
[docs/scenario_setup.md](docs/scenario_setup.md).

### Controllers and evaluations

The repository contains:

- phase-specific LQR controllers, gain studies, held-out evaluation, and
  full-mission rollouts;
- closing and final-approach SAC training, Optuna tuning, multiseed evaluation,
  and retained checkpoints;
- SAC curriculum, observation-space, action-hold, demonstration-replay,
  behavior-cloning, HER, and LQR-residual experiments;
- deterministic policy-selection and expanded-initial-condition robustness
  evaluations;
- mixed LQR-closing/SAC-final full-mission evaluation and plotting; and
- closing and final-approach CBF-QP filters with SciPy and OSQP solver paths,
  gain tuning, ablations, feasibility diagnostics, and paired filtered versus
  unfiltered evaluations.

The CBF formulation, protected constraints, evaluator outputs, limitations,
and example commands are documented in
[src/controllers/cbf_3DoF/README.md](src/controllers/cbf_3DoF/README.md).

### Benchmark problems

`benchmark_problems/` contains SAC training and evaluation examples for
Mountain Car Continuous and Pendulum. These scripts provide smaller reference
problems for checking the reinforcement-learning workflow independently of the
docking environment.

## Scope

The implemented simulator is a planar 3-DoF model. It does not include:

- R-bar translation, roll, or pitch dynamics;
- full 6-DoF rigid-body motion;
- absolute orbital propagation;
- vehicle mass or inertia variation;
- thruster allocation, pulse modulation, or propellant consumption;
- environmental disturbances; or
- physical contact and capture dynamics.

The CBF conditions are continuous-time constructions applied through a
sampled controller and finite-step numerical integrator. The analytic barriers
and rollout tests support the implementation, but they do not constitute a
formal discrete-time safety proof.

## Repository structure

```text
benchmark_problems/                 Mountain Car and Pendulum SAC examples
docs/                               Scenario and implementation documentation
src/envs/docking3dof/               Planar docking core and Gymnasium wrappers
src/controllers/LQR-3DoF/           LQR controllers, studies, and mission runs
src/controllers/SAC-3DoF/           SAC experiments and mixed-mission analysis
src/controllers/SAC-3DoF/closing/   Closing-phase SAC work
src/controllers/SAC-3DoF/final_approach/
                                    Final-approach SAC work
src/controllers/cbf_3DoF/           CBF-QP filters and evaluations
```

Tests, models, numerical results, and figures are generally colocated with the
subsystem that produces or consumes them. This keeps each experiment together
with its configuration and supporting artifacts.

## Installation

Python 3.12 is used by the project.

Create and activate a virtual environment on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On POSIX shells:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Git LFS datasets

Large LQR demonstration datasets are stored with Git Large File Storage. After
installing Git LFS, retrieve them with:

```bash
git lfs install
git lfs pull
```

The LFS-managed files are located in
`src/controllers/SAC-3DoF/closing/demonstrations/`:

- `lqr_closing_training_rewardv15_expanded_200successes.npz`;
- `lqr_closing_training_rewardv15_dt_dt0p1_200successes.partial.npz`;
- `lqr_closing_training_rewardv15_dt_dt0p1_replay_20episodes.npz`; and
- `lqr_closing_training_rewardv15_dt_dt0p1_bc_stride10_78episodes.npz`.

The `.partial.npz` file is a validated source archive containing 78 successful
episodes and 8,641,259 transitions. Collection stopped before the original
target of 200 successful episodes. The archive is retained because it produces
the smaller behavior-cloning and replay datasets and can be continued with the
generator's `--resume` option.

## Verification

Run the complete test suite from the repository root with `src` on
`PYTHONPATH`.

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

POSIX shells:

```bash
PYTHONPATH=src python -m pytest -q
```

Subsystem tests can be run by supplying the corresponding test directory. For
example:

```powershell
python -m pytest src/controllers/cbf_3DoF/tests -q
```

## Reproduction entry points

The repository contains many exploratory and comparison scripts. The following
locations provide the main entry points for understanding and reproducing the
implemented workflows:

- Environment definition:
  [`src/envs/docking3dof/core.py`](src/envs/docking3dof/core.py) and
  [`src/envs/docking3dof/gym_env.py`](src/envs/docking3dof/gym_env.py)
- LQR full mission:
  [`src/controllers/LQR-3DoF/evaluation/run_lqr_full_mission.py`](src/controllers/LQR-3DoF/evaluation/run_lqr_full_mission.py)
- SAC closing:
  [`src/controllers/SAC-3DoF/closing/`](src/controllers/SAC-3DoF/closing/)
- SAC final approach:
  [`src/controllers/SAC-3DoF/final_approach/`](src/controllers/SAC-3DoF/final_approach/)
- Mixed LQR/SAC full-mission evaluation:
  [`src/controllers/run_mixed_full_mission_evaluations.py`](src/controllers/run_mixed_full_mission_evaluations.py)
- Retained full-mission figures:
  [`src/controllers/SAC-3DoF/figures/representative_full_mission_figures/`](src/controllers/SAC-3DoF/figures/representative_full_mission_figures/)
- CBF-QP evaluation and tuning:
  [`src/controllers/cbf_3DoF/`](src/controllers/cbf_3DoF/)
- Benchmark problems:
  [`benchmark_problems/`](benchmark_problems/)

Individual training and evaluation scripts expose their configurable options
through `--help`. Retained experiment directories contain machine-readable
configuration, metrics, model-selection decisions, and model cards where
available.

## Engineering conventions

The repository's source-code, artifact-naming, reproducibility, and testing
conventions are defined in [CONTRIBUTING.md](CONTRIBUTING.md).
