# SAC Closing Script Naming Design

## Objective

Normalize script names and documentation across the remaining
`src/controllers/SAC-3DoF/closing` experiment families. Filenames should describe
the implemented behavior without attaching a reward version to reusable code.
Reward and experiment versions remain in filenames only when the implementation
or fixed input is specific to that version.

All changes remain uncommitted unless a later user request explicitly authorizes
a commit.

## Scope

The cleanup covers these closing families:

- `lqr_demonstration_learning`
- `phase_balanced_replay`
- `residual_lqr`
- `timestep_transfer`
- `two_stage_capture`
- `vanilla_sac`

The already organized `curriculum_and_p3_reach` family is used as the naming
reference and is not reorganized again.

The work includes:

- training, evaluation, tuning, analysis, data-generation, and plotting script
  names;
- test filenames when they correspond to renamed production modules;
- imports, executable references, and documentation links;
- comments and docstrings that are conversational, ambiguous, or tied to an
  obsolete filename;
- family READMEs that explain each script's purpose and whether it is reusable
  or experiment-specific.

Logs, models, results, saved configurations, datasets, and recorded artifact
names are outside the scope and remain unchanged.

## Shared modules

Shared modules remain directly under `closing`:

- `behavior_cloned_warmup_sac.py` is imported by both
  `lqr_demonstration_learning` and `residual_lqr`.
- `near_p3_reset_wrapper.py` is imported by `curriculum_and_p3_reach`,
  `phase_balanced_replay`, and `timestep_transfer`.
- `train_sac_closing_multiseed.py` remains at the closing root when its consumers
  span experiment families.

These locations indicate shared ownership and avoid assigning a common utility
to one experiment family.

## Naming rules

- Use snake case and content-based technical descriptions.
- Keep reusable trainers, evaluators, plotters, and support modules
  version-neutral.
- Include a reward or environment version only when the script is hard-coded to
  that implementation or to a fixed version-specific artifact.
- Retain version identifiers when separate hard-coded launchers would otherwise
  become indistinguishable.
- Replace opaque variant labels such as `pilot_e`, `v2`, or `v21` with the
  implemented distinction when the code provides enough evidence.
- Preserve established domain abbreviations such as SAC, LQR, HER, CBF, BC, P3,
  and `dt01` when they materially distinguish the implementation.
- Use `train_`, `evaluate_`, `tune_`, `plot_`, `analyze_`, `generate_`, `build_`,
  `collect_`, or `validate_` prefixes according to the script's primary action.
- Do not rename historical artifact paths merely to match a renamed script.

## Family treatment

### LQR demonstration learning

Names distinguish behavior-cloning warm starts, persistent behavior-cloning
constraints, replay prefill, mixed replay, dataset generation, and dataset
curation. Version labels are retained only for fixed policies or environments.

### Phase-balanced replay

Generic replay-buffer modules remain version-neutral when their interfaces are
not tied to a reward version. Training launchers retain a version only when they
select a fixed reward, checkpoint, environment, or CBF configuration. Analysis
and plotting names identify the retained diagnostic rather than a publication
purpose.

### Residual LQR

Names distinguish HER, reference-residual, lateral-residual, terminal-fade,
stratified replay, source warm-start, and constrained-residual experiments.
Opaque pilot labels are replaced with the implemented configuration when that
mapping can be established from code and stored configuration files.

### Timestep transfer

Names state the source-to-target timestep relationship and whether actor-only
transfer, replay warm-up, or horizon comparison is performed. A reward version
is omitted from reusable evaluation and plotting scripts.

### Two-stage capture

Names distinguish long-range transfer, capture-stage training, handover dataset
collection, systematic-action baselines, reset distributions, and evaluation
sweeps. Shared support modules receive descriptive names only when the existing
name does not communicate their role.

### Vanilla SAC

Names distinguish baseline training, timestep variants, action-hold variants,
port-observation variants, multiseed evaluation, reward-specific baselines, and
Optuna tuning. Generic entry points remain version-neutral; fixed reward or
checkpoint evaluations retain the relevant version.

## Code review boundaries

The review may correct:

- stale imports and references caused by renames;
- incorrect path construction that prevents renamed scripts from resolving
  repository files;
- import-time execution where a conventional `main()` guard can be added without
  changing runtime behavior;
- misleading or conversational comments and docstrings;
- obsolete commented-out code that has no reproducibility value.

The review must not change:

- reward calculations;
- environment dynamics or termination conditions;
- controller behavior;
- training hyperparameters;
- checkpoint-selection rules;
- evaluation metrics;
- numerical results or recorded artifacts.

Any suspected behavioral defect is reported separately rather than silently
changed during this naming cleanup.

## Documentation

Each family README will provide:

- the purpose of the family;
- a table of active scripts and their roles;
- an indication of reusable versus version-specific entry points;
- dependencies on shared closing modules;
- the relationship between scripts and the existing artifact directories;
- reproduction notes supported by the current command-line interfaces.

Comments and docstrings use objective technical wording. They describe why a
non-obvious operation exists, relevant units or constraints, and the meaning of
stored outputs. They do not address the reader conversationally or speculate
about intent.

## Verification

- Record the complete pre-change set and hashes of non-cache files within
  `closing`.
- Assert that logs, models, results, datasets, and recorded artifacts retain the
  same paths and contents.
- Search Python and Markdown files for every old script name after renaming.
- Compile all Python files under `closing`.
- Import executable modules without invoking their `main()` functions where the
  existing design permits safe import.
- Run the closing-family tests and structural layout tests.
- Run the complete repository test suite.
- Run `git diff --check` and confirm that no files are staged or committed.
- Remove only generated `__pycache__` and `.pytest_cache` directories.

## Non-goals

- No artifact directory or recorded run is renamed or moved.
- No logs, models, results, figures, datasets, or configuration files are
  deleted or regenerated.
- No training or evaluation is rerun.
- No new shared-code subfolder is introduced.
- No Git commit, merge, push, staging operation, or branch change is performed.
