# CBF-3DoF Full Audit and Documentation Design

## Purpose

Audit and clean the complete `src/controllers/CBF-3DoF` tree so that its
implementation, tests, scripts, documentation, and retained research artifacts
are understandable and internally consistent. Replace conversational,
promotional, or assistant-like prose with concise technical language while
preserving numerical behavior and research provenance.

The audited tree currently contains 4,546 files (approximately 6.33 GB): 33
Python files, 13 Markdown files, 243 CSV files, 125 JSON files, 4,081 NPZ
archives, 48 PNG files, two SQLite databases, and one `.gitignore` file.

## Scope

Every file below `src/controllers/CBF-3DoF` is in scope. The treatment depends
on the file's role:

- Maintained Python, tests, and Markdown may be edited when review identifies a
  concrete clarity, correctness, maintainability, or documentation issue.
- JSON and CSV artifacts are parsed and checked for structural consistency.
  Descriptive fields may be normalized only when the edit does not change the
  meaning or provenance of a recorded experiment.
- NPZ archives are opened and inventoried. Array names, shapes, dtypes, and
  non-finite values are checked without rewriting the archives.
- SQLite databases receive read-only integrity and schema checks.
- PNG files receive decoding, format, and dimension checks without image
  recompression.
- `.gitignore` rules are reviewed for consistency with the documented result
  retention policy.

Generated scientific outputs are evidence, not prose documents. Historical
measurements, seeds, timestamps, model paths, solver statuses, and recorded
configuration values must not be silently changed. Stale absolute paths in
historical metadata are documented as provenance unless an application reads
them as active paths and a compatibility-preserving normalization is tested.

Files outside `src/controllers/CBF-3DoF` may be changed only when required to
test or document this subtree. Existing unrelated working-tree changes remain
untouched.

## Documentation Structure

The root README will explain the package purpose, mathematical scope, directory
map, installation, tests, evaluation workflow, tuning workflow, plotting
workflow, result schema, limitations, and links to the detailed subdirectory
documentation.

Each meaningful maintained or archival collection will have a README at the
nearest useful boundary. A README must identify:

1. the directory's purpose;
2. its maintained entry points or artifact classes;
3. inputs, outputs, and important dependencies;
4. naming and schema conventions;
5. representative files or result directories and why they are representative;
6. commands for ordinary use when the directory contains executable scripts;
7. provenance or interpretation limitations where applicable.

READMEs will not be added mechanically to individual experiment-run folders.
Their parent README and generated inventory will describe repeated run layouts.
This avoids thousands of redundant documents while still covering every file.

An artifact inventory will provide a machine-readable record for every file and
a concise human-readable validation summary. Inventory entries will include at
least relative path, artifact class, byte size, and validation status. Binary
and structured formats will include appropriate metadata such as array counts,
table dimensions, image dimensions, or database table names. The inventory is
descriptive and must not duplicate multi-gigabyte content.

## Objective Writing Standard

Documentation, module docstrings, comments, command help, and human-readable
metadata will use neutral technical language. The audit will remove or rewrite:

- conversational asides and second-person coaching where direct instructions
  are not needed;
- claims such as "robust", "excellent", "comprehensive", or "safe" unless a
  cited result or precisely stated condition supports them;
- references to an assistant, prompt, conversation, or drafting process;
- speculative explanations presented as established facts;
- repeated summaries that do not add operational or scientific information;
- ambiguous temporal terms such as "current", "new", and "previous" when a
  schema version, date, or named configuration is available.

Safety claims must distinguish continuous-time CBF construction, sampled-data
simulation evidence, soft constraints, hard constraints, and nominal fallback
behavior. The text must not imply a formal guarantee beyond the evidence in the
repository.

## Code Review and Cleanup

The Python review covers the importable `cbf_3dof` package, evaluation entry
points, tuning scripts, plotters, and tests. Review criteria include:

- correctness of units, state/action ordering, frame conventions, barrier sign
  conventions, solver results, and recorded diagnostics;
- explicit public interfaces and consistent terminology across modules;
- removal of unused imports, unreachable branches, stale comments, duplicated
  local utilities, and misleading names;
- decomposition of oversized routines only where it reduces risk and can be
  protected by tests;
- portable path construction and clear command-line validation;
- deterministic output schemas and actionable errors;
- docstrings for public APIs and comments for non-obvious mathematical choices;
- tests that describe behavior rather than implementation details.

Behavioral changes and bug fixes require a failing regression test before the
implementation change. Pure prose and generated-inventory changes do not
require unit tests, but they require link, syntax, or schema validation as
appropriate.

The large evaluation and plotting scripts will not be split merely to satisfy
a line-count target. Refactoring requires a concrete boundary, unchanged CLI
behavior, and focused tests for extracted logic.

## Artifact Review

All 4,546 files must appear in the audit inventory. Validation is content-aware:

- Python: syntax compilation, import checks, tests, and static review.
- Markdown: local-link resolution, heading structure, path accuracy, and prose
  review.
- JSON: complete parsing, top-level type identification, and schema-key
  comparison within each artifact family.
- CSV: parseability, column uniqueness, row-width consistency, row counts, and
  schema comparison within each artifact family.
- NPZ: archive readability; member name, shape, and dtype inventory; object
  array identification; and numeric non-finite counts. Large arrays are checked
  in bounded chunks when needed.
- PNG: successful decoding plus width, height, mode, and format inventory.
- SQLite: read-only `PRAGMA integrity_check`, table inventory, and schema
  readability.
- Other files: explicit classification and a documented validation method.

Differences between historical schema versions are not errors by themselves.
The validator groups comparable artifacts by role and records intentional
schema variants. Corrupt, partial, duplicated, or unexplained artifacts are
reported individually. Deletion or replacement requires evidence and explicit
justification; ambiguous historical data is retained and documented.

## Representative Files

"Representative files" means curated examples referenced by each relevant
README, not synthetic replacements for retained results. Selection criteria are:

- the file is complete and parseable;
- its configuration and seed provenance are present;
- it represents a documented controller phase, filter variant, or output
  schema;
- it avoids known incomplete or diagnostic-only runs unless the purpose is to
  illustrate that category;
- another researcher can trace the example from configuration through summary
  metrics to step-level data and figures.

Where no single artifact covers the directory's roles, the README may name a
small set of complementary examples. Selection reasons are stated explicitly.

## Validation and Acceptance Criteria

The work is complete when:

1. every file in the subtree has a successful validation record or a specific
   documented issue;
2. every meaningful top-level or phase-level directory is explained by a
   README, and repeated run directories are covered by their parent schema;
3. each relevant README identifies representative files or directories;
4. all maintained prose has been reviewed against the objective writing
   standard;
5. all Python files compile and the focused CBF test suite passes;
6. the complete repository test suite is run and every failure is reported;
7. local Markdown links resolve;
8. JSON, CSV, NPZ, PNG, and SQLite validators complete without unreported
   errors;
9. `git diff --check` reports no whitespace errors;
10. a final inventory accounts for additions, edits, moves, and any approved
    deletions without silently losing retained research artifacts.

Warnings that arise from valid historical schema differences or known external
library behavior are documented rather than suppressed without analysis.

## Deliverables

- cleaned Python, tests, scripts, and maintained documentation;
- revised root and subdirectory README files;
- curated representative-file references;
- a reproducible artifact-audit tool or script;
- a machine-readable full inventory and concise validation report;
- regression tests for any behavioral corrections;
- a final summary of modified files, preserved historical limitations, test
  results, and unresolved issues.

## Non-goals

- rerunning training, tuning, or multi-seed evaluation campaigns;
- changing CBF safety specifications or controller gains without separate
  scientific justification;
- rewriting binary archives solely for formatting consistency;
- claiming formal sampled-data safety from continuous-time conditions and
  empirical rollouts;
- deleting historical artifacts merely because they are old, duplicated in
  presentation, or use an earlier schema.
