# CBF-3DoF Full Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit, clean, and document every file below `src/controllers/CBF-3DoF` while preserving controller behavior and historical research evidence.

**Architecture:** A tested, read-only artifact auditor will classify and validate the complete tree and emit a reproducible inventory plus summary. Maintained Python and prose will then be reviewed by functional area, with regression tests preceding behavioral changes; generated scientific artifacts will be validated and documented without being rewritten unless an edit is demonstrably provenance-safe.

**Tech Stack:** Python 3, pytest, NumPy, pandas/standard-library CSV, Pillow or Matplotlib image decoding, sqlite3, Markdown, Git

**Spec:** `docs/superpowers/specs/2026-09-30-cbf-3dof-full-audit-design.md`

## Global Constraints

- Every file below `src/controllers/CBF-3DoF` is in scope and must appear in the final inventory.
- Historical measurements, seeds, timestamps, model paths, solver statuses, and recorded configuration values must not be silently changed.
- NPZ, PNG, and SQLite artifacts are validated read-only and are not rewritten for formatting.
- JSON and CSV descriptive fields may be normalized only when meaning and provenance are unchanged.
- Behavioral changes require a failing regression test before implementation.
- Safety prose must distinguish continuous-time CBF construction, sampled-data evidence, soft constraints, hard constraints, and nominal fallback behavior.
- Existing unrelated working-tree changes, including the deleted 2026-09-29 planning documents, remain untouched.
- No training, tuning, or evaluation campaign is rerun as part of this audit.
- The baseline contains 4,546 files and 6,328,091,084 bytes; count changes must be explained by named audit deliverables or approved cleanup changes.

## Review Focus

- A truncated or corrupt NPZ/PNG/SQLite artifact must produce a path-specific validation error without aborting the remainder of the audit; Task 1 tests this.
- Numeric arrays containing `NaN` or infinity must report bounded per-member counts without loading the entire archive into a combined in-memory structure; Task 1 tests this.
- Legitimate historical JSON/CSV schema variants must be grouped and reported rather than flattened or labeled corrupt solely for differing keys; Tasks 1 and 2 test this.
- Symlinks or paths escaping the audit root must not be followed into unrelated data; Task 1 tests this.
- A file added, removed, or changed after the baseline must appear explicitly in the final inventory comparison; Tasks 2 and 8 test this.

---

### Task 1: Build the read-only artifact auditor

**Files:**
- Create: `src/controllers/CBF-3DoF/tools/audit_artifacts.py`
- Create: `src/controllers/CBF-3DoF/tools/__init__.py`
- Create: `src/controllers/CBF-3DoF/tests/test_artifact_audit.py`

**Interfaces:**
- Produces: `ArtifactRecord`, `ValidationIssue`, `classify_artifact(path: Path) -> str`, `validate_artifact(path: Path, root: Path) -> ArtifactRecord`, `audit_tree(root: Path) -> tuple[list[ArtifactRecord], list[ValidationIssue]]`, `write_inventory(records: Sequence[ArtifactRecord], destination: Path) -> None`, `write_summary(records: Sequence[ArtifactRecord], issues: Sequence[ValidationIssue], destination: Path) -> None`, and CLI `python tools/audit_artifacts.py ROOT --inventory PATH --summary PATH`.
- Guarantees: deterministic relative-path ordering, read-only validation, continued execution after individual-file failures, and nonzero CLI exit status when error-severity issues exist.

- [ ] **Step 1: Write classification and deterministic-order tests**

Add `test_classifies_supported_artifacts` and `test_audit_tree_orders_relative_paths` using temporary `.py`, `.md`, `.json`, `.csv`, `.npz`, `.png`, `.sqlite3`, and extensionless files. Assert the documented artifact-class names and lexicographic POSIX-style relative paths.

- [ ] **Step 2: Run the classification tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest src/controllers/CBF-3DoF/tests/test_artifact_audit.py -k "classifies or orders" -v`

Expected: FAIL because `tools.audit_artifacts` does not exist.

- [ ] **Step 3: Implement record types, classification, safe traversal, and deterministic ordering**

`ArtifactRecord` must contain `path`, `artifact_class`, `size_bytes`, `status`, `metadata`, and `issues`. `ValidationIssue` must contain `path`, `severity`, `code`, and `message`. Resolve the root once, do not follow directory symlinks, and reject any resolved file outside the root.

- [ ] **Step 4: Write structured-format validation tests**

Add tests for valid and malformed JSON; CSV with duplicate headings and inconsistent row widths; NPZ member names, shapes, dtypes, and non-finite counts; a decodable PNG; SQLite integrity and table names; and a corrupt file for each binary format. Assert that one corrupt artifact does not prevent later files from appearing in the results.

- [ ] **Step 5: Run the structured-format tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest src/controllers/CBF-3DoF/tests/test_artifact_audit.py -k "json or csv or npz or png or sqlite or corrupt" -v`

Expected: FAIL because format validators are not implemented.

- [ ] **Step 6: Implement content-aware validators**

Use streaming CSV iteration, per-member NumPy inspection, Pillow if already available (otherwise Matplotlib's installed image reader), and sqlite3 read-only URI mode with `PRAGMA integrity_check`. Report object arrays without deserializing them with pickle. Count non-finite numeric values per array member without constructing a combined archive-wide array.

- [ ] **Step 7: Write inventory, summary, CLI, and mutation-safety tests**

Add `test_inventory_contains_every_regular_file`, `test_summary_groups_schema_variants`, `test_cli_returns_nonzero_for_errors`, `test_symlink_outside_root_is_not_followed`, and `test_audit_does_not_change_source_hashes`. Inventory columns are `path,artifact_class,size_bytes,status,metadata_json,issues_json`; the summary contains counts by class/status, issue records, and JSON/CSV schema-family groups.

- [ ] **Step 8: Run the new test file and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest src/controllers/CBF-3DoF/tests/test_artifact_audit.py -v`

Expected: PASS.

- [ ] **Step 9: Commit Task 1**

```powershell
git add src/controllers/CBF-3DoF/tools src/controllers/CBF-3DoF/tests/test_artifact_audit.py
git commit -m "test: add CBF artifact auditor"
```

### Task 2: Establish the complete baseline and issue ledger

**Files:**
- Create: `src/controllers/CBF-3DoF/artifact_inventory.csv`
- Create: `src/controllers/CBF-3DoF/artifact_validation_summary.json`
- Create: `src/controllers/CBF-3DoF/AUDIT_REPORT.md`
- Test: `src/controllers/CBF-3DoF/tests/test_artifact_audit.py`

**Interfaces:**
- Consumes: Task 1 auditor and the approved specification.
- Produces: one inventory row per regular file present before output files are generated, plus explicit self-records for the three audit deliverables on the final run; a grouped issue ledger used by Tasks 3-7.

- [ ] **Step 1: Add a fixed-point output test**

Add `test_outputs_account_for_themselves_on_second_run`: audit a temporary tree, write inventory and summary inside it, audit again, and assert both generated files appear exactly once with no unstable ordering. Add a Markdown report fixture only after the machine outputs stabilize.

- [ ] **Step 2: Run the fixed-point test and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest src/controllers/CBF-3DoF/tests/test_artifact_audit.py::test_outputs_account_for_themselves_on_second_run -v`

Expected: FAIL until output-file accounting is supported.

- [ ] **Step 3: Implement stable output accounting and run the complete audit**

Run the auditor twice so the committed inventory includes `artifact_inventory.csv`, `artifact_validation_summary.json`, and `AUDIT_REPORT.md`. Record the original 4,546-file, 6,328,091,084-byte baseline separately from audit-created files. `AUDIT_REPORT.md` must list validation scope, counts, schema families, every error, warning categories, and the remediation task owning each maintained-content issue.

- [ ] **Step 4: Verify inventory coverage and preservation**

Run a PowerShell comparison between filesystem relative paths and inventory paths. Expected: empty symmetric difference. Hash all pre-existing files before and after the read-only audit. Expected: no changed hashes.

- [ ] **Step 5: Run Task 1 tests and inspect every reported error**

Run: `.\.venv\Scripts\python.exe -m pytest src/controllers/CBF-3DoF/tests/test_artifact_audit.py -v`

Expected: PASS. Classify each real-tree error as corruption, incomplete historical evidence, validator defect, or maintained-file defect before proceeding.

- [ ] **Step 6: Commit Task 2**

```powershell
git add src/controllers/CBF-3DoF/artifact_inventory.csv src/controllers/CBF-3DoF/artifact_validation_summary.json src/controllers/CBF-3DoF/AUDIT_REPORT.md src/controllers/CBF-3DoF/tests/test_artifact_audit.py src/controllers/CBF-3DoF/tools/audit_artifacts.py
git commit -m "docs: record CBF artifact audit baseline"
```

### Task 3: Review the importable CBF implementation

**Files:**
- Review/Modify: `src/controllers/CBF-3DoF/cbf_3dof/*.py`
- Review/Modify: `src/controllers/CBF-3DoF/tests/test_barriers.py`
- Review/Modify: `src/controllers/CBF-3DoF/tests/test_boundary_feasibility.py`
- Review/Modify: `src/controllers/CBF-3DoF/tests/test_controller.py`
- Review/Modify: `src/controllers/CBF-3DoF/tests/test_port_dynamics.py`
- Review/Modify: `src/controllers/CBF-3DoF/tests/test_qp_filter.py`
- Review/Modify: `src/controllers/CBF-3DoF/tests/test_manual_action_scenarios.py`

**Interfaces:**
- Preserves: exports from `cbf_3dof/__init__.py`, state order `[rx, ry, vx, vy, psi, omega]`, body-frame action order `[ax_Bc, ay_Bc, alpha_Bc]`, barrier signs, solver semantics, and diagnostic field names.
- Produces: objective docstrings/comments and tested corrections for any concrete implementation defect found in the issue ledger or manual review.

- [ ] **Step 1: Run the implementation-focused baseline tests**

Run: `.\.venv\Scripts\python.exe -m pytest src/controllers/CBF-3DoF/tests/test_barriers.py src/controllers/CBF-3DoF/tests/test_boundary_feasibility.py src/controllers/CBF-3DoF/tests/test_controller.py src/controllers/CBF-3DoF/tests/test_port_dynamics.py src/controllers/CBF-3DoF/tests/test_qp_filter.py src/controllers/CBF-3DoF/tests/test_manual_action_scenarios.py -q`

Expected: record exact pass/fail counts before editing.

- [ ] **Step 2: Review every implementation and associated test file**

For each file, record disposition in `AUDIT_REPORT.md`: unchanged after review, prose-only edit, refactor, or defect correction. Check units, frames, shape validation, finite inputs, CBF inequality direction, actuator bounds, solver fallbacks, diagnostics, unused code, duplication, and public documentation.

- [ ] **Step 3: Add one failing regression test per behavioral defect**

Each test must name the externally visible failure and assert the preserved or corrected interface. Run the individual test and confirm it fails for the identified defect rather than test setup.

- [ ] **Step 4: Apply minimal implementation and objective-prose edits**

Do not change gains, limits, state ordering, action ordering, or fallback policy. Replace conversational or unsupported claims in docstrings/comments. Refactor only boundaries protected by existing or newly failing tests.

- [ ] **Step 5: Run focused tests and compile the package**

Run the Step 1 command and `.\.venv\Scripts\python.exe -m compileall -q src/controllers/CBF-3DoF/cbf_3dof`.

Expected: PASS and compile exit code 0.

- [ ] **Step 6: Commit Task 3**

Stage only reviewed implementation/test files and `AUDIT_REPORT.md`; commit with `refactor: clean CBF implementation` or `fix:` if behavior changed.

### Task 4: Review evaluation, tuning, and plotting programs

**Files:**
- Review/Modify: `src/controllers/CBF-3DoF/evaluations/*.py`
- Review/Modify: `src/controllers/CBF-3DoF/tuning/*.py`
- Review/Modify: `src/controllers/CBF-3DoF/plotters/*.py`
- Review/Modify: `src/controllers/CBF-3DoF/tests/test_closing_cbf_shield_env.py`
- Review/Modify: `src/controllers/CBF-3DoF/tests/test_closing_evaluator.py`
- Review/Modify: `src/controllers/CBF-3DoF/tests/test_evaluation_initialization.py`
- Review/Modify: `src/controllers/CBF-3DoF/tests/test_evaluator_diagnostics.py`
- Review/Modify: `src/controllers/CBF-3DoF/tests/test_trajectory_comparison_figure_layout.py`
- Review/Modify: `src/controllers/CBF-3DoF/tests/test_tuning.py`

**Interfaces:**
- Preserves: CLI option names/defaults, paired-seed behavior, result filenames, schema version fields, and result-directory conventions unless a failing test proves a defect.
- Produces: portable path handling, objective help/docstrings/comments, explicit validation, and reduced duplication where testable.

- [ ] **Step 1: Run the script-focused baseline tests and capture `--help`**

Run all six listed test modules. Capture `--help` for each of the nine executable scripts and record exit codes and any import/path failures.

- [ ] **Step 2: Review all executable files and tests**

Record each file's disposition in `AUDIT_REPORT.md`. Check path portability, imports, CLI validation, seed pairing, output schemas, deterministic ordering, exception messages, duplicated utilities, stale comments, unsupported safety claims, and active references to historical absolute paths.

- [ ] **Step 3: Add failing tests for each behavioral or portability defect**

Prefer testing extracted pure functions or CLI parsing. Do not run campaigns. Verify every new test fails for the intended reason.

- [ ] **Step 4: Implement minimal fixes and prose cleanup**

Retain large-file structure unless a concrete extracted boundary reduces duplication or makes validation testable. Preserve CLI compatibility and result schemas.

- [ ] **Step 5: Verify scripts**

Run the script-focused tests, compile all evaluation/tuning/plotter files, and rerun each `--help` command. Expected: tests pass, compilation succeeds, and every help command exits 0 without starting a campaign.

- [ ] **Step 6: Commit Task 4**

Stage only reviewed script/test files and `AUDIT_REPORT.md`; commit with `refactor: clean CBF workflows` or `fix:` if behavior changed.

### Task 5: Rewrite maintained documentation and select representative artifacts

**Files:**
- Modify: `src/controllers/CBF-3DoF/README.md`
- Modify: `src/controllers/CBF-3DoF/evaluations/README.md`
- Modify: `src/controllers/CBF-3DoF/tuning/README.md`
- Modify: `src/controllers/CBF-3DoF/plotters/README.md`
- Modify: `src/controllers/CBF-3DoF/results/README.md`
- Modify: all existing README/report Markdown files below `src/controllers/CBF-3DoF/results/`
- Create only if a meaningful uncovered collection exists: nearest-parent `README.md`
- Modify: `src/controllers/CBF-3DoF/AUDIT_REPORT.md`

**Interfaces:**
- Consumes: audit inventory, schema-family summary, reviewed CLI behavior, and actual retained artifact paths.
- Produces: complete directory map, commands, schemas, provenance limitations, and curated representative-file references.

- [ ] **Step 1: Create a documentation coverage checklist**

Enumerate each meaningful maintained directory and each repeated artifact collection. Map it to the README that owns its explanation; identify only genuine gaps that need a new nearest-parent README.

- [ ] **Step 2: Verify representative candidates from configuration through output**

For each controller phase and artifact class, select the smallest complementary set whose configuration, summary metrics, step data, and figures are parseable and traceable. Record exact relative paths and selection reasons. Do not select incomplete runs except as explicitly labeled diagnostic examples.

- [ ] **Step 3: Rewrite all maintained Markdown**

Apply the specification's objective writing standard. Correct stale paths and names, state inputs/outputs and dependencies, document historical schema variants, distinguish safety evidence from guarantees, and add representative references. Avoid redundant README files in individual run directories.

- [ ] **Step 4: Validate Markdown structure and links**

Run a repository-local link checker or a small read-only Python check over every Markdown file in the subtree. Expected: every relative file link resolves; deliberately non-file placeholders such as `PATH_TO_MODEL.zip` are in code blocks and not links.

- [ ] **Step 5: Search for assistant-like and ambiguous prose**

Use targeted case-insensitive searches for `ChatGPT`, `AI`, `as requested`, `we can`, `you can`, `obviously`, `clearly`, `robust`, `comprehensive`, `current`, `new`, and `previous`. Review every match in context; rewrite unsupported or conversational uses and document legitimate technical uses.

- [ ] **Step 6: Commit Task 5**

Stage Markdown changes and commit with `docs: document CBF 3DoF artifacts`.

### Task 6: Review every JSON and CSV artifact without changing scientific values

**Files:**
- Review: all `src/controllers/CBF-3DoF/**/*.json`
- Review: all `src/controllers/CBF-3DoF/**/*.csv`
- Modify only when provenance-safe: affected JSON/CSV files
- Modify: `src/controllers/CBF-3DoF/AUDIT_REPORT.md`

**Interfaces:**
- Consumes: Task 1 schema grouping and Task 5 terminology/path decisions.
- Preserves: every numeric measurement, seed, timestamp, solver status, gain, model identity, and original-run provenance.
- Produces: disposition for all 368 structured-text artifacts and documented schema variants.

- [ ] **Step 1: Review schema families and outliers**

Inspect every unique JSON key set and CSV header set, then every path-specific parse/shape issue. Compare outliers with their run configuration and parent README before classifying them.

- [ ] **Step 2: Identify editable prose and active paths**

Search string-valued fields for conversational wording, unsupported claims, obsolete active paths, encoding defects, and inconsistent terminology. Treat absolute paths recorded by completed runs as provenance unless executable code relies on them.

- [ ] **Step 3: Prove any proposed structured-artifact edit preserves evidence**

Before editing, capture a canonical projection containing all numeric values and protected provenance fields. Apply only descriptive normalization, then assert the projection is byte-for-byte equal. If equality cannot be established, retain the file and document the issue.

- [ ] **Step 4: Rerun JSON/CSV validation and compare protected fields**

Expected: every file parses; documented historical schema variants remain grouped; no protected field changes; all unresolved anomalies are named in `AUDIT_REPORT.md`.

- [ ] **Step 5: Commit Task 6 if files changed**

Stage only provenance-safe structured-text edits and the report. If no artifact requires editing, commit only the report update or fold it into Task 7.

### Task 7: Inspect all binary and database artifacts and close the issue ledger

**Files:**
- Review read-only: all `src/controllers/CBF-3DoF/**/*.npz`
- Review read-only: all `src/controllers/CBF-3DoF/**/*.png`
- Review read-only: all `src/controllers/CBF-3DoF/**/*.sqlite3`
- Modify: `src/controllers/CBF-3DoF/AUDIT_REPORT.md`
- Regenerate: `src/controllers/CBF-3DoF/artifact_inventory.csv`
- Regenerate: `src/controllers/CBF-3DoF/artifact_validation_summary.json`

**Interfaces:**
- Consumes: Task 1 validators and Tasks 2-6 issue classifications.
- Produces: disposition for all 4,081 NPZ, 48 PNG, and two SQLite artifacts, including every corrupt, non-finite, object-array, integrity, or schema warning.

- [ ] **Step 1: Inspect every validator finding by artifact family**

Review all binary/database errors and warnings, representative metadata for every schema family, NPZ member-name/shape/dtype outliers, PNG dimension/mode outliers, and SQLite integrity/table results.

- [ ] **Step 2: Trace anomalies to configurations and documentation**

Determine whether each anomaly is expected historical evidence, an incomplete run, a corrupt artifact, or a validator defect. Update parent READMEs for interpretation-relevant limitations and record path-specific unresolved issues in `AUDIT_REPORT.md`.

- [ ] **Step 3: Confirm binary hashes are unchanged**

Compare SHA-256 hashes for every NPZ, PNG, and SQLite file with the Task 2 baseline. Expected: exact equality for every retained binary artifact.

- [ ] **Step 4: Regenerate final inventory and summary**

Run the auditor twice for stable self-accounting. Expected: every current regular file occurs exactly once and every error/warning is reflected in the report.

- [ ] **Step 5: Commit Task 7**

```powershell
git add src/controllers/CBF-3DoF/AUDIT_REPORT.md src/controllers/CBF-3DoF/artifact_inventory.csv src/controllers/CBF-3DoF/artifact_validation_summary.json
git commit -m "docs: finalize CBF artifact audit"
```

### Task 8: Full verification and preservation audit

**Files:**
- Verify: all files changed by Tasks 1-7
- Update if verification changes inventory only: audit deliverables

**Interfaces:**
- Consumes: all prior tasks and the acceptance criteria in the specification.
- Produces: fresh evidence for final completion claims and an exact explanation for every path added, edited, removed, or preserved with a warning.

- [ ] **Step 1: Run the focused CBF suite**

Run: `.\.venv\Scripts\python.exe -m pytest src/controllers/CBF-3DoF/tests -q`

Expected: all focused tests pass; record count and warnings.

- [ ] **Step 2: Run the complete repository suite**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests pass. If any fail, record every failing test by name and determine whether it is introduced, exposed, or pre-existing before making a completion claim.

- [ ] **Step 3: Compile every CBF Python file and exercise every CLI help path**

Run compileall on `src/controllers/CBF-3DoF`, then invoke `--help` for every evaluation, tuning, plotting, and audit entry point. Expected: exit code 0 throughout.

- [ ] **Step 4: Run final artifact and documentation checks**

Run the auditor, inventory/filesystem symmetric-difference check, Markdown link check, objective-prose search, SQLite integrity checks, and `git diff --check`. Expected: no unreported errors, no missing inventory paths, no broken links, and no whitespace errors.

- [ ] **Step 5: Compare the final tree with the baseline**

Produce an explicit list of added, edited, moved, and deleted paths. Verify that all binary research artifacts retain their baseline hashes. Explain count/byte differences using named audit outputs and approved edits. Confirm the unrelated deleted 2026-09-29 planning documents retain their pre-task status.

- [ ] **Step 6: Review the specification line by line**

Mark every acceptance criterion as satisfied with a command/result reference or as unresolved with a path-specific explanation. Do not classify documented warnings as success without stating them.

- [ ] **Step 7: Commit final generated updates if needed**

Stage only audit-scope files and commit with `chore: complete CBF 3DoF audit`. Never stage the unrelated deleted planning documents.
