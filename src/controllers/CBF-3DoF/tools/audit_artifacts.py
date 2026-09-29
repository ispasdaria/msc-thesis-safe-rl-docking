"""Read-only validation and inventory of CBF-3DoF artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
import sqlite3
from typing import Any, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    severity: str
    code: str
    message: str


@dataclass
class ArtifactRecord:
    path: str
    artifact_class: str
    size_bytes: int
    status: str = "valid"
    metadata: dict[str, Any] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)


_ARTIFACT_CLASSES = {
    ".py": "python",
    ".md": "markdown",
    ".json": "json",
    ".csv": "csv",
    ".npz": "npz",
    ".png": "png",
    ".sqlite3": "sqlite",
}


def classify_artifact(path: Path) -> str:
    """Return the validation class selected from a file's suffix."""
    return _ARTIFACT_CLASSES.get(path.suffix.lower(), "other")


def validate_artifact(path: Path, root: Path) -> ArtifactRecord:
    """Validate one artifact without modifying it."""
    root_resolved = root.resolve()
    resolved = path.resolve()
    relative = resolved.relative_to(root_resolved).as_posix()
    record = ArtifactRecord(relative, classify_artifact(path), path.stat().st_size)
    validators = {
        "json": _validate_json,
        "csv": _validate_csv,
        "npz": _validate_npz,
        "png": _validate_png,
        "sqlite": _validate_sqlite,
    }
    validator = validators.get(record.artifact_class)
    if validator is None:
        return record
    try:
        validator(path, record)
    except Exception as exc:  # A damaged artifact must not stop the tree audit.
        _add_issue(record, "error", f"invalid_{record.artifact_class}", str(exc))
    return record


def _add_issue(record: ArtifactRecord, severity: str, code: str, message: str) -> None:
    record.issues.append(ValidationIssue(record.path, severity, code, message))
    if severity == "error":
        record.status = "error"
    elif record.status == "valid":
        record.status = "warning"


def _validate_json(path: Path, record: ArtifactRecord) -> None:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    type_names = {dict: "object", list: "array", str: "string", int: "number", float: "number", bool: "boolean", type(None): "null"}
    record.metadata["top_level_type"] = type_names.get(type(value), type(value).__name__)
    if isinstance(value, dict):
        record.metadata["keys"] = sorted(value)


def _validate_csv(path: Path, record: ArtifactRecord) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        duplicate_headers = sorted({name for name in header if header.count(name) > 1})
        if duplicate_headers:
            _add_issue(record, "warning", "duplicate_csv_header", ", ".join(duplicate_headers))
        inconsistent: list[int] = []
        row_count = 0
        for row_count, row in enumerate(reader, start=1):
            if len(row) != len(header):
                inconsistent.append(row_count + 1)
        record.metadata.update({"columns": header, "row_count": row_count})
        if inconsistent:
            _add_issue(record, "warning", "inconsistent_csv_row_width", f"lines: {inconsistent[:20]}")


def _validate_npz(path: Path, record: ArtifactRecord) -> None:
    members: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            try:
                array = archive[name]
            except ValueError as exc:
                members[name] = {"object_array": True}
                _add_issue(record, "warning", "npz_object_array", f"{name}: {exc}")
                continue
            nonfinite = 0
            if np.issubdtype(array.dtype, np.number):
                flat = array.reshape(-1)
                for start in range(0, flat.size, 1_000_000):
                    nonfinite += int(np.count_nonzero(~np.isfinite(flat[start : start + 1_000_000])))
            members[name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "nonfinite_count": nonfinite,
            }
    record.metadata["members"] = members


def _validate_png(path: Path, record: ArtifactRecord) -> None:
    with Image.open(path) as image:
        image.verify()
        record.metadata.update({"width": image.width, "height": image.height, "mode": image.mode, "format": image.format})


def _validate_sqlite(path: Path, record: ArtifactRecord) -> None:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        integrity_value = str(integrity[0]) if integrity else "no result"
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    record.metadata.update({"integrity_check": integrity_value, "tables": tables})
    if integrity_value != "ok":
        _add_issue(record, "error", "sqlite_integrity", integrity_value)


def audit_tree(root: Path) -> tuple[list[ArtifactRecord], list[ValidationIssue]]:
    """Inventory regular files below *root* in deterministic path order."""
    root = root.resolve()
    records: list[ArtifactRecord] = []
    issues: list[ValidationIssue] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            record = validate_artifact(path, root)
            records.append(record)
            issues.extend(record.issues)
        except (OSError, ValueError) as exc:
            issue = ValidationIssue(
                path=path.relative_to(root).as_posix(),
                severity="error",
                code="unreadable_path",
                message=str(exc),
            )
            issues.append(issue)
    return records, issues


def write_inventory(records: Sequence[ArtifactRecord], destination: Path) -> None:
    """Write deterministic one-row-per-artifact CSV metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "artifact_class", "size_bytes", "status", "metadata_json", "issues_json"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.path):
            writer.writerow(
                {
                    "path": record.path,
                    "artifact_class": record.artifact_class,
                    "size_bytes": record.size_bytes,
                    "status": record.status,
                    "metadata_json": json.dumps(record.metadata, sort_keys=True, separators=(",", ":")),
                    "issues_json": json.dumps([asdict(issue) for issue in record.issues], sort_keys=True, separators=(",", ":")),
                }
            )


def write_summary(
    records: Sequence[ArtifactRecord],
    issues: Sequence[ValidationIssue],
    destination: Path,
) -> None:
    """Write aggregate counts, issues, and structured-text schema families."""
    families: dict[str, dict[str, list[str]]] = {"json": defaultdict(list), "csv": defaultdict(list)}
    for record in records:
        if record.artifact_class == "json" and "keys" in record.metadata:
            families["json"]["|".join(record.metadata["keys"])].append(record.path)
        elif record.artifact_class == "csv" and "columns" in record.metadata:
            families["csv"]["|".join(record.metadata["columns"])].append(record.path)
    payload = {
        "file_count": len(records),
        "total_size_bytes": sum(record.size_bytes for record in records),
        "counts_by_class": dict(sorted(Counter(record.artifact_class for record in records).items())),
        "counts_by_status": dict(sorted(Counter(record.status for record in records).items())),
        "issues": [asdict(issue) for issue in issues],
        "schema_families": {
            kind: {schema: paths for schema, paths in sorted(groups.items())}
            for kind, groups in families.items()
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the artifact audit CLI and return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    records, issues = audit_tree(args.root)
    write_inventory(records, args.inventory)
    write_summary(records, issues, args.summary)
    return int(any(issue.severity == "error" for issue in issues))


if __name__ == "__main__":
    raise SystemExit(main())
