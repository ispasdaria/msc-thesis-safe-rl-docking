from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np

import pytest

from tools.audit_artifacts import (
    ArtifactRecord,
    audit_tree,
    classify_artifact,
    main,
    validate_artifact,
    write_inventory,
    write_summary,
)


def test_classifies_supported_artifacts() -> None:
    expected = {
        "module.py": "python",
        "README.md": "markdown",
        "run.json": "json",
        "metrics.csv": "csv",
        "trace.npz": "npz",
        "figure.png": "png",
        "study.sqlite3": "sqlite",
        ".gitignore": "other",
    }

    assert {name: classify_artifact(Path(name)) for name in expected} == expected


def test_audit_tree_orders_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "z.json").write_text("{}", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.md").write_text("# B\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("VALUE = 1\n", encoding="utf-8")

    records, issues = audit_tree(tmp_path)

    assert [record.path for record in records] == ["a.py", "nested/b.md", "z.json"]
    assert issues == []


def test_json_reports_top_level_keys_and_parse_errors(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text('{"seed": 4, "solver": "osqp"}', encoding="utf-8")
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"seed":', encoding="utf-8")

    valid_record = validate_artifact(valid, tmp_path)
    malformed_record = validate_artifact(malformed, tmp_path)

    assert valid_record.metadata["top_level_type"] == "object"
    assert valid_record.metadata["keys"] == ["seed", "solver"]
    assert malformed_record.status == "error"
    assert malformed_record.issues[0].code == "invalid_json"


def test_csv_reports_duplicate_headers_and_inconsistent_rows(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("seed,value,value\n1,2,3\n2,4\n", encoding="utf-8")

    record = validate_artifact(path, tmp_path)

    assert record.metadata["row_count"] == 2
    assert {issue.code for issue in record.issues} == {
        "duplicate_csv_header",
        "inconsistent_csv_row_width",
    }


def test_npz_reports_members_shapes_dtypes_and_nonfinite_values(tmp_path: Path) -> None:
    path = tmp_path / "trace.npz"
    np.savez(path, state=np.array([[1.0, np.nan], [np.inf, 4.0]]), seed=np.array(7))

    record = validate_artifact(path, tmp_path)

    assert record.metadata["members"]["seed"] == {
        "shape": [],
        "dtype": "int64",
        "nonfinite_count": 0,
    }
    assert record.metadata["members"]["state"]["shape"] == [2, 2]
    assert record.metadata["members"]["state"]["nonfinite_count"] == 2


def test_png_reports_dimensions_mode_and_format(tmp_path: Path) -> None:
    path = tmp_path / "figure.png"
    plt.imsave(path, np.zeros((3, 5, 3), dtype=float))

    record = validate_artifact(path, tmp_path)

    assert record.metadata["width"] == 5
    assert record.metadata["height"] == 3
    assert record.metadata["format"] == "PNG"


def test_sqlite_reports_integrity_and_tables(tmp_path: Path) -> None:
    path = tmp_path / "study.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE trials (id INTEGER PRIMARY KEY, value REAL)")

    record = validate_artifact(path, tmp_path)

    assert record.metadata["integrity_check"] == "ok"
    assert record.metadata["tables"] == ["trials"]


def test_corrupt_binary_files_do_not_stop_audit(tmp_path: Path) -> None:
    for name in ("bad.npz", "bad.png", "bad.sqlite3"):
        (tmp_path / name).write_bytes(b"not a valid file")
    (tmp_path / "z.json").write_text("{}", encoding="utf-8")

    records, issues = audit_tree(tmp_path)

    assert [record.path for record in records] == [
        "bad.npz",
        "bad.png",
        "bad.sqlite3",
        "z.json",
    ]
    assert [record.status for record in records] == ["error", "error", "error", "valid"]
    assert len([issue for issue in issues if issue.severity == "error"]) == 3


def test_inventory_contains_every_regular_file(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.json").write_text("{}", encoding="utf-8")
    (root / "b.md").write_text("# B\n", encoding="utf-8")
    records, _ = audit_tree(root)
    destination = tmp_path / "inventory.csv"

    write_inventory(records, destination)

    with destination.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["path"] for row in rows] == ["a.json", "b.md"]
    assert list(rows[0]) == [
        "path",
        "artifact_class",
        "size_bytes",
        "status",
        "metadata_json",
        "issues_json",
    ]


def test_summary_groups_schema_variants(tmp_path: Path) -> None:
    records = [
        ArtifactRecord("a.json", "json", 2, metadata={"keys": ["seed"]}),
        ArtifactRecord("b.json", "json", 2, metadata={"keys": ["seed", "solver"]}),
        ArtifactRecord("c.csv", "csv", 4, metadata={"columns": ["seed", "value"]}),
    ]
    destination = tmp_path / "summary.json"

    write_summary(records, [], destination)

    summary = json.loads(destination.read_text(encoding="utf-8"))
    assert summary["schema_families"]["json"] == {
        "seed": ["a.json"],
        "seed|solver": ["b.json"],
    }
    assert summary["schema_families"]["csv"] == {"seed|value": ["c.csv"]}


def test_cli_returns_nonzero_for_errors(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")

    result = main(
        [
            str(tmp_path),
            "--inventory",
            str(tmp_path / "inventory.csv"),
            "--summary",
            str(tmp_path / "summary.json"),
        ]
    )

    assert result == 1


def test_symlink_outside_root_is_not_followed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    try:
        (root / "escape.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    records, _ = audit_tree(root)

    assert records == []


def test_audit_does_not_change_source_hashes(tmp_path: Path) -> None:
    source = tmp_path / "trace.npz"
    np.savez(source, values=np.arange(4))
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    audit_tree(tmp_path)

    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_outputs_account_for_themselves_on_second_run(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    inventory = root / "artifact_inventory.csv"
    summary = root / "artifact_validation_summary.json"
    report = root / "AUDIT_REPORT.md"
    report.write_text("# Audit report\n", encoding="utf-8")
    args = [str(root), "--inventory", str(inventory), "--summary", str(summary)]

    assert main(args) == 0
    with inventory.open(encoding="utf-8", newline="") as handle:
        first_paths = [row["path"] for row in csv.DictReader(handle)]
    assert first_paths.count("artifact_inventory.csv") == 1
    assert first_paths.count("artifact_validation_summary.json") == 1
    assert first_paths.count("AUDIT_REPORT.md") == 1

    assert main(args) == 0
    with inventory.open(encoding="utf-8", newline="") as handle:
        second_paths = [row["path"] for row in csv.DictReader(handle)]
    assert second_paths == first_paths == sorted(first_paths)
