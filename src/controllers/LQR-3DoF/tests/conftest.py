"""Shared import configuration for the LQR controller tests."""

from __future__ import annotations

import sys
from pathlib import Path


LQR_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LQR_ROOT.parents[1]

for import_path in (LQR_ROOT, SRC_ROOT):
    path = str(import_path)
    if path not in sys.path:
        sys.path.insert(0, path)
