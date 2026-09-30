"""Import-path setup for diagnostics executed directly from the repository."""

from pathlib import Path
import sys


def ensure_project_imports() -> None:
    """Expose the repository's ``src`` tree to a directly executed script."""
    src_dir = str(Path(__file__).resolve().parents[4])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
