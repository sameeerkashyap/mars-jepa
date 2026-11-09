"""Shared locations in the MarsWorld research checkout."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_SPEC = REPO_ROOT / "data" / "specification" / "pilot"
PILOT_DATA = REPO_ROOT / "data" / "synthetic" / "pilot"
PILOT_OUTPUT = REPO_ROOT / "outputs" / "pilot"
