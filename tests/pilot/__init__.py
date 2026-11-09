"""Paths used by pilot source-isolation tests after the repository move."""

from paths import REPO_ROOT

SCRIPT_ROOT = REPO_ROOT / "scripts" / "pilot"
MODEL_FILES = [str(p) for directory in (REPO_ROOT / "src/models", REPO_ROOT / "benchmarks")
               for p in directory.rglob("*.py")]
