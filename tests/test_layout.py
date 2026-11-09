"""The architecture move preserves imports, data defaults, and installed tools."""

import subprocess
import sys

import pytest

from benchmarks.registry import build_model
from models import PIJEPA
from models.pi_jepa.jepa import PIJEPA as SpatialWorldModel
from paths import PILOT_DATA, PILOT_SPEC, REPO_ROOT
from physics.surface import load_config
from physics.surface_torch import load_ebm_params
from scripts.pilot.train import load_phase2_config


def test_pilot_defaults_work_outside_repository(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert load_config()["grid"]["n"] == 64
    assert load_ebm_params()["C_eff"] > 0
    assert load_phase2_config()["data"]["data_dir"] == str(PILOT_DATA)
    assert (PILOT_SPEC / "thresholds.yaml").is_file()
    assert PIJEPA is SpatialWorldModel


@pytest.mark.parametrize("arm,module", [
    ("jepa", "marsworld.models.latent_jepa.model"),
    ("gru", "benchmarks.gru.model"),
    ("gru_noforce", "benchmarks.gru.model"),
    ("rssm", "benchmarks.rssm.model"),
    ("transformer", "benchmarks.transformer.model"),
    ("hybrid", "benchmarks.hybrid.model"),
    ("ebm_only", "benchmarks.hybrid.model"),
    ("linear", "benchmarks.linear.model"),
])
def test_registry_uses_architecture_packages(arm, module):
    stats = {"T_s_mean": 200, "T_s_std": 30, "S_mean": 100, "S_std": 80,
             "F_a_mean": 2, "F_a_std": 2}
    model = build_model(arm, stats, 64, load_phase2_config()["arch"])
    assert type(model).__module__ == module


def test_installed_pilot_command_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.pilot.train", "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "--arm" in result.stdout and "--data-dir" in result.stdout


def test_data_code_and_metadata_are_not_ignored():
    paths = ["src/marsworld/data/solver.py", "src/marsworld/data/dataset.py",
             "data/specification/pilot/config.yaml",
             "data/synthetic/pilot/manifest.json",
             "tests/pilot/fixtures/regression_ref.npz"]
    result = subprocess.run(["git", "check-ignore", *paths], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 1, result.stdout + result.stderr
