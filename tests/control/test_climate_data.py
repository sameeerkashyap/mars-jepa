import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("torchrl", minversion="0.14.0")

from benchmarks.dreamer_control.data import generate_dataset, load_dataset  # noqa: E402


def test_versioned_grouped_data_and_integrity(tmp_path):
    destination = tmp_path / "tiny"
    generated = generate_dataset(destination, split="train", condition_seeds=[5000])
    manifest, episodes = load_dataset(destination, split="train")
    assert manifest == generated
    assert manifest["data_namespace"] == "control-response-v1"
    assert manifest["condition_seeds"] == [5000]
    assert manifest["sealed_test_conditions"] == [5200, 5219]
    assert manifest["cost"]["wall_seconds"] > 0 and manifest["cost"]["gpu_hours"] == 0
    assert len(episodes) == 4
    assert episodes[0]["policy"]["rewards"].sum() == -480.
    for episode in episodes:
        assert set(episode) == {"policy", "daily", "oracle_diagnostics"}
        policy = episode["policy"]
        assert set(policy) == {"observations", "actions", "requested_actions", "rewards", "terminated", "truncated"}
        assert policy["observations"].shape == (13, 133)
        assert episode["daily"]["T_s"].shape == (12, 10, 64)
        assert episode["daily"]["t_sol"].shape == (12, 10)
        assert episode["oracle_diagnostics"]["tracking_rmse_K"] >= 0
        response = episode["oracle_diagnostics"]["daily_response_T_s"]
        assert response.shape == (12, 10, 64) and np.isfinite(response).all()
        np.testing.assert_allclose(response,
                                   episode["daily"]["T_s"] - episodes[0]["daily"]["T_s"])
        np.testing.assert_allclose(policy["rewards"].flatten(),
                                   -np.square(response[:, :, :4] - 2).mean(-1).sum(-1), rtol=1e-6)
        assert np.array_equal(policy["observations"][0], episodes[0]["policy"]["observations"][0])
    with pytest.raises(FileExistsError):
        generate_dataset(destination, split="train", condition_seeds=[5000])
    with pytest.raises(ValueError, match="split"):
        load_dataset(destination, split="validation")
    with pytest.raises(ValueError, match="split"):
        generate_dataset(tmp_path / "bad", split="validation", condition_seeds=[5000])
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="sealed"):
        generate_dataset(tmp_path / "sealed", split="test")
    data_path = destination / manifest["episodes"][0]["file"]
    original = data_path.read_bytes()
    with data_path.open("ab") as file:
        file.write(b"corruption")
    with pytest.raises(ValueError, match="hash"):
        load_dataset(destination, split="train")

    data_path.write_bytes(original)
    # A valid file hash must not allow a malformed or private policy schema.
    with np.load(data_path, allow_pickle=False) as file:
        data = {key: file[key] for key in file.files}
    data["reference_T_s"] = np.zeros((12, 10, 64))
    with data_path.open("wb") as file:
        np.savez_compressed(file, **data)
    manifest["episodes"][0]["sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    manifest.pop("data_sha256")
    manifest["data_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema"):
        load_dataset(destination, split="train")
