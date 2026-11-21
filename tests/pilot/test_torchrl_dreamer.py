"""TorchRL forecasting, forcing alignment, and restart safety on a tiny fixture."""

import pytest
import torch

pytest.importorskip("torchrl", minversion="0.14.0")

from benchmarks.registry import build_model, count_params
from scripts.pilot import train
from scripts.pilot.eval import load_model
from tests.pilot.fakedata import build_fake_dataset

STATS = {"T_s_mean": 200, "T_s_std": 30, "S_mean": 100, "S_std": 80,
         "F_a_mean": 2, "F_a_std": 2}


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return train.load_phase2_config("data/specification/pilot/torchrl_small.yaml")


def inputs():
    return (torch.full((2, 4, 64), 200.0), torch.full((2, 9, 64), 100.0),
            torch.zeros(2, 9), torch.linspace(-1.4, 1.4, 64))


def test_uses_torchrl_components_and_small_parameter_budget(config):
    from torchrl.modules.models.model_based import RSSMPosteriorV3, RSSMPriorV3

    model = build_model("torchrl_dreamer", STATS, 64, config["arch"])
    assert isinstance(model.prior, RSSMPriorV3)
    assert isinstance(model.posterior, RSSMPosteriorV3)
    counts = [count_params(model)] + [count_params(build_model(a, STATS, 64, config["arch"]))
                                    for a in ("gru", "rssm", "jepa")]
    assert max(counts) / min(counts) <= 1.1
    assert config["study"]["fraction"] == 0.3 and config["study"]["subset_seed"] == 0
    assert config["protocol"] == {"K": 40, "H": 120, "stride": 120}


def test_deterministic_validation_and_previous_interval_forcing(config):
    model = build_model("torchrl_dreamer", STATS, 64, config["arch"]).eval()
    ts, sun, heat, lat = inputs()
    rng = torch.get_rng_state()
    with torch.no_grad():
        baseline, _ = model.rollout(ts, sun, heat, lat)
        assert baseline.shape == (2, 5, 64) and torch.isfinite(baseline).all()
        assert torch.equal(rng, torch.get_rng_state())
        heat[:, 4] = 5
        changed, _ = model.rollout(ts, sun, heat, lat)
        assert torch.equal(baseline[:, 0], changed[:, 0])
        assert not torch.allclose(baseline[:, 1], changed[:, 1])
        heat.zero_()
        heat[:, 3] = 5
        previous, _ = model.rollout(ts, sun, heat, lat)
        assert not torch.allclose(baseline[:, 0], previous[:, 0])
        sampled, _ = model.rollout(ts, sun, heat, lat, sample=True)
        assert not torch.allclose(previous, sampled)


def test_loss_trains_all_components_and_targets_stay_out_of_rollout(config):
    model = build_model("torchrl_dreamer", STATS, 64, config["arch"])
    ts, sun, heat, lat = inputs()
    target = torch.full((2, 5, 64), 205.0)
    pred, aux = model.rollout(ts, sun, heat, lat)
    loss = ((pred - target) / STATS["T_s_std"]).square().mean() + model.latent_loss(aux, target)
    assert torch.isfinite(loss)
    loss.backward()
    for component in (model.enc, model.prior, model.posterior, model.dec):
        grads = [p.grad for p in component.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) for g in grads)
    model.eval()
    with torch.no_grad():
        before, aux = model.rollout(ts, sun, heat, lat)
        first = model.latent_loss(aux, target)
        changed = model.latent_loss(aux, target + 20)
        after, _ = model.rollout(ts, sun, heat, lat)
    assert not torch.allclose(first, changed)
    assert torch.equal(before, after)
    with pytest.raises(ValueError, match="targets"):
        model.latent_loss(aux, target[..., :1])


def test_small_data_fit_resume_and_evaluation_loader(config, tmp_path, monkeypatch):
    config["data"]["data_dir"] = build_fake_dataset(str(tmp_path / "data"), n_time=180)
    calls = []
    original_load = train.load_split

    def safe_load(root, family, split, **kwargs):
        calls.append(split)
        assert split in ("train", "val")
        return original_load(root, family, split, **kwargs)

    monkeypatch.setattr(train, "load_split", safe_load)
    kw = dict(arm="torchrl_dreamer", physics="off", fraction=0.3, subset_seed=0,
              seed=0, lam=0.0, p2=config, full_cfg=train.load_config(), device="cpu",
              max_batches=1, record_curve=True, calibration_mode=True)
    train.train_one(**kw, epochs=2, checkpoint_dir=tmp_path / "full")
    train.train_one(**kw, epochs=1, checkpoint_dir=tmp_path / "resumed")
    model, stats, info = train.train_one(**kw, epochs=2,
                                       checkpoint_dir=tmp_path / "resumed", resume=True)
    full = torch.load(tmp_path / "full/latest.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resumed/latest.pt", weights_only=False)
    assert full["val_curve"] == resumed["val_curve"]
    for key in full["model"]:
        assert torch.equal(full["model"][key], resumed["model"][key]), key
    assert info["train_windows"] == 4 and info["val_windows"] == 3
    assert set(calls) == {"train", "val"}
    assert not (tmp_path / "data/test_access.log").exists()
    train.save_run(str(tmp_path / "resumed"), model, stats, info, config)
    loaded, *_ = load_model(str(tmp_path / "resumed"), config, train.load_config(), "cpu")
    model.eval()
    with torch.no_grad():
        assert torch.equal(model.rollout(*inputs())[0], loaded.rollout(*inputs())[0])
