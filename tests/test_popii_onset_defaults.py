"""Adopted mixed-model defaults and paired/single-model routing regressions."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from auroralf.experiments import transition_workers as workers
from auroralf.ssp.ionizing import IonizingKernel
from scripts.submit.prepare_reionization_calibration import default_source

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def histories(monkeypatch):
    t = np.broadcast_to(np.linspace(0.1, 0.3, 21), (4, 21)).copy()
    m = 10 ** ((t - 0.1) * 20 - 2)
    sfr, cool = np.ones_like(t), np.ones_like(t)
    kernel = IonizingKernel(np.array([0.001, 100.0]), np.ones(2))
    cfg = SimpleNamespace(
        seed=31,
        n_tracks=4,
        track_chunk=4,
        max_lookback_myr=100,
        popiii_max_age_myr=100,
        q_log10_mean=0,
        q_log10_sigma=1,
        epsilon_b=0.03,
        fesc_popii=0.2,
        fesc_popiii=0.2,
    )
    cosmo = SimpleNamespace(omega_b=0.05, omega_m=0.3)
    monkeypatch.setattr(workers.source, "STATE", (cfg, cosmo, None, kernel, kernel), raising=False)
    monkeypatch.setattr(
        workers, "UV", (kernel, kernel.age_myr, kernel.rate_per_msun), raising=False
    )
    monkeypatch.setattr(workers, "histories", lambda *_: iter([(0, t, m, t > 0, sfr, cool)]))


@pytest.mark.parametrize("name", ["baseline", "delay0", "delay30"])
def test_single_selection_reproduces_paired_uv_and_rates(histories, monkeypatch, name):
    monkeypatch.setattr(workers, "SELECTED", workers.VARIANTS, raising=False)
    paired_rates = workers.rate_cell((0, 0, 8, 1e9))[2]
    edges = np.arange(-30.0, 10.0, 0.5)
    _, paired_uv, paired_diag = workers.uv_cell((0, 8, 1e9, edges))
    monkeypatch.setattr(workers, "SELECTED", (name,))
    single_rates = workers.rate_cell((0, 0, 8, 1e9))[2]
    _, single_uv, single_diag = workers.uv_cell((0, 8, 1e9, edges))
    i = workers.VARIANTS.index(name)
    for key in [
        "mean_rate",
        "se_rate",
        "rate_covariance_of_mean",
        "popii_quadrature_error",
        "popii_censored_upper_extra",
    ]:
        np.testing.assert_array_equal(single_rates[key][0], paired_rates[key][i])
    np.testing.assert_array_equal(single_uv[0], paired_uv[i])
    np.testing.assert_array_equal(single_diag, paired_diag[[i, -2, -1]])
    assert np.all(paired_rates["mean_rate"][2, 0] <= paired_rates["mean_rate"][1, 0])
    assert np.all(paired_rates["mean_rate"][1, 0] < paired_rates["mean_rate"][0, 0])


@pytest.mark.parametrize("variants", [[], ["delay0", "delay0"], ["unknown"], "delay0", [None]])
def test_invalid_selection_rejected(variants):
    with pytest.raises(ValueError, match="variants"):
        workers.validate_variants(variants)


def test_default_uv_and_reionization_share_adopted_zero_delay():
    plan = json.loads((ROOT / "configs/uvlf/popii_popiii.json").read_text())
    assert plan["variants"] == ["delay0"]
    assert workers.DELAYS_MYR[plan["variants"][0]] == 0
    assert default_source("popii_popiii") == ROOT / plan["adoption"]["validated_sources"]
    assert default_source("popii") == ROOT / "data_save/threshold_zero_all_20260918/ionizing"


def test_adopted_manifest_change_fails(tmp_path, monkeypatch):
    from scripts.submit import prepare_reionization_calibration as prepare

    (tmp_path / "configs/uvlf").mkdir(parents=True)
    (tmp_path / "source").mkdir()
    (tmp_path / "configs/uvlf/popii_popiii.json").write_text(
        json.dumps(
            {
                "variants": ["delay0"],
                "adoption": {"validated_sources": "source", "source_manifest_sha256": "wrong"},
            }
        )
    )
    (tmp_path / "source/manifest.json").write_text('{"status":"complete","variant":"delay0"}')
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="manifest changed"):
        prepare.default_source("popii_popiii")


def test_changed_selection_cannot_reuse_zero_delay_sources(tmp_path, monkeypatch):
    from scripts.submit import prepare_reionization_calibration as prepare

    (tmp_path / "configs/uvlf").mkdir(parents=True)
    (tmp_path / "configs/uvlf/popii_popiii.json").write_text('{"variants":["delay30"]}')
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Selected model differs"):
        prepare.default_source("popii_popiii")


def test_submit_dry_run_routes_both_jobs_to_current_plan(tmp_path, monkeypatch):
    import sys

    from scripts.submit import submit_popii_transition as submit

    monkeypatch.setattr(submit, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs/uvlf").mkdir(parents=True)
    (tmp_path / "configs/uvlf/popii_popiii.json").write_text('{"variants":["delay0"]}')
    calls = []

    def read_scheduler(command):
        calls.append(command)
        if command.startswith("sinfo"):
            return "node|idle|0/56/0/56"
        if command.startswith("squeue"):
            return ""
        raise AssertionError("Dry run attempted a remote mutation")

    monkeypatch.setattr(submit, "ssh", read_scheduler)
    monkeypatch.setattr(sys, "argv", ["submit", "--release", "causal-default", "--dry-run"])
    submit.main()
    record = json.loads((tmp_path / "outputs/causal-default/dry-run.json").read_text())
    assert len(calls) == 2 and len(record["commands"]) == 2
    for cmd in record["commands"]:
        assert "--partition=cp6" in cmd and "--cpus-per-task=56" in cmd
        assert "--plan configs/uvlf/popii_popiii.json" in cmd[-1]
        assert not any(arg.startswith(("--time", "--mem")) for arg in cmd)
