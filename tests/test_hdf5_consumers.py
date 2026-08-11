from __future__ import annotations

from dataclasses import replace
import importlib.util
import inspect
from pathlib import Path

import h5py
import numpy as np
import pytest

from auroralf.config import (
    CONFIG_SCHEMA_VERSION,
    CosmologyConfig,
    MAHConfig,
    OutputConfig,
    SamplingConfig,
    StarFormationConfig,
    StellarPopulationConfig,
    UVLFRunConfig,
)
from auroralf.io import ArtifactProvenance, UVLFArtifact, write_uvlf_artifact_atomic
from auroralf.results import (
    IMFModeResult,
    ModeRunDiagnostics,
    RedshiftResult,
    RunDiagnostics,
    UVLFRunResult,
)


SCRIPT_CASES = {
    "replot": (
        Path("scripts/plot/replot_uvlf_imf_no_delay_compare.py"),
        ["--hdf5-path"],
        ["--npz-path"],
        [],
    ),
    "burst": (
        Path("scripts/analysis/plot_uvlf_burst_comparison.py"),
        ["--no-burst-hdf5", "--burst-hdf5"],
        ["--no-burst-npz", "--burst-npz"],
        ["--no-burst-hdf5", "a.h5", "--burst-hdf5", "b.h5", "--output-prefix", "out"],
    ),
    "gate": (
        Path("scripts/analysis/plot_uvlf_gate_delay_burst_matrix.py"),
        [
            "--no-delay-no-burst-hdf5",
            "--no-delay-burst-hdf5",
            "--delay-no-burst-hdf5",
            "--delay-burst-hdf5",
        ],
        [
            "--no-delay-no-burst-npz",
            "--no-delay-burst-npz",
            "--delay-no-burst-npz",
            "--delay-burst-npz",
        ],
        [
            "--no-delay-no-burst-hdf5",
            "a.h5",
            "--no-delay-burst-hdf5",
            "b.h5",
            "--delay-no-burst-hdf5",
            "c.h5",
            "--delay-burst-hdf5",
            "d.h5",
            "--output-prefix",
            "out",
        ],
    ),
    "hmf_compare": (
        Path("scripts/analysis/compare_uvlf_mass_function_outputs.py"),
        ["--reference-hdf5", "--candidate-hdf5"],
        ["--reference-npz", "--candidate-npz"],
        ["--reference-hdf5", "a.h5", "--candidate-hdf5", "b.h5"],
    ),
    "fit_check": (
        Path("scripts/plot/plot_popii_uvlf_fit_check_z6_z12p5.py"),
        ["--hdf5-path"],
        ["--npz-path"],
        [],
    ),
    "canonical_hmf": (
        Path("scripts/plot/plot_uvlf_canonical_hmf_observations.py"),
        ["--reed07-hdf5"],
        ["--reed07-npz"],
        ["--reed07-hdf5", "a.h5"],
    ),
}


def _load_script(name: str, relative_path: Path) -> object:
    path = relative_path.resolve()
    spec = importlib.util.spec_from_file_location(f"hdf5_consumer_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(
    tmp_path: Path,
    *,
    run_id: str = "consumer",
    redshifts: tuple[float, ...] = (6.0, 8.0),
    modes: tuple[str, ...] = ("canonical", "mah_burst_mild_topheavy"),
    edges: tuple[float, ...] = (-24.0, -20.0, -16.0),
    star_formation: StarFormationConfig | None = None,
) -> UVLFRunConfig:
    source_dir = tmp_path / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    canonical = (source_dir / "canonical.dat").resolve()
    topheavy = (source_dir / "topheavy.hdf5").resolve()
    popiii = (source_dir / "popiii.dat").resolve()
    canonical.write_bytes(b"canonical")
    topheavy.write_bytes(b"topheavy")
    popiii.write_bytes(b"popiii")
    return UVLFRunConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        run_id=run_id,
        redshifts=redshifts,
        base_seed=123,
        cosmology=CosmologyConfig(),
        mah=MAHConfig(n_time_steps=8),
        star_formation=(
            StarFormationConfig(metallicity_source="none")
            if star_formation is None
            else star_formation
        ),
        stellar_population=StellarPopulationConfig(
            imf_modes=modes,
            enable_archived_imf_gate=any(mode != "canonical" for mode in modes),
            canonical_ssp_path=canonical,
            topheavy_ssp_path=topheavy,
            popiii_ssp_path=popiii,
            birth_metallicity_topheavy_max_zsun=None,
        ),
        sampling=SamplingConfig(
            mass_batch_size=1,
            n_halo_mass_samples=2,
            n_tracks_per_halo_mass=2,
            muv_bin_edges=edges,
            workers=1,
            apply_dust=False,
        ),
        output=OutputConfig((tmp_path / f"{run_id}.h5").resolve()),
    )


def _mode_result(mode: str, edges: tuple[float, ...], scale: float) -> IMFModeResult:
    edge_array = np.asarray(edges, dtype=float)
    centers = 0.5 * (edge_array[:-1] + edge_array[1:])
    widths = np.diff(edge_array)
    count = centers.size
    base = np.arange(1, count + 1, dtype=float) * scale
    return IMFModeResult(
        imf_mode=mode,
        bin_edges_muv=edge_array,
        bin_centers_muv=centers,
        bin_width_mag=widths,
        raw_counts=np.arange(count, 0, -1, dtype=np.int64),
        weighted_counts_per_mpc3=base * 1.0e-5,
        weight_squared_counts_per_mpc6=base * 1.0e-10,
        weighted_count_sigma_per_mpc3=base * 1.0e-5,
        effective_counts=base,
        phi_intrinsic_per_mpc3_per_mag=base * 2.0e-6,
        phi_intrinsic_sigma_per_mpc3_per_mag=base * 2.0e-7,
        phi_observed_per_mpc3_per_mag=base * 3.0e-6,
        phi_observed_sigma_per_mpc3_per_mag=base * 3.0e-7,
        halo_tracks=(),
    )


def _result(config: UVLFRunConfig) -> UVLFRunResult:
    redshift_results = tuple(
        RedshiftResult(
            redshift=redshift,
            imf_modes=tuple(
                _mode_result(mode, config.sampling.muv_bin_edges, 1.0 + z_index + mode_index)
                for mode_index, mode in enumerate(config.stellar_population.imf_modes)
            ),
        )
        for z_index, redshift in enumerate(config.redshifts)
    )
    diagnostics = tuple(
        ModeRunDiagnostics(
            redshift=redshift,
            imf_mode=mode,
            sampling_seconds=1.0,
            sample_count=4,
            valid_sample_count=3,
            topheavy_source_fraction=0.0 if mode == "canonical" else 0.2,
            popiii_source_fraction=0.0,
            sfrd_msun_per_yr_per_mpc3=1.0e-3,
            popiii_sfrd_msun_per_yr_per_mpc3=0.0,
        )
        for redshift in config.redshifts
        for mode in config.stellar_population.imf_modes
    )
    return UVLFRunResult(
        config=config,
        redshifts=redshift_results,
        diagnostics=RunDiagnostics(total_seconds=1.0, mode_runs=diagnostics),
    )


def _write_artifact(config: UVLFRunConfig, path: Path) -> Path:
    provenance = ArtifactProvenance.for_config(
        config,
        code_revision="a" * 40,
        code_dirty=False,
        seed_namespace="auroralf.pipeline.v1",
        source_paths=(
            ("canonical_ssp", config.stellar_population.canonical_ssp_path),
            ("topheavy_ssp", config.stellar_population.topheavy_ssp_path),
        ),
        created_utc="2026-07-11T12:00:00Z",
    )
    return write_uvlf_artifact_atomic(
        UVLFArtifact(result=_result(config), provenance=provenance),
        path=path.resolve(),
        overwrite=False,
    )


def test_analysis_loader_uses_public_reader_without_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from auroralf.io import analysis

    config = _config(tmp_path)
    expected = _result(config)
    calls: list[tuple[Path, bool]] = []

    class Artifact:
        result = expected

    def fake_reader(path: Path, *, load_samples: bool) -> Artifact:
        calls.append((Path(path), load_samples))
        return Artifact()

    monkeypatch.setattr(analysis, "read_uvlf_artifact", fake_reader)
    assert analysis.load_uvlf_result(tmp_path / "model.h5") is expected
    assert calls == [(tmp_path / "model.h5", False)]


def test_typed_series_selection_preserves_observed_intrinsic_counts_and_edges(
    tmp_path: Path,
) -> None:
    from auroralf.io.analysis import select_mode_result

    result = _result(_config(tmp_path))
    series = select_mode_result(result, redshift=6.0, mode="canonical")

    np.testing.assert_array_equal(series.bin_edges_muv, [-24.0, -20.0, -16.0])
    np.testing.assert_array_equal(series.raw_counts, [2, 1])
    np.testing.assert_array_equal(
        series.phi_observed_per_mpc3_per_mag,
        [3.0e-6, 6.0e-6],
    )
    np.testing.assert_array_equal(
        series.phi_intrinsic_per_mpc3_per_mag,
        [2.0e-6, 4.0e-6],
    )
    with pytest.raises(ValueError, match="redshift.*9"):
        select_mode_result(result, redshift=9.0, mode="canonical")
    with pytest.raises(ValueError, match="mode.*z10_mild_topheavy"):
        select_mode_result(result, redshift=6.0, mode="z10_mild_topheavy")


@pytest.mark.parametrize("name", SCRIPT_CASES)
def test_each_script_model_loader_reads_hdf5_artifact(
    tmp_path: Path,
    name: str,
) -> None:
    relative_path = SCRIPT_CASES[name][0]
    module = _load_script(name, relative_path)
    artifact_path = _write_artifact(_config(tmp_path, run_id=name), tmp_path / f"{name}.h5")

    loaded = module._load_model_result(artifact_path)

    assert type(loaded) is UVLFRunResult
    assert loaded.config.run_id == name


@pytest.mark.parametrize("name", SCRIPT_CASES)
def test_each_script_cli_uses_hdf5_model_arguments(name: str) -> None:
    relative_path, new_options, old_options, argv = SCRIPT_CASES[name]
    module = _load_script(name, relative_path)

    args = module._parse_args(argv)
    help_text = inspect.getsource(module._parse_args)

    for option in new_options:
        assert option in help_text
    for option in old_options:
        assert option not in help_text
    for value in vars(args).values():
        if isinstance(value, (str, Path)) and str(value).endswith((".h5", ".npz")):
            assert str(value).endswith(".h5")


def test_multi_artifact_compatibility_allows_only_declared_physical_fields(
    tmp_path: Path,
) -> None:
    from auroralf.io.analysis import (
        BURST_CONFIG_DIFFERENCES,
        GATE_DELAY_BURST_CONFIG_DIFFERENCES,
        require_compatible_results,
    )

    reference = _result(_config(tmp_path, run_id="reference"))
    runtime_config = replace(
        reference.config,
        run_id="runtime-only",
        sampling=replace(
            reference.config.sampling,
            workers=2,
            mass_batch_size=2,
        ),
        output=OutputConfig((tmp_path / "runtime-only.h5").resolve()),
    )
    require_compatible_results(
        reference,
        _result(runtime_config),
        allowed_config_differences=frozenset(),
        context="runtime-only comparison",
    )
    burst_sf = replace(reference.config.star_formation, burst_scatter_dex=0.3)
    burst = _result(_config(tmp_path, run_id="burst", star_formation=burst_sf))
    require_compatible_results(
        reference,
        burst,
        allowed_config_differences=BURST_CONFIG_DIFFERENCES,
        context="burst comparison",
    )
    delay_sf = replace(burst_sf, enable_time_delay=not burst_sf.enable_time_delay)
    delay = _result(_config(tmp_path, run_id="delay", star_formation=delay_sf))
    require_compatible_results(
        reference,
        delay,
        allowed_config_differences=GATE_DELAY_BURST_CONFIG_DIFFERENCES,
        context="gate comparison",
    )
    wrong_sf = replace(reference.config.star_formation, efficiency_normalization=0.2)
    wrong = _result(_config(tmp_path, run_id="wrong", star_formation=wrong_sf))
    with pytest.raises(ValueError, match="efficiency_normalization"):
        require_compatible_results(
            reference,
            wrong,
            allowed_config_differences=BURST_CONFIG_DIFFERENCES,
            context="burst comparison",
        )


@pytest.mark.parametrize("axis", ["redshift", "mode", "bin"])
def test_multi_artifact_compatibility_rejects_axis_conflicts(
    tmp_path: Path,
    axis: str,
) -> None:
    from auroralf.io.analysis import require_compatible_results

    reference = _result(_config(tmp_path, run_id="reference"))
    if axis == "redshift":
        candidate_config = _config(
            tmp_path,
            run_id="candidate-z",
            redshifts=(6.0, 9.0),
        )
    elif axis == "mode":
        candidate_config = _config(
            tmp_path,
            run_id="candidate-mode",
            modes=("canonical",),
        )
    else:
        candidate_config = _config(
            tmp_path,
            run_id="candidate-bin",
            edges=(-24.0, -21.0, -18.0, -16.0),
        )
    candidate = _result(candidate_config)

    with pytest.raises(ValueError, match=axis):
        require_compatible_results(
            reference,
            candidate,
            allowed_config_differences=frozenset(),
            context="strict comparison",
        )


def test_script_loader_surfaces_public_reader_tamper_rejection(tmp_path: Path) -> None:
    module = _load_script("replot", SCRIPT_CASES["replot"][0])
    artifact_path = _write_artifact(_config(tmp_path), tmp_path / "tampered.h5")
    with h5py.File(artifact_path, "r+") as handle:
        handle["results/z=6/canonical/phi_observed_per_mpc3_per_mag"][0] *= 2.0

    with pytest.raises(ValueError, match="checksum|digest|tamper"):
        module._load_model_result(artifact_path)


def test_script_loader_rejects_incomplete_hdf5_through_public_reader(
    tmp_path: Path,
) -> None:
    module = _load_script("replot", SCRIPT_CASES["replot"][0])
    incomplete = tmp_path / "incomplete.h5"
    with h5py.File(incomplete, "w") as handle:
        handle.create_group("config")

    with pytest.raises((FileNotFoundError, ValueError), match="marker|missing|groups|incomplete"):
        module._load_model_result(incomplete)


@pytest.mark.parametrize("name", SCRIPT_CASES)
def test_model_loader_source_has_no_legacy_npz_model_path(name: str) -> None:
    relative_path, _, old_options, _ = SCRIPT_CASES[name]
    module = _load_script(name, relative_path)
    source = relative_path.read_text(encoding="utf-8")

    assert "load_uvlf_result" in inspect.getsource(module._load_model_result)
    assert "np.load" not in inspect.getsource(module._load_model_result)
    for option in old_options:
        assert option not in source
