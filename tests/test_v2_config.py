from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import tomllib

import numpy as np
import pytest

from auroralf.config import (
    CONFIG_SCHEMA_VERSION,
    CosmologyConfig,
    MAHConfig,
    MZRConfig,
    OutputConfig,
    RegulatorConfig,
    SamplingConfig,
    StarFormationConfig,
    StellarPopulationConfig,
    UVLFRunConfig,
)
from auroralf.mah import Cosmology
from auroralf.sfr import PopIIISFRParameters, SFRModelParameters


def _cosmology() -> CosmologyConfig:
    return CosmologyConfig(h0_km_s_mpc=67.4, omega_m=0.315, omega_b=0.049)


def _mah(**overrides: object) -> MAHConfig:
    values: dict[str, object] = {
        "backend": "mcbride",
        "sampler": "mcbride",
        "z_start_max": 50.0,
        "n_time_steps": 32,
        "tng_cache_path": None,
        "tng_mass_bin_width_dex": 0.15,
        "tng_min_candidates": 5,
        "tng_smoothing_myr": 0.0,
        "tng_time_grid_mode": "snapshot",
        "thesan_cache_path": None,
        "thesan_mass_bin_width_dex": 0.15,
        "thesan_min_candidates": 5,
        "thesan_smoothing_myr": 0.0,
        "thesan_time_grid_mode": "snapshot",
    }
    values.update(overrides)
    return MAHConfig(**values)  # type: ignore[arg-type]


def _mzr() -> MZRConfig:
    return MZRConfig(
        relation="fire2_highz",
        returned_fraction=0.4,
        scatter_dex=0.0,
        stellar_mass_floor_msun=1.0e6,
    )


def _regulator() -> RegulatorConfig:
    return RegulatorConfig(
        solar_metallicity_mass_fraction=0.0142,
        gas_fraction_norm=0.02,
        gas_fraction_mass_scale_msun=1.0e10,
        gas_fraction_mass_slope=0.0,
        gas_fraction_redshift_scale=10.0,
        gas_fraction_redshift_slope=0.0,
        returned_fraction=0.4,
        metal_yield=0.01,
        inflow_metallicity_zsun=0.0,
        metal_loading_norm=20.0,
        metal_loading_mass_scale_msun=1.0e10,
        metal_loading_mass_slope=-0.5,
        metal_loading_redshift_scale=10.0,
        metal_loading_redshift_slope=0.0,
        stellar_mass_floor_msun=0.0,
        metallicity_scatter_dex=0.0,
    )


def _star_formation(**overrides: object) -> StarFormationConfig:
    values: dict[str, object] = {
        "enable_time_delay": True,
        "efficiency_normalization": 0.12,
        "characteristic_halo_mass_msun": 10.0**11.7,
        "low_mass_slope": 0.66,
        "high_mass_slope": 0.65,
        "enable_archived_burst_scatter": False,
        "burst_scatter_dex": 0.0,
        "burst_scatter_correlation_timescale_myr": 20.0,
        "burst_scatter_mass_conserving": True,
        "enable_archived_metallicity": False,
        "metallicity_source": "none",
        "mzr": None,
        "regulator": None,
    }
    values.update(overrides)
    return StarFormationConfig(**values)  # type: ignore[arg-type]


def _stellar_population(tmp_path: Path, **overrides: object) -> StellarPopulationConfig:
    values: dict[str, object] = {
        "imf_modes": ("canonical",),
        "enable_archived_imf_gate": False,
        "canonical_ssp_path": tmp_path / "canonical.dat",
        "topheavy_ssp_path": tmp_path / "topheavy.hdf5",
        "topheavy_ssp_template_metallicity_zsun": 0.05,
        "historical_topheavy_redshift_min": 10.0,
        "source_redshift_gate_enabled": False,
        "growth_time_threshold_myr": 50.0,
        "birth_metallicity_topheavy_max_zsun": 0.05,
        "enable_popiii": False,
        "popiii_ssp_path": tmp_path / "popiii.dat",
        "popiii_efficiency": 1.0e-3,
        "popiii_pivot_halo_mass_msun": 1.0e7,
        "popiii_low_mass_slope": 0.0,
        "popiii_high_mass_slope": 0.0,
        "lw_background_j21": 0.0,
        "popiii_upper_mass_mode": "atomic",
        "popiii_upper_mass_msun": None,
    }
    values.update(overrides)
    return StellarPopulationConfig(**values)  # type: ignore[arg-type]


def _sampling(**overrides: object) -> SamplingConfig:
    values: dict[str, object] = {
        "mass_batch_size": 1,
        "n_halo_mass_samples": 2,
        "n_tracks_per_halo_mass": 3,
        "log10_halo_mass_min_msun": 9.0,
        "log10_halo_mass_max_msun": 12.0,
        "muv_bin_edges": (-24.0, -20.0, -16.0),
        "workers": 1,
        "mass_function_model": "hmf_reed07",
        "hmf_dlog10m": 0.01,
        "apply_dust": False,
    }
    values.update(overrides)
    return SamplingConfig(**values)  # type: ignore[arg-type]


def valid_run_config(tmp_path: Path, **overrides: object) -> UVLFRunConfig:
    values: dict[str, object] = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "run_id": "v2-test_01",
        "redshifts": (6.0, 8.0),
        "base_seed": 123,
        "cosmology": _cosmology(),
        "mah": _mah(),
        "star_formation": _star_formation(),
        "stellar_population": _stellar_population(tmp_path),
        "sampling": _sampling(),
        "output": OutputConfig(artifact_path=(tmp_path / "result.h5").resolve()),
    }
    values.update(overrides)
    return UVLFRunConfig(**values)  # type: ignore[arg-type]


def test_config_schema_version_and_frozen_model_conversion(tmp_path: Path) -> None:
    config = valid_run_config(tmp_path)

    assert CONFIG_SCHEMA_VERSION == "2.2.0"
    with pytest.raises(FrozenInstanceError):
        config.run_id = "changed"  # type: ignore[misc]
    cosmology = config.cosmology.to_model()
    assert type(cosmology) is Cosmology
    assert cosmology.h0_km_s_mpc == pytest.approx(67.4)
    assert cosmology.omega_lambda == pytest.approx(1.0 - 0.315)
    assert type(config.star_formation.to_model()) is SFRModelParameters
    assert type(config.stellar_population.to_popiii_model()) is PopIIISFRParameters


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (_cosmology, "h0_km_s_mpc"),
        (_mah, "z_start_max"),
        (_star_formation, "efficiency_normalization"),
        (_sampling, "hmf_dlog10m"),
    ],
)
@pytest.mark.parametrize("value", [True, np.bool_(False), np.nan, np.inf, "1.0"])
def test_numeric_fields_reject_boolean_nonfinite_and_strings(
    factory: object,
    field: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        factory(**{field: value})  # type: ignore[operator]


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (_mah, "n_time_steps"),
        (_mah, "tng_min_candidates"),
        (_sampling, "n_halo_mass_samples"),
        (_sampling, "workers"),
    ],
)
@pytest.mark.parametrize("value", [True, np.bool_(False), 1.5, "2"])
def test_integer_fields_reject_boolean_float_and_string(
    factory: object,
    field: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        factory(**{field: value})  # type: ignore[operator]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"backend": "bad"}, "backend"),
        ({"sampler": "bad"}, "sampler"),
        ({"tng_time_grid_mode": "bad"}, "tng_time_grid_mode"),
        ({"thesan_time_grid_mode": "bad"}, "thesan_time_grid_mode"),
        ({"backend": "tng"}, "tng_cache_path"),
        ({"backend": "thesan"}, "thesan_cache_path"),
    ],
)
def test_mah_validates_existing_modes_and_requires_active_cache(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _mah(**kwargs)


def test_mah_cache_path_is_not_required_to_exist_at_construction(tmp_path: Path) -> None:
    cache = (tmp_path / "missing-cache.h5").resolve()
    config = _mah(backend="tng", tng_cache_path=cache)

    assert config.tng_cache_path == cache
    assert not cache.exists()


@pytest.mark.parametrize(
    ("source", "mzr", "regulator"),
    [
        ("none", _mzr(), None),
        ("none", None, _regulator()),
        ("mzr", None, None),
        ("mzr", _mzr(), _regulator()),
        ("regulator", None, None),
        ("regulator", _mzr(), _regulator()),
    ],
)
def test_metallicity_source_requires_exact_matching_nested_config(
    source: str,
    mzr: MZRConfig | None,
    regulator: RegulatorConfig | None,
) -> None:
    with pytest.raises(ValueError, match="metallicity_source"):
        _star_formation(
            enable_archived_metallicity=True,
            metallicity_source=source,
            mzr=mzr,
            regulator=regulator,
        )


def test_archived_burst_scatter_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="burst scatter is archived"):
        _star_formation(burst_scatter_dex=0.2)

    config = _star_formation(
        enable_archived_burst_scatter=True,
        burst_scatter_dex=0.2,
    )
    assert config.enable_archived_burst_scatter is True
    assert config.burst_scatter_dex == pytest.approx(0.2)


def test_archived_metallicity_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="metallicity models are archived"):
        _star_formation(metallicity_source="mzr", mzr=_mzr())

    config = _star_formation(
        enable_archived_metallicity=True,
        metallicity_source="mzr",
        mzr=_mzr(),
    )
    assert config.enable_archived_metallicity is True
    assert config.metallicity_source == "mzr"


def test_nested_metallicity_configs_convert_to_current_models() -> None:
    assert _mzr().to_model().relation == "fire2_highz"
    assert _regulator().to_model().gas_fraction_mass_scale_msun == 1.0e10


@pytest.mark.parametrize(
    ("modes", "match"),
    [
        (("z10_mild_topheavy", "canonical"), "canonical.*first"),
        (("canonical", "canonical"), "unique"),
        (("canonical", "bad"), "imf_mode"),
        ((), "non-empty"),
        (["canonical"], "tuple"),
    ],
)
def test_stellar_population_validates_imf_mode_tuple(
    tmp_path: Path,
    modes: object,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _stellar_population(tmp_path, imf_modes=modes)


def test_variant_requires_positive_topheavy_template_metallicity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="topheavy_ssp_template_metallicity_zsun"):
        _stellar_population(
            tmp_path,
            imf_modes=("canonical", "z10_mild_topheavy"),
            topheavy_ssp_template_metallicity_zsun=None,
        )


def test_fixed_popiii_upper_mass_requires_positive_mass(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="popiii_upper_mass_msun"):
        _stellar_population(tmp_path, popiii_upper_mass_mode="fixed", popiii_upper_mass_msun=None)


def test_atomic_popiii_upper_mass_requires_none(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="popiii_upper_mass_msun.*None.*atomic"):
        _stellar_population(
            tmp_path,
            popiii_upper_mass_mode="atomic",
            popiii_upper_mass_msun=5.0e7,
        )


def test_cross_config_metallicity_gate_requires_source_for_variant(tmp_path: Path) -> None:
    stellar = _stellar_population(
        tmp_path,
        imf_modes=("canonical", "mah_burst_mild_topheavy"),
        enable_archived_imf_gate=True,
        birth_metallicity_topheavy_max_zsun=0.05,
    )
    with pytest.raises(ValueError, match="metallicity_source"):
        valid_run_config(tmp_path, stellar_population=stellar)


def test_run_config_rejects_archived_imf_gate_without_explicit_opt_in(
    tmp_path: Path,
) -> None:
    stellar = _stellar_population(
        tmp_path,
        imf_modes=("canonical", "mah_burst_mild_topheavy"),
        enable_archived_imf_gate=False,
        birth_metallicity_topheavy_max_zsun=None,
    )
    with pytest.raises(ValueError, match="IMF gate modes are archived"):
        valid_run_config(tmp_path, stellar_population=stellar)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"muv_bin_edges": (-20.0,)}, "muv_bin_edges"),
        ({"muv_bin_edges": (-20.0, -20.0)}, "strictly increasing"),
        ({"muv_bin_edges": [-20.0, -19.0]}, "tuple"),
        ({"log10_halo_mass_min_msun": 12.0}, "log10_halo_mass"),
        ({"mass_function_model": "massfunc_st"}, "mass_function_model"),
        ({"n_tracks_per_halo_mass": 0}, "n_tracks_per_halo_mass"),
        ({"mass_batch_size": 0}, "mass_batch_size"),
        ({"mass_batch_size": True}, "mass_batch_size"),
    ],
)
def test_sampling_rejects_invalid_ranges_and_models(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _sampling(**kwargs)


@pytest.mark.parametrize(
    "path",
    [Path("relative.h5"), Path("/tmp/result.txt")],
)
def test_output_path_must_be_absolute_hdf5(path: Path) -> None:
    with pytest.raises(ValueError, match="artifact_path"):
        OutputConfig(artifact_path=path)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"schema_version": "2.2.1"}, "schema_version"),
        ({"run_id": "bad id"}, "run_id"),
        ({"run_id": "x" * 129}, "run_id"),
        ({"redshifts": ()}, "redshifts"),
        ({"redshifts": (8.0, 6.0)}, "strictly increasing"),
        ({"redshifts": (6.0, np.inf)}, "redshifts"),
        ({"base_seed": True}, "base_seed"),
        ({"base_seed": -1}, "base_seed"),
        ({"base_seed": 2**64}, "base_seed"),
        ({"cosmology": object()}, "cosmology"),
    ],
)
def test_run_config_validates_identity_axes_and_nested_exact_types(
    tmp_path: Path,
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        valid_run_config(tmp_path, **overrides)


def test_run_config_requires_every_redshift_below_mah_start_redshift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="redshifts.*z_start_max"):
        valid_run_config(
            tmp_path,
            redshifts=(6.0, 8.0),
            mah=_mah(z_start_max=8.0),
        )


def _valid_toml() -> str:
    return """
schema_version = "2.2.0"
run_id = "toml-run"
redshifts = [6.0, 8.0]
base_seed = 42

[cosmology]
h0_km_s_mpc = 67.4
omega_m = 0.315
omega_b = 0.049

[mah]
backend = "mcbride"
sampler = "mcbride"
z_start_max = 50.0
n_time_steps = 32
tng_mass_bin_width_dex = 0.15
tng_min_candidates = 5
tng_smoothing_myr = 0.0
tng_time_grid_mode = "snapshot"
thesan_mass_bin_width_dex = 0.15
thesan_min_candidates = 5
thesan_smoothing_myr = 0.0
thesan_time_grid_mode = "snapshot"

[star_formation]
enable_time_delay = true
efficiency_normalization = 0.12
characteristic_halo_mass_msun = 5.011872336e11
low_mass_slope = 0.66
high_mass_slope = 0.65
enable_archived_burst_scatter = false
burst_scatter_dex = 0.0
burst_scatter_correlation_timescale_myr = 20.0
burst_scatter_mass_conserving = true
enable_archived_metallicity = false
metallicity_source = "none"

[stellar_population]
imf_modes = ["canonical"]
enable_archived_imf_gate = false
canonical_ssp_path = "ssp/canonical.dat"
topheavy_ssp_path = "ssp/topheavy.hdf5"
topheavy_ssp_template_metallicity_zsun = 0.05
historical_topheavy_redshift_min = 10.0
source_redshift_gate_enabled = false
growth_time_threshold_myr = 50.0
birth_metallicity_topheavy_max_zsun = 0.05
enable_popiii = false
popiii_ssp_path = "ssp/popiii.dat"
popiii_efficiency = 0.001
popiii_pivot_halo_mass_msun = 10000000.0
popiii_low_mass_slope = 0.0
popiii_high_mass_slope = 0.0
lw_background_j21 = 0.0
popiii_upper_mass_mode = "atomic"

[sampling]
mass_batch_size = 1
n_halo_mass_samples = 2
n_tracks_per_halo_mass = 3
log10_halo_mass_min_msun = 9.0
log10_halo_mass_max_msun = 12.0
muv_bin_edges = [-24.0, -20.0, -16.0]
workers = 1
mass_function_model = "hmf_reed07"
hmf_dlog10m = 0.01
apply_dust = false

[output]
artifact_path = "artifacts/result.h5"
""".strip()


def _write_toml(tmp_path: Path, text: str) -> Path:
    directory = tmp_path / "config-dir"
    directory.mkdir(exist_ok=True)
    path = directory / "run.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_from_toml_resolves_all_paths_relative_to_toml_parent(tmp_path: Path) -> None:
    path = _write_toml(tmp_path, _valid_toml())
    config = UVLFRunConfig.from_toml(path)

    parent = path.parent.resolve()
    assert config.stellar_population.canonical_ssp_path == parent / "ssp/canonical.dat"
    assert config.stellar_population.topheavy_ssp_path == parent / "ssp/topheavy.hdf5"
    assert config.stellar_population.popiii_ssp_path == parent / "ssp/popiii.dat"
    assert config.output.artifact_path == parent / "artifacts/result.h5"


def test_from_toml_resolves_active_backend_cache_relative_to_toml_parent(tmp_path: Path) -> None:
    text = _valid_toml().replace('backend = "mcbride"', 'backend = "tng"\ntng_cache_path = "cache/tng.h5"')
    path = _write_toml(tmp_path, text)

    config = UVLFRunConfig.from_toml(path)

    assert config.mah.tng_cache_path == path.parent.resolve() / "cache/tng.h5"


def test_from_toml_rejects_later_redshift_at_mah_start_boundary(tmp_path: Path) -> None:
    text = _valid_toml().replace("z_start_max = 50.0", "z_start_max = 8.0")
    path = _write_toml(tmp_path, text)

    with pytest.raises(ValueError, match="redshifts.*z_start_max"):
        UVLFRunConfig.from_toml(path)


def test_from_toml_rejects_duplicate_keys(tmp_path: Path) -> None:
    text = _valid_toml().replace(
        "h0_km_s_mpc = 67.4",
        "h0_km_s_mpc = 67.4\nh0_km_s_mpc = 70.0",
    )
    path = _write_toml(tmp_path, text)

    with pytest.raises(tomllib.TOMLDecodeError, match="overwrite"):
        UVLFRunConfig.from_toml(path)


def test_from_toml_reports_unknown_before_missing_in_same_table(tmp_path: Path) -> None:
    text = _valid_toml().replace(
        "h0_km_s_mpc = 67.4\n",
        "unknown_cosmology = 1\n",
    )
    path = _write_toml(tmp_path, text)

    with pytest.raises(ValueError, match="cosmology.unknown_cosmology"):
        UVLFRunConfig.from_toml(path)


def test_from_toml_symlink_resolves_relative_paths_from_real_config_parent(
    tmp_path: Path,
) -> None:
    real_directory = tmp_path / "real-config"
    real_directory.mkdir()
    real_path = real_directory / "run.toml"
    real_path.write_text(_valid_toml(), encoding="utf-8")
    link_directory = tmp_path / "link-config"
    link_directory.mkdir()
    link_path = link_directory / "run.toml"
    link_path.symlink_to(real_path)

    config = UVLFRunConfig.from_toml(link_path)

    assert config.stellar_population.canonical_ssp_path == (
        real_directory / "ssp/canonical.dat"
    ).resolve()


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (_valid_toml().replace("base_seed = 42\n", ""), "base_seed"),
        (_valid_toml().replace("h0_km_s_mpc = 67.4\n", ""), "cosmology.h0_km_s_mpc"),
        (_valid_toml().replace("schema_version = \"2.2.0\"\n", ""), "schema_version"),
    ],
)
def test_from_toml_reports_missing_required_keys_precisely(
    tmp_path: Path,
    text: str,
    match: str,
) -> None:
    path = _write_toml(tmp_path, text)
    with pytest.raises(ValueError, match=match):
        UVLFRunConfig.from_toml(path)


def test_sampling_config_requires_explicit_mass_batch_size() -> None:
    with pytest.raises(TypeError, match="mass_batch_size"):
        SamplingConfig(  # type: ignore[call-arg]
            n_halo_mass_samples=2,
            n_tracks_per_halo_mass=3,
            log10_halo_mass_min_msun=9.0,
            log10_halo_mass_max_msun=12.0,
            muv_bin_edges=(-24.0, -20.0, -16.0),
            workers=1,
            mass_function_model="hmf_reed07",
            hmf_dlog10m=0.01,
            apply_dust=False,
        )


def test_from_toml_requires_mass_batch_size(tmp_path: Path) -> None:
    text = _valid_toml().replace("mass_batch_size = 1\n", "")
    path = _write_toml(tmp_path, text)
    with pytest.raises(ValueError, match="sampling.mass_batch_size"):
        UVLFRunConfig.from_toml(path)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (_valid_toml() + "\nunknown_root = 1\n", "unknown_root"),
        (_valid_toml().replace("omega_b = 0.049", "omega_b = 0.049\nunknown_nested = 1"), "cosmology.unknown_nested"),
        (
            _valid_toml().replace(
                'metallicity_source = "none"',
                'metallicity_source = "mzr"\n\n[star_formation.mzr]\nrelation = "fire2_highz"\nreturned_fraction = 0.4\nscatter_dex = 0.0\nstellar_mass_floor_msun = 1e6\nunknown_deep = 1',
            ),
            "star_formation.mzr.unknown_deep",
        ),
    ],
)
def test_from_toml_rejects_unknown_keys_recursively(
    tmp_path: Path,
    text: str,
    match: str,
) -> None:
    path = _write_toml(tmp_path, text)
    with pytest.raises(ValueError, match=match):
        UVLFRunConfig.from_toml(path)


def test_from_toml_rejects_unknown_key_in_inactive_optional_table(tmp_path: Path) -> None:
    text = _valid_toml().replace(
        'metallicity_source = "none"',
        'metallicity_source = "none"\n\n[star_formation.regulator]\nunknown = 1',
    )
    path = _write_toml(tmp_path, text)

    with pytest.raises(ValueError, match="star_formation.regulator.unknown"):
        UVLFRunConfig.from_toml(path)


@pytest.mark.parametrize(
    "replacement",
    [
        'base_seed = "42"',
        "base_seed = true",
        'h0_km_s_mpc = "67.4"',
        "h0_km_s_mpc = true",
    ],
)
def test_from_toml_does_not_treat_strings_or_booleans_as_numeric(
    tmp_path: Path,
    replacement: str,
) -> None:
    if replacement.startswith("base_seed"):
        text = _valid_toml().replace("base_seed = 42", replacement)
        match = "base_seed"
    else:
        text = _valid_toml().replace("h0_km_s_mpc = 67.4", replacement)
        match = "h0_km_s_mpc"
    path = _write_toml(tmp_path, text)

    with pytest.raises((TypeError, ValueError), match=match):
        UVLFRunConfig.from_toml(path)


def test_config_module_uses_required_indexing_not_generic_get() -> None:
    source = (Path(__file__).resolve().parents[1] / "auroralf/config.py").read_text(encoding="utf-8")
    assert ".get(" not in source
