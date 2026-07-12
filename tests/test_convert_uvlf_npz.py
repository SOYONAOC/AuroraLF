from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings
import zipfile

import numpy as np
import pytest

from auroralf.io import read_uvlf_artifact
from auroralf.io.legacy import convert_legacy_uvlf_npz
from auroralf.uvlf.dust import compute_dust_attenuated_uvlf


MODES = ("canonical", "mah_burst_mild_topheavy")
REDSHIFTS = (6.0, 8.0)


def _config_text() -> str:
    return """
schema_version = "2.0.0"
run_id = "legacy-conversion"
redshifts = [6.0, 8.0]
base_seed = 123

[cosmology]
h0_km_s_mpc = 67.4
omega_m = 0.315
omega_b = 0.04897

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
burst_scatter_dex = 0.0
burst_scatter_correlation_timescale_myr = 20.0
burst_scatter_mass_conserving = true
metallicity_source = "mzr"

[star_formation.mzr]
relation = "fire2_highz"
returned_fraction = 0.4
scatter_dex = 0.0
stellar_mass_floor_msun = 1000000.0

[stellar_population]
imf_modes = ["canonical", "mah_burst_mild_topheavy"]
canonical_ssp_path = "sources/canonical.dat"
topheavy_ssp_path = "sources/topheavy.hdf5"
topheavy_ssp_template_metallicity_zsun = 0.05
historical_topheavy_redshift_min = 10.0
source_redshift_gate_enabled = false
growth_time_threshold_myr = 50.0
birth_metallicity_topheavy_max_zsun = 0.05
enable_popiii = false
popiii_ssp_path = "sources/popiii.dat"
popiii_efficiency = 0.001
popiii_pivot_halo_mass_msun = 10000000.0
popiii_low_mass_slope = 0.0
popiii_high_mass_slope = 0.0
lw_background_j21 = 0.0
popiii_upper_mass_mode = "atomic"

[sampling]
mass_batch_size = 1
n_halo_mass_samples = 2
n_tracks_per_halo_mass = 2
log10_halo_mass_min_msun = 9.0
log10_halo_mass_max_msun = 12.0
muv_bin_edges = [-24.0, -20.0, -16.0]
workers = 1
mass_function_model = "hmf_reed07"
hmf_dlog10m = 0.01
apply_dust = false

[output]
artifact_path = "configured-output.h5"
""".strip()


def _write_config(tmp_path: Path) -> Path:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "canonical.dat").write_bytes(b"canonical")
    (source_dir / "topheavy.hdf5").write_bytes(b"topheavy")
    (source_dir / "popiii.dat").write_bytes(b"popiii")
    path = tmp_path / "run.toml"
    path.write_text(_config_text(), encoding="utf-8")
    return path


def _z_tag(redshift: float) -> str:
    return f"z{str(float(redshift)).replace('.', 'p')}"


def _legacy_payload(tmp_path: Path) -> dict[str, np.ndarray]:
    edges = np.array([-24.0, -20.0, -16.0], dtype=np.float64)
    payload: dict[str, np.ndarray] = {
        "z_values": np.array(REDSHIFTS, dtype=np.float64),
        "mode_names": np.array(MODES),
        "variant_mode_names": np.array(MODES[1:]),
        "shared_bin_edges": edges,
        "workers": np.array([1], dtype=np.int64),
        "N_mass": np.array([2], dtype=np.int64),
        "n_tracks": np.array([2], dtype=np.int64),
        "base_seed": np.array([123], dtype=np.uint64),
        "z_start_max": np.array([50.0]),
        "n_grid": np.array([32], dtype=np.int64),
        "mah_backend": np.array(["mcbride"]),
        "sampler": np.array(["mcbride"]),
        "tng_mah_cache_path": np.array([""]),
        "tng_mass_bin_width_dex": np.array([0.15]),
        "tng_min_candidates": np.array([5], dtype=np.int64),
        "tng_smoothing_myr": np.array([0.0]),
        "tng_time_grid_mode": np.array(["snapshot"]),
        "thesan_mah_cache_path": np.array([""]),
        "thesan_mass_bin_width_dex": np.array([0.15]),
        "thesan_min_candidates": np.array([5], dtype=np.int64),
        "thesan_smoothing_myr": np.array([0.0]),
        "thesan_time_grid_mode": np.array(["snapshot"]),
        "bins_count": np.array([2], dtype=np.int64),
        "muv_min": np.array([-24.0]),
        "muv_max": np.array([-16.0]),
        "logM_min": np.array([9.0]),
        "logM_max": np.array([12.0]),
        "apply_dust": np.array([False]),
        "enable_time_delay": np.array([True]),
        "mass_function_model": np.array(["hmf_reed07"]),
        "epsilon_0": np.array([0.12]),
        "fstar_characteristic_mass": np.array([5.011872336e11]),
        "fstar_beta": np.array([0.66]),
        "fstar_gamma": np.array([0.65]),
        "burst_scatter_dex": np.array([0.0]),
        "burst_scatter_timescale_myr": np.array([20.0]),
        "burst_scatter_preserve_mean": np.array([True]),
        "burst_scatter_mass_conserving": np.array([True]),
        "metallicity_source": np.array(["mzr"]),
        "mzr_metallicity_enabled": np.array([True]),
        "regulator_metallicity_enabled": np.array([False]),
        "mzr_relation": np.array(["fire2_highz"]),
        "mzr_stellar_mass_floor": np.array([1.0e6]),
        "mzr_scatter_dex": np.array([0.0]),
        "mzr_returned_fraction": np.array([0.4]),
        "regulator_gas_fraction_norm": np.array([0.02]),
        "regulator_gas_fraction_mass_slope": np.array([0.0]),
        "regulator_gas_fraction_redshift_slope": np.array([0.0]),
        "regulator_yield": np.array([0.01]),
        "regulator_returned_fraction": np.array([0.4]),
        "regulator_inflow_metallicity_zsun": np.array([0.0]),
        "regulator_metal_loading_norm": np.array([20.0]),
        "regulator_metal_loading_mass_slope": np.array([-0.5]),
        "regulator_metal_loading_redshift_slope": np.array([0.0]),
        "regulator_metallicity_scatter_dex": np.array([0.0]),
        "metallicity_topheavy_max_zsun": np.array([0.05]),
        "canonical_ssp_file": np.array([str(tmp_path / "sources/canonical.dat")]),
        "topheavy_ssp_file": np.array([str(tmp_path / "sources/topheavy.hdf5")]),
        "topheavy_ssp_metallicity": np.array([0.05]),
        "enable_popiii": np.array([False]),
        "popiii_ssp_file": np.array([str(tmp_path / "sources/popiii.dat")]),
        "popiii_epsilon_star": np.array([0.001]),
        "popiii_mp": np.array([1.0e7]),
        "popiii_alpha_star": np.array([0.0]),
        "popiii_beta_star": np.array([0.0]),
        "popiii_upper_mass_mode": np.array(["atomic"]),
        "popiii_upper_mass_msun": np.array([np.nan]),
        "lw_background_j21": np.array([0.0]),
        "z_topheavy_min": np.array([10.0]),
        "source_redshift_gate_enabled": np.array([False]),
        "growth_time_threshold_myr": np.array([50.0]),
        "total_seconds": np.array([9.0]),
    }
    for redshift in REDSHIFTS:
        z_tag = _z_tag(redshift)
        payload[f"{z_tag}_bin_edges"] = edges.copy()
        payload[f"{z_tag}_bin_centers"] = np.array([-22.0, -18.0])
        payload[f"{z_tag}_bin_width"] = np.array([4.0, 4.0])
        payload[f"{z_tag}_base_seed"] = np.array([123], dtype=np.uint64)
        for mode_index, mode in enumerate(MODES):
            prefix = f"{z_tag}_{mode}"
            scale = 1.0 + mode_index + redshift / 100.0
            intrinsic = np.array([1.0e-5, 2.0e-5]) * scale
            intrinsic_sigma = np.array([1.0e-6, 2.0e-6]) * scale
            observed = intrinsic.copy()
            observed_sigma = intrinsic_sigma.copy()
            payload[f"{prefix}_intrinsic_phi"] = intrinsic
            payload[f"{prefix}_intrinsic_phi_sigma"] = intrinsic_sigma
            payload[f"{prefix}_phi"] = observed
            payload[f"{prefix}_phi_sigma_mc"] = observed_sigma
            payload[f"{prefix}_raw_counts"] = np.array([2, 1], dtype=np.int64)
            payload[f"{prefix}_weighted_counts"] = observed * 4.0
            payload[f"{prefix}_intrinsic_weighted_counts"] = intrinsic * 4.0
            payload[f"{prefix}_weight_squared_counts"] = (intrinsic_sigma * 4.0) ** 2
            payload[f"{prefix}_effective_counts"] = np.divide(
                payload[f"{prefix}_intrinsic_weighted_counts"] ** 2,
                payload[f"{prefix}_weight_squared_counts"],
                out=np.zeros(2),
                where=payload[f"{prefix}_weight_squared_counts"] > 0.0,
            )
            payload[f"{prefix}_sampling_seconds"] = np.array([1.5 + mode_index])
            payload[f"{prefix}_topheavy_source_fraction"] = np.array(
                [0.0 if mode == "canonical" else 0.25]
            )
            payload[f"{prefix}_topheavy_light_fraction_median"] = np.array([0.0])
            payload[f"{prefix}_popiii_source_fraction"] = np.array([0.0])
            payload[f"{prefix}_popiii_light_fraction_median"] = np.array([0.0])
            payload[f"{prefix}_final_gas_metallicity_zsun_median_by_mass"] = np.array(
                [np.nan, np.nan]
            )
            payload[
                f"{prefix}_birth_metallicity_zsun_starforming_median_by_mass"
            ] = np.array([np.nan, np.nan])
            if mode != "canonical":
                payload[f"{prefix}_phi_ratio_over_canonical"] = np.divide(
                    observed,
                    payload[f"{z_tag}_canonical_phi"],
                )
    return payload


def _manifest(tmp_path: Path) -> dict[str, object]:
    diagnostics = []
    for redshift in REDSHIFTS:
        for mode_index, mode in enumerate(MODES):
            diagnostics.append(
                {
                    "redshift": redshift,
                    "imf_mode": mode,
                    "sampling_seconds": 1.5 + mode_index,
                    "sample_count": 4,
                    "valid_sample_count": 3,
                    "topheavy_source_fraction": 0.0 if mode == "canonical" else 0.25,
                    "popiii_source_fraction": 0.0,
                    "sfrd_msun_per_yr_per_mpc3": 1.0e-3,
                    "popiii_sfrd_msun_per_yr_per_mpc3": 0.0,
                }
            )
    return {
        "schema_version": "auroralf.legacy_uvlf_manifest.v1",
        "code_revision": "a" * 40,
        "code_dirty": False,
        "seed_namespace": "auroralf.pipeline.v1",
        "created_utc": "2026-07-11T12:00:00Z",
        "sources": [
            {
                "label": "canonical_ssp",
                "path": str((tmp_path / "sources/canonical.dat").resolve()),
            },
            {
                "label": "topheavy_ssp",
                "path": str((tmp_path / "sources/topheavy.hdf5").resolve()),
            },
        ],
        "diagnostics": diagnostics,
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, np.ndarray]]:
    config = _write_config(tmp_path)
    payload = _legacy_payload(tmp_path)
    npz_path = tmp_path / "legacy.npz"
    np.savez(npz_path, **payload)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(tmp_path)), encoding="utf-8")
    return npz_path, config, manifest_path, payload


def _apply_dust_to_fixture(
    payload: dict[str, np.ndarray],
    *,
    tamper_variant: bool = False,
) -> None:
    payload["apply_dust"] = np.array([True])
    for redshift in REDSHIFTS:
        z_tag = _z_tag(redshift)
        centers = payload[f"{z_tag}_bin_centers"]
        widths = payload[f"{z_tag}_bin_width"]
        for mode in MODES:
            prefix = f"{z_tag}_{mode}"
            intrinsic = payload[f"{prefix}_intrinsic_phi"]
            intrinsic_sigma = payload[f"{prefix}_intrinsic_phi_sigma"]
            observed = np.asarray(
                compute_dust_attenuated_uvlf(
                    intrinsic_muv=centers,
                    intrinsic_phi=intrinsic,
                    z=redshift,
                    muv_obs=centers,
                )["phi_obs"],
                dtype=np.float64,
            )
            if tamper_variant and redshift == REDSHIFTS[0] and mode == MODES[1]:
                observed = observed * 1.01
            payload[f"{prefix}_phi"] = observed
            payload[f"{prefix}_weighted_counts"] = observed * widths
            payload[f"{prefix}_phi_sigma_mc"] = observed * np.divide(
                intrinsic_sigma,
                intrinsic,
                out=np.full_like(intrinsic_sigma, np.nan),
                where=intrinsic > 0.0,
            )
        for mode in MODES[1:]:
            payload[f"{z_tag}_{mode}_phi_ratio_over_canonical"] = np.divide(
                payload[f"{z_tag}_{mode}_phi"],
                payload[f"{z_tag}_canonical_phi"],
                out=np.full_like(payload[f"{z_tag}_{mode}_phi"], np.nan),
                where=payload[f"{z_tag}_canonical_phi"] > 0.0,
            )


def _rewrite_npz_member(
    path: Path,
    member_name: str,
    replacement: bytes,
    *,
    compression: int = zipfile.ZIP_STORED,
) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        members = [(info.filename, archive.read(info)) for info in archive.infolist()]
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members:
            archive.writestr(
                name,
                replacement if name == member_name else data,
                compress_type=compression if name == member_name else zipfile.ZIP_STORED,
            )


def _npy_header_bytes(
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
    fortran_order: bool = False,
) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream,
        {
            "shape": shape,
            "fortran_order": fortran_order,
            "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
        },
    )
    return stream.getvalue()


def _mark_zip_members_encrypted(path: Path) -> None:
    content = bytearray(path.read_bytes())
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        start = 0
        while True:
            index = content.find(signature, start)
            if index < 0:
                break
            flags = int.from_bytes(content[index + flag_offset : index + flag_offset + 2], "little")
            content[index + flag_offset : index + flag_offset + 2] = (flags | 1).to_bytes(2, "little")
            start = index + 4
    path.write_bytes(content)


def _patch_central_uncompressed_sizes(
    path: Path,
    sizes: list[int],
    *,
    compressed_sizes: list[int] | None = None,
) -> None:
    content = bytearray(path.read_bytes())
    positions: list[int] = []
    start = 0
    while True:
        index = content.find(b"PK\x01\x02", start)
        if index < 0:
            break
        positions.append(index)
        start = index + 4
    if len(sizes) == 1:
        positions = positions[:1]
    assert len(positions) == len(sizes)
    if compressed_sizes is not None:
        assert len(compressed_sizes) == len(sizes)
    for index, size in zip(positions, sizes, strict=True):
        content[index + 24 : index + 28] = size.to_bytes(4, "little")
    if compressed_sizes is not None:
        for index, size in zip(positions, compressed_sizes, strict=True):
            content[index + 20 : index + 24] = size.to_bytes(4, "little")
    path.write_bytes(content)


@pytest.mark.parametrize(
    "tamper",
    [
        "duplicate_member",
        "directory_member",
        "extra_member",
        "encrypted",
        "unsupported_compression",
        "fortran_order",
        "huge_shape",
        "huge_string_dtype",
        "truncated_data",
        "bad_header",
    ],
)
def test_converter_preflights_npz_zip_and_npy_headers_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    import auroralf.io.legacy as legacy

    npz_path, config_path, manifest_path, _ = _write_inputs(tmp_path)
    member = "z_values.npy"
    if tamper == "duplicate_member":
        with zipfile.ZipFile(npz_path, "a") as archive:
            original = archive.read(member)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(member, original)
    elif tamper == "directory_member":
        with zipfile.ZipFile(npz_path, "a") as archive:
            archive.writestr("nested/", b"")
    elif tamper == "extra_member":
        with zipfile.ZipFile(npz_path, "a") as archive:
            archive.writestr("notes.txt", b"not an npy member")
    elif tamper == "encrypted":
        _mark_zip_members_encrypted(npz_path)
    elif tamper == "unsupported_compression":
        with zipfile.ZipFile(npz_path, "r") as archive:
            original = archive.read(member)
        _rewrite_npz_member(
            npz_path,
            member,
            original,
            compression=zipfile.ZIP_BZIP2,
        )
    elif tamper == "fortran_order":
        replacement = _npy_header_bytes(
            shape=(2,),
            dtype=np.dtype(np.float64),
            fortran_order=True,
        ) + np.array(REDSHIFTS, dtype=np.float64).tobytes()
        _rewrite_npz_member(npz_path, member, replacement)
    elif tamper == "huge_shape":
        _rewrite_npz_member(
            npz_path,
            member,
            _npy_header_bytes(shape=(10**12,), dtype=np.dtype(np.float64)),
        )
    elif tamper == "huge_string_dtype":
        _rewrite_npz_member(
            npz_path,
            "mode_names.npy",
            _npy_header_bytes(shape=(2,), dtype=np.dtype("U1000000")),
        )
    elif tamper == "truncated_data":
        with zipfile.ZipFile(npz_path, "r") as archive:
            original = archive.read(member)
        _rewrite_npz_member(npz_path, member, original[:-1])
    else:
        _rewrite_npz_member(npz_path, member, b"not-a-valid-npy-header")

    def forbidden_materialization(*args: object, **kwargs: object) -> object:
        raise AssertionError("np.load must not run before NPZ/NPY preflight succeeds")

    monkeypatch.setattr(legacy.np, "load", forbidden_materialization)
    with pytest.raises((TypeError, ValueError), match="NPZ|NPY|member|compression|encrypted|corrupt|shape|dtype|Fortran|data"):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            (tmp_path / "unsafe.h5").resolve(),
            overwrite=False,
        )


def test_converter_accepts_deflated_npz_after_header_preflight(tmp_path: Path) -> None:
    npz_path, config_path, manifest_path, payload = _write_inputs(tmp_path)
    np.savez_compressed(npz_path, **payload)

    convert_legacy_uvlf_npz(
        npz_path,
        config_path,
        manifest_path,
        (tmp_path / "compressed.h5").resolve(),
        overwrite=False,
    )


@pytest.mark.parametrize("payload_tamper", ["truncated", "oversized"])
def test_converter_reports_corrupt_npz_for_inconsistent_npy_payload_bytes(
    tmp_path: Path,
    payload_tamper: str,
) -> None:
    npz_path, config_path, manifest_path, _ = _write_inputs(tmp_path)
    member = "z_values.npy"
    with zipfile.ZipFile(npz_path, "r") as archive:
        original = archive.read(member)
    replacement = original[:-1] if payload_tamper == "truncated" else original + b"\x00"
    _rewrite_npz_member(npz_path, member, replacement)

    with pytest.raises(ValueError, match="corrupt legacy NPZ"):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            (tmp_path / "corrupt-payload.h5").resolve(),
            overwrite=False,
        )


@pytest.mark.parametrize(
    ("version", "header_length"),
    [
        ((1, 0), 50_000),
        ((2, 0), 50 * 1024 * 1024),
        ((2, 0), 0xFFFFFFFF),
    ],
)
def test_converter_rejects_npy_header_length_before_numpy_reads_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: tuple[int, int],
    header_length: int,
) -> None:
    npz_path, config_path, manifest_path, _ = _write_inputs(tmp_path)
    length_width = 2 if version == (1, 0) else 4
    malicious = (
        b"\x93NUMPY"
        + bytes(version)
        + header_length.to_bytes(length_width, "little")
    )
    _rewrite_npz_member(npz_path, "z_values.npy", malicious)
    real_read = zipfile.ZipExtFile.read
    requested_sizes: list[int] = []

    def bounded_read(self: zipfile.ZipExtFile, size: int = -1) -> bytes:
        requested_sizes.append(size)
        if size > 32:
            raise AssertionError("oversized header read reached ZipExtFile")
        return real_read(self, size)

    monkeypatch.setattr(zipfile.ZipExtFile, "read", bounded_read)
    with pytest.raises(ValueError, match="NPY header.*too large"):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            (tmp_path / "header-length.h5").resolve(),
            overwrite=False,
        )
    assert max(requested_sizes) <= 32


@pytest.mark.parametrize("resource", ["member", "total", "compression_ratio"])
def test_converter_rejects_npz_resource_abuse_before_header_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource: str,
) -> None:
    import auroralf.io.legacy as legacy

    npz_path, config_path, manifest_path, _ = _write_inputs(tmp_path)
    if resource == "member":
        boundary = 64 * 1024 * 1024 + 1
        _patch_central_uncompressed_sizes(
            npz_path,
            [boundary],
            compressed_sizes=[boundary],
        )
    elif resource == "total":
        with zipfile.ZipFile(npz_path, "r") as archive:
            member_count = len(archive.infolist())
        per_member = (256 * 1024 * 1024) // member_count + 1
        _patch_central_uncompressed_sizes(
            npz_path,
            [per_member] * member_count,
            compressed_sizes=[per_member] * member_count,
        )
    else:
        with zipfile.ZipFile(npz_path, "r") as archive:
            original = archive.read("z_values.npy")
        quality_like_payload = original + bytes(6_420_000)
        _rewrite_npz_member(
            npz_path,
            "z_values.npy",
            quality_like_payload,
            compression=zipfile.ZIP_DEFLATED,
        )
        with zipfile.ZipFile(npz_path, "r") as archive:
            info = archive.getinfo("z_values.npy")
        assert info.file_size > 6_400_000
        assert info.compress_size < 17_800

    def forbidden_header(*args: object, **kwargs: object) -> object:
        raise AssertionError("resource abuse reached NPY header parsing")

    monkeypatch.setattr(legacy, "_read_npy_header", forbidden_header)
    with pytest.raises(ValueError, match="member.*limit|total.*limit|compression ratio"):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            (tmp_path / "resource-abuse.h5").resolve(),
            overwrite=False,
        )


def test_converter_rejects_config_derived_npz_layout_over_resource_limit(
    tmp_path: Path,
) -> None:
    npz_path, config_path, manifest_path, _ = _write_inputs(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "n_halo_mass_samples = 2",
            "n_halo_mass_samples = 100000000",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="config-derived.*layout.*limit"):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            (tmp_path / "oversized-layout.h5").resolve(),
            overwrite=False,
        )


@pytest.mark.parametrize("tamper", ["root_duplicate", "nested_duplicate", "nan", "infinity"])
def test_converter_rejects_noncanonical_json_before_schema_validation(
    tmp_path: Path,
    tamper: str,
) -> None:
    npz_path, config_path, manifest_path, _ = _write_inputs(tmp_path)
    text = manifest_path.read_text(encoding="utf-8")
    if tamper == "root_duplicate":
        text = text.replace(
            '"schema_version":',
            '"schema_version": "auroralf.legacy_uvlf_manifest.v1", "schema_version":',
            1,
        )
    elif tamper == "nested_duplicate":
        text = text.replace(
            '"label": "canonical_ssp"',
            '"label": "canonical_ssp", "label": "canonical_ssp"',
            1,
        )
    elif tamper == "nan":
        text = text.replace('"sample_count": 4', '"sample_count": NaN', 1)
    else:
        text = text.replace('"sample_count": 4', '"sample_count": Infinity', 1)
    manifest_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="JSON.*duplicate|JSON.*constant"):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            (tmp_path / "bad-json.h5").resolve(),
            overwrite=False,
        )


@pytest.mark.parametrize("tamper", ["invalid_utf8", "too_large"])
def test_converter_rejects_unsafe_json_bytes(tmp_path: Path, tamper: str) -> None:
    npz_path, config_path, manifest_path, _ = _write_inputs(tmp_path)
    if tamper == "invalid_utf8":
        manifest_path.write_bytes(b"\xff")
    else:
        manifest_path.write_bytes(b" " * (8 * 1024 * 1024 + 1))

    with pytest.raises(ValueError, match="JSON.*UTF-8|JSON.*too large"):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            (tmp_path / "unsafe-json.h5").resolve(),
            overwrite=False,
        )


def test_legacy_conversion_roundtrip_maps_only_provable_fields(tmp_path: Path) -> None:
    npz_path, config_path, manifest_path, payload = _write_inputs(tmp_path)
    output = (tmp_path / "converted.h5").resolve()

    assert convert_legacy_uvlf_npz(
        npz_path,
        config_path,
        manifest_path,
        output,
        overwrite=False,
    ) == output
    artifact = read_uvlf_artifact(output)

    mode = artifact.result.redshifts[0].imf_modes[0]
    prefix = "z6p0_canonical"
    np.testing.assert_array_equal(
        mode.weighted_counts_per_mpc3,
        payload[f"{prefix}_intrinsic_weighted_counts"],
    )
    np.testing.assert_array_equal(
        mode.weighted_count_sigma_per_mpc3,
        np.sqrt(payload[f"{prefix}_weight_squared_counts"]),
    )
    np.testing.assert_array_equal(
        mode.phi_intrinsic_per_mpc3_per_mag,
        payload[f"{prefix}_intrinsic_phi"],
    )
    np.testing.assert_array_equal(
        mode.phi_observed_per_mpc3_per_mag,
        payload[f"{prefix}_phi"],
    )
    assert artifact.result.diagnostics.total_seconds == 9.0
    assert artifact.result.diagnostics.mode_runs[0].sample_count == 4
    checksums = {item.label: item for item in artifact.provenance.source_checksums}
    assert set(checksums) == {
        "canonical_ssp",
        "topheavy_ssp",
        "legacy_uvlf_npz",
        "conversion_config_toml",
        "conversion_manifest",
    }
    assert checksums["legacy_uvlf_npz"].path == npz_path.resolve()


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("missing", "missing.*key"),
        ("unknown", "unknown.*key"),
        ("partial_axis", "missing.*key|partial"),
        ("wrong_shape", "shape|scalar"),
        ("wrong_dtype", "dtype|integer"),
        ("nonfinite", "finite|infinity"),
        ("object", "allow_pickle|object"),
        ("inactive_global_dtype", "dtype"),
        ("sigma_semantic", "weight_squared_counts.*inconsistent"),
        ("observed_sigma_semantic", "phi_sigma_mc.*inconsistent"),
    ],
)
def test_converter_rejects_legacy_npz_schema_and_numeric_tampering(
    tmp_path: Path,
    tamper: str,
    match: str,
) -> None:
    npz_path, config_path, manifest_path, payload = _write_inputs(tmp_path)
    if tamper == "missing":
        del payload["z6p0_canonical_raw_counts"]
    elif tamper == "unknown":
        payload["unknown"] = np.array([1])
    elif tamper == "partial_axis":
        for key in tuple(payload):
            if key.startswith("z8p0_mah_burst_mild_topheavy"):
                del payload[key]
    elif tamper == "wrong_shape":
        payload["base_seed"] = np.array([123, 124], dtype=np.uint64)
    elif tamper == "wrong_dtype":
        payload["z6p0_canonical_raw_counts"] = np.array([2.0, 1.0])
    elif tamper == "nonfinite":
        payload["z6p0_canonical_intrinsic_phi"][0] = np.inf
    elif tamper == "object":
        payload["z6p0_canonical_raw_counts"] = np.array([object()], dtype=object)
    elif tamper == "inactive_global_dtype":
        payload["regulator_yield"] = np.array([1], dtype=np.int64)
    elif tamper == "sigma_semantic":
        payload["z6p0_canonical_weight_squared_counts"] *= 2.0
    else:
        payload["z6p0_canonical_phi_sigma_mc"] *= 2.0
    np.savez(npz_path, **payload)

    with pytest.raises((TypeError, ValueError), match=match):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            (tmp_path / "bad.h5").resolve(),
            overwrite=False,
        )


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("base_seed", np.array([124], dtype=np.uint64), "base_seed"),
        ("z_values", np.array([6.0, 9.0]), "redshift"),
        ("mode_names", np.array(["canonical"]), "mode"),
        ("shared_bin_edges", np.array([-24.0, -19.0, -16.0]), "bin"),
        ("epsilon_0", np.array([0.2]), "efficiency|epsilon"),
    ],
)
def test_converter_rejects_config_npz_identity_and_physics_mismatch(
    tmp_path: Path,
    key: str,
    value: np.ndarray,
    match: str,
) -> None:
    npz_path, config_path, manifest_path, payload = _write_inputs(tmp_path)
    payload[key] = value
    np.savez(npz_path, **payload)

    with pytest.raises((TypeError, ValueError), match=match):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            (tmp_path / "mismatch.h5").resolve(),
            overwrite=False,
        )


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("no_dust_phi", "apply_dust=False|observed phi"),
        ("effective_counts", "effective_counts"),
        ("negative_gas_metallicity", "metallicity.*non-negative"),
        ("negative_birth_metallicity", "metallicity.*non-negative"),
        ("infinite_metallicity", "infinity"),
        ("sample_count", "sample_count.*N_mass.*n_tracks"),
        ("raw_exceeds_valid", "raw_counts.*valid_sample_count"),
        ("valid_exceeds_sample", "valid_sample_count|sample_count"),
    ],
)
def test_converter_rejects_unprovable_physical_invariants(
    tmp_path: Path,
    tamper: str,
    match: str,
) -> None:
    npz_path, config_path, manifest_path, payload = _write_inputs(tmp_path)
    manifest = _manifest(tmp_path)
    prefix = "z6p0_mah_burst_mild_topheavy"
    if tamper == "no_dust_phi":
        payload[f"{prefix}_phi"] *= 0.8
        payload[f"{prefix}_weighted_counts"] = (
            payload[f"{prefix}_phi"] * payload["z6p0_bin_width"]
        )
        payload[f"{prefix}_phi_sigma_mc"] = payload[f"{prefix}_phi"] * np.divide(
            payload[f"{prefix}_intrinsic_phi_sigma"],
            payload[f"{prefix}_intrinsic_phi"],
        )
        payload[f"{prefix}_phi_ratio_over_canonical"] = np.divide(
            payload[f"{prefix}_phi"],
            payload["z6p0_canonical_phi"],
        )
    elif tamper == "effective_counts":
        payload[f"{prefix}_effective_counts"] *= 2.0
    elif tamper == "negative_gas_metallicity":
        payload[f"{prefix}_final_gas_metallicity_zsun_median_by_mass"][0] = -0.1
    elif tamper == "negative_birth_metallicity":
        payload[f"{prefix}_birth_metallicity_zsun_starforming_median_by_mass"][0] = -0.1
    elif tamper == "infinite_metallicity":
        payload[f"{prefix}_final_gas_metallicity_zsun_median_by_mass"][0] = np.inf
    elif tamper == "sample_count":
        manifest["diagnostics"][0]["sample_count"] = 5
    elif tamper == "raw_exceeds_valid":
        payload[f"{prefix}_raw_counts"] = np.array([3, 1], dtype=np.int64)
    else:
        manifest["diagnostics"][0]["valid_sample_count"] = 5
    np.savez(npz_path, **payload)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=match):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            (tmp_path / "physical-invariant.h5").resolve(),
            overwrite=False,
        )


def test_converter_validates_positive_dust_reconstruction(tmp_path: Path) -> None:
    npz_path, config_path, manifest_path, payload = _write_inputs(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "apply_dust = false",
            "apply_dust = true",
        ),
        encoding="utf-8",
    )
    _apply_dust_to_fixture(payload)
    np.savez(npz_path, **payload)

    convert_legacy_uvlf_npz(
        npz_path,
        config_path,
        manifest_path,
        (tmp_path / "dust-positive.h5").resolve(),
        overwrite=False,
    )


def test_converter_rejects_dust_phi_not_recomputed_from_intrinsic(
    tmp_path: Path,
) -> None:
    npz_path, config_path, manifest_path, payload = _write_inputs(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "apply_dust = false",
            "apply_dust = true",
        ),
        encoding="utf-8",
    )
    _apply_dust_to_fixture(payload, tamper_variant=True)
    np.savez(npz_path, **payload)

    with pytest.raises(ValueError, match="dust.*observed phi|observed phi.*dust"):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            (tmp_path / "dust-bad.h5").resolve(),
            overwrite=False,
        )


def _activate_required_source(
    tmp_path: Path,
    config_path: Path,
    payload: dict[str, np.ndarray],
    source_kind: str,
) -> tuple[str, Path]:
    if source_kind == "canonical":
        return "canonical_ssp", (tmp_path / "sources/canonical.dat").resolve()
    if source_kind == "topheavy":
        return "topheavy_ssp", (tmp_path / "sources/topheavy.hdf5").resolve()
    if source_kind == "popiii":
        config_path.write_text(
            config_path.read_text(encoding="utf-8").replace(
                "enable_popiii = false",
                "enable_popiii = true",
            ),
            encoding="utf-8",
        )
        payload["enable_popiii"] = np.array([True])
        return "popiii_ssp", (tmp_path / "sources/popiii.dat").resolve()
    cache = (tmp_path / f"sources/{source_kind}-cache.npz").resolve()
    cache.write_bytes(f"{source_kind}-cache".encode("ascii"))
    config_text = config_path.read_text(encoding="utf-8").replace(
        'backend = "mcbride"',
        f'backend = "{source_kind}"\n{source_kind}_cache_path = "sources/{source_kind}-cache.npz"',
    )
    config_path.write_text(config_text, encoding="utf-8")
    payload["mah_backend"] = np.array([source_kind])
    payload[f"{source_kind}_mah_cache_path"] = np.array([str(cache)])
    return f"{source_kind}_mah_cache", cache


@pytest.mark.parametrize(
    "source_kind",
    ["canonical", "topheavy", "popiii", "tng", "thesan"],
)
def test_converter_requires_every_active_science_source_in_manifest(
    tmp_path: Path,
    source_kind: str,
) -> None:
    npz_path, config_path, manifest_path, payload = _write_inputs(tmp_path)
    manifest = _manifest(tmp_path)
    _, required_path = _activate_required_source(
        tmp_path,
        config_path,
        payload,
        source_kind,
    )
    manifest["sources"] = [
        source
        for source in manifest["sources"]
        if Path(source["path"]).resolve() != required_path
    ]
    np.savez(npz_path, **payload)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest sources.*active|required.*source"):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            (tmp_path / "missing-active-source.h5").resolve(),
            overwrite=False,
        )


@pytest.mark.parametrize("source_kind", ["popiii", "tng", "thesan"])
def test_converter_accepts_manifest_covering_optional_active_source(
    tmp_path: Path,
    source_kind: str,
) -> None:
    npz_path, config_path, manifest_path, payload = _write_inputs(tmp_path)
    manifest = _manifest(tmp_path)
    label, required_path = _activate_required_source(
        tmp_path,
        config_path,
        payload,
        source_kind,
    )
    manifest["sources"].append({"label": label, "path": str(required_path)})
    np.savez(npz_path, **payload)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    convert_legacy_uvlf_npz(
        npz_path,
        config_path,
        manifest_path,
        (tmp_path / "covered-active-source.h5").resolve(),
        overwrite=False,
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "unknown",
        "missing",
        "cross_mismatch",
        "diagnostic_unknown",
        "source_unknown",
        "reserved_source",
        "diagnostic_missing_axis",
    ],
)
def test_converter_rejects_manifest_schema_and_cross_check_mismatch(
    tmp_path: Path,
    tamper: str,
) -> None:
    npz_path, config_path, manifest_path, _ = _write_inputs(tmp_path)
    manifest = _manifest(tmp_path)
    if tamper == "unknown":
        manifest["unknown"] = 1
    elif tamper == "missing":
        del manifest["code_revision"]
    elif tamper == "cross_mismatch":
        manifest["diagnostics"][0]["sampling_seconds"] = 99.0
    elif tamper == "diagnostic_unknown":
        manifest["diagnostics"][0]["unknown"] = 1
    elif tamper == "source_unknown":
        manifest["sources"][0]["unknown"] = 1
    elif tamper == "diagnostic_missing_axis":
        manifest["diagnostics"].pop()
    else:
        manifest["sources"][0]["label"] = "legacy_uvlf_npz"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        (TypeError, ValueError),
        match="unknown|missing|sampling_seconds|reserved|exactly cover",
    ):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            (tmp_path / "manifest-bad.h5").resolve(),
            overwrite=False,
        )


def test_converter_rejects_no_dust_empty_bin_sigma_not_equal_intrinsic(
    tmp_path: Path,
) -> None:
    npz_path, config_path, manifest_path, payload = _write_inputs(tmp_path)
    prefix = "z6p0_canonical"
    payload[f"{prefix}_intrinsic_phi"][0] = 0.0
    payload[f"{prefix}_intrinsic_phi_sigma"][0] = 0.0
    payload[f"{prefix}_phi"][0] = 0.0
    payload[f"{prefix}_phi_sigma_mc"][0] = np.nan
    payload[f"{prefix}_weighted_counts"][0] = 0.0
    payload[f"{prefix}_intrinsic_weighted_counts"][0] = 0.0
    payload[f"{prefix}_weight_squared_counts"][0] = 0.0
    payload[f"{prefix}_effective_counts"][0] = 0.0
    for mode in MODES[1:]:
        payload[f"z6p0_{mode}_phi_ratio_over_canonical"][0] = np.nan
    np.savez(npz_path, **payload)
    with pytest.raises(ValueError, match="apply_dust=False|observed sigma"):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            (tmp_path / "empty-bin.h5").resolve(),
            overwrite=False,
        )


def test_conversion_failure_and_overwrite_policy_preserve_old_output(
    tmp_path: Path,
) -> None:
    npz_path, config_path, manifest_path, payload = _write_inputs(tmp_path)
    output = (tmp_path / "existing.h5").resolve()
    convert_legacy_uvlf_npz(
        npz_path,
        config_path,
        manifest_path,
        output,
        overwrite=False,
    )
    old_artifact = output.read_bytes()
    old_marker = output.with_name(output.name + ".complete").read_bytes()

    with pytest.raises(FileExistsError):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            output,
            overwrite=False,
        )
    del payload["z6p0_canonical_raw_counts"]
    np.savez(npz_path, **payload)
    with pytest.raises(ValueError):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            output,
            overwrite=True,
        )
    assert output.read_bytes() == old_artifact
    assert output.with_name(output.name + ".complete").read_bytes() == old_marker


def test_converted_artifact_detects_manifest_source_change(tmp_path: Path) -> None:
    npz_path, config_path, manifest_path, _ = _write_inputs(tmp_path)
    output = (tmp_path / "source-change.h5").resolve()
    convert_legacy_uvlf_npz(
        npz_path,
        config_path,
        manifest_path,
        output,
        overwrite=False,
    )
    (tmp_path / "sources/canonical.dat").write_bytes(b"changed-source")

    with pytest.raises(ValueError, match="source.*checksum|source.*size"):
        read_uvlf_artifact(output)


def _manifest_toml(manifest: dict[str, object]) -> str:
    lines = [
        f'schema_version = "{manifest["schema_version"]}"',
        f'code_revision = "{manifest["code_revision"]}"',
        f'code_dirty = {str(manifest["code_dirty"]).lower()}',
        f'seed_namespace = "{manifest["seed_namespace"]}"',
        f'created_utc = "{manifest["created_utc"]}"',
    ]
    for source in manifest["sources"]:
        lines.extend(
            [
                "",
                "[[sources]]",
                f'label = "{source["label"]}"',
                f'path = "{source["path"]}"',
            ]
        )
    for diagnostic in manifest["diagnostics"]:
        lines.extend(["", "[[diagnostics]]"])
        for key, value in diagnostic.items():
            encoded = f'"{value}"' if isinstance(value, str) else str(value).lower()
            lines.append(f"{key} = {encoded}")
    return "\n".join(lines) + "\n"


def test_converter_accepts_exact_toml_manifest(tmp_path: Path) -> None:
    npz_path, config_path, _, _ = _write_inputs(tmp_path)
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(_manifest_toml(_manifest(tmp_path)), encoding="utf-8")
    output = (tmp_path / "toml-manifest.h5").resolve()

    convert_legacy_uvlf_npz(
        npz_path,
        config_path,
        manifest_path,
        output,
        overwrite=False,
    )

    assert read_uvlf_artifact(output).result.config.run_id == "legacy-conversion"


@pytest.mark.parametrize("tamper", ["invalid_utf8", "too_large"])
def test_converter_rejects_unsafe_toml_manifest_bytes(
    tmp_path: Path,
    tamper: str,
) -> None:
    npz_path, config_path, _, _ = _write_inputs(tmp_path)
    manifest_path = tmp_path / "manifest.toml"
    if tamper == "invalid_utf8":
        manifest_path.write_bytes(b"\xff")
    else:
        manifest_path.write_bytes(b" " * (8 * 1024 * 1024 + 1))

    with pytest.raises(ValueError, match="manifest TOML.*UTF-8|manifest TOML.*too large"):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            (tmp_path / "unsafe-toml.h5").resolve(),
            overwrite=False,
        )


def test_converter_materializes_npz_from_preflighted_file_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auroralf.io.legacy as legacy

    npz_path, config_path, manifest_path, _ = _write_inputs(tmp_path)
    real_load = legacy.np.load
    file_objects: list[object] = []

    def guarded_load(source: object, *args: object, **kwargs: object) -> object:
        if isinstance(source, (str, bytes, Path)):
            raise AssertionError("np.load reopened the NPZ path")
        file_objects.append(source)
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(legacy.np, "load", guarded_load)
    convert_legacy_uvlf_npz(
        npz_path,
        config_path,
        manifest_path,
        (tmp_path / "stable-fd.h5").resolve(),
        overwrite=False,
    )

    assert len(file_objects) == 1
    assert hasattr(file_objects[0], "fileno")


@pytest.mark.parametrize("replacement_kind", ["identical", "sparse_bomb"])
def test_converter_rejects_npz_path_replacement_after_preflight_without_reopening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    import auroralf.io.legacy as legacy

    npz_path, config_path, manifest_path, _ = _write_inputs(tmp_path)
    original_bytes = npz_path.read_bytes()
    real_preflight = legacy._preflight_npz
    real_load = legacy.np.load
    path_load_calls = 0

    def replace_path_after_preflight(*args: object, **kwargs: object) -> object:
        result = real_preflight(*args, **kwargs)
        replacement = tmp_path / f"{replacement_kind}.npz"
        if replacement_kind == "identical":
            replacement.write_bytes(original_bytes)
        else:
            with replacement.open("wb") as handle:
                handle.seek(100 * 1024 * 1024 - 1)
                handle.write(b"\x00")
        replacement.replace(npz_path)
        return result

    def forbid_path_load(source: object, *args: object, **kwargs: object) -> object:
        nonlocal path_load_calls
        if isinstance(source, (str, bytes, Path)):
            path_load_calls += 1
            raise AssertionError("replacement path reached np.load")
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(legacy, "_preflight_npz", replace_path_after_preflight)
    monkeypatch.setattr(legacy.np, "load", forbid_path_load)
    with pytest.raises(ValueError, match="NPZ.*changed|identity|checksum|size"):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            (tmp_path / "replaced-path.h5").resolve(),
            overwrite=False,
        )
    assert path_load_calls == 0


@pytest.mark.parametrize("input_name", ["npz", "config", "manifest"])
def test_conversion_reverifies_bound_inputs_before_atomic_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
) -> None:
    import auroralf.io.legacy as legacy

    npz_path, config_path, manifest_path, _ = _write_inputs(tmp_path)
    selected = {
        "npz": npz_path,
        "config": config_path,
        "manifest": manifest_path,
    }[input_name]
    real_write = legacy.write_uvlf_artifact_atomic

    def replace_input_before_write(*args: object, **kwargs: object) -> Path:
        replacement = tmp_path / f"replacement-{input_name}"
        replacement.write_bytes(b"changed conversion input")
        replacement.replace(selected)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(legacy, "write_uvlf_artifact_atomic", replace_input_before_write)

    with pytest.raises(ValueError, match="source.*checksum|source.*size"):
        convert_legacy_uvlf_npz(
            npz_path,
            config_path,
            manifest_path,
            (tmp_path / "changed-input.h5").resolve(),
            overwrite=False,
        )
    assert not (tmp_path / "changed-input.h5").exists()


def test_converter_cli_successfully_writes_public_artifact(tmp_path: Path) -> None:
    npz_path, config_path, manifest_path, _ = _write_inputs(tmp_path)
    output = (tmp_path / "cli-output.h5").resolve()
    script = Path("scripts/data/convert_uvlf_npz_to_v2_hdf5.py").resolve()

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--npz",
            str(npz_path),
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"converted_hdf5={output}" in result.stdout
    assert read_uvlf_artifact(output).result.config.run_id == "legacy-conversion"


def test_converter_script_help_import_and_nonzero_failure_have_no_side_effects(
    tmp_path: Path,
) -> None:
    script = Path("scripts/data/convert_uvlf_npz_to_v2_hdf5.py").resolve()
    before = set(tmp_path.iterdir())
    spec = importlib.util.spec_from_file_location("legacy_converter_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(tmp_path.iterdir()) == before

    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--npz" in help_result.stdout and "--manifest" in help_result.stdout
    failure = subprocess.run(
        [
            sys.executable,
            str(script),
            "--npz",
            str(tmp_path / "missing.npz"),
            "--config",
            str(tmp_path / "missing.toml"),
            "--manifest",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "should-not-exist.h5"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failure.returncode != 0
    assert not (tmp_path / "should-not-exist.h5").exists()
