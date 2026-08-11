from __future__ import annotations

from math import log10, pi
from pathlib import Path

import astropy.units as u
from astropy.constants import G, k_B, m_p
from astropy.cosmology import Planck15, Planck18
import pytest

from auroralf import constants
from auroralf.config import CosmologyConfig, UVLFRunConfig
from auroralf.mah import models
from auroralf.uvlf import hmf_sampling


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_production_cosmology_defaults_share_the_constants_source() -> None:
    config = CosmologyConfig()
    cosmology = config.to_model()

    assert config.h0_km_s_mpc == constants.PLANCK18_H0_KM_S_MPC
    assert config.omega_m == constants.PLANCK18_OMEGA_M
    assert config.omega_b == constants.PLANCK18_OMEGA_B
    assert cosmology.h0 == pytest.approx(constants.PLANCK18_H0_GYR, rel=1.0e-15, abs=0.0)
    assert cosmology.omega_lambda == constants.PLANCK18_OMEGA_LAMBDA
    assert hmf_sampling.MASS_FUNCTION_SIGMA8 == constants.PLANCK18_SIGMA8
    assert hmf_sampling.MASS_FUNCTION_NS == constants.PLANCK18_NS

    production = UVLFRunConfig.from_toml(PROJECT_ROOT / "configs/uvlf/production.toml")
    assert production.cosmology == config


def test_astropy_is_the_source_for_available_reference_constants() -> None:
    assert constants.SECONDS_PER_GYR == u.Gyr.to(u.s)
    assert constants.YEARS_PER_GYR == u.Gyr.to(u.yr)
    assert constants.KM_PER_MPC == u.Mpc.to(u.km)
    assert constants.GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN == G.to_value(
        u.Mpc * u.km**2 / (u.s**2 * u.Msun)
    )
    assert constants.PROTON_MASS_KG == m_p.to_value(u.kg)
    assert constants.BOLTZMANN_CONSTANT_J_K == k_B.to_value(u.J / u.K)
    assert constants.PLANCK18_H0_KM_S_MPC == Planck18.H0.to_value(
        u.km / (u.s * u.Mpc)
    )
    assert constants.PLANCK18_OMEGA_M == Planck18.Om0
    assert constants.PLANCK18_OMEGA_B == Planck18.Ob0
    assert constants.PLANCK18_OMEGA_LAMBDA == 1.0 - Planck18.Om0
    assert constants.PLANCK18_SIGMA8 == Planck18.meta["sigma8"]
    assert constants.PLANCK18_NS == Planck18.meta["n"]
    zero_flux = (0.0 * u.ABmag).to(u.erg / (u.s * u.cm**2 * u.Hz))
    reference_luminosity = (
        4.0 * pi * (10.0 * u.pc) ** 2 * zero_flux
    ).to_value(u.erg / (u.s * u.Hz))
    assert constants.AB_ZEROPOINT_LNU == 2.5 * log10(reference_luminosity)


def test_legacy_models_constant_exports_reference_shared_values() -> None:
    assert models.SECONDS_PER_GYR == constants.SECONDS_PER_GYR
    assert models.KM_PER_MPC == constants.KM_PER_MPC
    assert (
        models.GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN
        == constants.GRAVITATIONAL_CONSTANT_MPC_KMS2_MSUN
    )
    assert models.PLANCK18_H0_KM_S_MPC == constants.PLANCK18_H0_KM_S_MPC
    assert models.PLANCK18_OMEGA_M == constants.PLANCK18_OMEGA_M
    assert models.PLANCK18_OMEGA_B == constants.PLANCK18_OMEGA_B


def test_planck15_simulation_cosmology_is_flat_and_derived_once() -> None:
    assert constants.PLANCK15_H0_KM_S_MPC == Planck15.H0.to_value(
        u.km / (u.s * u.Mpc)
    )
    assert constants.PLANCK15_H == Planck15.h
    assert constants.PLANCK15_OMEGA_B == Planck15.Ob0
    assert constants.PLANCK15_OMEGA_LAMBDA == 1.0 - constants.PLANCK15_OMEGA_M
    assert constants.PLANCK15_H0_GYR == pytest.approx(
        constants.PLANCK15_H0_KM_S_MPC
        * constants.SECONDS_PER_GYR
        / constants.KM_PER_MPC,
        rel=1.0e-15,
        abs=0.0,
    )
