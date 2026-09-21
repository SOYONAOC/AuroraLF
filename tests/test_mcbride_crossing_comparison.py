"""Ensure diagnostic root finding represents the unchanged public MAH API."""

import numpy as np
import pytest

from auroralf.cooling import compute_atomic_cooling_mass_msun
from auroralf.mah import Cosmology, generate_halo_histories
from auroralf.mah.sampling import sample_parameters
from scripts.analysis.compare_atomic_crossing_mcbride import solve_crossings


@pytest.mark.parametrize("mass", [3e8, 2e9, 2e10])
def test_roots_agree_with_public_generator_and_scan_refinement(mass):
    cosmo = Cosmology()
    seed = 917
    pars, _ = sample_parameters(
        mass_ref=mass,
        size=128,
        sampler="mcbride",
        rng=np.random.default_rng(seed),
        pilot_samples=50000,
    )
    z, status = solve_crossings(*pars.T, mass, 6.0, cosmo)
    zfine, sfine = solve_crossings(*pars.T, mass, 6.0, cosmo, n_scan=2049)
    np.testing.assert_array_equal(status, sfine)
    np.testing.assert_allclose(z, zfine, atol=1e-8)
    tracks = generate_halo_histories(
        128,
        6.0,
        mass,
        cosmology=cosmo,
        random_seed=seed,
        z_start_max=50,
        store_inactive_history=True,
        dz=0.025,
    ).tracks
    zz = tracks["z"].reshape(128, -1)
    mm = tracks["Mh"].reshape(128, -1)
    j = (mm >= compute_atomic_cooling_mass_msun(zz, cosmology=cosmo)).argmax(axis=1)
    api_z = zz[np.arange(128), j]
    assert np.all(z >= api_z - 1e-8)
    assert np.all(z - api_z <= 0.025 + 1e-8)
    np.testing.assert_array_equal(status == 2, j == 0)
