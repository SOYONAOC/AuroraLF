"""Dimensional and noise-propagation checks for the instrument experiment."""

import numpy as np

from scripts.analysis.summarize_heii_instrument import snr_and_limit


def test_flux_limit_solves_full_source_plus_background_noise():
    basis = dict(
        signal_per_fref=2.0,
        variance_line_per_fref=0.7,
        variance_background=3.0,
        variance_continuum_per_cref=1.3,
        fref=1e-18,
        cref_mjy=1e-4,
        variance_ff=0.01,
        variance_cc=0.02,
        variance_fc=0.005,
    )
    continuum = np.array([0.0, 1e-4, 5e-4])
    _, limits = snr_and_limit(basis, np.zeros(3), continuum, 32)
    snr, _ = snr_and_limit(basis, limits, continuum, 32)
    np.testing.assert_allclose(snr, 5.0)
    assert np.all(np.diff(limits) > 0)
    _, deeper = snr_and_limit(basis, np.zeros(3), continuum, 160)
    assert np.all(deeper < limits)


def test_independent_repeated_ramps_scale_snr_and_preserve_zero_emitters():
    basis = dict(
        signal_per_fref=1.0,
        variance_line_per_fref=0.4,
        variance_background=2.0,
        variance_continuum_per_cref=1.0,
        fref=1e-18,
        cref_mjy=1e-4,
        variance_ff=0.01,
        variance_cc=0.02,
        variance_fc=0.005,
    )
    flux = np.array([0.0, 1e-18, 1e-17])
    a, _ = snr_and_limit(basis, flux, np.ones(3) * 1e-4, 1)
    b, _ = snr_and_limit(basis, flux, np.ones(3) * 1e-4, 5)
    np.testing.assert_allclose(b, np.sqrt(5) * a)
    assert a[0] == 0
