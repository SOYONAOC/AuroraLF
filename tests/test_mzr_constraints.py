from __future__ import annotations

import numpy as np

from auroralf.chemistry import (
    CALIBRATED_TOPHEAVY_YIELD_MULTIPLIER,
    equivalent_oxygen_abundance_from_zsun,
    fire2_highz_mzr_oh12,
    jades_lowmass_mzr_oh12,
    max_positive_mzr_offset_dex,
)


def test_equivalent_oxygen_abundance_uses_solar_normalization() -> None:
    np.testing.assert_allclose(equivalent_oxygen_abundance_from_zsun(np.array([1.0, 0.1])), [8.69, 7.69])


def test_literature_mzr_relations_match_adopted_forms() -> None:
    logmstar = np.array([8.0, 10.0])

    np.testing.assert_allclose(fire2_highz_mzr_oh12(logmstar), 8.69 + 0.37 * logmstar - 4.3)
    np.testing.assert_allclose(jades_lowmass_mzr_oh12(logmstar), 7.72 + 0.17 * (logmstar - 8.0))


def test_max_positive_mzr_offset_ignores_under_enrichment() -> None:
    model = np.array([7.1, 7.4, 8.2])
    reference = np.array([7.2, 7.3, 7.6])

    np.testing.assert_allclose(max_positive_mzr_offset_dex(model, reference), 0.6)


def test_calibrated_topheavy_yield_multiplier_is_mzr_constrained_value() -> None:
    np.testing.assert_allclose(CALIBRATED_TOPHEAVY_YIELD_MULTIPLIER, 1.28)
