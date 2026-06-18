# Pop III LW Minimum-Mass Validation

Date: 2026-06-18

Branch: `codex/popiii-new-idea`

## Scope

Added the Pop III minihalo lower mass boundary used by the HMF sampling-layer
stellar-channel routing:

- no-LW H2 floor:
  `M_min,H2 = 2.5e5 * (26 / (1 + z)) Msun`
- homogeneous LW correction:
  `M_min,PopIII = M_min,H2 * (1 + 22.87 * J_LW21**0.47)`
- default `lw_background_j21=0.0`, so the default threshold is the no-LW H2
  cooling floor

The stellar-channel routing is now:

- `Mh < M_min,PopIII`: `below_popiii_min`
- `M_min,PopIII <= Mh < M_atomic`: `popiii`
- `Mh >= M_atomic`: `popii`

This still only adds routing metadata. It does not add Pop III luminosity,
feedback, or enrichment physics.

## Literature And Package Checks

The H2 floor and LW correction follow the local Pop III literature library:

- Ventura et al. source: `external_data/literature_sources/popiii_uvlf_library/papers/Ventura2024SemiAnalyticModellingOf/source/meraxes.tex`
- Liu et al. source: `external_data/literature_sources/popiii_uvlf_library/papers/Liu2020WhenDidPopulationIII/source/main.tex`

The Jeans comparison uses:

- `massfunc.SFRD().M_Jeans(z)`

## Red-Green Checks

Red check:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_hmf_sampling.py
```

Expected failure before implementation:

- collection failed because `POPIII_H2_COOLING_MASS_NORMALIZATION_MSUN` and
  related Pop III LW lower-bound helpers were not yet defined in
  `auroralf.uvlf.hmf_sampling`

Green focused check:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_hmf_sampling.py
```

Result: `13 passed in 5.43s`

Full regression check:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests
```

Result: `74 passed in 53.56s`

## Diagnostic Plot

Command:

```bash
PYTHONPATH=. .venv/bin/python scripts/plot/plot_popiii_lw_min_mass_vs_jeans.py
```

Outputs:

- `outputs/popiii_lw_min_mass_vs_jeans_jlw0.png`
- `outputs/popiii_lw_min_mass_vs_jeans_jlw0.pdf`
- `outputs/popiii_lw_min_mass_vs_jeans_jlw0.csv`
- `outputs/popiii_lw_min_mass_vs_jeans_jlw0.txt`

Default `J_LW21=0` comparison summary:

- `z=10`: `M_min,PopIII / M_Jeans = 8.73064975e+01`
- `z=20`: `M_min,PopIII / M_Jeans = 1.73372432e+01`
- `z=30`: `M_min,PopIII / M_Jeans = 6.54823179e+00`

## Scientific Assumptions

- `lw_background_j21` is a homogeneous scalar background in units of
  `1e-21 erg s^-1 cm^-2 Hz^-1 sr^-1`.
- The current HMF Monte Carlo has no spatial galaxy field, so it does not yet
  compute local or self-consistent `J_LW(x,z)`.
- If the LW-corrected Pop III minimum exceeds the atomic-cooling threshold, the
  `popiii` interval becomes empty and halos at or above the atomic threshold
  remain routed to `popii`.

## Remaining Unverified Items

- No self-consistent LW background integration from SFRD has been implemented.
- No Pop III UV luminosity model has been connected to the `popiii` channel yet.
