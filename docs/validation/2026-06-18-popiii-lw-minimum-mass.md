# Pop III LW Minimum-Mass Validation

Date: 2026-06-18

Branch: `codex/popiii-venditti-cruz`

## Scope

Added the Pop III minihalo lower mass boundary used by the HMF sampling-layer
stellar-channel routing:

- no-LW Cruz/Venditti molecular-cooling floor:
  `M_mol,0 = 3.3e7 * (1 + z)**(-1.5) Msun`
- homogeneous LW correction:
  `M_min,PopIII = M_mol,0 * (1 + 2.0 * J_LW21**0.6)`
- default `lw_background_j21=0.0`, so the default threshold is the no-LW
  molecular-cooling floor

The stellar-channel routing is now:

- `Mh < M_min,PopIII`: `below_popiii_min`
- `M_min,PopIII <= Mh < M_atomic`: `popiii`
- `Mh >= M_atomic`: `popii`

This lower boundary is used by the Pop III channel routing and by the optional
Pop III SFR duty cycle. Pop III feedback and enrichment are still not fed back
into the Pop II metallicity regulator or IMF gate.

## Literature And Package Checks

The current molecular-cooling floor and LW correction follow the
Venditti/Cruz/Zeus21 Pop III duty-cycle model:

- Venditti source:
  `external_data/literature_sources/popiii_uvlf_library/papers/Venditti2025BurstyOrHeavyThe/source/main.tex`
- Cruz source:
  arXiv `2407.18294`, Eq. `M_mol`

The older helper used a Machacek/Fialkov-style comparison from the local Pop III
literature library:

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

- collection failed because `POPIII_MOLECULAR_COOLING_M0_NORMALIZATION_MSUN` and
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

- `z=10`: `M_min,PopIII / M_Jeans = 1.33644413e+02`
- `z=20`: `M_min,PopIII / M_Jeans = 1.92075060e+01`
- `z=30`: `M_min,PopIII / M_Jeans = 5.97095475e+00`

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
- Pop III UV luminosity is now connected behind `enable_popiii=True`, but it does
  not alter Pop II SFR, metallicity, or IMF-gate calculations.
