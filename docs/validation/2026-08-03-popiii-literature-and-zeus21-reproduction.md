# Pop III literature audit and Zeus21 reproduction

Date: 2026-08-03
Branch: `codex/popiii-literature-reproduction`

## Outcome

The first low-cost external reproduction is complete. The public Zeus21
fiducial Pop II+III calculation from Cruz et al. (2025) runs in about 16 s in the
AuroraLF project environment, produces a reusable redshift-dependent LW
history, and agrees point-by-point with AuroraLF's existing LW-only molecular
cooling equation to relative tolerance `2e-12`.

The literature audit also found that the historical AuroraLF function named
`compute_popiii_sfr_visbal2015_from_grids` was scientifically misattributed.
Visbal et al. (2015) compute global SFR densities from collapsed-fraction
derivatives, not a per-halo `Mh H(z) / eta_duty` law. The old entry point now
fails explicitly; the two mass thresholds actually stated in the paper are
implemented and tested.

## Reproduction priority matrix

| Priority | Model or data product | Public source status on 2026-08-03 | Expected cost | AuroraLF use |
|---|---|---|---|---|
| P0 complete | [Cruz et al. 2025 / Zeus21](https://doi.org/10.1103/PhysRevD.111.083503), [arXiv:2407.18294](https://arxiv.org/abs/2407.18294) | [MIT code](https://github.com/ZeusCosmo/Zeus21), pinned locally | 16 s for fiducial global history | Self-consistent `J21(z)` bridge; future streaming backend |
| P0 next | [Venditti et al. 2025](https://doi.org/10.3847/1538-4357/ae0610), [arXiv:2505.20263](https://arxiv.org/abs/2505.20263) | [Frozen notebook/code/data](https://doi.org/10.5281/zenodo.16907335), MIT, 14.6 MB | Notebook minutes; optional MCMC 1--8 CPU h | Reproduce Pop III UVLF bright tail and separate burstiness from high upper-mass cutoff |
| P0 next | [Kulkarni et al. 2021](https://doi.org/10.3847/1538-4357/ac08a3), [arXiv:2010.04169](https://arxiv.org/abs/2010.04169) | Published fit coefficients; no public repository found | Less than 1 s | Optional `Mcrit(z,J21,vbc)` cooling backend |
| P0 existing | [Ventura et al. 2024](https://doi.org/10.1093/mnras/stae567), [arXiv:2401.07396](https://arxiv.org/abs/2401.07396) | [Meraxes](https://github.com/meraxes-devs/meraxes) and [N-body products](https://doi.org/10.5281/zenodo.10608236) public | Existing local replay: 20 min run + 11 min extraction | Diagnostic `Q_Z(z,Mh)` / pristine-probability backend; not yet a complete enrichment closure |
| P1 | [Hegde & Furlanetto 2023](https://doi.org/10.1093/mnras/stad2308), [arXiv:2304.03358](https://arxiv.org/abs/2304.03358) | Equations public; data by request | Formula benchmark seconds | Alternative filtering/cooling/LW/streaming threshold |
| P1 | [A-SLOTH physics release](https://doi.org/10.3847/1538-4357/ac7150), [arXiv:2206.00223](https://arxiv.org/abs/2206.00223); [JOSS software paper](https://doi.org/10.21105/joss.04417), [arXiv:2209.07339](https://arxiv.org/abs/2209.07339) | [MIT public code](https://gitlab.com/thartwig/asloth) | EPS fiducial about 4 GB and minutes-to-hours; large N-body 24--64 GB | Extract enrichment/transition modules; do not embed the entire framework |
| P1 | [Hegde & Furlanetto 2025 / abcd](https://doi.org/10.33232/001c.145070), [arXiv:2507.19581](https://arxiv.org/abs/2507.19581) | Paper's GitHub URL returned 404; no public author repository found | Paper reports about 30 s; reimplementation effort is moderate | Best long-term dual pristine/enriched ISM/CGM reservoir model |
| P2 fit only | [Park & Ricotti 2026](https://doi.org/10.1093/mnras/stag1111), [arXiv:2603.26353](https://arxiv.org/abs/2603.26353) | Occupation fit published; data by request | Fit evaluation seconds | Optional stochastic halo occupation for LW-only or X-ray+LW backgrounds |
| P2 later | [Gurian et al. 2026](https://arxiv.org/abs/2604.26006) | Code/data by request; no public release found | Full multiscale model is non-trivial | Future replacement for constant Pop III efficiency, not a first reproduction |

The threshold-systematics set also includes [Schauer et al. 2021](https://doi.org/10.1093/mnras/stab1953)
([arXiv:2008.05663](https://arxiv.org/abs/2008.05663)) and the monotonic
Kulkarni-fit envelope proposed by [Ishiyama & Hirano 2025](https://doi.org/10.3847/1538-4357/ae102e)
([arXiv:2501.17540](https://arxiv.org/abs/2501.17540)). Their original
hydrodynamic/N-body simulations are not sensible first-round rerun targets;
their published fits are.

## Zeus21 source and environment

- Source: `https://github.com/ZeusCosmo/Zeus21.git`
- Exact commit: `9f2d2105e99e74096092e2061082a79c3f85eaca`
- License: MIT; local `LICENSE` SHA-256
  `0415534dc09fd660b69b123e8d5a053404c07d37622f062dd401360f8f1fc81f`
- Python: 3.13.7
- Added to the existing project environment:
  `zeus21==0.1.dev0`, `classy==3.3.4.0`, `mcfit==0.0.22`,
  `numexpr==2.14.2`, `powerbox==0.9.0`, and `pyfftw==0.15.1`.
- Upstream packaging gap: `setup.py` omits `powerbox` and `pyfftw` although
  the package imports both. They were installed explicitly and recorded in
  `external_data/source_manifests/zeus21.toml`.
- Upstream verification: 4 tests passed and 3 were skipped across
  `test_sfrd.py`, `test_inputs.py`, and `test_astrophysics.py`.

The reproduction follows the public Pop II+III fiducial tutorial:
`precisionboost=1.2`, `fstar_III=1e-3`, `A_LW=2`, `beta_LW=0.6`,
`A_vcb=1`, `beta_vcb=1.8`, and relative velocities enabled.

## Zeus21 numerical result

The output grid contains 77 samples over `10 <= z <= 35`.

| Quantity | Result |
|---|---:|
| Runtime | 15.762 s (repeat runs: 15--17 s) |
| Peak Pop III SFRD | `5.41348e-4 Msun yr^-1 Mpc^-3` |
| Peak redshift | `z=12.596` |
| Pop III SFRD at `z=20` | `2.30921e-4 Msun yr^-1 Mpc^-3` |
| Total `J21_LW` at `z=20` | `1.10480` |
| Total `J21_LW` at `z=10` | `12.0915` |
| LW-only `Mmol` at `z=10` | `8.97584e6 Msun` |
| LW + mean streaming `Mmol` at `z=10` | `2.90785e7 Msun` |
| Mean-streaming multiplier relative to LW-only | `3.23964` |

The generated products are:

- `data_save/zeus21_popiii_fiducial.csv`
- `data_save/zeus21_popiii_fiducial.metadata.json`
- `data_save/zeus21_popiii_mass_distribution.npz`
- `outputs/zeus21_popiii_fiducial.png`
- `outputs/zeus21_popiii_mass_distribution.png`

`scripts/analysis/reproduce_zeus21_popiii.py` validates the exact source
commit and tracked-source cleanliness before running. It then asserts that
Zeus21's `Mmol_LW` and AuroraLF's
`compute_popiii_lw_minimum_mass_msun` agree over every output redshift.

## Halo-mass-resolved distribution

Zeus21 does not define a stochastic count of discrete Pop III-hosting halos.
Its smooth duty prescription instead provides the physically relevant
mass-resolved contribution

```text
dSFRD_III/dln(Mh) = [dn/dMh] SFR_III(Mh,z) Mh.
```

The NPZ bridge product stores all three factors: `dn/dMh`, the Pop III SFR per
halo, and `dSFRD/dlog10(Mh)` for Pop II and Pop III. Integrating the saved
kernel over `log10(Mh)` closes to the previously saved global histories with
maximum relative errors `4.87e-4` for Pop II and `4.72e-4` for Pop III.

| Redshift | mode halo mass | 16th percentile | median halo mass | 84th percentile |
|---:|---:|---:|---:|---:|
| 10.0 | `4.31e7 Msun` | `1.88e7 Msun` | `4.47e7 Msun` | `1.07e8 Msun` |
| 15.1 | `1.57e7 Msun` | `5.89e6 Msun` | `1.46e7 Msun` | `3.87e7 Msun` |
| 20.0 | `5.70e6 Msun` | `2.24e6 Msun` | `5.68e6 Msun` | `1.57e7 Msun` |
| 25.2 | `2.08e6 Msun` | `9.14e5 Msun` | `2.31e6 Msun` | `6.59e6 Msun` |
| 30.2 | `7.55e5 Msun` | `4.50e5 Msun` | `1.06e6 Msun` | `2.95e6 Msun` |

These are SFRD-weighted masses, not number-weighted halo-occupation
percentiles. A count distribution requires a separate occupation model such
as Park & Ricotti (2026), which is not part of the Zeus21 fiducial.

The pinned upstream HMF uses 42 logarithmic points over `1e5--1e14 Msun`, or
`0.2195 dex` spacing. Global integrals and cumulative percentiles are stable,
but the plotted mode must land on a native mass bin and therefore has roughly
`0.1--0.2 dex` localization precision. A dense interpolation check moves the
mode by at most about 16% over the representative redshifts above.

One unrelated upstream issue surfaced during this audit: the reionization
coefficient `niondot_avg_III` uses `N_ion_perbaryon_II` rather than the defined
Pop III value. It does not affect the SFRD or mass distribution saved here,
but it must be resolved before treating the upstream ionization history as a
validated Pop III prediction.

## External attachment implemented

`auroralf.sfr.load_popiii_lw_background_history` loads and validates an
external `(z,J21)` CSV, sorts it, rejects duplicate/negative/non-finite data,
and refuses extrapolation. `compute_popiii_sfr_from_grids` now accepts
`lw_background_j21_grid`, so the Zeus21 history can drive the AuroraLF
molecular-cooling duty cycle without importing Zeus21 into production code.

This attachment is intentionally narrow. It does not pretend that AuroraLF
has adopted Zeus21's HMF, streaming distribution, enrichment, or radiative
transfer. A future global closure should iterate

```text
AuroraLF SFRD(z) -> LW light-cone integral -> J21(z)
                 -> Mcrit/duty -> AuroraLF SFRD(z).
```

Cooling, occupation, pristine probability, and SSP choice should remain
separate switches.

## Visbal 2015 correction

[Visbal, Haiman & Bryan 2015](https://doi.org/10.1093/mnras/stv1941)
([arXiv:1505.06359](https://arxiv.org/abs/1505.06359)) use

```text
SFRD_a = rho_b fstar_a [dFcoll_a/dt (1-Q) + dFcoll_i/dt]
SFRD_m = rho_b fstar_m  dFcoll_m/dt (1-Q),
```

not a per-halo Hubble-time SFR. Their atomic-cooling scale is

```text
Ma = 5.4e7 [(1+z)/11]^(-1.5) Msun,
```

and their LW-dependent minihalo threshold is

```text
Mm = 2.5e5 [(1+z)/26]^(-1.5)
     [1 + 6.96 (4 pi J_LW)^0.47] Msun.
```

The `[Mmin,2 Mmin]` interval is an explicitly idealized self-enrichment
window. The former AuroraLF diagnostic incorrectly combined that window with
an unrelated per-halo SFR normalization. It is now a hard migration error;
the two correct mass functions are tested independently.

## Pop III spectra and observables audit

The local Raiter/Schaerer tables give a second, table-level low-cost
reproduction:

- The default `pop3_ge0_sal_500_001_is5` is `Z=0`, Salpeter 1--500 Msun,
  instantaneous, normalized to one initial solar mass.
- Its `.25` `L_1500` kernel is stellar plus nebular continuum for an
  ionization-bounded `fesc=0` model, not a pure stellar spectrum. At 0.01 Myr
  the nebular fraction is 0.598, so total/stellar is 2.49. For the 50--500
  Msun table it is 0.681, so total/stellar is 3.13.
- In the default `.22` table, 299 non-sentinel rows satisfy
  `L(Hbeta) * I(HeII1640)/I(Hbeta) = 5.7e-12 Q2` to within 0.00497 dex. The
  ratio ranges from 0.98862 to 1.00133, with median 0.994765.
- The 702 late-time `logQ2=-99` no-emission sentinels are now parsed as exact
  zero rather than `1e-99`.

Primary public spectral follow-ups, in order of cost/value:

1. Full local Raiter/Schaerer grid regression against the [CDS A64 tables](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/523/A64)
   and [official model archive](https://obswww.unige.ch/Research/SFR_data/sfr_tls/pop32_models.html).
2. [Muspelheim](https://www.astro.uu.se/~ez/muspelheim/muspelheim.html) modern rotating/non-rotating stellar SED and `Q2(t)` tables.
3. [Yggdrasil](https://www.astro.uu.se/~ez/yggdrasil/yggdrasil.html) rest-frame SEDs for explicit `fcov/fesc` validation. Its README warns against using the affected Pop III `fcov=0` precomputed NIRCam magnitudes directly.
4. [Mas-Ribas et al. public SEDs/notebook](https://github.com/lluism/seds) for stochastic low-mass-cluster He II predictions.
5. The [Garching PISN STELLA archive](https://wwwmpa.mpa-garching.mpg.de/ccsnarchive/data/Kozyreva/PISN/) for a transient-contamination module, motivated by [Ferrara et al. 2026](https://doi.org/10.33232/001c.162107) rather than a claim of confirmed Pop III.

The current He II API multiplies `covering_factor * (1-escape_fraction)`.
Because Yggdrasil defines `fcov=1-fesc`, callers must not supply both as the
same geometry; a future interface should make those semantics explicit.

## Current AuroraLF gaps exposed by the review

- The current mean-field Pop III channel is Cruz/Zeus21-like, with fixed
  scalar `J21` in production and no streaming multiplier.
- There is no pristine probability, internal/external metal-enrichment
  closure, or Pop III to Pop II transition.
- The v2 pipeline computes component UV luminosities, but the production
  runner/HDF5 artifact does not yet retain Pop III UV or light fraction.
- Production HMF sampling currently starts at `log10(Mh/Msun)=8`, missing
  terminal minihalo contributions.
- Existing UV/He II/PISN diagnostic scripts do not all use the same Pop III
  IMF; comparisons must state the IMF explicitly.
- The local Meraxes `Q_Z` table is a useful diagnostic, but it is single-seed,
  noisy, and not a calibrated progenitor/pristine closure.

## Recommended implementation order

1. Reproduce Venditti et al. 2025 from the frozen Zenodo notebook and retain
   Pop II/Pop III UV components in v2 artifacts.
2. Add a cooling-backend enum with Cruz 2025, Kulkarni 2021, Schauer 2021,
   and Hegde 2023 fits; include a monotonic out-of-domain guard.
3. Add the Zeus21 global LW iteration and streaming factor as independent
   switches, using the completed CSV bridge as a regression baseline.
4. Port the already-computed Ventura/Meraxes `Q_Z` result as an explicitly
   diagnostic enrichment backend.
5. Implement the Hegde & Furlanetto 2025 dual-reservoir model only after the
   equation-level specification is tested; do not claim to use unavailable
   public code.
