# Pop III fractions near z = 6: AuroraLF and Venditti (2023)

The current random-q model combines rare recent bursts with very large mass per
burst. Its host-conditional young stellar mass fraction is much larger than the
mass fraction in massive Pop III hosts in Venditti et al. (2023). The global
star-formation-rate fraction differs by a more modest factor of about 2.4 at
matched redshift. These are different statistics.

Reference: [arXiv:2301.10259v2](https://arxiv.org/html/2301.10259v2),
doi:10.1093/mnras/stad1201, Section 2.2, Eq. (1), Figs. 1 and 3, Table 1.
The paper's SFRD curve stops at z = 6.5 and its lowest-redshift host sample is
z = 6.7. There is no exact z = 6 entry to quote from these figures.

## Conditional young stellar mass

The new diagnostic uses 256 current McBride histories per fixed final halo mass,
epsilon_b = 0.03, log10(q) ~ Normal(0.5, 1.5^2), z_start = 50, and the archived
source-run seeds/cosmology. Pop II uses the existing delayed SFR and active mask.
There is still no pristine-gas or enrichment veto.

For each history and threshold realization, define a host by a resolved first
burst in the preceding 3 Myr, matching the paper's young-host age cut. Integrate
the entire q distribution, including non-host outcomes in the probability
denominator. The reported mass fraction is

\[
E\left[\frac{M_{\mathrm{III,initial}}(\mathrm{age}<3\,\mathrm{Myr})}
{M_{\mathrm{II,formed,mainbranch}}+M_{\mathrm{III,initial}}}
\mid \mathrm{host}\right].
\]

This is an **initial formed-mass proxy**, not the paper's stellar-particle mass:
it excludes stellar mass return and accreted stellar populations. It is also
conditioned on halo mass, whereas Fig. 3 bins by stellar mass. No exact
paper/model mass-fraction ratio or matched-population significance is claimed.

| z | Halo mass [Msun] | Host probability [%] | Mean young Pop III mass in hosts [Msun] | Conditional mass fraction proxy [%] |
|---|---:|---:|---:|---:|
| 6.0 | 1e10 | 0.02557 | 4.721e7 | 74.58 ± 0.10 |
| 6.0 | 1e11 | 0.01864 | 4.715e8 | 40.81 ± 0.21 |
| 6.0 | 1e12 | 0.00930 | 4.689e9 | 24.49 ± 0.76 |
| 6.7 | 1e10 | 0.03607 | 4.712e7 | 74.97 ± 0.19 |
| 6.7 | 1e11 | 0.02425 | 4.705e8 | 41.15 ± 0.24 |
| 6.7 | 1e12 | 0.00989 | 4.690e9 | 23.71 ± 0.35 |

Uncertainties are one-standard-error MAH sampling estimates in percentage points,
with q integrated analytically/numerically, not physical-model uncertainties.
The mean conditional Pop II formed mass at 1e12 Msun is about 1.5e10 Msun:
this at least places this diagnostic in the broad massive-galaxy regime.

In the paper's z = 6.7 massive stellar-mass bins, about 10% of galaxies host young
Pop III. Their conditional mean mass fraction is roughly 0.01–0.1%. Fig. 3's
white hatched bars include non-hosts; its red bars are host-conditional averages.
Table 1's four objects with Mstar > 3e9 Msun at z = 6.7 have mass fractions
0.00891%, 0.02188%, 0.03162%, and 0.03631%. Their Pop III masses are each about
2.0–2.1e6 Msun, close to the simulation's stellar-particle scale; they are selected
examples, not a universal cluster-mass limit. The lower-mass U8H54 has 0.234%,
illustrating why 0.1% must not be applied to every galaxy.

The model's single burst consumes epsilon_b times the halo's cosmic baryon
allocation. A rare late threshold crossing therefore creates a large young
population. Paper hosts instead contain small, locally low-metallicity star
forming regions. Matching global SFRD cannot validate both event frequency and
event mass separately.

## Cosmic SFR and photons

For the same trailing 10 Myr definition, use f_SFR = PsiIII/(PsiII + PsiIII):

| z | AuroraLF fraction | Venditti Fig. 1 mean-curve fraction | Fraction ratio |
|---|---:|---:|---:|
| 6.0 | 5.008% | not provided | — |
| 6.7 | 7.242% | 3.067% | 2.361 |
| 7.0 | 8.461% | 3.465% | 2.442 |
| 8.0 | 14.660% | 6.153% | 2.383 |

At z = 6.7, PsiIII is 1.308e-3 versus 5.101e-4 Msun/yr/cMpc^3, a factor 2.565.
Paper values are interpolated from previously extracted vector-curve vertices;
model z = 6.7 and 7.0 entries are interpolated in log10(SFRD) vs redshift from
the saved z = 6.0, 6.99, 8.0 values. No extrapolation to z = 6 is made.
AuroraLF weights main branches by Reed07 HMF; the paper counts all stellar
particles in its volumes. Their resolution and IMF differ.

The existing z = 6 ionizing-rate diagnostic gives 26.69% Pop III for a 6 Myr
Pop III kernel window, 100 Myr Pop II window, and equal escape fractions 0.2.
This is a current photon-rate share, not a stellar mass fraction, SFR fraction,
or accumulated reionization contribution. The quoted paper mass fractions give
no directly comparable photon-rate prediction. The current Pop III SSP header
specifies Mlow = 1, Mup = 500 Msun, IMF type 4, Mc = 60, sigma = 1; the paper
adopts a Salpeter IMF over 100–500 Msun. The initial burst-mass diagnostic uses
no SSP luminosity weighting.

## Reproducibility and checks

- New calculation: `scripts/analysis/popiii_fraction_venditti.py`.
- SLURM jobs 159296 and 159299 completed on fat2; the latter adds sampling errors
  with identical histories. Final data: `data_save/popiii_fraction_venditti_20260917/mass_proxy.json`.
- First-passage probabilities agree with an independent normal-CDF calculation;
  16/32-point q integration agrees to relative error below 1e-8 (observed <1e-15).
- Existing SFRD input hashes and source-figure hash were verified. Derived
  comparison and paper-table values: `data_save/popiii_fraction_venditti_20260917/comparison.json`.
- The final paper's p. 8 Fig. 3 was visually checked against its caption and
  Section 2.2 definition. The PDF evidence crop and its hash/render manifest
  are under `outputs/popiii_fraction_venditti_20260917/paper_fig3_final.*`.
- Current experiment choices are preserved; no production model was changed.

For a strict mass-selected reproduction, full stellar assembly, mass return,
IMF consistency, and a pristine-gas model remain necessary. The paper itself
has finite mass resolution and incomplete radiative feedback; it is a model
comparison, not an observational upper limit.

## Higher-redshift extension

The same calculation was extended to z = 8.1, 10, 12.5, and 15 with halo masses
1e8, 1e9, 1e10, 1e11, and 1e12 Msun. Output:
`data_save/popiii_fraction_highz_20260917/mass_proxy.json` (job 159316, node6).
No scientific parameter was changed. The bounded diagnostic used three shared
CPUs (out of four idle), relaxing the submitter's default four-CPU throughput
floor while retaining its 80% idle-CPU ceiling. Both first-passage CDF and
16/32-order quadrature checks passed in all 20 cells.

Host-conditional initial mass fraction proxies, in percent:

| z | Mh = 1e9 Msun | 1e10 Msun | 1e11 Msun | 1e12 Msun |
|---|---:|---:|---:|---:|
| 8.1 | 93.26 | 75.06 | 41.80 | 23.89 |
| 10 | 93.35 | 75.29 | 42.75 | 24.34 |
| 12.5 | 93.54 | 76.10 | 42.97 | 25.59 |
| 15 | 93.62 | 76.59 | 46.00 | 26.10 |

Probability of a first burst within the preceding 3 Myr, in percent:

| z | Mh = 1e9 Msun | 1e10 Msun | 1e11 Msun | 1e12 Msun |
|---|---:|---:|---:|---:|
| 8.1 | 0.04914 | 0.05404 | 0.03503 | 0.01353 |
| 10 | 0.08313 | 0.08632 | 0.05464 | 0.02048 |
| 12.5 | 0.15578 | 0.15583 | 0.08632 | 0.02767 |
| 15 | 0.24940 | 0.23071 | 0.12963 | 0.04046 |

The 1e8 Msun cells, available in the JSON, have mass fractions about 98.9–100%.
At z = 8.1 their Pop II formed mass is effectively zero. This is an outcome of
the current random-q prescription (which permits q < 1), not a demonstration
of physical primordial-gas survival or molecular cooling. All fixed-mass
columns are diagnostics, not HMF-weighted typical host populations. In
particular, the high-redshift 1e12 Msun column must not be described as common.

Global SFR comparisons, using the same 10 Myr window:

| z | Model Pop III SFR fraction | Paper fraction | Fraction ratio | Pop III SFRD ratio |
|---|---:|---:|---:|---:|
| 8 | 14.66% | 6.15% | 2.383 | 3.780 |
| 10 | 32.09% | 15.10% | 2.125 | 6.612 |
| 12.5 | 58.75% | 42.22% | 1.391 | 14.883 |
| 15 | 79.16% | 64.97% | 1.219 | 54.584 |

Derived data: `data_save/popiii_fraction_highz_20260917/sfr_comparison.json`.
At z = 15, the Pop III SFRDs are 4.759e-4 and 8.719e-6 Msun/yr/cMpc^3.
The fractions look closer because the model's Pop II SFRD is also higher
(26.64 times the paper value). Paper Section 3.1 explicitly identifies strong
resolution dependence above z ~ 13; the 54.6 ratio is not an observational
rejection or a uniquely established physical overprediction.

The paper's Fig. 3 stellar-mass-fraction sample reaches only z = 8.1. Its
low-redshift 0.1% statement must not be extrapolated to z = 10–15. For example,
Table 1's selected z = 8.1 objects already span about 0.083–0.55%, with stellar
masses roughly 1e9–2.5e9 Msun. Neither those selected examples nor the fixed
halo-mass diagnostic above are a matched-population measurement.

The analysis CLI now accepts `--redshifts`, `--masses`, and `--output`;
defaults preserve the z = 6/6.7 diagnostic. Run it through the SLURM submitter:

```bash
.venv/bin/python ~/.codex/skills/dmde-compute/scripts/submit_python_job.py \
  --job-name popiii-fraction-highz --parallelism-mode none \
  scripts/analysis/popiii_fraction_venditti.py -- \
  --redshifts 8.1 10 12.5 15 --masses 1e8 1e9 1e10 1e11 1e12 \
  --output data_save/popiii_fraction_highz_20260917
```

Old run hashes continue to refer to the old script revision; their original
products were retained. Only argument selection and metadata were generalized.

## Overlay with observed total SFR density

`scripts/plot/plot_sfrd_observations.py` overlays the current 10 Myr total,
Pop II and Pop III SFRDs from `data_save/ionizing_sources/sfrd_v1/sfrd.csv`
on the observations shown in Venditti et al. (2023), Fig. 1, left panel.
It retains the source PDF's observation marker outlines, translating their
centres onto wider axes and mapping error-bar endpoints with the same coordinate
transformation. Text and marker shapes are not stretched. These are published figure elements,
not newly obtained survey catalogues. The original observation legend is kept,
including its preprint-year labels. Original simulation curves and bands are
omitted from this overlay; the original paper figure remains in external data.

The observations constrain total SFRD, not Pop III separately. The model uses
its full HMF integration; survey selections, limiting UV magnitudes, IMF and
dust conventions have not been homogenized. This is a visual comparison,
without fitting or a statistical exclusion claim.

```bash
PYTHONPATH=. .venv/bin/python scripts/plot/plot_sfrd_observations.py
```

Vector output: `slides/atomic_crossing_z6/assets/sfrd_observations.pdf`.
PNG, SVG and provenance/validation manifest:
`outputs/popiii_sfrd_observations_20260917/`.
The overlay appears on slide 4 of `slides/atomic_crossing_z6/crossing.pdf`.
The figure uses a 900-by-385 pt canvas, with a roughly 2:1 plotting area,
and spans the slide's text width.
