# Low-redshift PISN extrapolation

The HSC comparison requires rates at the redshifts where its light-curve
templates can be selected. Moriya et al. (2021), arXiv:2012.00171v1, Section 6,
Table 2 and Figure 8 give an order-of-magnitude bound near 100 events per source
year per comoving Gpc^3 for luminous, long-lasting PISNe, typically at z=1–3.
This is not a measured rate in three redshift bins, a specified-confidence
upper limit, or a constraint on all Pop III PISNe. Their simulations extend
over z=0–6 and assume a constant volumetric rate for each light-curve template.

## Calculation

`scripts/run/run_pisn_lowz.py --config configs/experiments/pisn_lowz_20260918.json`
requires a cp6 SLURM allocation and a verified immutable release. The computation
uses the existing McBride main-branch histories, Reed07 HMF, 1e5–1e12 Msun
descendant mass range, threshold log10 mean 0 and scatter 1.5 dex, epsilon_b=0.03,
logE stellar IMF, and Marigo H+He-burning lifetime approximation for 140–260
Msun progenitors. Redshifts 1, 2, 3 are extrapolation targets; 12.5 and 14.5
provide overlap checks with the existing calculation.

The new `threshold_averaged_pisn_rate` integrates over the original random
threshold distribution instead of drawing rare young bursts. It only integrates
new record levels of log10(Mhalo/Mcool), preserving first passage and preventing
repeat bursts on a single history. It uses the same piecewise interpolation in
time and log mass as `random_q.first_crossing`. Quadrature intervals are split
at every PISN lifetime knot, including both edges of its short lifetime window.
Its output is the expectation of Mhalo,birth times the per-initial-stellar-mass
PISN delay kernel, before multiplication by epsilon_b times the baryon fraction.
The runner integrates this result over the HMF with midpoint log-mass quadrature.

The 128/256 mass-node and 960/1920 time-node cases check numerical resolution.
Each mass node samples 512 MAHs. Reported MAH standard errors cover stochastic
MAH sampling only. The difference of 8- and 16-point quadrature is saved
separately. Results and input/product hashes are stored in
`data_save/pisn_lowz_20260918/`; execution records and diagnostic figures are
under `outputs/pisn_lowz_20260918/`. No SSP, UV host grouping, dust, or instrument
response is needed to compute the all-host intrinsic PISN event rate.

## Interpretation limits

This is an extrapolation of the existing phenomenological model. It does not
model metallicity, external enrichment, survival of pristine gas, or feedback.
It therefore does not establish the abundance of actual low-redshift Pop III
star formation. The halo mass interval is inherited from the earlier run and
does not automatically cover every low-redshift progenitor environment.

A comparison to HSC detections additionally needs the distribution of PISN
light curves/SEDs and their mapping to progenitors, redshifted filter fluxes,
survey epochs, detection thresholds, and selection/control times. The published
R250, R225, R200, R175, R150, B250, B200, He130, He100 and He80 models have
different efficiencies (Moriya Table 2). Neither a UV host cut nor one common
arbitrary bright-progenitor mass cut supplies these ingredients. The low-z
all-PISN calculation must not be labelled an HSC detection prediction or a
formal exclusion by the published luminous-template bound.

Validation: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_pisn_first_passage.py tests/test_random_q_pisn.py`.

## First completed run

Job 11693155 finished successfully in 4m14s. All 15 cases and 16 product hashes
were verified. For the inherited 1e5–1e12 Msun halo interval, the finest-grid
source-frame rates (events / year / comoving Gpc^3) are:

| z | Rate | MAH MC standard error |
|---|---:|---:|
| 1 | 14.41 | 0.29 |
| 2 | 104.17 | 1.12 |
| 3 | 354.45 | 2.63 |
| 12.5 | 1335.44 | 8.64 |
| 14.5 | 938.13 | 6.38 |

Doubling time resolution changes all rates by less than 0.061%; changing the
mass quadrature from 128 to 256 nodes changes them by less than 1.65%, with
different MAH samples at the changed mass nodes. This latter comparison includes
MC noise and is not an isolated deterministic quadrature error. The two high-z
points differ from previous direct-q-draw estimates by 0.23 and 1.26 combined
reported standard errors, respectively.

At z=1,2,3, the uppermost original halo-mass decade (1e11–1e12 Msun) contributes
64.9%, 47.7%, and 26.9% of the integrated rate. The inherited high-z mass limit
therefore does not establish a complete low-z rate. A separate diagnostic,
`configs/experiments/pisn_lowz_mass_tail_20260918.json`, covers the adjacent
1e12–1e15 Msun interval with the same physical model and independent seed. It
preserves the original calculation rather than overwriting it. Job 11693767
uses one cp6 node with 56 CPUs. Its mass and time convergence are checked in
the same way. The extra high-mass contribution must be inspected before quoting
an integrated low-z prediction.

`scripts/plot/plot_pisn_lowz.py` validates completed artifacts and writes CSV,
PNG and PDF diagnostics under `outputs/pisn_lowz_20260918/`. Pass
`--tail data_save/pisn_lowz_mass_tail_20260918` only after the second run is
complete. The main slide deck has not been changed.

## Combined result after the high-mass check

Job 11693767 completed in 2m25s. All nine tail cases, ten product hashes and
the frozen-deployment digest were verified; stderr was empty. Adding its
1e12–1e15 Msun contribution to the original mass interval gives:

| z | Original interval | Added high-mass interval | Combined rate | Combined MAH MC SE |
|---|---:|---:|---:|---:|
| 1 | 14.41 | 12.14 | 26.55 | 0.30 |
| 2 | 104.17 | 22.51 | 126.69 | 1.13 |
| 3 | 354.45 | 18.18 | 372.63 | 2.63 |

All rates above are events per source year per comoving Gpc^3. Independent
seeds were used in the two mass intervals; their MC variances were added.
These error bars do not cover physical model or survey-selection uncertainty.

For the added interval, doubling the time resolution changes the rate by at
most 0.039%; the two mass grids differ by at most 0.657%, including changed MAH
sampling. The quadrature differences are less than 2.2e-13 of the tail rate.
The 1e14–1e15 Msun decade contributes only 0.514%, 0.00726%, and 0.0000258%
of the combined z=1,2,3 rates. The uppermost 0.25 dex contributes 0.00293%,
6.17e-7%, and 9.62e-12%, respectively. This falling tail supports the adequacy
of the expanded upper mass boundary for this numerical calculation; it is
not a mathematical bound on all possible halo masses or on missing physics.

The corresponding pre-selection observer-frame rates are 0.1163, 0.5352,
and 1.1781 events per observer year per square degree per unit redshift.
They are not detection counts. In `rates_with_tail.csv`, this quantity is
the `observer_rate` column; the columns ending in `gpc3_yr` use source-frame
years and comoving Gpc^3.

`outputs/pisn_lowz_20260918/combined_validation.json` records the numerical
checks. The independent `rates_with_tail.png` / `.pdf` diagnostic was visually
reviewed. Its low-z points include 1e5–1e15 Msun halos; the two high-z overlap
points retain their original 1e5–1e12 Msun interval. The HSC line remains a
luminous-template context indicator, not a like-for-like exclusion threshold
for the total-PISN points. No metallicity/pristine closure or light-curve
selection has been added. Main slides remain unchanged.
