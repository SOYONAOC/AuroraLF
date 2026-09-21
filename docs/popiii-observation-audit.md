# Observational audit of the first three high-redshift slides

2026-09-17. The current fixed random-q model has a substantial bright-end UVLF
excess at z = 8 relative to Bowler et al. (2020). This is evidence against the
fixed combination of star formation, bursts and dust used in that comparison.
It is not a direct observational rejection of each of the three fraction/probability
tables, and the excess cannot be attributed entirely to Pop III: the Pop II
baseline is already high. The checked z ~ 12–14 measurements do not establish
a model rejection; this does not demonstrate universal observational consistency.

## What the first three pages measure

1. SFR-weighted Pop III fraction integrated over the model halo mass grid and a
   trailing 10 Myr window. Venditti (2023) is another simulation, not an observed
   Pop III fraction. Our SFRD grid covers 1e4–1e15 Msun.
2. Mean young initial stellar mass fraction conditional on a first burst in the
   last 3 Myr, at fixed halo mass. Pop II mass includes main-branch formed mass,
   not stellar accretion or surviving mass after stellar evolution.
3. Probability of that recent first burst over all histories and q realizations
   at fixed halo mass. It is not the fraction of a UV-selected sample.

At z = 12.5, Mh = 1e10 Msun, the conditional mean is 76.1036% and the event
probability is 0.155825%. Their product is the equal-halo mean young initial
mass-fraction proxy, 0.118589%, not a ratio of summed stellar masses and not the
cosmic SFR fraction. This identity is independently checked by the audit script.

Donnan et al. (2023), Section 5.3, explicitly integrates the UVLF to MUV = -17
and converts UV density with K_UV = 1.15e-28 Msun/yr/(erg/s/Hz). A full-halo-grid
10 Myr formed-mass rate with a different IMF is not the same observable.
Source: <https://arxiv.org/abs/2207.12356>, local PDF page 8. The existing overlay
is a visual comparison and cannot supply an exclusion confidence.

## The z = 8 discrepancy

The audit re-integrates saved 64x-sampling UVLFs over each original observed
magnitude bin, with the current dust mapping and epsilon = 0.03. All product
hashes and reproduction of the prior numerical summary are checked.

| MUV | Observed | Pop II baseline | Total model | Total / observed |
|---|---:|---:|---:|---:|
| -21.65 | 2.95 +/- 0.98 | 8.076 | 15.240 | 5.166 |
| -22.15 | 0.58 +/- 0.33 | 2.336 | 4.752 | 8.192 |
| -22.90 | 0.14 +/- 0.06 | 0.265 | 0.717 | 5.122 |

Number densities are in 1e-6 cMpc^-3 mag^-1. Observations are Bowler et al.
(2020), Table 6, PDF page 11; the magnitude-bin full widths are 0.5, 0.5 and
1.0 mag, respectively. The observation JSON was checked against the paper's
table. Source: <https://academic.oup.com/mnras/article/493/2/2059/5721544>.
McLure et al. (2013), Table 2, is an additional comparison: at MUV = -21.25,
the total model is 4.303 times its measured 8e-6 cMpc^-3 mag^-1.

These are substantial discrepancies with the quoted observational errors.
No Gaussian joint significance is assigned: the audit does not supply survey
selection/redshift averaging, full data/model covariance or marginalization over
dust. Pop II alone is 1.89–4.03 times the Bowler values, so changing Pop III alone
does not remove the baseline discrepancy under the same prescription.

## Higher redshifts and He II

Against all seven z = 12.5 bins of Donnan et al. (2024), Table 2, model/observed
UVLF ratios are 1.247–2.463. The largest difference relative to the quoted upper
error is at MUV = -18.75: model 1.9606e-4 versus observed
(0.80 +0.51/-0.36)e-4 cMpc^-3 mag^-1. The difference is 2.28 times that one
upper error, not a joint Gaussian significance. Source:
<https://arxiv.org/html/2403.03171v3#S4.T2>. The local PDF is v3; the
checked Table 2 values also agree with v2. Bowler's local PDF is
arXiv:1911.12832v2, and Donnan (2023)'s is arXiv:2207.12356v3.

At z = 14.5 and -21 < MUV < -20, the model bin average is 3.5029e-6 versus the
MoM+JADES spectroscopic estimate 4.3652e-6 (+5.8678e-6/-2.8163e-6), a ratio
0.8025. Source: <https://arxiv.org/abs/2505.11263>. The model is a snapshot;
the observed sample covers 14 < z < 15. These existing UV runs sample halo
masses 1e5–1e12 Msun and use a Pop II 1600 A + Pop III 1500 A proxy. They test
the same burst prescription through observable light, not the identical
integration domain as the SFRD table.

Existing exact-redshift He II results at epsilon = 0.03, with total MUV within
0.25 mag of each target, give the following fluxes in 1e-19 erg/s/cm^2:

| Target | Observed | Model median [16th,84th] | Below observed reference |
|---|---:|---:|---:|
| GHZ2, z=12.342 | 2.7 +/- 1.6 | 5.826 [0.459,25.109] | 37.26% |
| GS-z14-1, z=13.86 | <0.70 (3 sigma) | 0.810 [0.069,3.861] | 48.04% |

Observations: Castellano et al. (2024), Table 1,
<https://arxiv.org/html/2403.10238v2>; Wu et al. (2025), Table 2,
<https://arxiv.org/html/2507.22858v1>. Original primary HTML tables were opened
and checked. The upper limit is not a zero-flux measurement. Model quantiles
describe population diversity; below-reference fractions are not p values or
noise-convolved nondetection probabilities. Full MUV errors, selection, gas
response, escape fraction and SSP uncertainty have not been marginalized.
Acquiring the Wu PDF failed at HTTPS certificate validation, and the web PDF
fetch timed out; no Wu PDF crop is claimed. Its HTML Table 2 supplies the
verified numerical measurement.

Tang et al. (2026), Section III.3, provides a larger high-z target set: 26
grating spectra, two He II detections, and median 3-sigma EW limit 14 Angstrom
for the other 24 objects. Those limits constrain line EWs; they are not a 2/26
Pop III-host frequency measurement. Source: <https://arxiv.org/html/2507.08245v2>.

Fujimoto et al. (2025), Section IV.4, estimates a Pop III SFR share of about
0.01–1% at z ~ 6–7. This is relevant to the earlier z = 6 question, but is not
a high-z upper bound on pages 1–3. The paper explicitly notes that its [O III]
non-detection selection misses mixed Pop II+III systems, also discusses a null
clump search, and calls for further work on the fraction. Source:
<https://arxiv.org/html/2501.11678v2#S4.SS4>. We have not applied that photometric
selection to the model and do not promote this estimate to a model-independent
limit on all Pop III formation.

## Reproducibility and review

Run `PYTHONPATH=. .venv/bin/python scripts/analysis/audit_popiii_observations.py`.
Results and hashes are in `outputs/popiii_observation_audit_20260917/comparison.json`.
The checked Bowler PDF crop and rendering manifest are `bowler-table6.png/json`
in the same directory. That PDF was reused from the local literature archive;
2026-09-17 is the audit/revalidation date, not its original acquisition date.
Original paper vectors are retained in the slide asset. Evidence and scope
appear on pages 5–7 of `slides/atomic_crossing_z6/crossing.pdf`; the first page
now points explicitly to the z = 8 observational tension.
