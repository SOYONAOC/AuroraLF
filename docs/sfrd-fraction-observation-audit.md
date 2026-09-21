# Observational audit beyond UVLF (2026-09-17)

There are additional concerns, but they have different evidential status.
UV-inferred SFRD inherits the UVLF data; it must not be counted as an independent
likelihood. Existing fixed-parameter reionization predictions give an independent
tension. Pop III fractions require an observational selection model before an
exclusion claim can be made.

## Total SFRD: reproduce the observational estimator

Donnan2023 section5.3 and Donnan2024 section4.3 integrate Lnu*phi to MUV=-17
and use KUV=1.15e-28 (Salpeter calibration). The model's all-halo formed initial
stellar mass per trailing10Myr is not this quantity. We applied the same
UV-to-SFR conversion to the existing total UVLF (epsilon=0.03), with the current
dust mapping. Units below are Msun/yr/cMpc^3.

| z | Observational UV-inferred SFRD | Model UV-inferred SFRD | Ratio |
|---|---:|---:|---:|
|8|5.34190e-3|7.04788e-3|1.3194|
|12.5|5.01993e-4|1.05954e-3|2.1107|
|14.5|9.56528e-5|4.60882e-4|4.8183|

At z8 we reconstruct the central DPL from Donnan2023 Table8:
phi*=3.30e-4, M*=-20.02, alpha=-2.04, beta=-4.26. This is a reconstruction,
not a newly measured datum or an uncertainty estimate. At z12.5 and14.5,
Donnan2024 Table3 gives log10 rhoUV=24.64(+0.18/-0.32) and
23.92(+0.27/-0.81). The latter is tentative, with the faint slope and M* fixed;
the LF-integrated constraint contains substantial extrapolation. Do not convert
the central ratios into formal sigma significance. Contemporary surveys may
have different normalizations; this is a specified literature comparison.

Checks: manifest hashes, finite positive total LF, 4097/8193 quadrature convergence,
no LF extrapolation. The finite brightest available half-mag contributes <4.3e-6
of rhoUV; this diagnoses the edge but is not a rigorous bound on an unmodeled tail.
The PopII-only high-z histogram contains an empty bin, so no log-interpolated
PopII-only SFRD is reported; the requested total is fully positive.
High-z saved UV proxies mix PopII1600A and PopIII1500A. Cosmology is not homogenized.
Donnan's calibration is an observational estimator, not an assertion that a
Salpeter IMF describes our PopIII SSP. Removing dust gives ratios1.676,2.129,4.820.

Sources: https://arxiv.org/html/2207.12356v3 (Table8/section5.3),
https://arxiv.org/html/2403.03171v3 (Table3/section4.3).

## Pop III SFRD and fraction: update the GLIMPSE evidence

The older Fujimoto2025 GLIMPSE paper (2501.11678v2) reported an estimated
PopIII SFRD interval1.5e-6--1.5e-4 and a0.01%--1% contribution, conditional on
GLIMPSE-16043 being a PopIII candidate. It already warned that mixed
PopII+III systems can be missed. This is not a universal upper limit.

The follow-up **2512.11790v3 (20 December2025)** spectroscopically detects
[OIII] and rejects the zero-metallicity interpretation of GLIMPSE-16043.
Section5.3/Table4 instead uses AMORE6 as a candidate and gives
logSFRD=[-6.29,-3.98] at5.6<z<6.6. The upper endpoint is1.04713e-4.
The paper's Figure10 caption and section5.3 explicitly state that selection
prefers pure, young(<10Myr), high-nebular-covering-fraction systems and misses
partially enriched/mixed systems. The total population may exceed this interval.
The lower endpoint remains conditional on AMORE6 being a genuine PopIII source.

Our z6 formed-mass PopIII SFRD is1.26273e-3, 12.06 times the selected interval's
upper endpoint; the true SFR fraction is5.008%. This is a useful discrepancy
in scale, **not a matched-estimator exclusion**. Faint cut, SSP/IMF, time response
and selection all differ. Our z8,10,12.5,15 fractions are14.66,32.09,58.75,79.16%.
There is no inspected observation directly measuring those same cosmic fractions.
Venditti2023's fractions and <=0.1% conditional host mass fraction are simulation
predictions. Their disagreement is a model comparison, not observational proof.

Primary updated source: https://arxiv.org/html/2512.11790v3 (section5.3,
Table4, Figure10 caption, summary7). Local curl and requests PDF acquisition
failed certificate verification; web PDF fetch exceeded the tool's size limit.
The exact-version primary HTML was read, including all selection caveats. No
uninspected PDF crop is claimed for this paper.

## Independent observables

The existing full spatial reionization experiment has epsilon_b=.03 and both
escape fractions=.2. PopIII+II predicts neutral fraction0 atz7 and7.54,
versus Mason2018:0.59(+.11/-.15) and Davies2018:0.60(+.20/-.23).
PopII-alone values are.418 and.699. Combined model reaches99% ionization atz8.017.
This is an independent fixed-parameter tension, not a fitted rejection of the
full model family. It is specifically the **100Myr/100Myr emission-window run**,
not an uncomputed updated6Myr run. Density provenance remains
legacy-cosmology-unverified. Escape fractions, recombinations and source-time
windows matter; changing the UV dust postprocessing alone does not change the
already assumed escaping ionizing photon budget.

Read and hash-verified:
`outputs/21cm_map/observations/comparison.json`, plus its original source
`../SmallScale21cm/runs/aurora_maps/instantaneous_100myr_v1/{manifest,histories}.json`
and `external_data/observations/reionization/neutral_fraction.csv`.
Primary sources: https://arxiv.org/abs/1709.05356,
https://arxiv.org/abs/1802.06066.

Cai2015 BDF-521(z7.01) gives a2sigma PopIII SFR fraction<4% for a Salpeter
50--1000Msun IMF; Cai2011 IOK-1(z6.96) quotes<6% with strongly model-dependent
conversion. These are individual-object constraints, not a cosmic mean upper
bound and not the young-host mass fraction in slide2. Rare-burst models must be
compared with selected line-flux/EW distributions and a sample likelihood.
The previously evaluated GHZ2/GS-z14-1 population distributions do not establish
exclusion. Source: https://arxiv.org/abs/1412.3845,
https://arxiv.org/abs/1105.2319.

## Reproduction and review

`PYTHONPATH=. .venv/bin/python scripts/analysis/audit_sfrd_fraction_constraints.py`

Numerical results: `outputs/sfrd_fraction_audit_20260917/comparison.json`.
Evidence crop: `sfrd-method.png` with rendering/source-hash sidecar.
Slides: `slides/atomic_crossing_z6/sfrd_fraction_audit.tex`, main deck pages11--13.
No star-formation, dust, SSP, escape-fraction or production model parameter was changed.
