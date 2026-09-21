# z=8 dust sensitivity (2026-09-17)

This comparison keeps the existing 64x-sampled intrinsic Pop II and total
(epsilon_b=0.03) UVLF fixed. It tests attenuation only; it does not establish
that dust caused the discrepancy or refit the production model.

## Literature prescriptions

All magnitudes in the empirical mappings are **observed** M_UV. Attenuation
A_1600 is approximated as A_1500, as in the production comparison. All A values
are floored at zero. Unless stated otherwise, beta=-2.21-0.146(Mobs+19.5)
at z=8, following the current Williams2018 mapping.

- Current Koprowski2018: A=4.85+2.10 beta. Original dust calibration is z=3–5;
  current z=8 use is an extrapolation.
- Meurer1999: A=4.43+1.99 beta. Local empirical reference, not a new z=8 fit.
- SMC-like reference: A=2.45+1.10 beta, as stated in Vogelsberger2020 section
  3.2.1, citing Bouwens2016 and earlier SMC studies. This tests the A(beta)
  conversion while keeping beta(M) fixed, rather than changing dust amount
  at a fixed E(B-V).
- ALPINE+REBELS, Bowler2024 section 4.4: A=2.11(beta+2.3), or
  A=1.38(beta+2.5). The fitted slopes are 2.11±0.13 and 1.38±0.09 for the
  two assumed intrinsic UV slopes. Here only the central fits are compared;
  no confidence envelope or full stochastic dust transfer is inferred.
  These are fits to massive UV-selected galaxies, with the main REBELS stacks
  at 6.5<z<7.7; seven z>7.7 objects were examined separately. FIR assumptions:
  Td=46 K, emissivity index=2.0. The empirical beta(M) inherited from AuroraLF
  is an additional assumption, not part of the REBELS fit.
- Vogelsberger2020 TNG Model A, equations 3,6,7 and Table 3:
  beta=-2.66-0.34(Mobs+19.5), sigma_beta=0.34, and
  A=4.43+1.99 beta+0.2 ln(10)(1.99)^2(0.34)^2.
  The last term is an effective mean luminosity-correction term (0.211 mag),
  not a stochastic convolution of the luminosity function. Table 3 parameters
  were calibrated to TNG UVLF agreement and are not independent observations
  or an AuroraLF fit. This diagnostic transfers the published mapping only,
  retaining AuroraLF's cap.

Primary sources:
[Williams2018](https://arxiv.org/abs/1802.05272),
[Koprowski2018](https://arxiv.org/abs/1801.00791),
[Vogelsberger2020](https://doi.org/10.1093/mnras/staa137),
[Bowler2024](https://arxiv.org/html/2309.17386v2#S4.SS4).
Schulz2020's redshift-dependent TNG50 IRX-beta formula was also checked, but
its stated validity is 0<=z<=4; it is not included as an independently supported
z=8 alternative. Resolved gas-column/radiative-transfer recipes require gas,
metal, geometry and dust inputs absent from these saved one-dimensional LFs.

## Numerical meaning and checks

M_int=M_obs-A(M_obs), with Jacobian dM_int/dM_obs=1-dA/dM_obs.
We interpolate log(phi_int), multiply by this Jacobian, and preserve the
production cap min(phi_mapped,phi_no_dust). This cap is project-specific;
it need not conserve global counts and is not attributed to a paper.
Integration is over the original observational magnitude bins, not a ratio at
bin centers. No LF extrapolation is allowed.

The script checks source hashes, agreement with the live production mapping,
1025/2049-point bin convergence, and count conservation for the uncapped
change of variables in the bright-end comparison interval. It then solves for
an additional constant screen attenuation in **each individual Bowler bin**
to match that bin's observed central value. These separate roots quantify the
required change; they do not constitute an independent dust model, a shared
best-fit parameter or evidence of statistical agreement.

Run:
`PYTHONPATH=. .venv/bin/python scripts/analysis/compare_z8_dust.py`

Inputs: `data_save/uvlf_efficiency_scan_20260914_64x/z8.npz` and
`external_data/observations/uvlf/current_z6_z8_z10.json`.
Results: `outputs/dust_z8_20260917/comparison.json` and `comparison.png`.
Slide asset: `slides/atomic_crossing_z6/assets/z8_dust_alternatives.pdf`.
Original TNG PDF, institutional copy of accepted/published-layout paper:
`external_data/literature_sources/dust_z8_20260917/Vogelsberger2020.pdf`.
Table 3 crop and URL/hash provenance are under the output directory.

## Scope of interpretation

Dust changes the observed UVLF but does not change the model's intrinsic
formed-mass Pop III SFR fraction or its burst probability. UV-slope, FIR/IRX
and UVLF constraints must be checked jointly before claiming the discrepancy
has been removed. In particular, REBELS finds bright UV-selected galaxies
bluer than an extrapolated faint-galaxy color-magnitude relation; simply making
all bright objects redder would require an independent observational check.

## Results

For Bowler bins M=-21.65,-22.15,-22.90, respectively, total model/observed
ratios are current [5.166,8.192,5.122], Meurer [7.121,12.324,8.482],
SMC [9.994,20.426,19.101], REBELS beta0=-2.3 [5.307,8.454,5.301],
REBELS beta0=-2.5 [4.993,8.863,6.549], TNG A [7.560,8.627,3.196].
Current center attenuations are [0.868,1.021,1.251] mag.
Matching central observed densities needs additional screens
[0.889,0.961,0.676] mag for total, or [0.541,0.579,0.211] mag for
Pop II alone. Required total center attenuations are [1.757,1.983,1.927] mag.

These tested literature recipes do not remove the discrepancy. They establish
sensitivity, not exclusion of all dust explanations. No formal likelihood,
selection marginalization, IRX or UV-color joint fit has been done.
