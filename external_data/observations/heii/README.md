# JWST He II 1640 comparison inputs

`jwst_targets.json` is a manually transcribed, source-attributed two-object input
for the fixed epsilon=0.03 comparison. Retrieved 2026-09-10. It is not a survey
catalog or a complete sample. Flux units are erg s^-1 cm^-2; MUV is AB absolute
magnitude and equivalent widths are rest-frame angstroms.

- GHZ2: Castellano et al. 2024, arXiv:2403.10238v2, Table 1 and sections II,
  III.1, III.2, III.5. MUV=-20.53±0.01 is delensed; HeII flux=(2.7±1.6)e-19
  is slit/aperture corrected but still magnified, mu=1.3. HeII and OIII] are
  blended and fit jointly. SNR=5 in the table comes from direct integration,
  not division of the Gaussian-fit flux by its quoted error.
- GS-z14-1: Wu et al. 2025, arXiv:2507.22858, Introduction, Table 2 and
  sections III.3/III.4. z=13.86(-0.05,+0.04), MUV=-19.0±0.4, HeII <7e-20
  at 3 sigma, EW<12 A. NIRSpec PRISM 56h, unresolved Gaussian width fixed to
  spectral resolution, noise covariance and redshift uncertainty included.
  The source gives a limit, not a measured zero flux. We assume mu=1 for this
  non-lensed field object; no magnification uncertainty is modeled.

The `population_z` fields are OUR analysis mapping to existing snapshots, not
paper measurements. Using the target luminosity distance does not correct the
mismatch in formation histories or halo abundance. The input records treatment
of slit losses; it does not contain an aperture response, noise covariance
matrix, magnification posterior, UV posterior or survey selection function.

Original sources:
https://arxiv.org/html/2403.10238v2
https://arxiv.org/html/2507.22858
