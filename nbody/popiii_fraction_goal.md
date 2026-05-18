# Pop III Fraction Goal

Goal: derive a defensible relation that gives the Pop III contribution as a
function of halo mass and redshift,

```text
f_III(M_h, z)
```

The relation should come from a metal-enrichment semi-analytic model driven by
halo histories, not from a direct claim that dark-matter-only simulations
predict Pop III star formation.

## Output Definitions

Use three related quantities and keep them separate:

```text
P_active,III(M_h, z) = probability that a halo hosts active Pop III star formation
f_SFR,III(M_h, z)    = SFR_III / (SFR_III + SFR_II)
f_star,III(M_h, z)   = M_star,III / (M_star,III + M_star,II)
```

The primary deliverable should be `f_SFR,III(M_h, z)` with 16/50/84 per cent
uncertainty bands. `P_active,III` is needed for observability and duty-cycle
work.

## Physical Model

Each halo history tracks:

```text
M_h(z), M_gas(z), M_star,III(z), M_star,II(z), Z_gas(z), f_pristine(z)
```

Pop III star formation is allowed when:

```text
M_h > M_crit(z, J_LW, v_bc, photoheating)
and f_pristine > 0
and Z_local < Z_crit
```

Pop II star formation uses the enriched gas reservoir:

```text
Z_local >= Z_crit
```

Do not use only one perfectly mixed halo metallicity. The key uncertain
variable is the pristine gas fraction `f_pristine`, because inefficient metal
mixing can allow Pop III pockets to survive in already enriched haloes.

## Metal Evolution

Track both self-enrichment and external enrichment:

- Self-enrichment: Pop III SNe/winds inject metals after stellar lifetimes.
- External enrichment: neighbouring SNe enrich haloes through expanding bubbles.
- Mixing: metals do not instantly homogenize; model with a mixing time-scale or
  retained pristine fraction.

Minimum parameter set:

```text
Z_crit
epsilon_III
epsilon_II
t_mix
metal_retention_fraction
external_enrichment_radius_normalization
LW_feedback_strength
photoheating_suppression_mass
```

The final `f_III(M_h,z)` must marginalize over these nuisance parameters or
show sensitivity to them.

## Simulation Strategy

### Phase 0: Semi-analytic prototype

Use Monte Carlo merger trees or existing halo histories. This is enough to test
model behaviour and parameter degeneracies.

Output:

```text
data_save/popiii_fraction_grid_v0.parquet
columns: z, log10_Mh, P_active_p16/p50/p84, f_SFR_p16/p50/p84, f_star_p16/p50/p84
```

### Phase 1: N-body calibrated halo histories

Use GADGET-4 dark-matter-only runs only to provide halo abundance and assembly
histories. Then run halo finder and merger trees.

The current `512^3`, 10 cMpc smoke run is useful for pipeline testing, but it
does not resolve minihalo Pop III hosts. It is more suitable for atomic-cooling
halo and post-processing workflow tests.

### Phase 2: High-resolution or literature calibration

Minihalo Pop III formation requires much higher mass resolution than our current
uniform-box tests. Use one of:

- zoom-in simulations for selected regions;
- published hydrodynamic calibration data;
- semi-analytic merger trees calibrated to Renaissance/Aeos/MERAXES-like
  results.

## Resource Policy

Use the `dmde-compute` skill before submitting jobs.

- Small semi-analytic grids: ordinary `cpu` nodes, or `amd1` only if live memory
  is sufficient and the job is CPU-bound.
- Large halo-history tables, snapshot scans, merger-tree joins, or dense
  parameter grids: `fat2`.
- Avoid `amd1` for memory-heavy metal-history grids; it has about 376 GiB RAM,
  much less than `fat2`.
- SLURM memory is not enforced reliably on this cluster, so scripts need
  chunked I/O, checkpointing, and explicit memory estimates.

## Literature Anchors

- Liu & Bromm 2020, "When did Population III star formation end?", MNRAS 497,
  2839, DOI 10.1093/mnras/staa2143.
- Liu et al. 2021, "Stellar winds and metal enrichment from fast-rotating
  Population III stars", MNRAS 506, 5247, DOI 10.1093/mnras/stab2057.
- Ventura et al. 2024, "Semi-analytic modelling of Pop. III star formation and
  metallicity evolution - I", MNRAS 529, 628, DOI 10.1093/mnras/stae567.
- Visbal, Haiman & Bryan 2015, "Limits on Population III star formation in
  minihaloes implied by Planck", MNRAS 453, 4456, DOI
  10.1093/mnras/stv1941.
- Wise et al. 2014, "The birth of a galaxy - III", MNRAS 442, 2560, DOI
  10.1093/mnras/stu979.

## Acceptance Criteria

The first useful version is complete when it can:

1. Generate halo histories or load merger trees.
2. Track pristine/enriched gas fractions and metal injection.
3. Produce a binned `f_III(M_h,z)` table with uncertainty bands.
4. Reproduce the qualitative trends in Liu & Bromm 2020 and Ventura et al. 2024.
5. Report parameter sensitivity, especially to metal mixing and LW feedback.
