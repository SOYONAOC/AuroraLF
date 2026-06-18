# Pop III Atomic-Cooling Channel Routing Validation

Date: 2026-06-17

Branch: `codex/popiii-new-idea`

## Scope

Added the first HMF sampling-layer Pop III / Pop II routing boundary:

- `M_atomic(z) = massfunc.SFRD().M_vir(mu=0.61, Tvir=1e4, z)`
- halos with `Mh < M_atomic(z_obs)` are marked as `popiii`
- halos with `Mh >= M_atomic(z_obs)` are marked as `popii`

The change records routing metadata in `sample_uvlf_from_hmf()` outputs. It does
not add a Pop III luminosity, feedback, or enrichment model; the current main
branch still has only the existing Pop II UV luminosity pipeline.

## Red-Green Checks

Red check:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_hmf_sampling.py
```

Expected failure before implementation:

- collection failed because `STELLAR_CHANNEL_POPII` and related atomic-cooling
  channel helpers were not yet defined in `auroralf.uvlf.hmf_sampling`

Green checks:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_hmf_sampling.py
```

Result: `11 passed in 5.31s`

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_hmf_sampling.py tests/test_chemistry.py
```

Result: `22 passed in 10.34s`

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests
```

Result: `72 passed in 26.88s`

## Scientific Assumptions

- The atomic-cooling threshold uses the same `massfunc` virial-temperature
  conversion already used by the MAH generator default floor.
- The gas mean molecular weight is fixed to `mu=0.61`.
- Equality with the threshold is assigned to the Pop II channel, so only
  strictly sub-threshold halos are marked Pop III.
- The default HMF sampling range remains `logM_min=9` to avoid changing the
  production UVLF mass range in this routing-only step.

## Remaining Unverified Items

- No Pop III luminosity or chemistry model is connected yet.
- No production SLURM UVLF run was submitted for this routing-only change.
- Default production redshifts with `logM_min=9` are expected to be nearly all
  Pop II because `Tvir=1e4 K` corresponds to about `3e7-2e8 Msun` over the
  relevant high-redshift range.
