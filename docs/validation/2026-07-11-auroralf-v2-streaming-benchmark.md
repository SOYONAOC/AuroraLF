# AuroraLF v2 streaming benchmark validation

Date: 2026-07-11

## Scope

This validation ran the real canonical BPASS + McBride UVLF pipeline at
`z = 10` with 8 halo-mass samples, 16 tracks per halo mass, 64 time steps,
12 UV-magnitude bins, a mass batch size of 2, and base seed `20260711`.
The three cases ran in separate child processes so their resident-memory peaks
are comparable:

- serial execution, samples disabled;
- two-worker execution, samples disabled;
- two-worker execution, HDF5 sample shards enabled.

## SLURM execution

- Job: `145848`
- Node: `amd1` (`AMD EPYC 9654 96-Core Processor`)
- Allocation: 2 CPUs, 30 minute limit
- Result: `COMPLETED`, exit code `0:0`, elapsed time 29 seconds
- Report: `outputs/uvlf_v2_streaming_benchmark_20260711.json`
- Git revision recorded by the report: `ad8f19aeeda8e75cfd7528b7daee630b868b7eb3`
- Worktree recorded by the report: dirty

The node had about 176.6 GiB physically available immediately before
submission. Slurm reports `RealMemory=1M` for this node, so the job omitted an
`--mem` request after a 16 GiB test request was rejected as an unavailable node
configuration. This is a cluster-accounting limitation, not an observed memory
shortage.

## Results

| Case | Workers | Samples | Wall time (s) | Peak RSS (bytes) | Peak RSS (MiB) |
|---|---:|---|---:|---:|---:|
| `serial_disabled` | 1 | disabled | 2.604 | 185,761,792 | 177.2 |
| `parallel_disabled` | 2 | disabled | 6.463 | 450,355,200 | 429.5 |
| `parallel_samples` | 2 | HDF5 shard | 8.312 | 461,586,432 | 440.2 |

The sample sink added 11,231,232 bytes (10.7 MiB, 2.49%) over the matching
two-worker run without samples. The sample shard contained all 128 expected
tracks and passed mass/track order validation.

All three cases produced the same science digest:

`59a15201fba21b19ba352207e539b7ee25ecc29dd66aed5c40e24bc9aa034eb6`

The controller published the report only after every child exited successfully,
all science digests matched, and the sample-shard count and ordering checks
passed. A separate strict JSON validation rejected duplicate keys and non-finite
tokens and rechecked the complete marker, case order, exit codes, digest
equality, positive timing/RSS values, SLURM provenance, and exact sample count.
