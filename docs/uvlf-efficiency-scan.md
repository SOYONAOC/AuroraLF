# UVLF efficiency comparison, 2026-09-14

Compare Pop II alone with Pop II+III at burst efficiencies 0.01, 0.03 and 0.1
for z=6, 8, 12.5 and 14.5. Keep the threshold distribution and all histories
fixed. This is an efficiency comparison, not a joint fit of the threshold
distribution and efficiency.

The high-redshift saved UVLFs already contain all three efficiencies. At low
redshift, rescale the saved conditional Pop III luminosity of each halo by
epsilon/0.03, add its original Pop II light, and rebuild the probability-weighted
histogram. Do not scale the binned total luminosity function. The job verifies
that epsilon=0.03 reproduces the saved LF and mass-cluster MC standard error.

Both low-redshift panels apply the existing dust mapping; no dust-free curves
are displayed. Low-redshift dust curves show central predictions, while the
intrinsic MC errors are retained in NPZ products. High-redshift panels show
their saved MC standard errors. Preserve the original observations and the
historical wavelength distinction: Pop II1600/Pop III1500 at high redshift,
1500 A for both populations at low redshift.

Run from the repository root:

```bash
PYTHONPATH=. .venv/bin/python scripts/analysis/build_uvlf_efficiency_scan.py \
  --plan configs/experiments/uvlf_efficiency_scan_20260914_mailfixed.json --validate-only
bash scripts/submit/submit_uvlf_efficiency_scan_20260914.sh --dry-run
bash scripts/submit/submit_uvlf_efficiency_scan_20260914.sh
```

The frozen plan records source hashes and output paths. A changed input or an
existing output directory is an error; use a new plan/output name for a new run.
The launcher uses the DMDE scheduler and the project virtual environment, with
END/FAIL notifications to the verified agent mailbox. It requests no wall-time
or memory limit. This task needs only saved-sample post-processing, so no remote
supercomputer or new formation-history run is required.

Products:

- `data_save/uvlf_efficiency_scan_20260914/`: four UVLF NPZ files, observational
  bin comparisons and completion manifest.
- `outputs/uvlf_efficiency_scan_20260914/`: figure previews, slide previews and
  XeLaTeX build log.
- `slides/popiii_heii_pisn_eps_scan_20260914/`: separate source/PDF deck with the
  two UVLF pages updated. The original deck remains the previous edition.

Successful compilation checks the log and exports slide previews. Its QA record
explicitly leaves visual review pending. Per user instruction, no recurring job
monitor or follow-up task is configured.

Job 157741 completed successfully on DMDE node4 (cpu partition). The 0.03
reproduction check passed, and both updated slide pages were visually reviewed.
The original submitted plan remains immutable. The launcher now uses the
`_mailfixed.json` plan and explicit sbatch mail options for SLURM 20.11; its
dry-run was checked. This job finished before its notification settings could
be updated, so a completion email with the PDF was separately accepted for
queueing by Agently. The submitted launcher and workflow note are archived
under the diagnostic output directory. No monitoring task is active.

## Increased low-redshift sampling

The 2026-09-14 follow-up increases z=6 and z=8 independently from 720 mass
samples x 512 MAHs to 5760 x 1024 (5,898,240 histories per redshift, 16x).
The mass range, HMF, time grid, SSPs, first-crossing distribution, age strata,
efficiencies and dust prescription are unchanged. Uniform log-mass draws and
MAH/conditional-age random streams are nested, so the original 720 x 512 samples
must reproduce the parent luminosities, probabilities, LF and MC errors.
A single whole-node SLURM job performs new MAH/SFR/SSP calculations, saves all
conditional samples, rebuilds the three efficiency curves and exports slides.
The high-redshift results are reused. No smoothing is applied.

```bash
PYTHONPATH=. .venv/bin/python scripts/run/increase_lowz_uvlf_sampling.py \
  --plan configs/experiments/uvlf_lowz_sampling_20260914.json --validate-only
bash scripts/submit/submit_lowz_uvlf_sampling_20260914.sh --dry-run
bash scripts/submit/submit_lowz_uvlf_sampling_20260914.sh
```

Outputs are `data_save/uvlf_lowz_sampling_20260914/` for full samples and nested
convergence arrays, `data_save/uvlf_efficiency_scan_20260914_16x/` for rebuilt
curves, `outputs/uvlf_lowz_sampling_20260914/` for convergence plots/error tables,
and `slides/popiii_heii_pisn_eps_scan_20260914_16x/` for the generated deck.
The full-size, half-size and original-size comparisons preserve mass-cluster
MC errors. Two disjoint equal halves also supply binwise differences in units
of their quadrature MC error. These diagnostics measure sampling error, not
time-grid convergence or model uncertainty; 16x sampling is not an automatic
claim of convergence. The old numerical slide caption is replaced by the new
sample counts; updated observational ratios are saved in the scan summary.

END/FAIL notifications use explicit scheduler mail flags for
`lighmisamisa@agent.qq.com`. No recurring monitor is created. Compiled slide QA
remains pending visual review until the generated pages are inspected.

The active retry plan is `configs/experiments/uvlf_lowz_sampling_20260914_r02.json`,
and its full samples are saved to `data_save/uvlf_lowz_sampling_20260914_r02/`.
Job 157745 stopped at its startup mass check: exp10 on different compute CPUs
changed 52 of 720 values by at most 2.18e-16 relative. The retry verifies agreement
to 1e-14, then uses the exact archived masses for the nested original samples.
No physical parameter or random draw is changed. The failed attempt is retained.

Job 157746 also failed during recomputation of the old samples: one conditional
Pop III luminosity differed by 1.175e-10 relative, exceeding the 1e-11 replay
check. No expanded UVLF or slide PDF was produced by either failed attempt.
The r03 plan implements direct sample extension: load and preserve the exact
720 x 512 parent luminosities and probabilities, and compute only histories
512..1023 at the original 720 masses and 0..1023 at the 5040 additional masses.
The seed uses the absolute history chunk index, so no new history duplicates an
archived one. Parent weights are renormalized by 1/16; the HMF calculation must
agree within 1e-12. Exact preservation of all parent samples and reproduction
of their nested LF remain enforced. This replaces unnecessary replay of valid
archived samples with explicit reuse; the physical model is unchanged.
The active plan is `uvlf_lowz_sampling_20260914_r03.json`; raw samples are in
`data_save/uvlf_lowz_sampling_20260914_r03/`. Earlier attempts remain archived.

Job 157748 stopped at the retained HMF-weight comparison (maximum relative
replay difference 1.425e-11). The r04 plan uses a declared 1e-8 relative
cross-CPU HMF replay tolerance, while still retaining the original weights
exactly after the sampling-normalization change. This is many orders below
sampling errors and is not a change to the HMF or its numerical calculation.
The active plan/raw-sample directory now end in `_r04`; neither failed retry
produced a UVLF result. Sample values and original-subset LF checks remain as
specified above. Failures and source snapshots are retained in outputs/.

Per the user's explicit request, job 157749 was cancelled and the r05 retry
requests exactly 30 CPUs without exclusive allocation. The mail submitter's
`--fixed-cpus 30` option retains live scheduler-state and speed-based node
selection while capping the allocation to the requested count; it only accepts
nodes with at least 30 idle CPUs and more than 30 total CPUs. Worker count is
read from the resulting SLURM allocation. This explicit request overrides the
usual 80-percent shared-node heuristic. Scientific inputs and sample counts are
unchanged. r04 partial arrays remain archived; r05 restarts in a separate output
directory using the validated original samples, not unfinished arrays.

## Further extension to 23040 mass draws

After job 157752, median relative intrinsic MC SE for epsilon=0.03 was 6.13%
at z=6 and 5.41% at z=8 (bin centres -24 <= M_UV <= -16). The user requested
more sampling. The 64x plan extends the completed r05 sample to 23040 independent
mass draws x 1024 MAHs: four times the last run, 64 times the initial 720 x 512.
All 5760 x 1024 existing samples per redshift are retained exactly. Only the
additional 17280 masses are calculated, adding 17,694,720 histories per redshift.
The expected factor-two reduction in MC SE is an estimate, to be tested by the
new nested and independent-half diagnostics. No physical parameters are changed.

The extension runner now reads parent dimensions from the actual completed
manifest/arrays, skips all fully retained masses, normalizes archived weights
by the old/new total sample ratio, and records dimensions in the convergence
NPZs. `original_size` denotes the immediate parent run. Plot labels and slide
sample counts reflect the current run rather than hardcoded initial dimensions.
Focused tests check exact coverage and no duplicated or discarded sample indices.

Plan: `configs/experiments/uvlf_lowz_sampling_64x_20260914.json`.
Launcher: `scripts/submit/submit_lowz_uvlf_sampling_64x_20260914.sh`.
Submit through `~/.local/bin/codex-job-submit -- bash <launcher>` so the job is
registered with the current Codex task. Scheduler allocation uses available CPUs
minus one, non-exclusive; END/FAIL mail goes to lighmisamisa@agent.qq.com.
The event-only watcher triggers result/slide review once the job ends.

Raw samples: `data_save/uvlf_lowz_sampling_64x_20260914/`.
Curves: `data_save/uvlf_efficiency_scan_20260914_64x/`.
Diagnostics: `outputs/uvlf_lowz_sampling_64x_20260914/`.
Current deck: `slides/popiii_heii_pisn_complete_20260916/popiii_heii_pisn.pdf`
(historical 2026-09-16 version: 20 pages, including the Pop II baseline introduction, V24 figure reference and a combined five-redshift He II comparison at epsilon=0.03; the cumulative He II flux page remains removed). The superseded
16-page PDF and its build intermediates were removed on 2026-09-16; the original
export source, figure assets and provenance remain in
`slides/popiii_heii_pisn_eps_scan_20260914_64x/` for reproducibility.

At this submission, no node had 30 idle CPUs. The 64x launcher therefore submits
a queued 30-CPU job to the cpu partition without pinning or exclusive allocation;
SLURM places it when sufficient CPUs become available. The plan and scientific
sample count are unchanged. The dry run displays the real sbatch command and
never substitutes a smaller allocation or a login-node calculation.

The user subsequently removed the 30-CPU constraint to prioritize immediate
startup. Pending job 157879 was cancelled before running. The launcher now uses
`--leave-idle-cpus 1`: fresh live availability minus one CPU, shared placement,
ranked by available aggregate benchmark throughput. Down/drained/debug nodes
remain excluded. This explicit resource preference overrides the generic
80-percent and per-core-first policy; scientific sampling is unchanged. The
spare CPU is left unrequested, not reserved against other users.


## Threshold mean zero: 2026-09-18

The current 42-page main deck uses log10(q) ~ Normal(0, 1.5^2) for the two UVLF
pages (8/9). cp6 array 11690033 completed all ten tasks with exit 0: R040/R041
at z=12.5/14.5 each use 3600 masses x 1000 histories; R042/R043 at z=6/8 each
use 23040 masses x 1024 histories, split into four disjoint global mass shards.
All histories and first crossings were recomputed. Efficiencies remain
0.01/0.03/0.1. Dust treatment and observational inputs are unchanged.

Configs: `configs/experiments/random_q_R040.toml` through `random_q_R043.toml`.
Products and validated manifests: `data_save/threshold_zero_20260918/`.
Finalizer: `PYTHONPATH=. .venv/bin/python scripts/analysis/finish_threshold_zero_uvlf.py`.
The low-z raw samples remain in the frozen remote release
`/fs2/home/xuelei/zhuhr/AuroraLF/releases/threshold-zero-20260918-01/data_save/`;
local shard curves and manifests retain their integrity hashes. High-z raw
samples are also available locally. Low-z Pop II baselines reproduce the old
curves to <5e-14 relative error. The same-seed high-z baseline, masses, weights,
and shifted logq were independently checked. High-z 1-mag bins have about
8–10% MC errors at efficiency 0.03; this is a new run, not reweighting.

Main deck: `slides/popiii_heii_pisn_complete_20260916/popiii_heii_pisn.pdf`.
He II, PISN and imported observation-audit predictions retain their original
mean-0.5 results, explicitly labeled in the deck. They were not recalculated
in this two-page UVLF update. Diagnostics and visual review are under
`outputs/threshold_zero_campaign_20260918/`.
