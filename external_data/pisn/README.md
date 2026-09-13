# PISN stellar inputs

`marigo2003_nonrotating_lifetimes.csv` transcribes the 120, 250 and 500 Msun
rows of Table 1 (nonrotating, radiatively driven mass loss) in Marigo, Chiosi
& Kudritzki (2003), A&A 399, 617, DOI
[10.1051/0004-6361:20021756](https://doi.org/10.1051/0004-6361:20021756).
[Source](https://arxiv.org/html/astro-ph/0212057v1), `h3997.tex` lines 603–605.
The models have Z=0 and are evolved to central carbon ignition. Columns are
initial mass in Msun, core hydrogen-burning duration in years and core
helium-burning duration in years. Their sum is an explicitly approximate
explosion delay; subsequent burning stages are omitted. Log lifetime is
interpolated in log initial mass, without extrapolation.

The classical 140–260 Msun PISN initial-mass interval is taken separately
from Heger & Woosley (2002), DOI
[10.1086/338487](https://doi.org/10.1086/338487), section 3 and Figure 2.
It corresponds approximately to 64–133 Msun helium cores for metal-free
stars with negligible mass loss. This fate window is a benchmark, not a
fate calculation using the Marigo tracks. Rotation, mass loss, binaries,
and different core-growth prescriptions can change it.
Marigo et al. section 2.3 (PDF page 5, right column) explicitly gives
127–252 Msun for the same helium-core interval in their models; this is
computed as a separate fate-window sensitivity, with the same lifetime table.

The IMF follows the current Raiter, Schaerer & Fosbury (2010) Table 1 logE/TE
SSP: Mc=60 Msun, sigma_lnM=1, 1–500 Msun. Its functional definition is
Tumlinson (2006), section 3.1.3, logarithmic-number IMF equation:
dN/dlnM proportional to exp[-ln(M/Mc)^2/(2 sigma^2)]. Tumlinson's own mass
limits are not substituted for the Raiter SSP limits.

Source PDFs and TeX archives were retrieved into
`external_data/literature_sources/pisn_random_q/`; adjacent JSON files record
URL, retrieval time and SHA-256. The analysis records this CSV's SHA-256.
The lifetime and fate inputs come from different stellar evolution models
than the UV SSP and do not constitute a single self-consistent grid.
