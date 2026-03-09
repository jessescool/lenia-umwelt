# Color Scaling in Sweep Plots

## Power-law normalization (gamma)

All three summary maps use `PowerNorm(gamma=0.5)` — a square-root transform on the color mapping.

The idea: `color_position = ((value - vmin) / (vmax - vmin)) ^ gamma`

- **gamma = 1.0** — linear. Color is proportional to value.
- **gamma = 0.5** — sqrt. Expands the lower range, compresses the upper range. Small differences between low values become visually distinguishable.
- **gamma < 0.5** — more aggressive. Pushes more of the colormap budget toward low values.
- **gamma → 0** — step function. Everything except zero maps to ~1.

Sqrt (0.5) is a good middle ground: it reveals structure in the low-to-mid range without flattening the high end into a uniform blob. Log would be more aggressive but has edge cases near zero and is harder to interpret on a colorbar.

The constant lives at the top of `viz/maps.py` as `_GAMMA = 0.5`.

## Percentile clipping (p2/p98)

Before applying the power norm, we clip the colorbar range to the 2nd and 98th percentiles of the recovered-pixel values.

**Why this matters:** Recovery time and max distance distributions are typically heavy-tailed — a handful of extreme outliers can be 10x larger than the bulk of the data. Without clipping, the colorbar range spans [min, max], and 95%+ of pixels get squeezed into a tiny sliver at the bottom of the colormap. The result looks like a single flat color with a few bright spots.

By clipping at p2/p98:
- The full colormap gradient spans the range where most of the data actually lives
- The 2% of pixels below p2 saturate at the bottom color (no information lost — they're "fast" anyway)
- The 2% above p98 saturate at the top color (they're the extreme outliers)
- The middle 96% gets the full visual dynamic range

The heading plot clips at p98 only (vmin is naturally 0 for deflection angles).

Both the clipping percentiles and the gamma are noted in small text at the bottom of each plot for reproducibility.
