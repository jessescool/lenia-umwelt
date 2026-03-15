"""Single-run heading-change scatter: TTR & max distance vs |heading change|.

Hexbin density plots for one orientation directory showing recovery severity
vs absolute heading change, with Spearman correlation. High severity is
necessary but not sufficient for large heading change.

TTR y-axis is anchored at 5 (warmup floor = 5T) with adaptive tick spacing.

Usage:
    python experiments/correlate_heading.py results/sweep/O2u/O2u_x4/O2u_x4_i3/O2u_x4_i3_o0

Output:
    {run_dir}/analysis/{prefix}_heading_scatter.png
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr
from pathlib import Path

STATUS_RECOVERED = 1


def detect_prefix(analysis_dir: Path) -> str:
    """Auto-detect the file prefix from *_recovery_map.npy in the analysis dir."""
    hits = list(analysis_dir.glob("*_recovery_map.npy"))
    if not hits:
        sys.exit(f"No *_recovery_map.npy found in {analysis_dir}")
    return hits[0].name.replace("_recovery_map.npy", "")


def load_run(run_dir: Path):
    """Load recovered-only TTR, max_distance, and |heading_change| from one run dir."""
    analysis = run_dir / "analysis"
    if not analysis.is_dir():
        sys.exit(f"No analysis/ subdirectory in {run_dir}")

    prefix = detect_prefix(analysis)

    status  = np.load(analysis / f"{prefix}_recovery_status_map.npy")
    ttr     = np.load(analysis / f"{prefix}_recovery_map.npy")
    maxd    = np.load(analysis / f"{prefix}_max_distance.npy")
    hc      = np.load(analysis / f"{prefix}_heading_change.npy")

    mask = status == STATUS_RECOVERED
    return ttr[mask], maxd[mask], np.degrees(np.abs(hc[mask])), prefix, mask.sum()


def main():
    parser = argparse.ArgumentParser(
        description="Heading-change scatter plots for a single run directory."
    )
    parser.add_argument("run_dir", type=Path,
                        help="Path to one run dir (e.g. results/sweep/O2u/O2u_x4/O2u_x4_i2/O2u_x4_i2_o0)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    ttr, maxd, heading, prefix, n = load_run(run_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (y, ylabel) in zip(axes, [
        (ttr,  "TTR (sim steps / T)"),
        (maxd, "Max distance"),
    ]):
        hb = ax.hexbin(heading, y, gridsize=40, cmap="viridis",
                        mincnt=1, linewidths=0.2,
                        norm=LogNorm(vmin=1))

        rho, _ = spearmanr(heading, y)
        ax.set_xlabel("Absolute heading change (\u00b0)")
        ax.set_ylabel(ylabel)

        # TTR: anchor ticks at 5 (warmup floor) with proportional padding
        if "TTR" in ylabel:
            ttr_hi = ax.get_ylim()[1]
            span = max(ttr_hi - 5, 2)  # floor so all-5 panels aren't degenerate
            margin = 0.05 * span
            ax.set_ylim(bottom=5 - margin, top=5 + span + margin)
            raw_step = span / 5
            step = max(1, round(raw_step))
            ticks = np.arange(5, 5 + span + 1, step)
            ax.set_yticks(ticks)
            ax.set_yticklabels(
                ["5 (min)" if t == 5 else f"{t:g}" for t in ticks]
            )
        fig.colorbar(hb, ax=ax, label="count")

        print(f"{prefix}  {ylabel:<28s}  \u03c1={rho:+.4f}  n={n:,}")

    fig.suptitle(f"Heading change correlations \u2014 {prefix}  (n={n:,})", fontsize=13)
    fig.tight_layout()

    out = run_dir / "analysis" / f"{prefix}_heading_scatter.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved \u2192 {out}")


if __name__ == "__main__":
    main()
