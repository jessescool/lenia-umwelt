"""Cross-animal heading-change scatter: 2xN grid (TTR + max_distance) x creatures.

For each creature, pools all orientations at a given scale and intervention size,
then plots hexbin density of recovery severity (TTR, max displacement) vs absolute
heading change. Shared LogNorm color scale across creatures per metric row.

TTR y-axis is anchored at 5 (warmup floor = 5T) with adaptive tick spacing.

Usage:
    python experiments/correlate_heading_cross.py --size 3
    python experiments/correlate_heading_cross.py --size 1 --scale 4 --gridsize 30

Output:
    results/sweep/cross_animal/heading_scatter_x{SCALE}_i{SIZE}.png
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

STATUS_RECOVERED = 1
CREATURES = ["O2u", "O2v", "K4s", "K6s", "S1s", "P4al"]
SWEEP_ROOT = Path(__file__).resolve().parent.parent / "results" / "sweep"


def detect_prefix(analysis_dir: Path) -> str:
    """Auto-detect the file prefix from *_recovery_map.npy in the analysis dir."""
    hits = list(analysis_dir.glob("*_recovery_map.npy"))
    if not hits:
        return None
    return hits[0].name.replace("_recovery_map.npy", "")


def load_run(run_dir: Path):
    """Load recovered-only TTR, max_distance, and |heading_change| from one run dir."""
    analysis = run_dir / "analysis"
    if not analysis.is_dir():
        return None

    prefix = detect_prefix(analysis)
    if prefix is None:
        return None

    try:
        status = np.load(analysis / f"{prefix}_recovery_status_map.npy")
        ttr    = np.load(analysis / f"{prefix}_recovery_map.npy")
        maxd   = np.load(analysis / f"{prefix}_max_distance.npy")
        hc     = np.load(analysis / f"{prefix}_heading_change.npy")
    except FileNotFoundError:
        return None

    mask = status == STATUS_RECOVERED
    tested = int((status != -1).sum())  # exclude STATUS_EMPTY
    return ttr[mask], maxd[mask], np.degrees(np.abs(hc[mask])), tested


def pool_orientations(code: str, scale: int, size: int):
    """Pool all orientation runs for one creature at a given scale and size."""
    size_dir = SWEEP_ROOT / code / f"{code}_x{scale}" / f"{code}_x{scale}_i{size}"
    if not size_dir.is_dir():
        return None, None, None, 0

    ttr_all, maxd_all, hc_all = [], [], []
    total = 0
    for ori_dir in sorted(size_dir.iterdir()):
        if not ori_dir.is_dir():
            continue
        result = load_run(ori_dir)
        if result is None:
            continue
        ttr, maxd, hc, n_total = result
        ttr_all.append(ttr)
        maxd_all.append(maxd)
        hc_all.append(hc)
        total += n_total

    if not ttr_all:
        return None, None, None, 0, 0

    return (np.concatenate(ttr_all),
            np.concatenate(maxd_all),
            np.concatenate(hc_all),
            len(ttr_all),
            total)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-animal heading-change scatter (2x6 grid)."
    )
    parser.add_argument("--size", type=int, default=3,
                        help="Intervention size (default: 3)")
    parser.add_argument("--scale", type=int, default=4,
                        help="Scale / magnification (default: 4)")
    parser.add_argument("--gridsize", type=int, default=40,
                        help="Hexbin gridsize (default: 40)")
    args = parser.parse_args()

    # --- collect data per creature ---
    data = {}  # code -> (ttr, maxd, heading, n_oris, total)
    for code in CREATURES:
        ttr, maxd, hc, n_oris, total = pool_orientations(code, args.scale, args.size)
        if ttr is None:
            print(f"  SKIP {code}: no data at x{args.scale}_i{args.size}")
            continue
        data[code] = (ttr, maxd, hc, n_oris, total)
        print(f"  {code}: {len(ttr):,}/{total:,} recovered from {n_oris} orientations")

    if not data:
        sys.exit("No data found for any creature.")

    codes = [c for c in CREATURES if c in data]
    ncols = len(codes)

    metrics = [
        ("ttr",  "TTR (sim steps / T)", 0),
        ("maxd", "Max distance",             1),
    ]

    # shared LogNorm floor: 1 count in the largest creature
    max_n = max(len(data[c][0]) for c in codes)
    vmin_frac = 1 / max_n

    # --- main figure ---
    col_w = 3.2
    w = col_w * ncols
    panel = w / (ncols + 0.3)  # approx panel width after colorbar
    fig, axes = plt.subplots(2, ncols, figsize=(w, 2 * panel + 0.6),
                             squeeze=False, layout='compressed')

    for col, code in enumerate(codes):
        ttr, maxd, hc, n_oris, total = data[code]
        n = len(ttr)

        for key, ylabel, row in metrics:
            ax = axes[row, col]
            y = ttr if key == "ttr" else maxd

            hb = ax.hexbin(hc, y, gridsize=args.gridsize, cmap="viridis",
                           mincnt=1, linewidths=0.2)
            ax.set_box_aspect(1)
            hb.set_array(hb.get_array() / n)
            hb.set_norm(LogNorm(vmin=vmin_frac, vmax=1))

            # axis labels: only left column gets y-label, only bottom row gets x-label
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            else:
                ax.set_ylabel("")
            ax.tick_params(axis='y', labelsize=6)

            if row < 1:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            # column header on top row
            if row == 0:
                ax.set_title(f"{code}  (n={n:,}/{total:,})", fontsize=9)

        # --- TTR: put "5" at same relative offset as "0" in max_distance ---
        ttr_ax = axes[0, col]
        ttr_hi = ttr_ax.get_ylim()[1]
        span = max(ttr_hi - 5, 2)  # floor at 2 so all-5 panels still get visible padding
        margin = 0.05 * span  # match matplotlib's default y-margin
        ttr_ax.set_ylim(bottom=5 - margin, top=5 + span + margin)

        # adaptive tick spacing anchored at 5
        ymax = 5 + span
        raw_step = span / 5
        step = max(1, round(raw_step))
        ticks = np.arange(5, ymax + 1, step)
        ttr_ax.set_yticks(ticks)
        ttr_ax.set_yticklabels(
            ["5 (min)" if t == 5 else f"{t:g}" for t in ticks]
        )

    # --- single shared colorbar ---
    mappable = axes[0, -1].collections[0]
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(),
                        location="right", shrink=0.85, pad=0.02)
    cbar.set_label("concentration", fontsize=8)

    # shared x-axis label
    fig.supxlabel("Absolute heading change (\u00b0)", fontsize=10)

    fig.suptitle(
        f"Heading change vs severity \u2014 x{args.scale}, intervention size {args.size}",
        fontsize=12
    )
    # --- save ---
    out_dir = SWEEP_ROOT / "cross_animal"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"heading_scatter_x{args.scale}_i{args.size}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    print(f"\nSaved \u2192 {out}")


if __name__ == "__main__":
    main()
