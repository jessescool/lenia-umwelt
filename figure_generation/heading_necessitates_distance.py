"""Scatter plots: heading change requires centroid displacement.

Output:
    figure_generation/heading_necessitates_distance_x{SCALE}_i{SIZE}.png
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CREATURE_COLORS

STATUS_RECOVERED = 1
CREATURES = ["O2u", "S1s", "K4s", "K6s"]
ROOT = Path(__file__).resolve().parent.parent
SWEEP_ROOT = ROOT / "results" / "sweep"


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
        return None, None, None, 0, 0

    ttr_all, maxd_all, hc_all = [], [], []
    total = 0
    for candidate in sorted(size_dir.iterdir()):
        if not candidate.is_dir():
            continue
        result = load_run(candidate)
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

    data = {}
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
        ("ttr",  "Steps until recovery", 0),
        ("maxd", "Max distortion",             1),
    ]

    # shared LogNorm floor: 1 count in the largest creature
    max_n = max(len(data[c][0]) for c in codes)
    vmin_frac = 1 / max_n

    col_w = 3.2
    w = col_w * ncols
    panel = w / (ncols + 0.3)  # approx panel width after colorbar
    fig, axes = plt.subplots(2, ncols, figsize=(w, 2 * panel - 0.8),
                             squeeze=False, layout='compressed')

    titles_to_dot = []  # (ax, color) pairs for post-render dot placement
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
                ax.set_ylabel(ylabel, fontsize=11)
            else:
                ax.set_ylabel("")
            ax.tick_params(axis='y', labelsize=6)

            if row < 1:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            # column header on top row with creature color dot
            if row == 0:
                color = CREATURE_COLORS.get(code, "#333333")
                ax.set_title(f"   $\\bf{{{code}}}$  (n={n:,}/{total:,})", fontsize=11)
                titles_to_dot.append((ax, color))

        # anchor TTR y-axis at 5 with same relative offset as 0 in max_distance
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
            [f"{t:g}" for t in ticks]
        )

    mappable = axes[0, -1].collections[0]
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(),
                        location="right", shrink=0.85, pad=0.02)
    cbar.set_label("concentration", fontsize=11)

    # place colored dots just left of each creature title
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, color in titles_to_dot:
        bb = ax.title.get_window_extent(renderer=renderer)
        # convert left edge of title bbox to axes fraction
        inv = ax.transAxes.inverted()
        x_left, y_mid = inv.transform((bb.x0, (bb.y0 + bb.y1) / 2 + 3))
        ax.plot(x_left, y_mid, 'o', color=color, markersize=6,
                transform=ax.transAxes, clip_on=False, zorder=10)

    # shared x-axis label
    fig.supxlabel("Absolute heading change (\u00b0)", fontsize=12)

    out_dir = ROOT / "figure_generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"heading_necessitates_distance_x{args.scale}_i{args.size}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"\nSaved \u2192 {out}")


if __name__ == "__main__":
    main()
