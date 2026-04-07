"""Correlation between morphological disruption severity and heading change."""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CREATURES = ["O2u", "O2v", "K4s", "K6s", "S1s", "P4al"]
SIZES = [1, 2, 3, 4]
SWEEP_ROOT = Path(__file__).resolve().parent.parent / "results" / "sweep"

STATUS_RECOVERED = 1
STATUS_EMPTY = -1


def detect_prefix(analysis_dir: Path) -> str:
    """Auto-detect file prefix from *_recovery_map.npy."""
    hits = list(analysis_dir.glob("*_recovery_map.npy"))
    if not hits:
        return None
    return hits[0].name.replace("_recovery_map.npy", "")


def load_run(run_dir: Path):
    """Load max_distance and |heading_change| for recovered positions from one orientation dir."""
    analysis = run_dir / "analysis"
    if not analysis.is_dir():
        return None

    prefix = detect_prefix(analysis)
    if prefix is None:
        return None

    try:
        status = np.load(analysis / f"{prefix}_recovery_status_map.npy")
        maxd   = np.load(analysis / f"{prefix}_max_distance.npy")
        hc     = np.load(analysis / f"{prefix}_heading_change.npy")
    except FileNotFoundError:
        return None

    mask = status == STATUS_RECOVERED
    return maxd[mask], np.abs(hc[mask])  # heading_change in radians, take abs


def pool_creature(code: str, scale: int, sizes: list[int]):
    """Pool all orientations × sizes for one creature.
    Returns (max_distance, |heading_change| in degrees) arrays for recovered positions."""
    maxd_all, hc_all = [], []

    for size in sizes:
        size_dir = SWEEP_ROOT / code / f"{code}_x{scale}" / f"{code}_x{scale}_i{size}"
        if not size_dir.is_dir():
            continue
        for ori_dir in sorted(size_dir.iterdir()):
            if not ori_dir.is_dir():
                continue
            result = load_run(ori_dir)
            if result is None:
                continue
            maxd, hc = result
            maxd_all.append(maxd)
            hc_all.append(hc)

    if not maxd_all:
        return None

    return {
        "max_distance": np.concatenate(maxd_all),
        "hc_degrees": np.degrees(np.concatenate(hc_all)),
        "n": sum(len(m) for m in maxd_all),
    }


def bin_and_summarize(maxd, hc_deg, n_bins):
    """Bin max_distance into fixed-width bins, return bin centers, mean |heading change|, SEM.
    Clips to 95th percentile to avoid noisy sparse tails."""
    hi = np.percentile(maxd, 95)
    lo = maxd.min()
    if hi - lo < 1e-12:
        return None, None, None

    # clip to 95th percentile
    keep = maxd <= hi
    maxd, hc_deg = maxd[keep], hc_deg[keep]

    edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.digitize(maxd, edges[1:-1])  # 0 .. n_bins-1
    centers, means, sems = [], [], []
    for i in range(len(edges) - 1):
        mask = bin_idx == i
        if mask.sum() < 2:
            continue
        vals = hc_deg[mask]
        centers.append((edges[i] + edges[i + 1]) / 2)
        means.append(vals.mean())
        sems.append(vals.std(ddof=1) / np.sqrt(len(vals)))

    return np.array(centers), np.array(means), np.array(sems)


def main():
    parser = argparse.ArgumentParser(
        description="Heading sacrifice correlation: disruption severity vs heading change."
    )
    parser.add_argument("--size", type=int, default=None,
                        help="Single intervention size (default: pool all 1-4)")
    parser.add_argument("--scale", type=int, default=4,
                        help="Scale / magnification (default: 4)")
    parser.add_argument("--bins", type=int, default=15,
                        help="Number of quantile bins (default: 15)")
    args = parser.parse_args()

    sizes = [args.size] if args.size is not None else SIZES

    data = {}
    for code in CREATURES:
        d = pool_creature(code, args.scale, sizes)
        if d is None:
            print(f"  SKIP {code}: no data at x{args.scale}")
            continue
        data[code] = d
        print(f"  {code}: {d['n']:,} recovered positions, "
              f"median max_d = {np.median(d['max_distance']):.3f}, "
              f"median |Δθ| = {np.median(d['hc_degrees']):.1f}°")

    if not data:
        sys.exit("No data found.")

    codes = [c for c in CREATURES if c in data]
    tab10 = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, code in enumerate(codes):
        d = data[code]
        centers, means, sems = bin_and_summarize(
            d["max_distance"], d["hc_degrees"], args.bins
        )
        if centers is None:
            print(f"  SKIP {code}: not enough bins after dedup")
            continue

        color = tab10[i % 10]
        ax.plot(centers, means, color=color, linewidth=1.8, label=code, zorder=3)
        ax.fill_between(centers, means - sems, means + sems,
                        color=color, alpha=0.2, zorder=2)

    ax.set_xlabel("Max distance (peak profile displacement)", fontsize=11)
    ax.set_ylabel("|Heading change| (°)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    size_label = f"i{args.size}" if args.size else "i1–i4 pooled"
    ax.set_title(
        f"Heading sacrifice: disruption severity → heading change\n"
        f"x{args.scale}, {size_label}, {args.bins} fixed-width bins (≤95th pctl)",
        fontsize=12, fontweight="bold"
    )

    fig.tight_layout()

    out_dir = SWEEP_ROOT / "cross_animal"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_i{args.size}" if args.size else ""
    out = out_dir / f"heading_sacrifice_correlation{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
