# experiments/competency_profile.py — Jesse Cool (jessescool)
"""Competency profiles: attractor basin cross-sections across intervention sizes.

The *shape* of these curves -- linear? threshold? gradual? -- tells you about
the topology of the attractor basin. Sharp transitions suggest a cliff-edged
basin; gradual degradation suggests a smooth bowl.
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

STATUS_RECOVERED = 1
STATUS_EMPTY = -1
CREATURES = ["O2u", "O2v", "K4s", "K6s", "S1s", "P4al"]
SWEEP_ROOT = Path(__file__).resolve().parent.parent / "results" / "sweep"

# distinct colors per creature (colorblind-friendly palette)
COLORS = {
    "O2u":  "#1b9e77",
    "O2v":  "#d95f02",
    "K4s":  "#7570b3",
    "K6s":  "#e7298a",
    "S1s":  "#66a61e",
    "P4al": "#e6ab02",
}


def detect_prefix(analysis_dir: Path) -> str:
    hits = list(analysis_dir.glob("*_recovery_map.npy"))
    if not hits:
        return None
    return hits[0].name.replace("_recovery_map.npy", "")


def load_run(run_dir: Path):
    """Load ALL data (not just recovered) for one run dir.

    Returns: (ttr, maxd, heading_change, status) arrays, or None.
    """
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

    return ttr, maxd, hc, status


def pool_orientations(code: str, scale: int, size: int):
    """Pool all orientation runs, returning full arrays (not just recovered)."""
    size_dir = SWEEP_ROOT / code / f"{code}_x{scale}" / f"{code}_x{scale}_i{size}"
    if not size_dir.is_dir():
        return None

    ttr_all, maxd_all, hc_all, status_all = [], [], [], []
    for ori_dir in sorted(size_dir.iterdir()):
        if not ori_dir.is_dir():
            continue
        result = load_run(ori_dir)
        if result is None:
            continue
        ttr, maxd, hc, status = result
        ttr_all.append(ttr)
        maxd_all.append(maxd)
        hc_all.append(hc)
        status_all.append(status)

    if not ttr_all:
        return None

    return {
        "ttr":    np.concatenate(ttr_all),
        "maxd":   np.concatenate(maxd_all),
        "hc":     np.concatenate(hc_all),
        "status": np.concatenate(status_all),
    }


def compute_stats(d):
    """Compute summary stats from pooled data dict.

    Returns dict with: recovery_frac, median_ttr, p90_ttr, median_hc, p90_hc,
                        median_maxd, p90_maxd, n_tested, n_recovered
    """
    tested = d["status"] != STATUS_EMPTY
    recovered = d["status"] == STATUS_RECOVERED
    n_tested = tested.sum()
    n_recovered = recovered.sum()

    # recovery fraction: among tested, how many recovered?
    rec_frac = n_recovered / n_tested if n_tested > 0 else 0.0

    # stats on recovered subset only
    ttr_rec  = d["ttr"][recovered]
    maxd_rec = d["maxd"][recovered]
    hc_rec   = np.degrees(np.abs(d["hc"][recovered]))

    if n_recovered == 0:
        return {
            "recovery_frac": rec_frac,
            "median_ttr": np.nan, "p90_ttr": np.nan,
            "median_hc":  np.nan, "p90_hc":  np.nan,
            "median_maxd": np.nan, "p90_maxd": np.nan,
            "n_tested": int(n_tested), "n_recovered": int(n_recovered),
        }

    return {
        "recovery_frac": rec_frac,
        "median_ttr":  np.median(ttr_rec),
        "p90_ttr":     np.percentile(ttr_rec, 90),
        "median_hc":   np.median(hc_rec),
        "p90_hc":      np.percentile(hc_rec, 90),
        "median_maxd": np.median(maxd_rec),
        "p90_maxd":    np.percentile(maxd_rec, 90),
        "n_tested":    int(n_tested),
        "n_recovered": int(n_recovered),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Competency profiles: attractor basin cross-sections."
    )
    parser.add_argument("--scale", type=int, default=4,
                        help="Scale / magnification (default: 4)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Intervention sizes to include (default: 1 2 3 4)")
    args = parser.parse_args()

    sizes = sorted(args.sizes)

    # stats_by[code][size] = stats_dict
    stats_by = {}
    for code in CREATURES:
        stats_by[code] = {}
        for sz in sizes:
            d = pool_orientations(code, args.scale, sz)
            if d is None:
                print(f"  SKIP {code} i{sz}: no data")
                continue
            st = compute_stats(d)
            stats_by[code][sz] = st
            print(f"  {code} i{sz}: {st['n_recovered']:,}/{st['n_tested']:,} "
                  f"recovered ({st['recovery_frac']:.1%}), "
                  f"median TTR={st['median_ttr']:.1f}")

    # filter to creatures with at least some data
    codes = [c for c in CREATURES if stats_by[c]]
    if not codes:
        sys.exit("No data found.")

    metric_defs = [
        ("recovery_frac", "Recovery fraction",          (0, 1.05)),
        ("median_ttr",    "Median TTR (steps)",         None),
        ("p90_ttr",       "90th pctl TTR (steps)",      None),
        ("median_hc",     "Median |heading change| (deg)", None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.ravel()

    for ax, (key, ylabel, forced_ylim) in zip(axes, metric_defs):
        for code in codes:
            xs = sorted(stats_by[code].keys())
            ys = [stats_by[code][sz].get(key, np.nan) for sz in xs]
            ax.plot(xs, ys, "o-", color=COLORS.get(code, "gray"),
                    label=code, linewidth=2, markersize=6)

        ax.set_xlabel("Intervention size", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(sizes)
        ax.grid(True, alpha=0.3)
        if forced_ylim:
            ax.set_ylim(forced_ylim)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Competency profiles \u2014 attractor basin cross-sections (x{args.scale})",
        fontsize=13
    )

    out_dir = SWEEP_ROOT / "cross_animal"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"competency_profiles_x{args.scale}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
