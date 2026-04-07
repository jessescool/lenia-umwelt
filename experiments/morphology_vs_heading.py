# experiments/morphology_vs_heading.py — Jesse Cool (jessescool)
"""Two-panel figure: morphological disruption vs heading change."""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CREATURES = ["O2u", "O2v", "K4s", "K6s", "S1s", "P4al"]
SIZES = [1, 2, 3, 4]
SWEEP_ROOT = Path(__file__).resolve().parent.parent / "results" / "sweep"

# status codes
STATUS_EMPTY = -1
STATUS_DIED = 0
STATUS_RECOVERED = 1
STATUS_NEVER = 2


def detect_prefix(analysis_dir: Path) -> str:
    """Auto-detect file prefix from *_recovery_map.npy."""
    hits = list(analysis_dir.glob("*_recovery_map.npy"))
    if not hits:
        return None
    return hits[0].name.replace("_recovery_map.npy", "")


def load_run(run_dir: Path):
    """Load status and |heading_change| from one orientation dir."""
    analysis = run_dir / "analysis"
    if not analysis.is_dir():
        return None

    prefix = detect_prefix(analysis)
    if prefix is None:
        return None

    try:
        status = np.load(analysis / f"{prefix}_recovery_status_map.npy")
        hc = np.load(analysis / f"{prefix}_heading_change.npy")
    except FileNotFoundError:
        return None

    return status, hc


def pool_creature(code: str, scale: int, sizes: list[int]):
    """Pool all orientations × sizes for one creature. Returns aggregate counts and
    the |heading_change| array (degrees) for recovered positions."""
    n_alive, n_recovered, n_died, n_never = 0, 0, 0, 0
    hc_recovered = []

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
            status, hc = result
            alive_mask = status != STATUS_EMPTY
            n_alive += int(alive_mask.sum())
            n_recovered += int((status == STATUS_RECOVERED).sum())
            n_died += int((status == STATUS_DIED).sum())
            n_never += int((status == STATUS_NEVER).sum())
            # heading change for recovered positions only
            rec_mask = status == STATUS_RECOVERED
            hc_recovered.append(np.degrees(np.abs(hc[rec_mask])))

    if n_alive == 0:
        return None

    return {
        "n_alive": n_alive,
        "n_recovered": n_recovered,
        "n_died": n_died,
        "n_never": n_never,
        "recovery_rate": n_recovered / n_alive,
        "died_rate": n_died / n_alive,
        "never_rate": n_never / n_alive,
        "hc_degrees": np.concatenate(hc_recovered) if hc_recovered else np.array([]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Morphology convergence vs heading divergence (two-panel figure)."
    )
    parser.add_argument("--size", type=int, default=None,
                        help="Single intervention size (default: pool all 1-4)")
    parser.add_argument("--scale", type=int, default=4,
                        help="Scale / magnification (default: 4)")
    args = parser.parse_args()

    sizes = [args.size] if args.size is not None else SIZES

    data = {}
    for code in CREATURES:
        d = pool_creature(code, args.scale, sizes)
        if d is None:
            print(f"  SKIP {code}: no data at x{args.scale}")
            continue
        data[code] = d
        print(f"  {code}: {d['n_recovered']:,}/{d['n_alive']:,} recovered "
              f"({d['recovery_rate']:.1%}), median |Δθ| = {np.median(d['hc_degrees']):.1f}°")

    if not data:
        sys.exit("No data found.")

    codes = [c for c in CREATURES if c in data]
    ncols = len(codes)

    clr_recovered = "#4CAF50"   # green
    clr_died = "#E53935"        # red
    clr_never = "#FF9800"       # orange
    tab10 = plt.cm.tab10.colors

    fig, (ax_bar, ax_vio) = plt.subplots(1, 2, figsize=(12, 5),
                                          gridspec_kw={"width_ratios": [1, 1.3]})

    x = np.arange(ncols)
    rec_rates = [data[c]["recovery_rate"] for c in codes]
    died_rates = [data[c]["died_rate"] for c in codes]
    never_rates = [data[c]["never_rate"] for c in codes]

    ax_bar.bar(x, rec_rates, color=clr_recovered, label="Recovered", edgecolor="white", linewidth=0.5)
    ax_bar.bar(x, died_rates, bottom=rec_rates, color=clr_died, label="Died", edgecolor="white", linewidth=0.5)
    ax_bar.bar(x, never_rates, bottom=[r + d for r, d in zip(rec_rates, died_rates)],
               color=clr_never, label="Never recovered", edgecolor="white", linewidth=0.5)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(codes, fontsize=10)
    ax_bar.set_ylabel("Fraction of tested positions", fontsize=10)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.legend(fontsize=8, loc="lower right")
    ax_bar.set_title("A   Perturbation outcomes", fontsize=11, fontweight="bold", loc="left")

    # annotate recovery % on each bar
    for i, c in enumerate(codes):
        ax_bar.text(i, rec_rates[i] / 2, f"{rec_rates[i]:.0%}",
                    ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    violin_data = [data[c]["hc_degrees"] for c in codes]
    positions = np.arange(ncols)

    parts = ax_vio.violinplot(violin_data, positions=positions, showmedians=False,
                               showextrema=False, widths=0.7)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(tab10[i % 10])
        body.set_alpha(0.5)
        body.set_edgecolor("black")
        body.set_linewidth(0.6)

    # box-plot quartiles inside violins
    for i, vd in enumerate(violin_data):
        if len(vd) == 0:
            continue
        q25, med, q75 = np.percentile(vd, [25, 50, 75])
        ax_vio.vlines(i, q25, q75, color="black", linewidth=2.5, zorder=3)
        ax_vio.scatter(i, med, color="white", s=20, zorder=4, edgecolor="black", linewidth=0.8)

        # median annotation above violin
        ymax = np.percentile(vd, 95)
        ax_vio.text(i, ymax + 3, f"{med:.1f}°", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    ax_vio.set_xticks(positions)
    ax_vio.set_xticklabels(codes, fontsize=10)
    ax_vio.set_ylabel("|Heading change| (°)", fontsize=10)
    ax_vio.set_title("B   Heading change among recovered", fontsize=11, fontweight="bold", loc="left")
    ax_vio.set_xlim(-0.6, ncols - 0.4)

    size_label = f"i{args.size}" if args.size else "i1–i4 pooled"
    fig.suptitle(
        f"Morphology converges, heading diverges — x{args.scale}, {size_label}",
        fontsize=13, fontweight="bold", y=1.02
    )

    fig.tight_layout()

    out_dir = SWEEP_ROOT / "cross_animal"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_i{args.size}" if args.size else ""
    out = out_dir / f"morphology_vs_heading{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
