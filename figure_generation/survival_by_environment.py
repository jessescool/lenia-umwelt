"""Generate survival-by-environment range plot.

Reads per-creature .npz files from results/env_competency/ and produces
a grouped range plot: median (solid dot), IQR (thick bar), min–max (hollow dots).

Survival is defined as the last time the creature is within lambda * d_max
of its orbit, computed from per-timestep distance arrays.

Usage:
    python figure_generation/survival_by_environment.py
    python figure_generation/survival_by_environment.py --lambda 5
    python figure_generation/survival_by_environment.py --codes O2u S1s K4s K6s
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── display constants ──

from config import CREATURE_ORDER, CREATURE_COLORS, ENV_ORDER

DEATH_THRESH = 0.01


def load_d_max(code: str, scale: int, input_dir: Path) -> float:
    orbit_path = Path(f"orbits/{code}/s{scale}/{code}_s{scale}_orbit.pt")
    if not orbit_path.exists():
        return None
    orbit = torch.load(orbit_path, weights_only=False)
    return float(orbit["d_max"])


def load_all(input_dir: Path, codes: list[str] | None = None,
             lam: float = 5.0, scale: int = 4) -> dict:
    """Load creature data and compute last-return from distance arrays."""
    data = {}
    for d in sorted(input_dir.iterdir()):
        if not d.is_dir():
            continue
        if codes and d.name not in codes:
            continue
        code = d.name

        # Load centroids npz (has distance array)
        cen_path = d / f"{code}_centroids.npz"
        comp_path = d / f"{code}_competency.npz"
        if not cen_path.exists() or not comp_path.exists():
            continue

        cen = np.load(cen_path, allow_pickle=True)
        comp = np.load(comp_path, allow_pickle=True)

        # Need distance array
        if 'distance' not in cen:
            print(f"  {code}: no distance array, falling back to last_return")
            data[code] = {
                'env_names': list(comp['env_names']),
                'last_return': comp['last_return'],
                'total_steps': int(comp['total_steps']),
            }
            continue

        d_max = load_d_max(code, scale, input_dir)
        if d_max is None:
            print(f"  {code}: no orbit file, skipping")
            continue

        env_names = list(cen['env_names'])
        distance = cen['distance']   # [n_envs, 360, n_frames]
        mass = cen['mass']           # [n_envs, 360, n_frames]
        n_envs, n_ori, n_frames = distance.shape
        total_steps = int(comp['total_steps'])
        thresh = lam * d_max

        # Compute last_return per env × orientation as % of total frames
        lr = np.zeros((n_envs, n_ori), dtype=np.float64)
        for ei in range(n_envs):
            for oi in range(n_ori):
                ok = ((mass[ei, oi] >= DEATH_THRESH) &
                      ~np.isnan(mass[ei, oi]) &
                      (distance[ei, oi] <= thresh))
                if ok.any():
                    lr[ei, oi] = np.where(ok)[0][-1] / (n_frames - 1) * 100
                else:
                    lr[ei, oi] = 0.0

        data[code] = {
            'env_names': env_names,
            'last_return_pct': lr,  # already in percentage
        }

    return data


def make_range_plot(data, envs, codes, save_path, lam):
    """Grouped range plot: median (dot), IQR (bar), min–max (hollow dots)."""
    n_creatures = len(codes)
    n_envs = len(envs)

    # build median/p25/p75/min/max matrices
    med = np.full((n_creatures, n_envs), np.nan)
    p25 = np.full((n_creatures, n_envs), np.nan)
    p75 = np.full((n_creatures, n_envs), np.nan)
    lo = np.full((n_creatures, n_envs), np.nan)
    hi = np.full((n_creatures, n_envs), np.nan)
    for i, code in enumerate(codes):
        d = data[code]
        for j, env in enumerate(envs):
            if env in d['env_names']:
                idx = d['env_names'].index(env)
                if 'last_return_pct' in d:
                    lr_pct = d['last_return_pct'][idx]
                else:
                    ts = d['total_steps']
                    lr_pct = d['last_return'][idx].astype(float) / ts * 100
                med[i, j] = np.median(lr_pct)
                p25[i, j] = np.percentile(lr_pct, 25)
                p75[i, j] = np.percentile(lr_pct, 75)
                lo[i, j] = lr_pct.min()
                hi[i, j] = lr_pct.max()

    # order environments
    col_order = [envs.index(e) for e in ENV_ORDER if e in envs]
    col_order += [i for i in range(n_envs) if i not in col_order]
    envs_sorted = [envs[k] for k in col_order]
    med = med[:, col_order]
    p25 = p25[:, col_order]
    p75 = p75[:, col_order]
    lo = lo[:, col_order]
    hi = hi[:, col_order]

    # order creatures
    ordered = [c for c in CREATURE_ORDER if c in codes]
    ordered += [c for c in codes if c not in ordered]
    order_idx = [codes.index(c) for c in ordered]
    med = med[order_idx]
    p25 = p25[order_idx]
    p75 = p75[order_idx]
    lo = lo[order_idx]
    hi = hi[order_idx]
    codes = ordered

    # layout
    intra = 0.09   # spacing between creatures within an environment
    inter = 0.22   # extra gap between environment groups
    row_pitch = (n_creatures - 1) * intra + inter

    fig, ax = plt.subplots(figsize=(4.0, row_pitch * n_envs + 0.8))

    for k, code in enumerate(codes):
        color = CREATURE_COLORS.get(code, '#333333')
        for j in range(n_envs):
            m_val = med[k, j]
            if np.isnan(m_val):
                continue
            l, q1, q3, h = lo[k, j], p25[k, j], p75[k, j], hi[k, j]
            y = j * row_pitch + k * intra
            # thin whisker from min to max
            ax.plot([l, h], [y, y], color=color, linewidth=0.8, alpha=0.35)
            # thick bar for IQR (25th–75th)
            ax.plot([q1, q3], [y, y], color=color, linewidth=3,
                    solid_capstyle='round', alpha=0.7)
            # hollow dots at min and max
            ax.plot(l, y, 'o', markerfacecolor='white', markeredgecolor=color,
                    markersize=4, markeredgewidth=1.1, zorder=4)
            ax.plot(h, y, 'o', markerfacecolor='white', markeredgecolor=color,
                    markersize=4, markeredgewidth=1.1, zorder=4)
            # solid dot at median
            ax.plot(m_val, y, 'o', color=color, markersize=4, zorder=5)

    ax.set_xlim(-5, 105)
    pad = inter / 2
    first_bar_y = 0
    last_bar_y = (n_envs - 1) * row_pitch + (n_creatures - 1) * intra
    ax.set_ylim(last_bar_y + pad, first_bar_y - pad)
    env_centers = [j * row_pitch + (n_creatures - 1) * intra / 2 for j in range(n_envs)]
    ax.set_yticks(env_centers)
    ax.set_yticklabels(envs_sorted, fontsize=10)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f'% of simulation within {lam}$\\times$d_max of orbit', fontsize=10)
    ax.grid(axis='x', alpha=0.15)
    # separators between environment groups
    for j in range(1, n_envs):
        prev_bottom = (j - 1) * row_pitch + (n_creatures - 1) * intra
        next_top = j * row_pitch
        sep_y = (prev_bottom + next_top) / 2
        ax.axhline(sep_y, color='black', linewidth=0.4,
                   linestyle='--', alpha=1.0)
    ax.yaxis.tick_right()
    ax.tick_params(left=False, right=False, labelsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate survival-by-environment figure")
    parser.add_argument("--codes", nargs="+", default=["O2u", "S1s", "K4s", "K6s"])
    parser.add_argument("--input", default="results/env_competency")
    parser.add_argument("--output", default="figure_generation/survival_by_environment.png")
    parser.add_argument("--lambda", dest="lam", type=float, default=5.0,
                        help="Survival threshold: lambda * d_max (default: 5.0)")
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    input_dir = Path(args.input)
    data = load_all(input_dir, codes=args.codes, lam=args.lam, scale=args.scale)
    if not data:
        print(f"No competency data found in {input_dir}")
        return

    codes = sorted(data.keys())
    all_envs = set()
    for d in data.values():
        all_envs.update(d['env_names'])
    envs = list(all_envs)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_range_plot(data, envs, codes, out, args.lam)


if __name__ == "__main__":
    main()
