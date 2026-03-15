"""Analyze environment competency results: heatmaps + polar plots.

Primary metric: last_return — last simulation step where the creature's
profile was within 1·d_max of its free-field orbit. Proxy for "how long
did the creature remain functional in this environment."

Reads per-creature .npz files from results/env_competency/ and produces:
  - Last-return heatmap (creatures x environments), normalized to % of run
  - Orientation sensitivity heatmap (std of last_return)
  - Radar overlay comparing all creatures across environments
  - Per-creature polar plots (last_return by heading for each env)
  - Summary CSV

Usage:
    python experiments/analyze_env_competency.py
    python experiments/analyze_env_competency.py --input results/env_competency --output results/env_competency
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Canonical display order
ENV_ORDER = [
    "pegs", "chips", "shuriken",
    "guidelines", "membrane-1px", "membrane-3px",
    "box", "capsule", "ring",
    "corridor", "funnel", "noise",
]


def load_all(input_dir: Path) -> dict:
    """Load all creature .npz files."""
    data = {}
    for d in sorted(input_dir.iterdir()):
        if not d.is_dir():
            continue
        npz_path = d / f"{d.name}_competency.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path, allow_pickle=True)
        entry = {
            'env_names': list(npz['env_names']),
            'M': npz['M'],
            'V': npz['V'],
            'F': npz['F'],
        }
        if 'last_return' in npz:
            entry['last_return'] = npz['last_return']
            entry['total_steps'] = int(npz['total_steps'])
        data[d.name] = entry
    return data


def make_heatmap(matrix, envs, codes, title, save_path,
                 cmap='RdYlGn', vmin=0, vmax=1, fmt='.0f', pct=False):
    """Generic creature x environment heatmap."""
    fig, ax = plt.subplots(figsize=(12, max(3, len(codes) * 0.8 + 1)))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(envs, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(codes, fontsize=10, fontweight='bold')

    for i in range(len(codes)):
        for j in range(len(envs)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            # pick text color for readability
            frac = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            color = 'white' if frac < 0.35 else 'black'
            label = f'{val:{fmt}}%' if pct else f'{val:{fmt}}'
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(title, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  {save_path.name}")


def make_polar_plots(data, metric_key, envs, codes, save_path,
                     title_suffix, ylim=None, normalize=None):
    """Per-creature polar plot for a metric by orientation."""
    n_envs = len(envs)
    for code in codes:
        d = data[code]
        if metric_key not in d:
            continue
        fig, axes = plt.subplots(3, 4, figsize=(14, 10),
                                 subplot_kw=dict(projection='polar'))
        axes = axes.flatten()

        for j, env in enumerate(envs):
            ax = axes[j]
            if env in d['env_names']:
                idx = d['env_names'].index(env)
                vals = d[metric_key][idx].astype(float)
                if normalize is not None:
                    vals = vals / normalize
                n_ori = len(vals)
                angles = np.linspace(0, 2 * np.pi, n_ori, endpoint=False)
                angles_c = np.append(angles, angles[0])
                vals_c = np.append(vals, vals[0])
                ax.fill(angles_c, vals_c, alpha=0.25, color='steelblue')
                ax.plot(angles_c, vals_c, linewidth=0.8, color='steelblue')
            if ylim is not None:
                ax.set_ylim(0, ylim)
            ax.set_title(env, fontsize=8, fontweight='bold', pad=8)
            ax.set_yticklabels([])
            ax.tick_params(labelsize=6)

        for j in range(n_envs, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f'{code} — {title_suffix}', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = save_path / f"{code}_polar_{metric_key}.png"
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  {p.name}")


def make_radar_comparison(data, envs, codes, save_path):
    """Radar overlay: one polygon per creature, 12 spokes = environments.

    Radial axis = mean last_return as % of total run (0 center, 100 edge).
    Shaded band shows 25th–75th percentile spread across orientations.
    """
    n_envs = len(envs)
    angles = np.linspace(0, 2 * np.pi, n_envs, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    for k, code in enumerate(codes):
        d = data[code]
        if 'last_return' not in d:
            continue
        ts = d['total_steps']
        mean_vals, p25_vals, p75_vals = [], [], []
        for env in envs:
            if env in d['env_names']:
                idx = d['env_names'].index(env)
                lr_pct = d['last_return'][idx].astype(float) / ts * 100
                mean_vals.append(lr_pct.mean())
                p25_vals.append(np.percentile(lr_pct, 25))
                p75_vals.append(np.percentile(lr_pct, 75))
            else:
                mean_vals.append(0)
                p25_vals.append(0)
                p75_vals.append(0)
        # close polygon for mean line
        mean_vals += mean_vals[:1]
        color = cmap(k)
        ax.plot(angles, mean_vals, linewidth=2, label=code, color=color)
        # perpendicular tick marks on each spoke at p25 and p75
        tick_delta = 0.04  # angular half-width of tick
        for j in range(n_envs):
            theta = angles[j]
            for r in (p25_vals[j], p75_vals[j]):
                ax.plot([theta - tick_delta, theta + tick_delta],
                        [r, r], linewidth=1.5, color=color, alpha=0.7)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(envs, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7, color='grey')
    ax.set_title('competency by animal by environment',
                 fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  {save_path.name}")


def save_csv(data, envs, codes, save_path):
    """Save summary CSV."""
    has_lr = any('last_return' in d for d in data.values())
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['creature', 'environment', 'M_mean', 'M_std', 'n_ori']
        if has_lr:
            header += ['last_return_mean', 'last_return_std', 'last_return_pct']
        writer.writerow(header)

        for code in codes:
            d = data[code]
            for env in envs:
                if env not in d['env_names']:
                    continue
                idx = d['env_names'].index(env)
                row = [
                    code, env,
                    f"{d['M'][idx].mean():.4f}", f"{d['M'][idx].std():.4f}",
                    len(d['M'][idx]),
                ]
                if has_lr and 'last_return' in d:
                    lr = d['last_return'][idx]
                    total = d['total_steps']
                    row += [
                        f"{lr.mean():.1f}", f"{lr.std():.1f}",
                        f"{lr.mean() / total * 100:.1f}",
                    ]
                writer.writerow(row)
    print(f"  {save_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Analyze environment competency results")
    parser.add_argument("--input", default="results/env_competency", help="Input directory")
    parser.add_argument("--output", default="results/env_competency", help="Output directory")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_all(input_dir)
    if not data:
        print(f"No competency data found in {input_dir}")
        return

    codes = sorted(data.keys())
    all_envs = set()
    for d in data.values():
        all_envs.update(d['env_names'])
    envs = [e for e in ENV_ORDER if e in all_envs]
    for e in sorted(all_envs):
        if e not in envs:
            envs.append(e)

    has_lr = any('last_return' in d for d in data.values())
    n_ori = data[codes[0]]['M'].shape[1]

    print(f"Creatures: {codes}")
    print(f"Environments: {envs}")
    print(f"Orientations: {n_ori}")
    if has_lr:
        total = data[codes[0]].get('total_steps', '?')
        print(f"Total steps: {total}")
    print()

    # ── last_return heatmaps (primary metric) ──
    if has_lr:
        print("Last-return heatmaps:")
        # percentage of run
        total_steps = data[codes[0]]['total_steps']
        lr_pct = np.full((len(codes), len(envs)), np.nan)
        lr_std_pct = np.full((len(codes), len(envs)), np.nan)
        for i, code in enumerate(codes):
            d = data[code]
            if 'last_return' not in d:
                continue
            ts = d['total_steps']
            for j, env in enumerate(envs):
                if env in d['env_names']:
                    idx = d['env_names'].index(env)
                    lr = d['last_return'][idx]
                    lr_pct[i, j] = lr.mean() / ts * 100
                    lr_std_pct[i, j] = lr.std() / ts * 100

        make_heatmap(lr_pct, envs, codes,
                     'Last return to orbit (% of run)',
                     output_dir / 'heatmap_last_return.png',
                     cmap='RdYlGn', vmin=0, vmax=100, fmt='.0f', pct=True)
        make_heatmap(lr_std_pct, envs, codes,
                     'Orientation sensitivity — last return (σ, % of run)',
                     output_dir / 'heatmap_last_return_std.png',
                     cmap='YlOrRd', vmin=0,
                     vmax=max(1, np.nanmax(lr_std_pct)), fmt='.0f', pct=True)
        print()

    # ── radar comparison ──
    if has_lr:
        print("Radar comparison:")
        make_radar_comparison(data, envs, codes,
                              output_dir / 'radar_comparison.png')
        print()

    # ── polar plots (per-creature orientation detail) ──
    if has_lr:
        print("Polar plots:")
        for code in codes:
            d = data[code]
            if 'last_return' in d:
                make_polar_plots(data, 'last_return', envs, [code], output_dir,
                                 'Last return to orbit (step)',
                                 ylim=d['total_steps'])
        print()

    # ── CSV ──
    print("Summary:")
    save_csv(data, envs, codes, output_dir / 'competency_summary.csv')

    print(f"\nOutput: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
