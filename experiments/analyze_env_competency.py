# experiments/analyze_env_competency.py — Jesse Cool (jessescool)
"""Analyze environment competency results: heatmaps and polar plots."""

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from config import ENV_ORDER


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


def make_range_plot(data, envs, codes, save_path):
    """Small-multiples range plot: dot at median, whisker from min to max.

    One panel per creature, shared x-axis (0-100% survival),
    environments on y-axis sorted by overall median difficulty.
    """
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
        if 'last_return' not in d:
            continue
        ts = d['total_steps']
        for j, env in enumerate(envs):
            if env in d['env_names']:
                idx = d['env_names'].index(env)
                lr_pct = d['last_return'][idx].astype(float) / ts * 100
                med[i, j] = np.median(lr_pct)
                p25[i, j] = np.percentile(lr_pct, 25)
                p75[i, j] = np.percentile(lr_pct, 75)
                lo[i, j] = lr_pct.min()
                hi[i, j] = lr_pct.max()

    # fixed display order for environments
    ENV_DISPLAY_ORDER = [
        'pegs', 'chips', 'shuriken', 'guidelines',
        'membrane-1px', 'membrane-3px',
        'box', 'capsule', 'ring', 'corridor', 'funnel', 'noise',
    ]
    col_order = [envs.index(e) for e in ENV_DISPLAY_ORDER if e in envs]
    # append any envs not in display order
    col_order += [i for i in range(n_envs) if i not in col_order]
    envs_sorted = [envs[k] for k in col_order]
    med = med[:, col_order]
    p25 = p25[:, col_order]
    p75 = p75[:, col_order]
    lo = lo[:, col_order]
    hi = hi[:, col_order]

    # single narrow panel, grouped: tight within env, clear gap between envs
    intra = 0.09  # spacing between creatures within an environment
    inter = 0.22  # extra gap between environment groups
    row_pitch = (n_creatures - 1) * intra + inter

    # fixed display order and colors
    DISPLAY_ORDER = ['O2u', 'S1s', 'K4s', 'K6s']
    _tab10 = plt.cm.tab10
    COLOR_MAP = {'O2u': _tab10(3), 'S1s': _tab10(1), 'K4s': _tab10(0), 'K6s': _tab10(2)}
    # reorder codes to match display order, keeping any extras at end
    ordered = [c for c in DISPLAY_ORDER if c in codes]
    ordered += [c for c in codes if c not in ordered]
    # reorder matrices to match
    order_idx = [codes.index(c) for c in ordered]
    med = med[order_idx]
    p25 = p25[order_idx]
    p75 = p75[order_idx]
    lo = lo[order_idx]
    hi = hi[order_idx]
    codes = ordered

    fig, ax = plt.subplots(figsize=(4.0, row_pitch * n_envs + 0.8))

    for k, code in enumerate(codes):
        color = COLOR_MAP.get(code, '#333333')
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
    # separator sits inter/2 from the nearest bar; match that for top/bottom padding
    pad = inter / 2
    first_bar_y = 0
    last_bar_y = (n_envs - 1) * row_pitch + (n_creatures - 1) * intra
    ax.set_ylim(last_bar_y + pad, first_bar_y - pad)
    # label at the center of each environment group
    env_centers = [j * row_pitch + (n_creatures - 1) * intra / 2 for j in range(n_envs)]
    ax.set_yticks(env_centers)
    ax.set_yticklabels(envs_sorted, fontsize=10)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('% duration of simulation survived', fontsize=10)
    ax.grid(axis='x', alpha=0.15)
    # separators between environment groups (midpoint between last bar of prev and first bar of next)
    for j in range(1, n_envs):
        prev_bottom = (j - 1) * row_pitch + (n_creatures - 1) * intra
        next_top = j * row_pitch
        sep_y = (prev_bottom + next_top) / 2
        ax.axhline(sep_y, color='black', linewidth=0.4,
                   linestyle='--', alpha=1.0)
    ax.yaxis.tick_right()
    ax.tick_params(left=False, right=False, labelsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  {save_path.name}")


def make_competency_heatmap_with_marginals(data, envs, codes, save_path):
    """Publication heatmap: creatures × environments with marginal summaries.

    - Main panel: median last_return as % of run, colored by magma
    - Right marginal: row medians (overall creature competency)
    - Bottom marginal: column medians (overall env difficulty)
    - Hatching on high-IQR cells (orientation-sensitive)
    - Rows sorted by median competency, cols sorted by median difficulty
    """
    from matplotlib.patches import Rectangle
    import scipy.cluster.hierarchy as sch

    n_creatures = len(codes)
    n_envs = len(envs)

    # build median and IQR matrices
    med_matrix = np.full((n_creatures, n_envs), np.nan)
    iqr_matrix = np.full((n_creatures, n_envs), np.nan)
    for i, code in enumerate(codes):
        d = data[code]
        if 'last_return' not in d:
            continue
        ts = d['total_steps']
        for j, env in enumerate(envs):
            if env in d['env_names']:
                idx = d['env_names'].index(env)
                lr_pct = d['last_return'][idx].astype(float) / ts * 100
                med_matrix[i, j] = np.median(lr_pct)
                iqr_matrix[i, j] = np.percentile(lr_pct, 75) - np.percentile(lr_pct, 25)

    # sort cols by median difficulty (easiest=highest median on left)
    col_med = np.nanmedian(med_matrix, axis=0)
    col_order = np.argsort(-col_med)  # descending = easiest first
    med_matrix = med_matrix[:, col_order]
    iqr_matrix = iqr_matrix[:, col_order]
    envs_sorted = [envs[k] for k in col_order]

    # sort rows by median competency (highest on top)
    row_med = np.nanmedian(med_matrix, axis=1)
    row_order = np.argsort(-row_med)
    med_matrix = med_matrix[row_order]
    iqr_matrix = iqr_matrix[row_order]
    codes_sorted = [codes[k] for k in row_order]
    row_med = row_med[row_order]

    # recalculate marginals after sorting
    col_med_sorted = np.nanmedian(med_matrix, axis=0)
    row_med_sorted = np.nanmedian(med_matrix, axis=1)

    # IQR threshold for hatching (top quartile of IQR values)
    iqr_thresh = np.nanpercentile(iqr_matrix, 75)

    # ── layout: main heatmap + right marginal + bottom marginal ──
    fig = plt.figure(figsize=(10, 4.5))
    # gridspec: main | right-margin | colorbar, bottom-margin below main
    gs = fig.add_gridspec(2, 3, width_ratios=[n_envs, 1.5, 0.4],
                          height_ratios=[n_creatures, 1.2],
                          hspace=0.08, wspace=0.08)

    ax_main = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 0])
    ax_cb = fig.add_subplot(gs[0, 2])
    # hide bottom-right corner
    ax_empty = fig.add_subplot(gs[1, 1])
    ax_empty.axis('off')
    ax_empty2 = fig.add_subplot(gs[1, 2])
    ax_empty2.axis('off')

    cmap = plt.cm.magma
    vmin, vmax = 0, 100

    # ── main heatmap ──
    im = ax_main.imshow(med_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                         aspect='auto', interpolation='nearest')

    # cell annotations + hatching for high IQR
    for i in range(len(codes_sorted)):
        for j in range(len(envs_sorted)):
            val = med_matrix[i, j]
            if np.isnan(val):
                continue
            frac = (val - vmin) / (vmax - vmin)
            color = 'white' if frac < 0.5 else 'black'
            ax_main.text(j, i, f'{val:.0f}', ha='center', va='center',
                         fontsize=8, color=color, fontweight='bold')
            # hatch high-IQR cells
            if iqr_matrix[i, j] >= iqr_thresh:
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 fill=False, edgecolor='white',
                                 linewidth=1.2, linestyle='--', alpha=0.7)
                ax_main.add_patch(rect)

    ax_main.set_xticks(range(len(envs_sorted)))
    ax_main.set_xticklabels([])  # labels go on bottom marginal
    ax_main.set_yticks(range(len(codes_sorted)))
    ax_main.set_yticklabels(codes_sorted, fontsize=11, fontweight='bold')
    ax_main.tick_params(length=0)

    # ── right marginal: horizontal bars for row medians ──
    y_pos = np.arange(len(codes_sorted))
    bars = ax_right.barh(y_pos, row_med_sorted, color=[cmap(v / 100) for v in row_med_sorted],
                         edgecolor='black', linewidth=0.5, height=0.7)
    for i, v in enumerate(row_med_sorted):
        ax_right.text(v + 1.5, i, f'{v:.0f}%', va='center', fontsize=9, fontweight='bold')
    ax_right.set_xlim(0, 110)
    ax_right.set_ylim(len(codes_sorted) - 0.5, -0.5)
    ax_right.set_yticks([])
    ax_right.set_xlabel('median %', fontsize=9)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.tick_params(labelsize=8)

    # ── bottom marginal: vertical bars for column medians ──
    x_pos = np.arange(len(envs_sorted))
    bars_b = ax_bot.bar(x_pos, col_med_sorted,
                        color=[cmap(v / 100) for v in col_med_sorted],
                        edgecolor='black', linewidth=0.5, width=0.7)
    for j, v in enumerate(col_med_sorted):
        ax_bot.text(j, v + 1.5, f'{v:.0f}', ha='center', fontsize=7, fontweight='bold')
    ax_bot.set_ylim(0, 110)
    ax_bot.set_xlim(-0.5, len(envs_sorted) - 0.5)
    ax_bot.set_xticks(range(len(envs_sorted)))
    ax_bot.set_xticklabels(envs_sorted, rotation=45, ha='right', fontsize=9)
    ax_bot.set_ylabel('median %', fontsize=9)
    ax_bot.spines['top'].set_visible(False)
    ax_bot.spines['right'].set_visible(False)
    ax_bot.tick_params(labelsize=8)

    # ── colorbar ──
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label('last return to orbit (% of run)', fontsize=9)

    # ── title ──
    fig.suptitle('Competency by creature × environment',
                 fontsize=13, fontweight='bold', y=1.02)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
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
    parser.add_argument("--codes", nargs="+", default=None, help="Creature codes to include")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_all(input_dir)
    if not data:
        print(f"No competency data found in {input_dir}")
        return

    if args.codes:
        data = {k: v for k, v in data.items() if k in args.codes}
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

    # ── range plot (primary overview) ──
    if has_lr:
        print("Range plot:")
        make_range_plot(data, envs, codes,
                        output_dir / 'survival_by_environment.png')
        print()

    # ── heatmap with marginals ──
    if has_lr:
        print("Competency heatmap with marginals:")
        make_competency_heatmap_with_marginals(data, envs, codes,
                              output_dir / 'competency_heatmap_marginals.png')
        print()

    # ── radar comparison (legacy) ──
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
