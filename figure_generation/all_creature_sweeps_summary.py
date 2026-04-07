#!/usr/bin/env python3
"""Cross-animal comparison grid: render sweep maps from raw numpy data.

Single matplotlib figure at high DPI. Each cell uses imshow(aspect='equal',
interpolation='nearest') so matplotlib renders crisp pixels at the target
resolution — no pixel-space upscaling.

Usage:
    python figure_generation/all_creature_sweeps_summary.py --scale 4 --size 1 --ori 0
    python figure_generation/all_creature_sweeps_summary.py --codes O2u S1s K4s K6s
    python figure_generation/all_creature_sweeps_summary.py                  # all combos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import PowerNorm, SymLogNorm, LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from viz import (
    STATUS_EMPTY, STATUS_DIED, STATUS_RECOVERED, STATUS_NEVER,
    COLOR_RED, COLOR_PURPLE,
    HEX_RED, HEX_PURPLE,
)
from config import CREATURE_COLORS

_CMAP_HEADING = LinearSegmentedColormap.from_list(
    'OrBu', ['#E87F24', '#ffffff', '#3A7ABF'])  # orange–white–blue

_GAMMA = 0.5
_DPI = 600
_CELL_INCHES = 5.3  # ~800px per cell at 150 DPI
_HEADING_LINTHRESH = 20  # symlog linear threshold (degrees)


def _load_animals() -> list[dict]:
    with open(ROOT / 'animals_to_run.json') as f:
        return json.load(f)['animals']


def _load_data(code: str, scale: int, size: int, ori: int):
    """Load .npy arrays, tight square bounding-box crop."""
    prefix = f'{code}_x{scale}_i{size}_o{ori}'
    # Try with orientation subdirectory first, then without
    analysis_dir = (ROOT / 'results' / 'sweep' / code / f'{code}_x{scale}' /
                    f'{code}_x{scale}_i{size}' / prefix / 'analysis')
    if not analysis_dir.exists():
        analysis_dir = (ROOT / 'results' / 'sweep' / code / f'{code}_x{scale}' /
                        f'{code}_x{scale}_i{size}' / 'analysis')
    if not analysis_dir.exists():
        return None

    keys = ['recovery_map', 'recovery_status_map', 'max_distance',
            'heading_change']
    data = {}
    for key in keys:
        p = analysis_dir / f'{prefix}_{key}.npy'
        if not p.exists():
            return None
        data[key] = np.load(p)

    # Tight bounding-box crop → make square
    status = data['recovery_status_map']
    nonempty = status != STATUS_EMPTY
    if nonempty.any():
        rows_any = np.where(nonempty.any(axis=1))[0]
        cols_any = np.where(nonempty.any(axis=0))[0]
        pad = 2
        r0 = max(0, rows_any[0] - pad)
        r1 = min(status.shape[0], rows_any[-1] + pad + 1)
        c0 = max(0, cols_any[0] - pad)
        c1 = min(status.shape[1], cols_any[-1] + pad + 1)
        # Expand shorter side to make square
        h, w = r1 - r0, c1 - c0
        side = max(h, w)
        rc, cc = (r0 + r1) // 2, (c0 + c1) // 2
        r0 = max(0, rc - side // 2)
        r1 = r0 + side
        c0 = max(0, cc - side // 2)
        c1 = c0 + side
        if r1 > status.shape[0]:
            r1 = status.shape[0]; r0 = max(0, r1 - side)
        if c1 > status.shape[1]:
            c1 = status.shape[1]; c0 = max(0, c1 - side)
        data = {k: v[r0:r1, c0:c1] for k, v in data.items()}
    return data


# ── Per-cell renderers ───────────────────────────────────────────────────

def _overlay_status(ax, status, H, W):
    """Draw died/never-recovered overlays."""
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    overlay[status == STATUS_DIED] = [*COLOR_RED, 1.0]
    overlay[status == STATUS_NEVER] = [*COLOR_PURPLE, 1.0]
    ax.imshow(overlay, aspect='equal', interpolation='nearest')


def _render_map(ax, values, status, cmap_name='viridis'):
    """Imshow with per-creature normalization + status overlays."""
    H, W = values.shape
    recovered = status == STATUS_RECOVERED
    vals = values[recovered]

    if len(vals) > 0:
        vmin, vmax = np.percentile(vals, 2), np.percentile(vals, 98)
        if vmax <= vmin:
            vmin, vmax = vals.min(), vals.max()
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0, 1

    display = np.full((H, W), np.nan, dtype=np.float32)
    display[recovered] = values[recovered]

    cmap = matplotlib.colormaps[cmap_name].copy()
    cmap.set_bad(alpha=0)
    norm = PowerNorm(gamma=_GAMMA, vmin=vmin, vmax=vmax, clip=True)

    ax.set_facecolor('white')
    ax.imshow(display, cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')
    _overlay_status(ax, status, H, W)
    ax.set_xticks([])
    ax.set_yticks([])


def _render_heading(ax, data):
    """Signed heading change — OrBu diverging with symlog norm."""
    status = data['recovery_status_map']
    H, W = status.shape
    heading = data['heading_change']  # signed radians

    recovered = status == STATUS_RECOVERED

    display = np.full((H, W), np.nan, dtype=np.float32)
    display[recovered] = np.degrees(heading[recovered])

    cmap = _CMAP_HEADING.copy()
    cmap.set_bad(color='white', alpha=0)
    norm = SymLogNorm(linthresh=_HEADING_LINTHRESH, linscale=1, vmin=-180, vmax=180)

    ax.set_facecolor('white')
    ax.imshow(display, cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')
    _overlay_status(ax, status, H, W)

    # Outline creature body so shape is visible in near-zero regions
    mask = (status != STATUS_EMPTY).astype(np.float32)
    ax.contour(mask, levels=[0.5], colors=['black'], linewidths=[0.8])

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


# ── Main grid builder ────────────────────────────────────────────────────

def build_grid(scale: int, size: int, ori: int,
               animals: list[dict], cmap_name: str = 'viridis',
               use_subdir: bool = False) -> Path | None:
    sweep_root = ROOT / 'results' / 'sweep'

    rows = []
    for animal in animals:
        code = animal['code']
        data = _load_data(code, scale, size, ori)
        if data is None:
            print(f"  Skipping {code} — missing analysis data")
            continue
        rows.append((code, data))

    if not rows:
        print(f"  No data found for scale={scale} size={size} ori={ori}")
        return None

    n = len(rows)

    fig, axes = plt.subplots(
        n, 3,
        figsize=(3 * _CELL_INCHES, n * _CELL_INCHES),
        gridspec_kw={'wspace': 0, 'hspace': 0},
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    # Render cells
    for i, (code, data) in enumerate(rows):
        status = data['recovery_status_map']
        _render_map(axes[i, 0], data['recovery_map'], status, cmap_name)
        _render_map(axes[i, 1], data['max_distance'], status, cmap_name)
        _render_heading(axes[i, 2], data)

    # Remove spines
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Title + column headers
    col_labels = ['Frames Until Recovery', 'Max Distortion', 'Heading Change']
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=22, fontweight='bold', pad=5)

    # Row labels
    for i, (code, _) in enumerate(rows):
        axes[i, 0].set_ylabel(code, fontsize=24, fontweight='bold',
                               rotation=0, labelpad=40, va='center')

    # Colorbars
    fig.subplots_adjust(bottom=0.06)
    cbar_h = 0.012
    cbar_y = 0.02
    p0 = axes[-1, 0].get_position()
    p1 = axes[-1, 1].get_position()
    p2 = axes[-1, 2].get_position()
    pad = 0.02
    opad = 0.015

    # Viridis under cols 0 & 1
    cax0 = fig.add_axes([p0.x0 + opad, cbar_y, p1.x1 - p0.x0 - pad - opad, cbar_h])
    sm0 = plt.cm.ScalarMappable(cmap=matplotlib.colormaps[cmap_name],
                                 norm=plt.Normalize(0, 1))
    sm0.set_array([])
    cb0 = plt.colorbar(sm0, cax=cax0, orientation='horizontal')
    cb0.set_ticks([0, 1])
    cb0.set_ticklabels(['less', 'more'])
    cb0.ax.tick_params(labelsize=18)

    # OrBu under col 2
    cax1 = fig.add_axes([p2.x0 + pad, cbar_y, p2.width - pad - opad, cbar_h])
    sm1 = plt.cm.ScalarMappable(cmap=_CMAP_HEADING,
                                 norm=SymLogNorm(linthresh=_HEADING_LINTHRESH,
                                                 linscale=1, vmin=-180, vmax=180))
    sm1.set_array([])
    cb1 = plt.colorbar(sm1, cax=cax1, orientation='horizontal')
    cb1.set_ticks([-180, -60, -20, 0, 20, 60, 180])
    cb1.set_ticklabels(['180°', '60°', '20°', '0°', '20°', '60°', '180°'])
    cb1.ax.xaxis.set_ticks_position('top')
    cb1.ax.xaxis.set_label_position('top')
    cb1.ax.tick_params(labelsize=10, top=True, bottom=False)
    # Direction arrow (creature heading) centered below the colorbar
    arrow_sz = 0.06
    arrow_cx = p2.x0 + pad + (p2.width - pad - opad) / 2  # center of colorbar
    arrow_cy = cbar_y - 0.02
    ax_arrow = fig.add_axes([arrow_cx - arrow_sz / 2, arrow_cy - arrow_sz / 2,
                              arrow_sz, arrow_sz])
    ax_arrow.set_xlim(-1.2, 1.2)
    ax_arrow.set_ylim(-1.2, 1.2)
    ax_arrow.set_aspect('equal')
    ax_arrow.axis('off')
    rad = np.radians(30)
    dx, dy = np.cos(rad), np.sin(rad)
    ax_arrow.annotate('', xy=(dx * 1.0, dy * 1.0),
                      xytext=(0, 0),
                      arrowprops=dict(arrowstyle='->,head_width=0.5,head_length=0.35',
                                      color='black', lw=2.5),
                      zorder=2)
    circle = plt.Circle((0, 0), 0.3, fill=True, facecolor='white',
                         edgecolor='black', lw=1.5, zorder=1)
    ax_arrow.add_patch(circle)
    ax_arrow.text(0, -1.0, '(initial heading)', fontsize=16,
                  ha='center', va='top', color='0.3')

    cb1.ax.text(0, -0.6, 'left', transform=cb1.ax.transAxes,
                fontsize=16, ha='left', va='top')
    cb1.ax.text(1, -0.6, 'right', transform=cb1.ax.transAxes,
                fontsize=16, ha='right', va='top')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=HEX_PURPLE, edgecolor='black', label='Never recovered'),
        mpatches.Patch(facecolor=HEX_RED, edgecolor='black', label='Died'),
    ]
    # Center legend under the viridis colorbar (cols 0 & 1)
    cbar0_center = (p0.x0 + opad + p1.x1 - pad) / 2
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               fontsize=19, framealpha=0.9, bbox_to_anchor=(cbar0_center, 0.015))

    # Creature color dots directly under each row label
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for i, (code, _) in enumerate(rows):
        color = CREATURE_COLORS.get(code, '#333333')
        bb = axes[i, 0].yaxis.label.get_window_extent(renderer=renderer)
        inv = fig.transFigure.inverted()
        x_c, y_b = inv.transform(((bb.x0 + bb.x1) / 2, bb.y0))
        fig.text(x_c, y_b - 0.001, '\u25CF', fontsize=28,
                 color=color, ha='center', va='top')

    if use_subdir:
        out_dir = ROOT / 'figure_generation' / f'all_creature_sweeps_summary_x{scale}_i{size}'
    else:
        out_dir = ROOT / 'figure_generation'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'all_creature_sweeps_summary_x{scale}_i{size}_o{ori}.png'
    fig.savefig(out_path, dpi=_DPI, bbox_inches='tight', pad_inches=0.02,
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


# ── Discovery + CLI ──────────────────────────────────────────────────────

def _discover_combos(animals: list[dict], scale: int) -> list[tuple[int, int]]:
    sweep_root = ROOT / 'results' / 'sweep'
    combos = set()
    for animal in animals:
        code = animal['code']
        scale_dir = sweep_root / code / f'{code}_x{scale}'
        if not scale_dir.exists():
            continue
        for size_dir in sorted(scale_dir.iterdir()):
            if not size_dir.is_dir():
                continue
            for ori_dir in sorted(size_dir.iterdir()):
                if not ori_dir.is_dir():
                    continue
                parts = ori_dir.name.split('_')
                try:
                    size_val = int([p for p in parts if p.startswith('i')][0][1:])
                    ori_val = int([p for p in parts if p.startswith('o')][0][1:])
                    combos.add((size_val, ori_val))
                except (IndexError, ValueError):
                    continue
    return sorted(combos)


def main():
    parser = argparse.ArgumentParser(
        description='Cross-animal comparison grid from raw numpy data')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--size', type=int, default=3,
                        help='Intervention size (default: 3)')
    parser.add_argument('--ori', type=int, default=2,
                        help='Orientation index (default: 2)')
    parser.add_argument('--codes', nargs='+', default=['O2u', 'S1s', 'K4s', 'K6s'],
                        help='Creature codes in desired order')
    parser.add_argument('--cmap', default='viridis',
                        help='Colormap for recovery/distance columns (default: viridis)')
    parser.add_argument('--all', action='store_true',
                        help='Generate all available size/ori combos')
    args = parser.parse_args()

    animals = _load_animals()
    if args.codes:
        by_code = {a['code']: a for a in animals}
        animals = [by_code[c] for c in args.codes if c in by_code]

    if not args.all:
        build_grid(args.scale, args.size, args.ori, animals, args.cmap,
                   use_subdir=False)
    else:
        combos = _discover_combos(animals, args.scale)

        if not combos:
            print(f"No results found for scale={args.scale}")
            return

        print(f"Found {len(combos)} (size, ori) combo(s) at scale={args.scale}")
        for size_val, ori_val in combos:
            print(f"\n--- size={size_val}, ori={ori_val} ---")
            build_grid(args.scale, size_val, ori_val, animals, args.cmap,
                       use_subdir=True)


if __name__ == '__main__':
    main()
