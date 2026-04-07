#!/usr/bin/env python3
"""Environment gallery with competency overlays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import CREATURE_COLORS
from figure_generation.survival_by_environment import load_all as _load_survival_data

# Curated subset — membrane-3px and funnel removed, membrane-1px renamed
ENVS = [
    'pegs', 'chips', 'shuriken', 'guidelines',
    'membrane-1px',
    'box', 'capsule', 'ring', 'corridor', 'noise',
]
ENV_DISPLAY_NAMES = {
    'membrane-1px': 'membrane',
}

_DPI = 600
_THUMB_SCALE = 0.75  # shrink thumbnails to 75%
_BAR_WIDTH_MULT = 2.0  # bar plot width = this * thumbnail width


def _load_env_mask(env_name: str) -> np.ndarray | None:
    pt_path = ROOT / 'environments' / f'{env_name}.pt'
    if not pt_path.exists():
        return None
    return torch.load(pt_path, weights_only=False).cpu().numpy()


def _render_env_bars(ax, env_name, data, codes):
    """Render survival bars for one environment."""
    n_creatures = len(codes)
    intra = 0.5  # tighter spacing between creature lines

    for k, code in enumerate(codes):
        if code not in data:
            continue
        d = data[code]
        color = CREATURE_COLORS.get(code, '#333333')

        if env_name not in d['env_names']:
            continue
        idx = d['env_names'].index(env_name)
        if 'last_return_pct' in d:
            lr_pct = d['last_return_pct'][idx]
        else:
            ts = d['total_steps']
            lr_pct = d['last_return'][idx].astype(float) / ts * 100

        med_val = np.median(lr_pct)
        q1, q3 = np.percentile(lr_pct, 25), np.percentile(lr_pct, 75)
        lo_val, hi_val = lr_pct.min(), lr_pct.max()

        y = k * intra
        ax.plot([lo_val, hi_val], [y, y], color=color, linewidth=0.8, alpha=0.35)
        ax.plot([q1, q3], [y, y], color=color, linewidth=1.75,
                solid_capstyle='round', alpha=0.7)
        ax.plot(lo_val, y, 'o', markerfacecolor='white', markeredgecolor=color,
                markersize=4, markeredgewidth=1.1, zorder=4)
        ax.plot(hi_val, y, 'o', markerfacecolor='white', markeredgecolor=color,
                markersize=4, markeredgewidth=1.1, zorder=4)
        ax.plot(med_val, y, 'o', color=color, markersize=4, zorder=5)

    ax.set_xlim(-5, 105)
    # More padding on top for the label, less on bottom
    total_lines = (n_creatures - 1) * intra
    pad_top = 1.0  # space above for env name
    pad_bot = 0.4
    ax.set_ylim(total_lines + pad_bot, -pad_top)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    for x in [0, 25, 50, 75, 100]:
        ax.axvline(x, color='#cccccc', linewidth=0.5, linestyle=':', zorder=0)


def build_figure(codes, input_dir, output_path, lam=5.0):
    data = _load_survival_data(input_dir, codes=codes, lam=lam)
    envs = list(ENVS)
    n_envs = len(envs)

    # Load one mask to get aspect ratio
    sample_mask = _load_env_mask(envs[0])
    mask_h, mask_w = sample_mask.shape  # e.g. 512 x 1024
    aspect = mask_w / mask_h  # e.g. 2.0

    # Compute figure dimensions from thumbnail size
    thumb_w_inches = 2.0 * _THUMB_SCALE  # base width in inches
    thumb_h_inches = thumb_w_inches / aspect
    bar_w_inches = thumb_w_inches * _BAR_WIDTH_MULT
    row_h_inches = thumb_h_inches
    gap_inches = 0.05

    fig_w = thumb_w_inches + bar_w_inches + 0.3  # small padding
    fig_h = n_envs * (row_h_inches + gap_inches) + 0.5  # extra for bottom label

    fig = plt.figure(figsize=(fig_w, fig_h))

    for j, env in enumerate(envs):
        # Position each row manually in figure coords
        y_top = 1.0 - (j * (row_h_inches + gap_inches) + 0.1) / fig_h
        rh = row_h_inches / fig_h
        thumb_rw = thumb_w_inches / fig_w
        bar_rw = bar_w_inches / fig_w
        x_thumb = 0.02
        x_bar = x_thumb + thumb_rw + 0.01

        # Thumbnail
        ax_thumb = fig.add_axes([x_thumb, y_top - rh, thumb_rw, rh])
        mask = _load_env_mask(env)
        if mask is not None:
            ax_thumb.imshow(1.0 - mask, cmap='gray', vmin=0, vmax=1,
                            aspect='auto', interpolation='nearest')
        ax_thumb.set_xticks([])
        ax_thumb.set_yticks([])
        for spine in ax_thumb.spines.values():
            spine.set_linewidth(0.5)

        # Bar plot — same height as thumbnail
        ax_bars = fig.add_axes([x_bar, y_top - rh, bar_rw, rh])
        _render_env_bars(ax_bars, env, data, codes)
        # Env name inside the box, top-left
        display_name = ENV_DISPLAY_NAMES.get(env, env)
        ax_bars.text(0.02, 0.93, display_name, transform=ax_bars.transAxes,
                     ha='left', va='top', fontsize=8, fontweight='bold',
                     color='black', zorder=10,
                     bbox=dict(facecolor='#e4e4e4', edgecolor='none',
                               pad=1.5, alpha=0.85))
        for spine in ax_bars.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor('black')

        # X-axis only on bottom
        if j == n_envs - 1:
            ax_bars.set_xticks([0, 25, 50, 75, 100])
            ax_bars.tick_params(left=False, right=False, top=False, labelsize=8)
            ax_bars.spines['bottom'].set_edgecolor('black')
            ax_bars.set_xlabel('% duration of simulation survived', fontsize=9)
        else:
            ax_bars.set_xticks([])
            ax_bars.tick_params(bottom=False)

    # Render once to get actual positions
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Get y-position of the bars xlabel ("% duration...")
    xlabel_bbox = ax_bars.xaxis.get_label().get_window_extent(renderer)
    xlabel_y = fig.transFigure.inverted().transform((0, xlabel_bbox.y0))[1]

    # Get y-position of the bars tick labels (0, 25, 50...)
    tick_labels = ax_bars.get_xticklabels()
    if tick_labels:
        tick_bbox = tick_labels[0].get_window_extent(renderer)
        tick_y = fig.transFigure.inverted().transform((0, tick_bbox.y0))[1]
    else:
        tick_y = xlabel_y + 0.02

    label_x = x_thumb + thumb_rw / 2
    fig.text(label_x, tick_y + 0.004, '(black regions occluded)',
             ha='center', va='bottom', fontsize=7, color='black')
    fig.text(label_x, xlabel_y, 'Environments',
             ha='center', va='bottom', fontsize=9)


    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_DPI, bbox_inches='tight', pad_inches=0.02,
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Environments + competencies figure")
    parser.add_argument('--codes', nargs='+', default=['O2u', 'S1s', 'K4s', 'K6s'])
    parser.add_argument('--input', default='results/env_competency')
    parser.add_argument('--output', default='figure_generation/envs_and_competencies.png')
    parser.add_argument('--lambda', dest='lam', type=float, default=10.0)
    args = parser.parse_args()

    build_figure(
        codes=args.codes,
        input_dir=ROOT / args.input,
        output_path=ROOT / args.output,
        lam=args.lam,
    )


if __name__ == '__main__':
    main()
