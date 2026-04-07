#!/usr/bin/env python3
"""Trajectory overlay figure.

Plots 360 centroid trajectories per creature overlaid on an environment
barrier mask, colored by per-timestep occlusion (or optionally W1 distance).
Death detection uses W1 distance from orbit (lambda * d_max threshold).

Data source: results/env_competency/{CODE}/{CODE}_centroids.npz

Usage:
    python figure_generation/plot_trajectory_overlay.py --env guidelines --highlight
    python figure_generation/plot_trajectory_overlay.py --env guidelines --wasserstein
    python figure_generation/plot_trajectory_overlay.py --env guidelines --weighted
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from environments.environments import make_env
from config import CREATURE_COLORS

NAMES = {"O2u": "Orbium", "S1s": "Scutium", "K4s": "Kronia",
         "K6s": "Kronia", "O2v": "Orbium", "P4al": "Pteridium"}

DEATH_THRESH = 0.01


# ── data helpers ───────────────────────────────────────────────────────

def load_data(code: str, env_name: str):
    """Return (centroids, mass, distance, occlusion_uniform, occlusion_weighted)."""
    path = Path(f"results/env_competency/{code}/{code}_centroids.npz")
    d = np.load(path, allow_pickle=True)
    envs = list(d["env_names"])
    idx = envs.index(env_name)
    return (d["centroids"][idx], d["mass"][idx],
            d["distance"][idx], d["occlusion_uniform"][idx],
            d["occlusion_weighted"][idx])


def load_d_max(code: str, scale: int = 4) -> float:
    orbit = torch.load(f"orbits/{code}/s{scale}/{code}_s{scale}_orbit.pt",
                        weights_only=False)
    return float(orbit["d_max"])


def get_env_mask(env_name: str, H: int, W: int) -> np.ndarray:
    mask = make_env(env_name, (H, W), torch.device("cpu"),
                    torch.float32, scaled=True)
    return mask.numpy()


def detect_deaths(mass, distance=None, d_max=None, lam=10.0):
    """Return death_frame [N] (-1 = survived).

    Death = last frame where distance < lam * d_max and mass > threshold.
    """
    N, T = mass.shape
    death_frame = np.full(N, -1, dtype=int)
    for i in range(N):
        if distance is not None and d_max is not None:
            ok = ((mass[i] >= DEATH_THRESH) & ~np.isnan(mass[i]) &
                  (distance[i] <= lam * d_max))
            if ok.any():
                last_ok = int(np.where(ok)[0][-1])
                if last_ok < T - 1:
                    death_frame[i] = last_ok + 1
            else:
                death_frame[i] = 0
        else:
            bad = (mass[i] < DEATH_THRESH) | np.isnan(mass[i])
            if bad.any():
                death_frame[i] = int(np.argmax(bad))
    return death_frame


# ── plotting ───────────────────────────────────────────────────────────

def pick_longest_lived(mass, distance=None, d_max=None, lam=10.0):
    """Return orientation index with the longest lifespan."""
    death_frame = detect_deaths(mass, distance, d_max, lam)
    T = mass.shape[1]
    lifespan = np.where(death_frame > 0, death_frame, T)
    return int(np.argmax(lifespan))


def plot_panel(ax, centroids, mass, distance, occlusion, d_max, env_mask,
               code, highlight=False, lam=10.0, use_wasserstein=False):
    """Render one creature's trajectory panel.

    Distance is always used for death detection. Line coloring uses
    occlusion by default, or distance if use_wasserstein=True.
    """
    color_data = distance if use_wasserstein else occlusion
    H, W = env_mask.shape
    color = CREATURE_COLORS.get(code, "#333333")

    ax.imshow(1.0 - env_mask, cmap="gray", vmin=0, vmax=1,
              aspect="equal", interpolation="nearest")

    death_frame = detect_deaths(mass, distance, d_max, lam)
    N = centroids.shape[0]

    # Normalize: clip at 95th percentile of alive frames
    alive_vals = color_data[mass > DEATH_THRESH]
    alive_vals = alive_vals[~np.isnan(alive_vals)]
    vmax = np.percentile(alive_vals, 95) if len(alive_vals) > 0 else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax, clip=True)

    if highlight is True:
        hi_idx = pick_longest_lived(mass, distance, d_max, lam)
    elif isinstance(highlight, int):
        hi_idx = highlight
    else:
        hi_idx = -1

    # Collect line segments for batch rendering
    all_segs = []
    all_vals = []

    for i in range(N):
        if i == hi_idx:
            continue
        df = death_frame[i]
        end = df if df > 0 else centroids.shape[1]
        traj = centroids[i, :end]
        cvals = color_data[i, :end]

        valid = ~np.isnan(traj).any(axis=1) & ~np.isnan(cvals[:end])
        traj = traj[valid]
        cvals = cvals[valid[:len(cvals)]] if len(cvals) > len(traj) else cvals[valid]
        if len(traj) < 2:
            continue

        # Skip toroidal wraps
        dr = np.abs(np.diff(traj[:, 0]))
        dc = np.abs(np.diff(traj[:, 1]))
        no_wrap = (dr < H * 0.3) & (dc < W * 0.3)

        points = np.column_stack([traj[:, 1], traj[:, 0]])  # (x, y)
        segments = np.stack([points[:-1], points[1:]], axis=1)

        all_segs.append(segments[no_wrap])
        all_vals.append(cvals[:-1][no_wrap])

    if all_segs:
        all_segs = np.concatenate(all_segs, axis=0)
        all_vals = np.concatenate(all_vals, axis=0)
        lc = LineCollection(all_segs, cmap="RdYlGn_r", norm=norm,
                            linewidths=0.5, alpha=0.25)
        lc.set_array(all_vals)
        ax.add_collection(lc)

    # Death markers
    for i in range(N):
        if i == hi_idx:
            continue
        df = death_frame[i]
        if df > 0:
            pos = centroids[i, df - 1]
            if not np.isnan(pos).any():
                ax.plot(pos[1], pos[0], "x", color="k",
                        markersize=2.5, markeredgewidth=0.6, alpha=0.45)

    # Highlighted trajectory (gray+white outline)
    if hi_idx >= 0:
        df = death_frame[hi_idx]
        end = df if df > 0 else centroids.shape[1]
        traj = centroids[hi_idx, :end]
        valid = ~np.isnan(traj).any(axis=1)
        traj = traj[valid]
        if len(traj) >= 2:
            segs, start = [], 0
            for j in range(1, len(traj)):
                if (abs(traj[j, 0] - traj[j-1, 0]) > H * 0.3 or
                        abs(traj[j, 1] - traj[j-1, 1]) > W * 0.3):
                    if j - start > 1:
                        segs.append(traj[start:j])
                    start = j
            if len(traj) - start > 1:
                segs.append(traj[start:])
            for seg in segs:
                ax.plot(seg[:, 1], seg[:, 0], color="white", alpha=0.95,
                        linewidth=2.5, solid_capstyle="round", zorder=7)
                ax.plot(seg[:, 1], seg[:, 0], color="#2463EB", alpha=0.9,
                        linewidth=1.3, solid_capstyle="round", zorder=8)
        if df > 0:
            pos = centroids[hi_idx, df - 1]
            if not np.isnan(pos).any():
                ax.plot(pos[1], pos[0], "x", color="k",
                        markersize=5, markeredgewidth=1.2, alpha=0.9, zorder=9)
        print(f"    {code} highlight: orientation {hi_idx}, "
              f"lifespan {'full' if df < 0 else df}")

    # Start dot
    ax.plot(W / 2, H / 2, "o", color="white", markersize=6,
            markeredgecolor="#555555", markeredgewidth=1.0, zorder=10)

    ax.set_title(code, fontsize=14, fontweight='bold', pad=6)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')


# ── main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Trajectory overlay figure")
    ap.add_argument("--env", default="guidelines")
    ap.add_argument("--codes", nargs="+", default=["O2u", "K4s", "K6s"])
    ap.add_argument("--highlight", action="store_true",
                    help="Bold the longest-lived trajectory per creature")
    ap.add_argument("--highlight-idx", nargs="+", type=int, default=None,
                    help="Override highlight orientation per creature (e.g. 70 -1 -1)")
    ap.add_argument("--wasserstein", action="store_true",
                    help="Color by W1 distance instead of occlusion")
    ap.add_argument("--weighted", action="store_true",
                    help="Use weighted occlusion instead of uniform")
    ap.add_argument("--lambda", dest="lam", type=float, default=10.0,
                    help="Death threshold: lambda * d_max (default: 10.0)")
    ap.add_argument("--snapshots", nargs="+", default=None,
                    help="Per-creature snapshot frames: 'CODE:ORI:f1,f2,f3,f4' e.g. O2u:120:740,760,770,780")
    ap.add_argument("--output", default=None,
                    help="Output stem (default: trajectory_overlay_{env})")
    args = ap.parse_args()

    codes = args.codes
    env = args.env
    n = len(codes)

    # Grid shape from scale in data
    d0 = np.load(f"results/env_competency/{codes[0]}/{codes[0]}_centroids.npz",
                 allow_pickle=True)
    scale = int(d0["scale"])
    H, W = 128 * scale, 256 * scale

    env_mask = get_env_mask(env, H, W)

    # Figure layout
    aspect = H / W
    panel_w = 7
    fig_w = panel_w * n
    dpi = 600
    title_h = 0.25
    bottom_h = 0.55
    gap_h = 0.15

    side_px = 4
    gap_px = 30
    gap_in = gap_px / dpi  # inter-panel gap in inches

    # 4 square placeholders per panel, same gap between them
    n_sq = 4
    sq_size_est = (panel_w - (n_sq - 1) * gap_in) / n_sq

    fig_h = (panel_w * aspect + title_h + gap_h
             + gap_in + sq_size_est + gap_in + bottom_h)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), dpi=dpi)

    left = side_px / (fig_w * dpi)
    right = 1 - left
    top = 1 - (title_h + gap_h) / fig_h
    bottom = (bottom_h + gap_in + sq_size_est + gap_in) / fig_h
    wspace = gap_px / (panel_w * dpi)
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace)
    if n == 1:
        axes = [axes]

    # Parse snapshot specs: {code: (ori, [frame_indices])}
    snap_map = {}
    if args.snapshots:
        for spec in args.snapshots:
            parts = spec.split(':')
            code_s, ori_s, frames_s = parts[0], int(parts[1]), [int(x) for x in parts[2].split(',')]
            snap_map[code_s] = (ori_s, frames_s)

    for i, code in enumerate(codes):
        print(f"  {code} …")
        centroids, mass, distance, occ_uniform, occ_weighted = load_data(code, env)
        occlusion = occ_weighted if args.weighted else occ_uniform
        d_max = load_d_max(code, scale)
        if args.highlight_idx and i < len(args.highlight_idx) and args.highlight_idx[i] >= 0:
            hl = args.highlight_idx[i]
        elif args.highlight:
            hl = True
        else:
            hl = False
        plot_panel(axes[i], centroids, mass, distance, occlusion, d_max,
                   env_mask, code, highlight=hl, lam=args.lam,
                   use_wasserstein=args.wasserstein)

        # Numbered snapshot markers on the trajectory
        if code in snap_map:
            ori, snap_indices = snap_map[code]
            metric_stride = 2  # 2000 steps → 1000 centroids
            for j, fidx in enumerate(snap_indices):
                cidx = fidx // metric_stride
                if cidx < centroids.shape[1]:
                    pos_rc = centroids[ori, cidx]
                    if not np.isnan(pos_rc).any():
                        # Nudge overlapping markers apart
                        nudge_r, nudge_c = 0, 0
                        if j > 0:
                            prev_cidx = snap_indices[j - 1] // metric_stride
                            prev_rc = centroids[ori, prev_cidx]
                            if not np.isnan(prev_rc).any():
                                dist = np.sqrt((pos_rc[0] - prev_rc[0])**2 + (pos_rc[1] - prev_rc[1])**2)
                                if dist < 20:  # too close — nudge this one away
                                    dr = pos_rc[0] - prev_rc[0]
                                    dc = pos_rc[1] - prev_rc[1]
                                    norm = max(dist, 1e-3)
                                    nudge_r = dr / norm * (20 - dist)
                                    nudge_c = dc / norm * (20 - dist)

                        axes[i].plot(pos_rc[1] + nudge_c, pos_rc[0] + nudge_r, 'o', color='#2463EB',
                                     markersize=11, markeredgewidth=0, zorder=11)
                        axes[i].text(pos_rc[1] + nudge_c, pos_rc[0] + nudge_r, f'$\\mathbf{{{j + 1}}}$',
                                     fontsize=7, fontweight='bold',
                                     ha='center', va='center', color='white', zorder=12)

    # Colored dots next to creature names (measure title bbox, place dot left)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for i, (ax, code) in enumerate(zip(axes, codes)):
        color = CREATURE_COLORS.get(code, "#333333")
        bbox = ax.title.get_window_extent(renderer=renderer)
        inv = fig.transFigure.inverted()
        x_left, y_mid = inv.transform((bbox.x0, (bbox.y0 + bbox.y1) / 2 + 5))
        dot_off = 10 / (fig_w * dpi)
        fig.text(x_left - dot_off, y_mid, "●", fontsize=12,
                 color=color, ha='right', va='center')

    # Snapshot / placeholder squares below each panel
    from matplotlib import colormaps as _cmaps
    _magma_r = _cmaps['magma_r']
    crop_px = 32 * scale

    sq_y = (bottom_h + gap_in) / fig_h
    for i, (ax, code) in enumerate(zip(axes, codes)):
        pos = ax.get_position()
        sq_gap_x = gap_in / fig_w
        sq_w = (pos.width - (n_sq - 1) * sq_gap_x) / n_sq
        sq_h = (sq_w * fig_w) / fig_h  # keep square in absolute terms

        # Load frames tensor if snapshots requested for this creature
        snap_frames = None
        snap_indices = None
        if code in snap_map:
            ori, snap_indices = snap_map[code]
            pt_path = Path(f"results/new/{code}_{env}_o{ori}_frames.pt")
            if pt_path.exists():
                snap_frames = torch.load(pt_path, map_location='cpu', weights_only=False)

        for j in range(n_sq):
            sq_x = pos.x0 + j * (sq_w + sq_gap_x)
            sq_ax = fig.add_axes([sq_x, sq_y, sq_w, sq_h])
            sq_ax.set_xticks([])
            sq_ax.set_yticks([])

            if snap_frames is not None and j < len(snap_indices):
                fidx = snap_indices[j]
                f = snap_frames[fidx].numpy()
                fH, fW = f.shape
                ys, xs = np.where(f > 0.01)
                if len(ys) > 0:
                    cy, cx = int(np.median(ys)), int(np.median(xs))
                    # Toroidal centering for edge cases
                    rolled = np.roll(np.roll(f, fH // 2 - cy, axis=0), fW // 2 - cx, axis=1)
                    r0 = fH // 2 - crop_px // 2
                    c0 = fW // 2 - crop_px // 2
                    cropped = rolled[r0:r0 + crop_px, c0:c0 + crop_px]
                else:
                    cropped = f[:crop_px, :crop_px]
                rgba = _magma_r(np.clip(cropped, 0, 1))[:, :, :3]
                rgb = (rgba * 255).astype(np.uint8)
                rgb[cropped < 0.01] = [255, 255, 255]
                sq_ax.imshow(rgb, aspect='equal', interpolation='nearest')
                # Blue numbered dot in top-left
                sq_ax.plot(0.10, 0.90, 'o', color='#2463EB', markersize=18,
                           markeredgewidth=0, transform=sq_ax.transAxes, zorder=11,
                           clip_on=False)
                sq_ax.text(0.10, 0.90, f'$\\mathbf{{{j + 1}}}$', fontsize=12, fontweight='bold',
                           ha='center', va='center', color='white',
                           transform=sq_ax.transAxes, zorder=12)
            else:
                sq_ax.set_facecolor('#E0E0E0')

            for sp in sq_ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(1)
                sp.set_color('black')

    # Horizontal colorbar at bottom, stretching up to ~3px below snapshot squares
    cbar_top = sq_y - 3 / (fig_h * dpi)  # 3px gap below squares
    cbar_bottom = 0.025
    cbar_height = 0.06
    cax = fig.add_axes([1/8, cbar_bottom, 3/4, cbar_height])
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([])
    cax.text(-0.005, 0.5, 'unoccluded', transform=cax.transAxes,
             ha='right', va='center', fontsize=16)
    cax.text(1.005, 0.5, 'more occluded', transform=cax.transAxes,
             ha='left', va='center', fontsize=16)

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output or f"trajectory_overlay_{env}"
    p = out_dir / f"{stem}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, facecolor='white', edgecolor='none',
                bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
