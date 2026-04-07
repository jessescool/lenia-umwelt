"""Six-panel barrier avoidance figure with trajectory flow fields."""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import uniform_filter1d
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from environments.environments import make_env
from viz._helpers import heading_from_centroids, toroidal_diff


# ── config ──────────────────────────────────────────────────────────────
PANEL_ENVS = ["funnel", "box", "shuriken", "capsule"]
OCC_THRESHOLD = 0.02       # minimum occlusion to count as "event"
EVENT_GAP = 5              # merge events closer than this many frames
MIN_EVENT_LEN = 3          # drop events shorter than this
WINDOW = 30                # heading-change window (frames)
CURATED_N = 8              # orientations per panel with arrows
FLOW_ALPHA = 0.03          # alpha for all-orientation flow lines


def angular_diff(a, b):
    """Signed angular difference b - a, wrapped to [-pi, pi]."""
    d = b - a
    return d - 2 * np.pi * np.round(d / (2 * np.pi))


def detect_occlusion_events(occ, threshold=OCC_THRESHOLD,
                            gap=EVENT_GAP, min_len=MIN_EVENT_LEN):
    """Find contiguous occlusion events. Returns list of (start, end) slices."""
    above = occ > threshold
    events = []
    in_event = False
    start = 0
    gap_count = 0
    for i in range(len(above)):
        if above[i]:
            if not in_event:
                start = i
                in_event = True
            gap_count = 0
        else:
            if in_event:
                gap_count += 1
                if gap_count > gap:
                    end = i - gap_count
                    if end - start >= min_len:
                        events.append((start, end))
                    in_event = False
    if in_event:
        end = len(above)
        if end - start >= min_len:
            events.append((start, end))
    return events


def compute_heading_change(headings, window=WINDOW):
    """Absolute heading change over rolling window. Returns same-length array."""
    # headings shape: (T-1,) — one per consecutive frame pair
    if len(headings) < window:
        return np.abs(angular_diff(headings[:1], headings[-1:]))
    # compare heading at t with heading at t+window
    n = len(headings) - window
    hc = np.abs(angular_diff(headings[:n], headings[window:]))
    # pad to match length
    pad = np.full(window, np.nan)
    return np.concatenate([hc, pad])


def load_barrier_mask(env_name, grid_shape, scale):
    """Generate barrier mask via make_env."""
    mask = make_env(env_name, grid_shape, torch.device('cpu'),
                    scaled=True).numpy()
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="O2u")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--input", default="results/env_competency")
    parser.add_argument("--output", default="results/new/O2u_barrier_avoidance.png")
    args = parser.parse_args()

    # ── load data ───────────────────────────────────────────────────────
    npz_path = Path(args.input) / args.code / f"{args.code}_centroids.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found"); sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)
    centroids = data['centroids']           # [n_envs, 360, 1000, 2]
    occ = data['occlusion_weighted']        # [n_envs, 360, 1000]
    mass = data['mass']                     # [n_envs, 360, 1000]
    env_names = list(data['env_names'])
    scale = int(data['scale'])

    grid_shape = (128 * scale, 256 * scale)  # (512, 1024)
    H, W = grid_shape
    n_ori = centroids.shape[1]

    # ── precompute headings for all envs/oris ───────────────────────────
    headings = heading_from_centroids(
        centroids.reshape(-1, centroids.shape[2], 2), grid_shape
    ).reshape(centroids.shape[0], n_ori, -1)  # [n_envs, 360, T-1]

    # ── figure layout ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14), facecolor='black')
    # top row: 4 trajectory panels
    gs_top = fig.add_gridspec(1, 4, left=0.03, right=0.97, top=0.95, bottom=0.45,
                              wspace=0.08)
    # bottom row: 2 aggregate panels
    gs_bot = fig.add_gridspec(1, 2, left=0.08, right=0.92, top=0.38, bottom=0.05,
                              wspace=0.35)

    # ── TOP ROW: trajectory flow fields ─────────────────────────────────
    for pi, env_name in enumerate(PANEL_ENVS):
        if env_name not in env_names:
            print(f"WARNING: {env_name} not in data, skipping"); continue
        ei = env_names.index(env_name)
        ax = fig.add_subplot(gs_top[0, pi])
        ax.set_facecolor('black')

        # barrier mask
        mask = load_barrier_mask(env_name, grid_shape, scale)
        barrier_rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        barrier_rgba[..., :3] = 0.35  # dark gray
        barrier_rgba[..., 3] = mask * 0.7
        ax.imshow(barrier_rgba, origin='upper', extent=[0, W, H, 0])

        # all 360 trajectories as low-alpha flow lines
        wrap_thresh = min(H, W) * 0.4
        for oi in range(n_ori):
            alive = mass[ei, oi] > 0
            valid = ~np.isnan(centroids[ei, oi, :, 0]) & alive
            traj = centroids[ei, oi][valid]
            if len(traj) < 10:
                continue
            rows, cols = traj[:, 0], traj[:, 1]
            jump = np.sqrt(np.diff(rows)**2 + np.diff(cols)**2)
            breaks = np.where(jump > wrap_thresh)[0] + 1
            for seg in np.split(np.arange(len(rows)), breaks):
                if len(seg) > 1:
                    ax.plot(cols[seg], rows[seg], color='white',
                            alpha=FLOW_ALPHA, linewidth=0.5, zorder=1)

        # curated subset with occlusion coloring + heading arrows
        # pick orientations that have the most occlusion events (most interesting)
        occ_totals = np.nansum(occ[ei], axis=1)  # sum per orientation
        alive_frac = (mass[ei] > 0).mean(axis=1)
        # only pick orientations that survived a decent portion
        viable = alive_frac > 0.3
        if viable.sum() < CURATED_N:
            viable[:] = True
        # among viable, pick those with highest occlusion exposure
        candidates = np.where(viable)[0]
        sort_idx = np.argsort(occ_totals[candidates])[::-1]
        # pick evenly spaced from the top half
        top_half = candidates[sort_idx[:max(len(sort_idx)//2, CURATED_N)]]
        step = max(1, len(top_half) // CURATED_N)
        curated = top_half[::step][:CURATED_N]

        # occlusion colormap: cyan to magenta
        occ_cmap = plt.cm.cool

        for oi in curated:
            alive = mass[ei, oi] > 0
            valid = ~np.isnan(centroids[ei, oi, :, 0]) & alive
            traj = centroids[ei, oi][valid]
            occ_v = occ[ei, oi][valid]
            if len(traj) < 10:
                continue
            rows, cols = traj[:, 0], traj[:, 1]

            # colored dots by occlusion
            occ_norm = occ_v / max(np.nanmax(occ_v), 1e-6)
            colors = occ_cmap(occ_norm)
            colors[:, 3] = 0.5 + 0.4 * occ_norm  # alpha scales with occlusion
            ax.scatter(cols, rows, c=colors, s=1.5, zorder=3, linewidths=0)

            # heading arrows at occlusion onset points
            h = headings[ei, oi]
            valid_idx = np.where(valid)[0]
            events = detect_occlusion_events(occ[ei, oi])
            for (es, ee) in events:
                if es < 2 or es >= len(valid_idx) - 5:
                    continue
                # find matching index in the truncated trajectory
                t_pre = max(0, es - 3)
                t_post = min(len(h) - 1, es + 5)
                if t_pre >= centroids.shape[2] or t_post >= centroids.shape[2]:
                    continue
                if not valid[es]:
                    continue
                # position at event start
                r0, c0 = centroids[ei, oi, es, 0], centroids[ei, oi, es, 1]
                if np.isnan(r0):
                    continue
                # pre heading
                if es - 1 < len(h):
                    h_pre = h[max(0, es-3):es].mean() if es > 2 else h[0]
                    h_post = h[es:min(es+5, len(h))].mean()
                    arrow_len = 15
                    # pre arrow (dim)
                    dr_pre = arrow_len * np.sin(h_pre)  # heading uses arctan2(dr, dc)
                    dc_pre = arrow_len * np.cos(h_pre)
                    ax.annotate('', xy=(c0 + dc_pre, r0 + dr_pre), xytext=(c0, r0),
                                arrowprops=dict(arrowstyle='->', color='#888888',
                                                lw=1.0), zorder=4)
                    # post arrow (bright)
                    dr_post = arrow_len * np.sin(h_post)
                    dc_post = arrow_len * np.cos(h_post)
                    ax.annotate('', xy=(c0 + dc_post, r0 + dr_post), xytext=(c0, r0),
                                arrowprops=dict(arrowstyle='->', color='#00FF88',
                                                lw=1.5), zorder=5)

            # start/end markers
            ax.plot(cols[0], rows[0], 'o', color='#33FF33', markersize=3,
                    markeredgecolor='white', markeredgewidth=0.3, zorder=6)
            ax.plot(cols[-1], rows[-1], 'o', color='#FF3333', markersize=3,
                    markeredgecolor='white', markeredgewidth=0.3, zorder=6)

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect('equal')
        ax.set_title(env_name, fontsize=14, color='white', fontweight='bold', pad=8)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # ── BOTTOM ROW: aggregate analysis ──────────────────────────────────

    # Collect per-environment deflection stats
    env_deflection_occ = {}    # {env: [mean |heading change| during occlusion]}
    env_deflection_base = {}   # {env: [mean |heading change| baseline]}
    all_scatter_occ = []       # peak occlusion per event
    all_scatter_defl = []      # total heading deflection per event
    all_scatter_env = []       # environment label

    for ei, env_name in enumerate(env_names):
        occ_hc_list = []
        base_hc_list = []
        for oi in range(n_ori):
            alive = mass[ei, oi] > 0
            valid = ~np.isnan(centroids[ei, oi, :, 0]) & alive
            if valid.sum() < 20:
                continue

            h = headings[ei, oi]  # length T_h = n_frames - 1
            T_h = len(h)
            if T_h < WINDOW + 5:
                continue

            # heading change per frame (absolute angular velocity)
            # hc[i] = |h[i+1] - h[i]|, length T_h - 1
            hc = np.abs(angular_diff(h[:-1], h[1:]))
            # match occlusion to hc: hc[i] ~ change around frame i+1
            occ_hc = occ[ei, oi][1:len(hc)+1]  # same length as hc

            # classify frames
            occ_frames = occ_hc > OCC_THRESHOLD
            base_frames = occ_hc < 0.001

            if occ_frames.sum() > 3:
                occ_hc_list.append(np.nanmean(hc[occ_frames]))
            if base_frames.sum() > 3:
                base_hc_list.append(np.nanmean(hc[base_frames]))

            # per-event stats for scatter
            events = detect_occlusion_events(occ[ei, oi][:T_h])
            for (es, ee) in events:
                peak_occ = np.nanmax(occ[ei, oi][es:ee])
                # total heading deflection over event
                if es < len(h) and ee <= len(h):
                    event_h = h[max(0,es-1):min(ee, len(h))]
                    if len(event_h) > 1:
                        total_defl = np.sum(np.abs(angular_diff(
                            event_h[:-1], event_h[1:])))
                        all_scatter_occ.append(peak_occ)
                        all_scatter_defl.append(np.degrees(total_defl))
                        all_scatter_env.append(env_name)

        if occ_hc_list:
            env_deflection_occ[env_name] = np.degrees(np.mean(occ_hc_list))
        if base_hc_list:
            env_deflection_base[env_name] = np.degrees(np.mean(base_hc_list))

    # ── Bar chart: deflection during occlusion vs baseline ──────────────
    ax_bar = fig.add_subplot(gs_bot[0, 0])
    ax_bar.set_facecolor('#111111')

    common_envs = sorted(set(env_deflection_occ) & set(env_deflection_base))
    if common_envs:
        x = np.arange(len(common_envs))
        bar_w = 0.35
        base_vals = [env_deflection_base[e] for e in common_envs]
        occ_vals_bar = [env_deflection_occ[e] for e in common_envs]

        bars_base = ax_bar.bar(x - bar_w/2, base_vals, bar_w, label='Baseline',
                               color='#4488CC', alpha=0.85, edgecolor='white',
                               linewidth=0.5)
        bars_occ = ax_bar.bar(x + bar_w/2, occ_vals_bar, bar_w, label='During occlusion',
                              color='#FF6644', alpha=0.85, edgecolor='white',
                              linewidth=0.5)

        # annotate deflection ratios
        for i, env in enumerate(common_envs):
            if base_vals[i] > 0:
                ratio = occ_vals_bar[i] / base_vals[i]
                ax_bar.text(x[i] + bar_w/2, occ_vals_bar[i] + 0.2,
                           f'{ratio:.1f}x', ha='center', va='bottom',
                           color='white', fontsize=8, fontweight='bold')

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(common_envs, rotation=35, ha='right',
                               fontsize=9, color='white')
        ax_bar.set_ylabel('Mean |heading change| (°/frame)', color='white', fontsize=10)
        ax_bar.set_title('Heading deflection: occlusion vs baseline',
                        color='white', fontsize=12, fontweight='bold', pad=10)
        ax_bar.legend(loc='upper left', fontsize=9, facecolor='#222222',
                     edgecolor='#555555', labelcolor='white')
        ax_bar.tick_params(colors='white')
        ax_bar.spines['bottom'].set_color('#555555')
        ax_bar.spines['left'].set_color('#555555')
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)

    # ── Dose-response scatter ───────────────────────────────────────────
    ax_scatter = fig.add_subplot(gs_bot[0, 1])
    ax_scatter.set_facecolor('#111111')

    if all_scatter_occ:
        scatter_occ = np.array(all_scatter_occ)
        scatter_defl = np.array(all_scatter_defl)
        scatter_env = np.array(all_scatter_env)

        # color by environment
        unique_envs = sorted(set(scatter_env))
        env_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_envs)))
        env_color_map = {e: env_colors[i] for i, e in enumerate(unique_envs)}

        for env in unique_envs:
            mask_e = scatter_env == env
            ax_scatter.scatter(scatter_occ[mask_e], scatter_defl[mask_e],
                             c=[env_color_map[env]], label=env,
                             alpha=0.4, s=12, edgecolors='none', zorder=2)

        # trend line (all data)
        if len(scatter_occ) > 10:
            # bin and compute median
            bins = np.linspace(scatter_occ.min(), scatter_occ.max(), 15)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_medians = []
            for i in range(len(bins) - 1):
                in_bin = (scatter_occ >= bins[i]) & (scatter_occ < bins[i+1])
                if in_bin.sum() > 2:
                    bin_medians.append((bin_centers[i], np.median(scatter_defl[in_bin])))
            if bin_medians:
                bx, by = zip(*bin_medians)
                ax_scatter.plot(bx, by, color='white', linewidth=2, alpha=0.8,
                              zorder=3, label='median trend')

        ax_scatter.set_xlabel('Peak occlusion', color='white', fontsize=10)
        ax_scatter.set_ylabel('Total heading deflection (°)', color='white', fontsize=10)
        ax_scatter.set_title('Dose-response: occlusion severity → deflection',
                           color='white', fontsize=12, fontweight='bold', pad=10)
        ax_scatter.legend(loc='upper left', fontsize=8, facecolor='#222222',
                        edgecolor='#555555', labelcolor='white', ncol=2)
        ax_scatter.tick_params(colors='white')
        ax_scatter.spines['bottom'].set_color('#555555')
        ax_scatter.spines['left'].set_color('#555555')
        ax_scatter.spines['top'].set_visible(False)
        ax_scatter.spines['right'].set_visible(False)

    # ── save ────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
