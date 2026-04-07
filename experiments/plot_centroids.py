"""Per-environment centroid displacement plots."""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from substrate.animals import load_animals


def select_heading_indices(n_ori: int, headings: int) -> list[int]:
    """Pick orientation indices closest to evenly-spaced target angles.

    Same logic as --gif-headings in run_env_competency.py (lines 365-373).
    Headings are reconstructed from n_ori assuming uniform 0-360 spacing.
    """
    if headings <= 0:
        return list(range(n_ori))

    all_angles = [i * 360.0 / n_ori for i in range(n_ori)]
    target_angles = [i * 360.0 / headings for i in range(headings)]

    indices = set()
    for target in target_angles:
        best = min(range(n_ori),
                   key=lambda i: min(abs(all_angles[i] - target),
                                     360 - abs(all_angles[i] - target)))
        indices.add(best)
    return sorted(indices)


def main():
    parser = argparse.ArgumentParser(description="Plot centroid trajectories on environments")
    parser.add_argument("--code", required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--input", default="results/env_competency")
    parser.add_argument("--envs", nargs="+", default=None, help="Subset of envs (default: all in NPZ)")
    parser.add_argument("--ori", type=int, default=None, help="Single orientation index (default: all)")
    parser.add_argument("--headings", type=int, default=None,
                        help="Evenly-spaced headings to plot (4=cardinal, 8=semicardinal, 0=all)")
    parser.add_argument("--occlusion", choices=["weighted", "uniform", "max"],
                        default="weighted", help="Which occlusion array to color by (default: weighted)")
    parser.add_argument("--output", default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()

    code = args.code
    # resolve species name for title (e.g. "Orbium unicaudatus")
    try:
        animals = load_animals("animals.json", codes=[code])
        species_name = animals[0].name if animals else code
    except Exception:
        species_name = code
    in_dir = Path(args.input) / code
    npz_path = in_dir / f"{code}_centroids.npz"

    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found")
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)
    centroids = data['centroids']           # [n_envs, n_ori, T_m, 2]
    mass = data['mass'] if 'mass' in data else None  # [n_envs, n_ori, T_m]
    env_names = list(data['env_names'])
    # load requested occlusion array, fall back to occlusion_mean for old NPZs
    occ_key = f'occlusion_{args.occlusion}'
    if occ_key in data:
        occ_mean = data[occ_key]
    elif 'occlusion_mean' in data:
        occ_mean = data['occlusion_mean']
    else:
        occ_mean = None

    # filter envs
    if args.envs:
        indices = [env_names.index(e) for e in args.envs if e in env_names]
        env_names = [env_names[i] for i in indices]
        centroids = centroids[indices]
        if mass is not None:
            mass = mass[indices]
        if occ_mean is not None:
            occ_mean = occ_mean[indices]

    n_envs = len(env_names)
    n_ori = centroids.shape[1]

    # resolve which orientations to plot
    if args.ori is not None:
        ori_indices = [args.ori]
    elif args.headings is not None:
        ori_indices = select_heading_indices(n_ori, args.headings)
    else:
        ori_indices = list(range(n_ori))

    out_dir = Path(args.output) if args.output else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # preload barrier masks once
    env_masks = {}
    for env_name in env_names:
        env_pt = Path(f"environments/{env_name}.pt")
        if env_pt.exists():
            env_masks[env_name] = torch.load(env_pt, weights_only=False).numpy()

    # yellow→red colormap for perceptive occlusion
    occ_cmap = mcolors.LinearSegmentedColormap.from_list(
        'occ_yellow_red', ['#FFD700', '#FF4500', '#CC0000'])

    # color normalization: power-law stretches the low end so small
    # differences in occlusion are visible (most values cluster near 0)
    if occ_mean is not None:
        occ_global_max = np.nanmax(occ_mean)
        if occ_global_max < 1e-6:
            occ_global_max = 1.0
        occ_norm = mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=occ_global_max)
    else:
        occ_global_max = 1.0
        occ_norm = None

    count = 0
    for idx, env_name in enumerate(env_names):
        for oi in ori_indices:
            traj = centroids[idx, oi]  # [T_m, 2] — row, col
            valid = ~np.isnan(traj[:, 0])
            # also mask out dead steps (mass=0 → centroid defaults to center)
            if mass is not None:
                alive = mass[idx, oi] > 0
                valid = valid & alive
            traj = traj[valid]
            if len(traj) == 0:
                continue

            # get occlusion values for this trajectory
            if occ_mean is not None:
                occ_vals = occ_mean[idx, oi][valid]
            else:
                occ_vals = None

            fig, ax = plt.subplots(figsize=(8, 4), facecolor='black')
            ax.set_facecolor('black')

            # draw barrier mask
            if env_name in env_masks:
                mask = env_masks[env_name]
                barrier_rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
                barrier_rgba[..., :3] = 200 / 255  # light grey, matching GIF style
                barrier_rgba[..., 3] = mask  # fully opaque where barrier=1
                ax.imshow(barrier_rgba, origin='upper',
                          extent=[0, mask.shape[1], mask.shape[0], 0])
                grid_w, grid_h = mask.shape[1], mask.shape[0]
            else:
                grid_h = centroids.shape[2]
                grid_w = grid_h * 2  # fallback assumes 1:2 aspect

            rows, cols = traj[:, 0], traj[:, 1]

            # derive velocity (px/frame) from centroid displacements
            dr = np.diff(rows)
            dc = np.diff(cols)
            speed = np.sqrt(dr**2 + dc**2)
            # clamp toroidal jumps so they don't blow up the size scale
            wrap_thresh = min(grid_h if env_name in env_masks else centroids.shape[2],
                              grid_w if env_name in env_masks else centroids.shape[2] * 2) * 0.4
            speed[speed > wrap_thresh] = 0.0
            # first point gets same speed as second
            speed = np.concatenate([[speed[0]], speed])
            # dot size anchored to cruising speed: half at rest, 4x at fast
            s_base = 2.0  # matplotlib s is area in pts²
            median_speed = np.median(speed[speed > 0]) if np.any(speed > 0) else 1.0
            ratio = np.clip(speed / median_speed, 0.5, 4.0)
            sizes = s_base * ratio ** 4

            # thin connecting line — break at toroidal wraps
            jump = np.sqrt(np.diff(rows)**2 + np.diff(cols)**2)
            breaks = np.where(jump > wrap_thresh)[0] + 1
            for seg in np.split(np.arange(len(rows)), breaks):
                if len(seg) > 1:
                    ax.plot(cols[seg], rows[seg], color='#555555', alpha=0.3, linewidth=0.5, zorder=1)

            # colored dots: occlusion-mapped or uniform yellow fallback
            # size encodes velocity (larger = faster)
            if occ_vals is not None and np.nanmax(occ_vals) > 1e-6:
                ax.scatter(cols, rows, c=occ_vals, cmap=occ_cmap,
                           norm=occ_norm,
                           s=sizes, alpha=0.7, zorder=2, linewidths=0)
            else:
                ax.scatter(cols, rows, c='#FFD700', s=sizes, alpha=0.6, zorder=2, linewidths=0)

            # start & end markers — white edge so they pop against the trail
            ax.plot(cols[-1], rows[-1], 'o', color='#FF3333', markersize=7,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=5)
            ax.plot(cols[0], rows[0], 'o', color='#33FF33', markersize=5,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=6)

            ax.set_xlim(0, grid_w)
            ax.set_ylim(grid_h, 0)
            ax.set_aspect('equal')

            deg = int(round(oi * 360.0 / n_ori))
            ax.set_title(f"{code} - {env_name} - {deg}°",
                         fontsize=10, color='white', pad=6)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            fig.tight_layout()
            env_subdir = out_dir / env_name
            env_subdir.mkdir(parents=True, exist_ok=True)
            out_path = env_subdir / f"{code}_{env_name}_{deg}deg_centroids.png"
            fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close(fig)
            print(f"Saved: {out_path}")
            count += 1

    print(f"\n{count} PNGs saved to {out_dir}")


if __name__ == "__main__":
    main()
