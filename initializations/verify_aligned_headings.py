"""Verify heading alignment across orientations.
    python initializations/verify_aligned_headings.py --scale 2 --grid 128 --orientations 0 15 30 45
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from substrate import Config, Simulation, load_animals
from substrate.lenia import _auto_device
from substrate.scaling import recenter_field
from metrics_and_machinery.trajectory_metrics import centroid
from utils.core import rotate_tensor
from viz.gif import _to_rgb_batch, _encode_gif, draw_dot
from viz._helpers import heading_from_centroids


DEFAULT_GRID = 128
OBSERVE_STEPS = 200


def draw_arrow(image, r0, c0, angle_rad, length, color):
    H, W, _ = image.shape
    dc = np.cos(angle_rad)
    dr = np.sin(angle_rad)
    for i in range(length):
        r = int(round(r0 + dr * i)) % H
        c = int(round(c0 + dc * i)) % W
        for off in range(-1, 2):
            image[(r + off) % H, c] = color
            image[r, (c + off) % W] = color
    tip_r = r0 + dr * length
    tip_c = c0 + dc * length
    for side in [-1, 1]:
        perp = angle_rad + side * 2.5
        for i in range(length // 3):
            r = int(round(tip_r + np.sin(perp) * i)) % H
            c = int(round(tip_c + np.cos(perp) * i)) % W
            image[r, c] = color


def main():
    parser = argparse.ArgumentParser(description="Generate heading-aligned inits + verification GIFs")
    parser.add_argument("--scale", "-s", type=int, default=2)
    parser.add_argument("--grid", "-g", type=int, default=DEFAULT_GRID)
    parser.add_argument("--orientations", type=float, nargs='+', default=[0, 15, 30, 45],
                        help="Orientation angles in degrees (default: 0 15 30 45)")
    parser.add_argument("--observe-steps", type=int, default=OBSERVE_STEPS)
    parser.add_argument("--settle-mult", type=float, default=30.0)
    parser.add_argument("--catalog", type=str, default="animals.json")
    parser.add_argument("--gif-dir", type=str, default="initializations/heading_gifs")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    device = _auto_device()
    ori_degs = args.orientations

    # Load heading offsets
    offsets_path = root / "initializations" / "heading_offsets.json"
    if not offsets_path.exists():
        print(f"Error: {offsets_path} not found")
        sys.exit(1)
    with open(offsets_path) as f:
        offsets = json.load(f)

    catalog_path = root / args.catalog
    gif_dir = Path(args.gif_dir)
    gif_dir.mkdir(parents=True, exist_ok=True)

    for code, info in offsets.items():
        heading_deg = info["heading_deg"]
        # rotate_tensor(θ) shifts heading by -θ in image coords, so base = +offset
        base = heading_deg
        angles = [base + d for d in ori_degs]

        creatures = load_animals(catalog_path, codes=[code])
        if not creatures:
            print(f"  {code}: skipping (not in catalog)")
            continue
        creature = creatures[0]
        T_param = creature.params.get("T", 10)
        settle_steps = int(args.settle_mult * T_param)

        cfg = Config.from_animal(creature, args.grid, scale=args.scale)
        H, W = cfg.grid_shape

        pt_dir = Path(f"initializations/{code}/s{args.scale}")
        pt_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{code}: heading_offset={heading_deg:.1f}°, orientations={ori_degs}")

        for oi, (ori_deg, angle) in enumerate(zip(ori_degs, angles)):
            t0 = time.time()

            # Rotate
            creature_rotated = copy.deepcopy(creature)
            cells = torch.as_tensor(creature.cells, device=device, dtype=torch.float32)
            cells_rotated = rotate_tensor(cells, angle, device)
            creature_rotated.cells = cells_rotated.cpu().numpy()

            # Settle
            sim = Simulation(cfg)
            sim.place_animal(creature_rotated, center=True)
            for _ in range(settle_steps):
                sim.lenia.step()
            sim.board.replace_tensor(recenter_field(sim.board.tensor))

            settled_tensor = sim.board.tensor.detach().cpu()

            # Save .pt
            stem = f"{code}_s{args.scale}_o{oi}"
            pt_path = pt_dir / f"{stem}.pt"
            torch.save({
                'tensor': settled_tensor,
                'code': code,
                'name': creature.name,
                'scale': args.scale,
                'base_grid': args.grid,
                'angle': angle,
                'sit_idx': oi,
                'T': float(T_param),
                'heading_offset': heading_deg,
            }, pt_path)

            # Save .png preview
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(settled_tensor.numpy(), cmap='inferno', interpolation='nearest')
            ax.set_title(f"{creature.name}  o{oi} ({ori_deg:.0f}°)", fontsize=10)
            ax.axis('off')
            png_path = pt_dir / f"{stem}.png"
            fig.savefig(png_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
            plt.close(fig)

            # Run forward for GIF
            frames = []
            centroids_list = []
            for _ in range(args.observe_steps):
                sim.lenia.step()
                board = sim.board.tensor.detach().cpu()
                frames.append(board)
                r, c = centroid(sim.board.tensor)
                centroids_list.append((r, c))

            # Measure actual heading
            centroids_arr = np.array(centroids_list)[np.newaxis, :, :]
            headings = heading_from_centroids(centroids_arr, (H, W))
            sin_mean = np.sin(headings).mean()
            cos_mean = np.cos(headings).mean()
            actual_deg = np.degrees(np.arctan2(sin_mean, cos_mean))

            # Make GIF
            frames_tensor = torch.stack(frames)
            rgb_batch = _to_rgb_batch(frames_tensor, colormap="magma")

            upscale = 4
            green = np.array([0, 255, 0], dtype=np.uint8)
            white = np.array([255, 255, 255], dtype=np.uint8)
            red = np.array([255, 80, 80], dtype=np.uint8)

            expected_rad = np.radians(ori_deg)
            actual_rad = np.radians(actual_deg)

            composed = []
            for t in range(len(frames)):
                rgb = np.repeat(np.repeat(rgb_batch[t], upscale, axis=0), upscale, axis=1).copy()
                r, c = centroids_list[t]
                ru, cu = int(r * upscale), int(c * upscale)
                draw_dot(rgb, ru, cu, 3, white)
                draw_arrow(rgb, ru, cu, expected_rad, 20, green)
                draw_arrow(rgb, ru, cu, actual_rad, 15, red)
                composed.append(rgb)

            gif_path = gif_dir / f"{code}_o{oi}.gif"
            _encode_gif(composed, gif_path, fps=15, upscale=1)

            elapsed = time.time() - t0
            print(f"  o{oi}: rot={angle:6.1f}° → heading={actual_deg:6.1f}° (expect {ori_deg:5.1f}°)  {elapsed:.1f}s")
            print(f"       {pt_path}  {png_path.name}  {gif_path.name}")

    print(f"\nAll outputs saved. GIFs in {gif_dir}/")


if __name__ == "__main__":
    main()
