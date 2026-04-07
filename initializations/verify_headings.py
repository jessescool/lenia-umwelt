# initializations/verify_headings.py — Jesse Cool (jessescool)
"""Verify heading calibration with arrow overlay plots."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from substrate import Config, Simulation, load_animals
from substrate.lenia import _auto_device
from substrate.scaling import recenter_field
from metrics_and_machinery.trajectory_metrics import centroid
from viz.gif import _to_rgb_batch, _encode_gif, draw_dot


DEFAULT_GRID = 128
OBSERVE_STEPS = 200


def draw_arrow(image: np.ndarray, r0: int, c0: int, angle_rad: float, length: int, color: np.ndarray):
    """Draw a simple arrow on an RGB image."""
    # angle_rad: arctan2(dr, dc), 0 = rightward
    dc = np.cos(angle_rad)
    dr = np.sin(angle_rad)
    H, W, _ = image.shape
    for i in range(length):
        r = int(round(r0 + dr * i)) % H
        c = int(round(c0 + dc * i)) % W
        for off in range(-1, 2):
            image[(r + off) % H, c] = color
            image[r, (c + off) % W] = color

    # Arrowhead
    tip_r = r0 + dr * length
    tip_c = c0 + dc * length
    for side in [-1, 1]:
        perp_angle = angle_rad + side * 2.5  # ~143° back
        for i in range(length // 3):
            r = int(round(tip_r + np.sin(perp_angle) * i)) % H
            c = int(round(tip_c + np.cos(perp_angle) * i)) % W
            image[r, c] = color


def main():
    parser = argparse.ArgumentParser(description="Generate heading verification GIFs")
    parser.add_argument("--scale", "-s", type=int, default=2)
    parser.add_argument("--grid", "-g", type=int, default=DEFAULT_GRID)
    parser.add_argument("--observe-steps", type=int, default=OBSERVE_STEPS)
    parser.add_argument("--settle-mult", type=float, default=30.0)
    parser.add_argument("--catalog", type=str, default="animals.json")
    parser.add_argument("--output-dir", "-o", type=str, default="initializations/heading_gifs")
    args = parser.parse_args()

    root = Path(__file__).parent.parent

    # Load heading offsets
    offsets_path = root / "initializations" / "heading_offsets.json"
    if not offsets_path.exists():
        print(f"Error: {offsets_path} not found. Run calibrate_headings.py first.")
        sys.exit(1)
    with open(offsets_path) as f:
        offsets = json.load(f)

    catalog_path = root / args.catalog
    device = _auto_device()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for code, info in offsets.items():
        print(f"\n{code}: heading = {info['heading_deg']:.1f}°")
        creatures = load_animals(catalog_path, codes=[code])
        if not creatures:
            print(f"  skipping (not in catalog)")
            continue
        creature = creatures[0]
        T = creature.params.get("T", 10)
        settle_steps = int(args.settle_mult * T)

        cfg = Config.from_animal(creature, args.grid, scale=args.scale)
        H, W = cfg.grid_shape

        # Settle
        sim = Simulation(cfg)
        sim.place_animal(creature, center=True)
        for _ in range(settle_steps):
            sim.lenia.step()
        sim.board.replace_tensor(recenter_field(sim.board.tensor))

        # Collect frames
        frames = []
        centroids = []
        for _ in range(args.observe_steps):
            sim.lenia.step()
            board = sim.board.tensor.detach().cpu()
            frames.append(board)
            r, c = centroid(sim.board.tensor)
            centroids.append((r, c))

        # Convert to RGB
        frames_tensor = torch.stack(frames)
        rgb_batch = _to_rgb_batch(frames_tensor, colormap="magma")  # (T, H, W, 3)

        heading_rad = info["heading_rad"]
        arrow_color = np.array([0, 255, 0], dtype=np.uint8)
        dot_color = np.array([255, 255, 255], dtype=np.uint8)

        # Draw centroid dot + heading arrow on each frame
        upscale = 4
        composed = []
        for t in range(len(frames)):
            # Upscale
            rgb = np.repeat(np.repeat(rgb_batch[t], upscale, axis=0), upscale, axis=1).copy()
            r, c = centroids[t]
            ru, cu = int(r * upscale), int(c * upscale)
            draw_dot(rgb, ru, cu, 3, dot_color)
            draw_arrow(rgb, ru, cu, heading_rad, 20, arrow_color)
            composed.append(rgb)

        gif_path = out_dir / f"{code}_heading.gif"
        _encode_gif(composed, gif_path, fps=15, upscale=1)
        print(f"  saved {gif_path}")

    print(f"\nAll GIFs saved to {out_dir}/")


if __name__ == "__main__":
    main()
