"""
Precompute a dense set of heading-aligned settled orientations for a creature.

Produces a single (N, H, W) tensor where index i = creature heading at i degrees
(CCW from +x). Also generates a spinning GIF for visual verification.

Usage:
    python initializations/generate_all_orientations.py -c O2u
    python initializations/generate_all_orientations.py -c O2u -n 36 --settle-mult 30
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

from substrate import Config, Simulation, load_animals
from substrate.lenia import _auto_device
from substrate.scaling import recenter_field
from utils.core import rotate_tensor
from PIL import Image, ImageDraw, ImageFont
from viz.gif import _to_rgb_batch, _encode_gif, _upscale_nn


DEFAULT_GRID = 64


def main():
    parser = argparse.ArgumentParser(
        description="Precompute dense heading-aligned orientations"
    )
    parser.add_argument("--code", "-c", type=str, required=True,
                        help="Creature code (e.g., O2u)")
    parser.add_argument("--scale", "-s", type=int, default=4,
                        help="Scale factor (default: 4)")
    parser.add_argument("--grid", "-g", type=int, default=DEFAULT_GRID,
                        help=f"Base grid size (default: {DEFAULT_GRID})")
    parser.add_argument("--num-orientations", "-n", type=int, default=360,
                        help="Number of orientations (default: 360)")
    parser.add_argument("--settle-mult", type=float, default=30.0,
                        help="Settle multiplier on T (default: 30.0)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory (default: initializations/{CODE}/s{SCALE}/)")
    parser.add_argument("--catalog", type=str, default="animals.json",
                        help="Creature catalog path")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    device = _auto_device()
    verbose = not args.quiet

    # Load heading offsets
    offsets_path = root / "initializations" / "heading_offsets.json"
    if not offsets_path.exists():
        print(f"Error: {offsets_path} not found")
        sys.exit(1)
    with open(offsets_path) as f:
        offsets = json.load(f)

    if args.code not in offsets:
        print(f"Error: '{args.code}' not in heading_offsets.json")
        print(f"  Available: {list(offsets.keys())}")
        print(f"  Run: python initializations/calibrate_headings.py -c {args.code} --scale {args.scale}")
        sys.exit(1)

    heading_offset = offsets[args.code]["heading_deg"]

    # Load creature
    catalog_path = root / args.catalog
    creatures = load_animals(catalog_path, codes=[args.code])
    if not creatures:
        print(f"Error: creature '{args.code}' not found in {args.catalog}")
        sys.exit(1)
    creature = creatures[0]

    T_param = creature.params.get("T", 10)
    settle_steps = int(args.settle_mult * T_param)
    N = args.num_orientations
    step_deg = 360.0 / N  # degrees per orientation

    cfg = Config.from_animal(creature, args.grid, scale=args.scale)
    H, W = cfg.grid_shape

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(f"initializations/{args.code}/s{args.scale}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Generating {N} orientations for {args.code} (scale={args.scale}, grid={args.grid})")
        print(f"  heading_offset={heading_offset:.2f}°, settle={settle_steps} steps ({args.settle_mult}*T)")
        print(f"  output: {out_dir}/")

    # Desired heading angles: [0, step, 2*step, ...] degrees
    headings = [i * step_deg for i in range(N)]
    # Body rotation = heading_offset + desired heading
    body_angles = [heading_offset + h for h in headings]

    all_tensors = []
    t_total = time.time()

    for i, (heading, body_angle) in enumerate(zip(headings, body_angles)):
        t0 = time.time()

        # Rotate
        creature_rotated = copy.deepcopy(creature)
        cells = torch.as_tensor(creature.cells, device=device, dtype=torch.float32)
        cells_rotated = rotate_tensor(cells, body_angle, device)
        creature_rotated.cells = cells_rotated.cpu().numpy()

        # Settle
        sim = Simulation(cfg)
        sim.place_animal(creature_rotated, center=True)
        for _ in range(settle_steps):
            sim.lenia.step()
        sim.board.replace_tensor(recenter_field(sim.board.tensor))

        settled = sim.board.tensor.detach().cpu()
        all_tensors.append(settled)

        elapsed = time.time() - t0
        if verbose and (i % max(1, N // 20) == 0 or i == N - 1):
            print(f"  [{i+1:3d}/{N}] heading={heading:6.1f}° body_rot={body_angle:7.1f}° {elapsed:.1f}s")

    # Stack into (N, H, W) tensor
    stacked = torch.stack(all_tensors)  # (N, H, W)

    # Save .pt
    pt_path = out_dir / f"{args.code}_s{args.scale}_all_orientations.pt"
    torch.save({
        'tensor': stacked,                           # (N, H, W) float32
        'headings': headings,                         # desired heading angles in degrees
        'body_angles': body_angles,                   # actual body rotations applied
        'heading_offset': heading_offset,             # creature's natural heading offset
        'code': args.code,
        'scale': args.scale,
        'base_grid': args.grid,
        'T': float(T_param),
        'settle_steps': settle_steps,
    }, pt_path)

    total_elapsed = time.time() - t_total
    mb = stacked.element_size() * stacked.nelement() / 1e6
    if verbose:
        print(f"\nSaved {pt_path} ({stacked.shape}, {mb:.1f} MB)")
        print(f"Total time: {total_elapsed:.0f}s")

    # Generate spinning GIF with degree labels
    gif_path = out_dir / f"{args.code}_s{args.scale}_spin.gif"
    rgb_batch = _to_rgb_batch(stacked, colormap="magma")
    upscale = 4

    # Try to load a font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()

    labeled_frames = []
    for i in range(len(rgb_batch)):
        frame = _upscale_nn(rgb_batch[i], upscale)
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        label = f"{headings[i]:.0f}\u00b0"
        draw.text((6, 4), label, fill=(255, 255, 255), font=font)
        labeled_frames.append(np.array(img))

    _encode_gif(np.stack(labeled_frames), gif_path, fps=30, upscale=1)
    if verbose:
        print(f"Spinning GIF: {gif_path}")


if __name__ == "__main__":
    main()
