"""
Precompute a dense set of heading-aligned settled orientations for a creature.

Produces a single (N, H, W) tensor where index i = creature heading at i degrees
(CCW from +x). Self-calibrates heading offset at the target scale.

Also generates a spinning GIF for visual verification and extracts o0.pt for sweeps.

Usage:
    python initializations/generate_all_orientations.py -c O2u -s 4 -g 48
    python initializations/generate_all_orientations.py -c O2u -s 4 -g 48 --heading-offset 72.3
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
from initializations.calibrate_headings import measure_heading
from PIL import Image, ImageDraw, ImageFont
from viz.gif import _to_rgb_batch, _encode_gif, _upscale_nn


DEFAULT_GRID = 64
SWEEP_BASE_GRID = 128  # base grid used by sweep.py


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
    parser.add_argument("--heading-offset", type=float, default=None,
                        help="Skip calibration and use this heading offset in degrees")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory (default: initializations/{CODE}/s{SCALE}/)")
    parser.add_argument("--catalog", type=str, default="animals.json",
                        help="Creature catalog path")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    device = _auto_device()
    verbose = not args.quiet

    # Load creature
    catalog_path = root / args.catalog
    creatures = load_animals(catalog_path, codes=[args.code])
    if not creatures:
        print(f"Error: creature '{args.code}' not found in {args.catalog}")
        sys.exit(1)
    creature = creatures[0]

    # Calibrate heading at the target scale (or use provided offset)
    if args.heading_offset is not None:
        heading_offset_deg = args.heading_offset
        heading_offset_rad = np.radians(heading_offset_deg)
        if verbose:
            print(f"Using provided heading offset: {heading_offset_deg:.2f}°")
    else:
        if verbose:
            print(f"Calibrating heading for {args.code} at scale={args.scale}, grid={args.grid}...")
        heading_offset_rad = measure_heading(
            creature, scale=args.scale, grid_size=args.grid,
            settle_mult=args.settle_mult, verbose=verbose,
        )
        heading_offset_deg = float(np.degrees(heading_offset_rad))
        if verbose:
            print(f"  → heading_offset = {heading_offset_deg:.2f}°")

        # Write back to heading_offsets.json (scale-aware)
        _update_heading_offsets(root, args.code, args.scale, args.grid,
                                heading_offset_rad, heading_offset_deg)

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
        print(f"  heading_offset={heading_offset_deg:.2f}°, settle={settle_steps} steps ({args.settle_mult}*T)")
        print(f"  output: {out_dir}/")

    # Desired heading angles: [0, step, 2*step, ...] degrees CCW from +x
    headings = [i * step_deg for i in range(N)]
    # Body rotation = heading_offset + desired heading
    body_angles = [heading_offset_deg + h for h in headings]

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
        'headings': headings,                         # [0, 1, 2, ..., 359] degrees CCW from +x
        'body_angles': body_angles,                   # actual body rotations applied
        'heading_offset_deg': heading_offset_deg,     # measured at THIS scale
        'heading_offset_rad': heading_offset_rad,
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

    # Extract o0.pt (heading 0° = positive x axis) for sweep consumption
    _extract_o0(stacked[0], args.code, args.scale, args.grid,
                body_angles[0], out_dir, verbose)

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


def _extract_o0(tensor_0: torch.Tensor, code: str, scale: int,
                base_grid: int, body_angle: float, out_dir: Path,
                verbose: bool):
    """Extract heading-0° tensor and save as o0.pt, centered on sweep grid."""
    sweep_grid = SWEEP_BASE_GRID * scale  # e.g. 128*4 = 512
    sH, sW = tensor_0.shape

    if sH < sweep_grid or sW < sweep_grid:
        # Center the smaller tensor on the sweep grid
        canvas = torch.zeros(sweep_grid, sweep_grid, dtype=tensor_0.dtype)
        r0 = (sweep_grid - sH) // 2
        c0 = (sweep_grid - sW) // 2
        canvas[r0:r0 + sH, c0:c0 + sW] = tensor_0
        out_tensor = canvas
    else:
        out_tensor = tensor_0

    pt_path = out_dir / f"{code}_s{scale}_o0.pt"
    torch.save({
        'tensor': out_tensor,
        'sit_idx': 0,
        'angle': body_angle,
        'code': code,
        'name': code,
    }, pt_path)

    if verbose:
        print(f"Extracted o0.pt: {pt_path} ({out_tensor.shape[0]}x{out_tensor.shape[1]})")


def _update_heading_offsets(root: Path, code: str, scale: int, grid: int,
                            heading_rad: float, heading_deg: float):
    """Write calibration result to heading_offsets.json (scale-aware format)."""
    offsets_path = root / "initializations" / "heading_offsets.json"

    if offsets_path.exists():
        with open(offsets_path) as f:
            all_offsets = json.load(f)
    else:
        all_offsets = {}

    # Migrate old flat format if needed: {code: {heading_deg, scale}} → {code: {scale: {...}}}
    if code in all_offsets and "heading_deg" in all_offsets[code]:
        old = all_offsets[code]
        old_scale = str(old.get("scale", 2))
        all_offsets[code] = {
            old_scale: {
                "heading_rad": old["heading_rad"],
                "heading_deg": old["heading_deg"],
                "grid": old.get("grid", 128),
            }
        }

    if code not in all_offsets:
        all_offsets[code] = {}

    all_offsets[code][str(scale)] = {
        "heading_rad": round(heading_rad, 6),
        "heading_deg": round(heading_deg, 2),
        "grid": grid,
    }

    with open(offsets_path, "w") as f:
        json.dump(all_offsets, f, indent=2)


if __name__ == "__main__":
    main()
