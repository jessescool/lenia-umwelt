"""
Visualize raw orbit datasets as GIFs (one per rotation angle).

Loads a .pt file produced by `orbits/orbits.py raw` and writes
animated GIFs. When warmup frames are present, stitches warmup + trial
into a single GIF with a blue border during warmup.

Usage:
    python -m viz.orbits results/orbits/O2u/s1/O2u_s1_raw.pt
    python -m viz.orbits results/orbits/O2u/s1/O2u_s1_raw.pt --angles 0 30 60 --upscale 4
    python -m viz.orbits results/orbits/O2u/s1/O2u_s1_raw.pt --no-warmup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from viz.gif import _to_rgb_batch, _encode_gif


def _render_stitched_gif(
    warmup: torch.Tensor | None,
    trial: torch.Tensor,
    path: Path,
    fps: int = 15,
    upscale: int = 4,
    colormap: str = "magma",
) -> Path:
    """Render warmup + trial frames into one GIF.

    Blue 2px border on warmup frames, no border on trial frames.
    """
    # Stack all frames: warmup (if any) then trial
    parts = []
    if warmup is not None and warmup.shape[0] > 0:
        parts.append(warmup)
    parts.append(trial)
    all_frames = torch.cat(parts, dim=0)  # (total, H, W)
    n_warmup = warmup.shape[0] if warmup is not None else 0

    rgb_batch = _to_rgb_batch(all_frames, colormap)

    # Blue border on warmup frames (2px)
    if n_warmup > 0:
        blue = np.array([80, 140, 255], dtype=np.uint8)
        b = 2  # border width in native pixels
        rgb_batch[:n_warmup, :b, :] = blue    # top
        rgb_batch[:n_warmup, -b:, :] = blue   # bottom
        rgb_batch[:n_warmup, :, :b] = blue    # left
        rgb_batch[:n_warmup, :, -b:] = blue   # right

    return _encode_gif(rgb_batch, path, fps=fps, upscale=upscale)


def main():
    parser = argparse.ArgumentParser(description="Render orbit GIFs from saved .pt data")
    parser.add_argument("input", type=Path, help="Path to orbit .pt file")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                        help="Output directory (default: results/new/orbits/<stem>)")
    parser.add_argument("--angles", type=float, nargs="*", default=None,
                        help="Specific angles to render (default: all)")
    parser.add_argument("--upscale", type=int, default=4,
                        help="Nearest-neighbor upscale factor (default: 4)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames per second (default: 15)")
    parser.add_argument("--colormap", type=str, default="magma")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warmup frames even if present")

    args = parser.parse_args()

    data = torch.load(args.input, weights_only=False)
    frames = data["frames"]                          # (num_rotations, num_frames, H, W)
    warmup_frames = data.get("warmup_frames", None)  # (num_rotations, num_warmup, H, W) or None
    angles = data["angles"]
    code = data.get("code", "unknown")

    has_warmup = warmup_frames is not None and not args.no_warmup

    out_dir = args.output_dir or Path(f"results/new/orbits/{args.input.stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter to requested angles
    if args.angles is not None:
        angle_set = set(args.angles)
        indices = [i for i, a in enumerate(angles) if a in angle_set]
        if not indices:
            print(f"No matching angles. Available: {angles}")
            sys.exit(1)
    else:
        indices = list(range(len(angles)))

    n_warmup = warmup_frames.shape[1] if has_warmup else 0
    n_trial = frames.shape[1]
    print(f"Rendering {len(indices)} GIFs from {args.input}")
    print(f"  {code}, {frames.shape[2]}x{frames.shape[3]} grid")
    if has_warmup:
        print(f"  {n_warmup} warmup frames (blue border) + {n_trial} trial frames")
    else:
        print(f"  {n_trial} trial frames" + (" (no warmup data)" if warmup_frames is None else " (warmup skipped)"))

    for i in indices:
        angle = angles[i]
        warmup_i = warmup_frames[i] if has_warmup else None
        trial_i = frames[i]
        out_path = out_dir / f"{code}_angle{angle:05.1f}.gif"
        _render_stitched_gif(warmup_i, trial_i, out_path,
                             fps=args.fps, upscale=args.upscale, colormap=args.colormap)
        print(f"  {out_path}")

    print(f"\nDone — {len(indices)} GIFs in {out_dir}/")


if __name__ == "__main__":
    main()
