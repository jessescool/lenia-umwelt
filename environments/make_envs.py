"""Generate all environment .pt files at simulation resolution.

Usage:
    python environments/make_envs.py                         # default: 128x256 base, scale 4
    python environments/make_envs.py --grid 128x256 --scale 2
    python environments/make_envs.py -g 64x128 -s 4 -o environments/
"""

import argparse
from pathlib import Path

import torch

from environments.environments import ENVIRONMENTS, make_env


def parse_grid(s: str) -> tuple[int, int]:
    parts = s.lower().split("x")
    return (int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(description="Generate precomputed environment .pt files")
    parser.add_argument("--grid", "-g", default="128x256", help="Base grid as HxW (default: 128x256)")
    parser.add_argument("--scale", "-s", type=int, default=4, help="Scale factor (default: 4)")
    parser.add_argument("--output-dir", "-o", default="environments/", help="Output directory (default: environments/)")
    args = parser.parse_args()

    base_h, base_w = parse_grid(args.grid)
    shape = (base_h * args.scale, base_w * args.scale)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(ENVIRONMENTS)} environment masks at {shape[0]}x{shape[1]} "
          f"(base {base_h}x{base_w}, scale {args.scale})")

    device = torch.device("cpu")
    dtype = torch.float32

    for name in ENVIRONMENTS:
        mask = make_env(name, shape, device, dtype, scaled=True)
        path = out_dir / f"{name}.pt"
        torch.save(mask, path)
        print(f"  {name}.pt  ({mask.shape[0]}x{mask.shape[1]}, {mask.sum().item():.0f} barrier px)")

    print(f"\nDone! {len(ENVIRONMENTS)} .pt files saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
