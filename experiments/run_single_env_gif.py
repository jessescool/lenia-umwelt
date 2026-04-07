"""Run a single creature through one environment, save a GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from substrate import Simulation, Config, load_animals
from environments import load_env
from experiments.run_env_batch import extract_pattern, get_spawn_position
from viz.gif import write_gif


def parse_grid(grid_str: str) -> tuple[int, int]:
    parts = grid_str.lower().split("x")
    return (int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(description="Single-orientation env GIF + frame tensor")
    parser.add_argument("--code", required=True, help="Creature code")
    parser.add_argument("--ori", required=True, type=int, help="Orientation index (0-359)")
    parser.add_argument("--env", default="guidelines", help="Environment name")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--grid", default="128x256", help="Base grid as HxW")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output", default="results/new", help="Output directory")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--catalog", default="animals.json")
    args = parser.parse_args()

    base_grid = parse_grid(args.grid)
    grid_h, grid_w = base_grid[0] * args.scale, base_grid[1] * args.scale
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load creature
    animals = load_animals(Path(args.catalog), codes=[args.code])
    if not animals:
        raise SystemExit(f"Creature '{args.code}' not found in {args.catalog}")
    animal = animals[0]

    # Load orientation from all_orientations.pt
    ao_path = Path(f"initializations/{args.code}/s{args.scale}/{args.code}_s{args.scale}_all_orientations.pt")
    if not ao_path.exists():
        raise SystemExit(f"Not found: {ao_path}")
    ao = torch.load(ao_path, weights_only=False)
    board_tensor = ao['tensor'][args.ori]  # (H_small, W_small)

    # Place on full grid
    sH, sW = board_tensor.shape
    if sH < grid_h or sW < grid_w:
        canvas = torch.zeros(grid_h, grid_w, dtype=board_tensor.dtype)
        r0 = (grid_h - sH) // 2
        c0 = (grid_w - sW) // 2
        canvas[r0:r0 + sH, c0:c0 + sW] = board_tensor
        board_tensor = canvas

    # Extract pattern bounding box for spawn positioning
    pattern = extract_pattern(board_tensor)
    ph, pw = pattern.shape

    # Set up simulation
    cfg = Config.from_animal(animal, base_grid=base_grid, scale=args.scale)
    sim = Simulation(cfg)

    mask = load_env(args.env, cfg.device, cfg.dtype)
    sim.set_barrier(mask)

    pos = get_spawn_position(args.env, grid_h, grid_w, ph, pw)
    pos = (max(0, min(pos[0], grid_h - ph)),
           max(0, min(pos[1], grid_w - pw)))
    sim.add_animal(pattern.numpy(), position=pos, wrap=False)

    # Run and collect frames
    print(f"Running {args.code} ori={args.ori} in {args.env} for {args.steps} steps...")
    rollout = sim.run(args.steps)
    all_frames = rollout.frames

    frames_tensor = torch.stack(all_frames)  # (T, H, W)

    # Save
    prefix = f"{args.code}_{args.env}_o{args.ori}"
    gif_path = output_dir / f"{prefix}.gif"
    pt_path = output_dir / f"{prefix}_frames.pt"

    write_gif(all_frames, gif_path, fps=args.fps, upscale=1, barrier_mask=mask)
    torch.save(frames_tensor, pt_path)

    print(f"GIF:    {gif_path}")
    print(f"Frames: {pt_path} ({frames_tensor.shape})")


if __name__ == "__main__":
    main()
