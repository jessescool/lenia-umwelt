"""
Orbit pipeline for Lenia creatures.

Consolidated pipeline: raw frames → sorted activation profiles → W1 distances.

Subcommands:
    python orbits/orbits.py raw -c O2u -s 2
    python orbits/orbits.py profile orbits/O2u/s2/O2u_s2_raw.pt
    python orbits/orbits.py distances orbits/O2u/s2/O2u_s2_profile.pt
    python orbits/orbits.py orbit orbits/O2u/s2/O2u_s2_profile.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics_and_machinery.distance_metrics import prepare_profile, wasserstein
from substrate import Animal, Config, Simulation, load_animals
from substrate.lenia import _auto_device
from substrate.scaling import recenter_field
from utils.core import rotate_tensor


DEFAULT_ROTATIONS = 90
DEFAULT_FRAMES = 64
DEFAULT_WARMUP_MULT = 30.0   # settle for 30*T before recording
DEFAULT_GRID = 128


# Thin wrappers around the portable distance primitives in
# metrics_and_machinery.distance_metrics (prepare_profile, wasserstein).

def sorted_profile(frame: Tensor, m: int) -> Tensor:
    """Convert a single (H, W) frame to a fixed-length sorted activation profile.

    Delegates to prepare_profile (threshold=0.0 → exact-zero trim).
    """
    return prepare_profile(frame, m, threshold=0.0)


def sorted_profiles_batched(frames: Tensor, m: int) -> Tensor:
    """Batch version: (B, H, W) → (B, m).  Delegates to prepare_profile."""
    return prepare_profile(frames, m, threshold=0.0)


def pairwise_w1(profiles: Tensor, **kwargs) -> Tensor:
    """All-by-all W1 distance matrix between sorted activation profiles.

    Auto-chunks to ~20 GB peak (fits comfortably on 40+ GB cards).
    """
    N, m = profiles.shape
    idx_i, idx_j = torch.triu_indices(N, N, offset=1)
    n_pairs = idx_i.shape[0]
    # ~3 tensors of (chunk, m) float32 per iteration; target 20 GB
    chunk_size = max(1024, int(20e9 / (3 * m * 4)))
    dists = torch.empty(n_pairs, device=profiles.device, dtype=profiles.dtype)
    for start in range(0, n_pairs, chunk_size):
        end = min(start + chunk_size, n_pairs)
        dists[start:end] = wasserstein(
            profiles[idx_i[start:end]], profiles[idx_j[start:end]], m
        )
    mat = torch.zeros(N, N, device=profiles.device, dtype=profiles.dtype)
    mat[idx_i, idx_j] = dists
    mat[idx_j, idx_i] = dists
    return mat


def build_orbit(
    creature: Animal,
    scale: int,
    grid_size: int = DEFAULT_GRID,
    num_rotations: int = DEFAULT_ROTATIONS,
    num_frames: int = DEFAULT_FRAMES,
    warmup_mult: float = DEFAULT_WARMUP_MULT,
    save_warmup: bool = False,
    verbose: bool = True,
    creature_base_grid: int | None = None,
) -> dict:
    """
    Collect raw orbit frames for a creature across rotation angles.

    For each angle: rotate cells -> scale + settle -> record frames.

    Returns dict with 'frames' tensor (num_rotations, num_frames, H, W)
    and metadata.
    """
    device = _auto_device()
    T = creature.params.get("T", 10)
    warmup_steps = int(warmup_mult * T)

    # Cover one quarter-turn (0°–90°) at uniform spacing; 90° = identity on a square grid
    angles = [i * (90.0 / num_rotations) for i in range(num_rotations)]

    cfg = Config.from_animal(creature, grid_size, scale=scale)
    H, W = cfg.grid_shape

    all_frames = torch.empty(num_rotations, num_frames, H, W, device="cpu")
    all_warmup = (
        torch.empty(num_rotations, warmup_steps, H, W, device="cpu")
        if save_warmup else None
    )

    for rot_idx, angle in enumerate(angles):
        t0 = time.time()

        creature_rotated = copy.deepcopy(creature)
        cells = torch.as_tensor(creature.cells, device=device, dtype=torch.float32)
        cells_rotated = rotate_tensor(cells, angle, device)
        creature_rotated.cells = cells_rotated.cpu().numpy()

        sim = Simulation(cfg)
        sim.place_animal(creature_rotated, center=True)

        for step in range(warmup_steps):
            if save_warmup:
                all_warmup[rot_idx, step] = sim.board.tensor.detach().cpu()
            sim.lenia.step()

        sim.board.replace_tensor(recenter_field(sim.board.tensor))

        for f in range(num_frames):
            all_frames[rot_idx, f] = sim.board.tensor.detach().cpu()
            sim.lenia.step()

        elapsed = time.time() - t0
        if verbose:
            mass = all_frames[rot_idx].sum(dim=(1, 2))
            print(
                f"  angle {angle:5.1f}° ({rot_idx+1}/{num_rotations}): "
                f"mass range [{mass.min():.1f}, {mass.max():.1f}], "
                f"{elapsed:.1f}s"
            )

    result = {
        "frames": all_frames,                # (num_rotations, num_frames, H, W)
        "angles": angles,
        "code": creature.code,
        "scale": scale,
        "grid_size": H,
        "warmup_steps": warmup_steps,
        "T": T,
        "num_rotations": num_rotations,
        "num_frames": num_frames,
    }
    if creature_base_grid is not None:
        result["creature_base_grid"] = creature_base_grid
    if save_warmup:
        result["warmup_frames"] = all_warmup  # (num_rotations, warmup_steps, H, W)
    return result


def build_profiles(data: dict, verbose: bool = True) -> dict:
    """Transform raw orbit data into sorted activation profiles."""
    frames = data["frames"]              # (R, F, H, W)

    R, F, H, W = frames.shape
    # m = round(μ + 2σ) of nonzero pixel count — captures tail without bloat
    nnz = (frames > 0).sum(dim=(2, 3)).flatten().float()
    m = round((nnz.mean() + 2 * nnz.std()).item())

    if verbose:
        print(f"Input: {R} rotations, {F} trial frames")
        print(f"Grid: {H}x{W}, m={m} (μ+2σ nnz)")
        print(f"Nonzero pixels: median={int(nnz.median())}, range=[{int(nnz.min())}, {int(nnz.max())}]")

    # Process trial frames: reshape (R, F, H, W) → (R*F, H, W), then back
    t0 = time.time()
    trial_flat = frames.reshape(R * F, H, W)
    trial_profiles = sorted_profiles_batched(trial_flat, m).reshape(R, F, m).contiguous()
    if verbose:
        print(f"Trial profiles: {trial_profiles.shape} [{time.time() - t0:.1f}s]")

    result = {
        "trial_profiles": trial_profiles,    # (R, F, m)
        "m": m,
        "grid_h": H,
        "grid_w": W,
    }
    for key in ("angles", "code", "scale", "grid_size", "warmup_steps",
                "T", "num_rotations", "num_frames", "creature_base_grid"):
        if key in data:
            result[key] = data[key]

    return result


def build_distances(data: dict, verbose: bool = True) -> dict:
    """Compute pairwise W1 distance matrix from orbit profiles."""
    profiles = data["trial_profiles"]  # (R, F, m)
    R, F, m = profiles.shape

    if verbose:
        print(f"Input: {R} rotations, {F} frames, profile length m={m}")
        print(f"Computing {R*F}x{R*F} pairwise W1 distances...")

    # Flatten rotations and frames into a single axis
    flat = profiles.reshape(R * F, m)

    t0 = time.time()
    dist_matrix = pairwise_w1(flat)
    if verbose:
        print(f"Distance matrix: {dist_matrix.shape} [{time.time() - t0:.1f}s]")

    result = {
        "distance_matrix": dist_matrix,  # (R*F, R*F)
        "m": m,
        "grid_h": data.get("grid_h"),
        "grid_w": data.get("grid_w"),
    }
    for key in ("angles", "code", "scale", "grid_size", "warmup_steps",
                "T", "num_rotations", "num_frames", "creature_base_grid"):
        if key in data:
            result[key] = data[key]

    return result


def compute_c_bar(profiles: Tensor) -> Tensor:
    """Fréchet median under W1 (componentwise median of activation profiles)."""
    flat = profiles.reshape(-1, profiles.shape[-1])  # (N, m)
    return flat.median(dim=0).values


def compute_c_hat(profiles: Tensor, c_bar: Tensor) -> tuple[float, float]:
    """Mean and std of d(c_i, c̄) — orbit radius in profile space."""
    flat = profiles.reshape(-1, profiles.shape[-1])  # (N, m)
    dists = (flat - c_bar).abs().mean(dim=1)          # L1 distance to c̄
    return dists.mean().item(), dists.std().item()


def build_orbit_summary(profile_data: dict) -> dict:
    """Bundle orbit barycenter and recovery threshold from profile data."""
    profiles = profile_data["trial_profiles"]  # (R, F, m)

    c_bar = compute_c_bar(profiles)
    c_hat, sigma = compute_c_hat(profiles, c_bar)

    result = {
        "c_bar": c_bar,
        "c_hat": c_hat,
        "sigma": sigma,
        "m": profile_data["m"],
    }
    # Carry forward metadata from profile data
    for key in ("angles", "code", "scale", "grid_size", "warmup_steps",
                "T", "num_rotations", "num_frames", "grid_h", "grid_w",
                "creature_base_grid"):
        if key in profile_data:
            result[key] = profile_data[key]

    return result


def _cmd_raw(args):
    """Subcommand: build raw orbit frames."""
    catalog_path = Path(__file__).parent.parent / args.catalog
    creatures = load_animals(catalog_path, codes=[args.code])
    if not creatures:
        print(f"Error: creature '{args.code}' not found")
        sys.exit(1)
    creature = creatures[0]

    verbose = not args.quiet
    if verbose:
        T = creature.params.get("T", 10)
        print(f"Building orbit: {args.code} scale={args.scale} grid={args.grid}")
        print(f"  {args.rotations} rotations × {args.frames} frames")
        print(f"  T={T}, warmup={int(args.warmup_multiplier * T)} steps")

    result = build_orbit(
        creature,
        scale=args.scale,
        grid_size=args.grid,
        num_rotations=args.rotations,
        num_frames=args.frames,
        warmup_mult=args.warmup_multiplier,
        save_warmup=args.save_warmup,
        verbose=verbose,
        creature_base_grid=args.creature_base_grid,
    )

    out_dir = Path(args.output_dir) if args.output_dir else Path(f"orbits/{args.code}/s{args.scale}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.code}_s{args.scale}_raw.pt"
    torch.save(result, out_path)

    frames = result["frames"]
    size_mb = frames.element_size() * frames.numel() / 1e6
    if verbose:
        print(f"\nSaved {out_path} ({frames.shape}, {size_mb:.1f} MB)")


def _cmd_profile(args):
    """Subcommand: build sorted activation profiles from raw frames."""
    verbose = not args.quiet
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if verbose:
        print(f"Loading {input_path}...")
    data = torch.load(input_path, weights_only=False)

    result = build_profiles(data, verbose=verbose)

    if args.output:
        out_path = Path(args.output)
    else:
        stem = input_path.stem
        if stem.endswith("_raw"):
            out_stem = stem[:-4] + "_profile"
        else:
            out_stem = stem + "_profile"
        out_path = input_path.parent / f"{out_stem}.pt"

    torch.save(result, out_path)
    if verbose:
        tp = result["trial_profiles"]
        size_mb = tp.element_size() * tp.numel() / 1e6
        print(f"\nSaved {out_path} ({size_mb:.1f} MB)")


def _cmd_distances(args):
    """Subcommand: build pairwise W1 distance matrix from profiles."""
    verbose = not args.quiet
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if verbose:
        print(f"Loading {input_path}...")
    data = torch.load(input_path, weights_only=False)

    result = build_distances(data, verbose=verbose)

    if args.output:
        out_path = Path(args.output)
    else:
        stem = input_path.stem
        if stem.endswith("_profile"):
            out_stem = stem[:-8] + "_distances"
        else:
            out_stem = stem + "_distances"
        out_path = input_path.parent / f"{out_stem}.pt"

    torch.save(result, out_path)
    if verbose:
        dm = result["distance_matrix"]
        size_mb = dm.element_size() * dm.numel() / 1e6
        print(f"\nSaved {out_path} ({size_mb:.1f} MB)")


def _cmd_orbit(args):
    """Subcommand: compute orbit summary (c̄ barycenter + ĉ threshold)."""
    verbose = not args.quiet
    profile_path = Path(args.profile_input)

    if not profile_path.exists():
        print(f"Error: {profile_path} not found")
        sys.exit(1)

    if verbose:
        print(f"Loading profiles: {profile_path}")
    profile_data = torch.load(profile_path, weights_only=False)

    result = build_orbit_summary(profile_data)

    if verbose:
        print(f"  c̄ shape: {result['c_bar'].shape}")
        print(f"  ĉ (mean d to c̄): {result['c_hat']:.6f}")
        print(f"  σ (std d to c̄):  {result['sigma']:.6f}")
        print(f"  ĉ+3σ (recovery threshold): {result['c_hat'] + 3 * result['sigma']:.6f}")

    if args.output:
        out_path = Path(args.output)
    else:
        stem = profile_path.stem
        if stem.endswith("_profile"):
            out_stem = stem[:-8] + "_orbit"
        else:
            out_stem = stem + "_orbit"
        out_path = profile_path.parent / f"{out_stem}.pt"

    torch.save(result, out_path)

    # Human-readable JSON sidecar (everything except the c_bar tensor)
    json_path = out_path.with_suffix(".json")
    json_data = {k: v for k, v in result.items() if not isinstance(v, Tensor)}
    json_path.write_text(json.dumps(json_data, indent=2) + "\n")

    if verbose:
        print(f"\nSaved {out_path}")
        print(f"Saved {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Orbit pipeline for Lenia creatures"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_raw = sub.add_parser("raw", help="Build raw orbit frames")
    p_raw.add_argument("--code", "-c", type=str, required=True,
                        help="Creature code (e.g., O2u)")
    p_raw.add_argument("--scale", "-s", type=int, default=1,
                        help="Scale factor (default: 1)")
    p_raw.add_argument("--grid", "-g", type=int, default=DEFAULT_GRID,
                        help=f"Base grid size (default: {DEFAULT_GRID})")
    p_raw.add_argument("--rotations", "-r", type=int, default=DEFAULT_ROTATIONS,
                        help=f"Number of rotation angles (default: {DEFAULT_ROTATIONS})")
    p_raw.add_argument("--frames", "-f", type=int, default=DEFAULT_FRAMES,
                        help=f"Frames to record per rotation (default: {DEFAULT_FRAMES})")
    p_raw.add_argument("--warmup-multiplier", "-w", type=float, default=DEFAULT_WARMUP_MULT,
                        help=f"Warmup = T * this (default: {DEFAULT_WARMUP_MULT})")
    p_raw.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory (default: orbits/<code>/s<scale>)")
    p_raw.add_argument("--catalog", type=str, default="animals.json",
                        help="Creature catalog path")
    p_raw.add_argument("--creature-base-grid", type=int, default=None,
                        help="Creature bounding-box grid size (from animals_to_run.json)")
    p_raw.add_argument("--save-warmup", action="store_true",
                        help="Store warmup frames in output (off by default)")
    p_raw.add_argument("--quiet", "-q", action="store_true")
    p_raw.set_defaults(func=_cmd_raw)

    p_prof = sub.add_parser("profile", help="Build sorted activation profiles from raw frames")
    p_prof.add_argument("input", type=str,
                        help="Path to raw orbit .pt file")
    p_prof.add_argument("--output", "-o", type=str, default=None,
                        help="Output filename (default: replaces _raw with _profile)")
    p_prof.add_argument("--quiet", "-q", action="store_true")
    p_prof.set_defaults(func=_cmd_profile)

    p_dist = sub.add_parser("distances", help="Build pairwise W1 distance matrix from profiles")
    p_dist.add_argument("input", type=str,
                        help="Path to profiles .pt file")
    p_dist.add_argument("--output", "-o", type=str, default=None,
                        help="Output filename (default: <input_stem>_distances.pt)")
    p_dist.add_argument("--quiet", "-q", action="store_true")
    p_dist.set_defaults(func=_cmd_distances)

    p_orbit = sub.add_parser("orbit", help="Compute orbit summary (barycenter + recovery threshold)")
    p_orbit.add_argument("profile_input", type=str,
                         help="Path to profiles .pt file")
    p_orbit.add_argument("--output", "-o", type=str, default=None,
                         help="Output filename (default: <profile_stem>_orbit.pt)")
    p_orbit.add_argument("--quiet", "-q", action="store_true")
    p_orbit.set_defaults(func=_cmd_orbit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
