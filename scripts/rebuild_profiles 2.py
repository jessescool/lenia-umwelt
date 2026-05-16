"""Rebuild stages 2-4 (profile → distances → orbit) from existing raw frames.

Skips the expensive simulation stage — just recomputes profiles with the
updated topk-based profile() and median(nnz) for m.

Usage:
    python scripts/rebuild_profiles.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from orbits.orbits import build_profiles, build_distances, build_orbit_summary

ORBITS_DIR = Path("orbits")


def rebuild_from_raw(raw_path: Path):
    """Run profile → distances → orbit from a single raw .pt file."""
    out_dir = raw_path.parent
    stem = raw_path.stem.replace("_raw", "")

    print(f"  Loading {raw_path.name}...")
    raw_data = torch.load(raw_path, weights_only=False)

    # Stage 2: profiles (quantile sampling, m = median nnz)
    t0 = time.time()
    print(f"  [2/4] Profiles...")
    prof = build_profiles(raw_data, verbose=True)
    torch.save(prof, out_dir / f"{stem}_profile.pt")
    print(f"  Saved {stem}_profile.pt [{time.time() - t0:.1f}s]")

    # Stage 3: pairwise distances
    t0 = time.time()
    print(f"  [3/4] Distances...")
    dist = build_distances(prof, verbose=True)
    torch.save(dist, out_dir / f"{stem}_distances.pt")
    print(f"  Saved {stem}_distances.pt [{time.time() - t0:.1f}s]")

    # Stage 4: orbit summary
    print(f"  [4/4] Orbit summary...")
    orb = build_orbit_summary(prof)
    orbit_path = out_dir / f"{stem}_orbit.pt"
    torch.save(orb, orbit_path)

    json_path = orbit_path.with_suffix(".json")
    json_data = {k: v for k, v in orb.items() if not isinstance(v, torch.Tensor)}
    json_path.write_text(json.dumps(json_data, indent=2) + "\n")

    print(f"  m={orb['m']}, ĉ={orb['c_hat']:.6f}, d_max={orb['d_max']:.6f}")
    print(f"  Saved {json_path}")


def main():
    raw_files = sorted(ORBITS_DIR.glob("*/s*/*_raw.pt"))
    print(f"Found {len(raw_files)} raw orbit files to rebuild")
    print()

    t0 = time.time()
    for i, raw_path in enumerate(raw_files):
        code_scale = raw_path.stem.replace("_raw", "")
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(raw_files)}] {code_scale}")
        print(f"{'='*60}")
        rebuild_from_raw(raw_path)

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done: rebuilt {len(raw_files)} orbits (stages 2-4) in {total/60:.1f} min")
    print(f"Output: {ORBITS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
