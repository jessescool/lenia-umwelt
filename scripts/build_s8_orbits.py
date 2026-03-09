"""Build scale-8 orbits for all creatures."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from substrate import load_animals
from orbits.orbits import (
    build_orbit, build_profiles, build_distances, build_orbit_summary,
    DEFAULT_GRID, DEFAULT_ROTATIONS, DEFAULT_FRAMES, DEFAULT_WARMUP_MULT,
)

CREATURES = ["3R4s", "H3s", "K4s", "K6s", "O2u", "O2v", "P4al", "S1s"]
SCALE = 8


def main():
    catalog = load_animals(Path("animals.json"))
    by_code = {c.code: c for c in catalog}

    manifest = json.loads(Path("animals_to_run.json").read_text())
    bg_lookup = {a["code"]: a["crop"] for a in manifest["animals"]}

    print(f"Building scale-{SCALE} orbits for {len(CREATURES)} creatures")
    t0 = time.time()

    for i, code in enumerate(CREATURES):
        creature = by_code.get(code)
        if creature is None:
            print(f"WARNING: {code} not found, skipping")
            continue
        creature_bg = bg_lookup.get(code)
        if creature_bg is None:
            print(f"WARNING: {code} not in animals_to_run.json, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(CREATURES)}] {creature.name} ({code}) scale={SCALE}")
        print(f"  m = ({creature_bg} * {SCALE})^2 = {(creature_bg * SCALE)**2}")
        print(f"{'='*60}")

        out_dir = Path(f"orbits/{code}/s{SCALE}")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{code}_s{SCALE}"

        print(f"  [1/4] Raw frames...")
        raw = build_orbit(creature, scale=SCALE, grid_size=DEFAULT_GRID,
                          num_rotations=DEFAULT_ROTATIONS, num_frames=DEFAULT_FRAMES,
                          warmup_mult=DEFAULT_WARMUP_MULT, verbose=True,
                          creature_base_grid=creature_bg)
        torch.save(raw, out_dir / f"{stem}_raw.pt")

        print(f"  [2/4] Profiles...")
        prof = build_profiles(raw, verbose=True)
        torch.save(prof, out_dir / f"{stem}_profile.pt")

        print(f"  [3/4] Distances...")
        dist = build_distances(prof, verbose=True)
        torch.save(dist, out_dir / f"{stem}_distances.pt")

        print(f"  [4/4] Orbit summary...")
        orb = build_orbit_summary(prof)
        torch.save(orb, out_dir / f"{stem}_orbit.pt")
        json_path = out_dir / f"{stem}_orbit.json"
        json_data = {k: v for k, v in orb.items() if not isinstance(v, torch.Tensor)}
        json_path.write_text(json.dumps(json_data, indent=2) + "\n")

        print(f"  m={orb['m']}, ĉ={orb['c_hat']:.6f}, d_max={orb['d_max']:.6f}")

    print(f"\nDone: {len(CREATURES)} orbits in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
