# experiments/preview_creatures.py — Jesse Cool (jessescool)
"""Render preview GIFs for creatures in the manifest."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from substrate import load_animals, Config, Simulation
from viz.gif import write_gif

STEPS = 300
OUT_DIR = Path("results/new/previews")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load manifest
with open("animals_to_run.json") as f:
    manifest = json.load(f)

entries = manifest["animals"]
print(f"Rendering {len(entries)} creatures, {STEPS} frames each, to {OUT_DIR}/")

failed = []

for i, entry in enumerate(entries):
    code = entry["code"]
    grid = 96
    try:
        creatures = load_animals(Path("animals.json"), codes=[code])
        if not creatures:
            failed.append((code, "NOT FOUND"))
            continue

        creature = creatures[0]
        name = creature.name or "?"
        cfg = Config.from_animal(creature, grid_shape=grid)
        sim = Simulation(cfg)
        sim.place_animal(creature, center=True)

        frames = sim.run(STEPS)
        out = OUT_DIR / f"{code}.gif"
        write_gif(frames, out, fps=30)
        print(f"  [{i+1}/{len(entries)}] {code:12s} ({name}) grid={grid}")

    except Exception as e:
        failed.append((code, str(e)))
        print(f"  [{i+1}/{len(entries)}] {code:12s} ERROR: {e}")

print(f"\nDone. {len(entries) - len(failed)}/{len(entries)} succeeded.")
if failed:
    print(f"Failed ({len(failed)}):")
    for c, err in failed:
        print(f"  {c}: {err}")
print(f"\nAll previews at: {OUT_DIR}/")
