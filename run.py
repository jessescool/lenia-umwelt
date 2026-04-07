# run.py — Jesse Cool (jessescool)
"""Quick Lenia preview. Usage: python run.py <animal_code>"""
import sys
from pathlib import Path
from substrate import load_animals, Config, Simulation
from viz.gif import write_gif

code = sys.argv[1] if len(sys.argv) > 1 else "O2u"
creatures = load_animals(Path("animals.json"), codes=[code])

if not creatures:
    # List some valid codes for reference
    all_animals = load_animals(Path("animals.json"))
    valid_codes = [a.code for a in all_animals if not a.code.startswith(">") and not a.code.startswith("~")][:20]
    print(f"Error: '{code}' not found. Some valid codes: {', '.join(valid_codes)}")
    sys.exit(1)

creature = creatures[0]
cfg = Config.from_animal(creature, grid_shape=128)
sim = Simulation(cfg)
sim.place_animal(creature, center=True)

frames = sim.run(500)
out = Path(f"results/{code}_preview.gif")
write_gif(frames, out, fps=30)
print(f"Saved {out}")
