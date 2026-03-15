"""Quick timing benchmark: O2v on 512x1024 grid with barrier."""
import time, sys, torch
sys.path.insert(0, '.')
from substrate import Config, Simulation, load_animals
from substrate.lenia import _auto_device
from environments import load_env

device = _auto_device()
animal = load_animals('animals.json', codes=['O2v'])[0]
T = animal.params.get('T', 10)
steps = min(200 * T, 100_000)
print(f"O2v: T={T}, steps={steps}")

cfg = Config.from_animal(animal, base_grid=(128, 256), scale=4)
shape = cfg.grid_shape
print(f"Grid: {shape}")

sim = Simulation(cfg)
mask = load_env('box', cfg.device, cfg.dtype)
sim.set_barrier(mask)

init = torch.load('initializations/O2v/s4/O2v_s4_all_orientations.pt', weights_only=False)
tensor = init['tensor'][0]
active = tensor > 0.01
rows, cols = torch.where(active)
r0 = max(0, rows.min().item() - 2)
r1 = min(tensor.shape[0], rows.max().item() + 3)
c0 = max(0, cols.min().item() - 2)
c1 = min(tensor.shape[1], cols.max().item() + 3)
pattern = tensor[r0:r1, c0:c1]
ph, pw = pattern.shape
pos = (shape[0] // 2 - ph // 2, shape[1] // 2 - pw // 2)
sim.add_animal(pattern.numpy(), position=pos, wrap=False)

current = sim.board.tensor.detach().clone()
automaton = sim.lenia.automaton

# warmup
for _ in range(10):
    current = automaton.step_batched(current.unsqueeze(0), blind_masks=mask).squeeze(0)
if torch.cuda.is_available():
    torch.cuda.synchronize()

t0 = time.time()
with torch.no_grad():
    for _ in range(steps):
        current = automaton.step_batched(current.unsqueeze(0), blind_masks=mask).squeeze(0)
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = time.time() - t0

ms_per_step = elapsed / steps * 1000
print(f"{steps} steps in {elapsed:.1f}s ({ms_per_step:.3f} ms/step)")
print(f"Projected 360 ori x 12 env = {360 * 12 * elapsed / 3600:.1f} hours")
