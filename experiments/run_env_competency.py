# experiments/run_env_competency.py — Jesse Cool (jessescool)
"""Score one creature's competency across barrier environments."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.fft import rfft2, irfft2

sys.path.insert(0, str(Path(__file__).parent.parent))

from substrate import Simulation, Config, load_animals
from substrate.lenia import _auto_device
from environments import ENVIRONMENTS, load_env
from metrics_and_machinery.distance_metrics import prepare_profile
from metrics_and_machinery.competency import orbit_residence_fraction, aggregate_competency
from viz.gif import write_gif


# ── constants ──────────────────────────────────────────────────────────

COMPETENCY_LAMBDA_MULT = 3.0   # competency threshold = mult * lambda * d_max
N_PERIODS = 200                # T-periods to simulate
DEATH_THRESHOLD = 0.01         # fraction of initial mass
EXPLOSION_THRESHOLD = 3.0      # multiple of initial mass


# ── self-contained helpers ─────────────────────────────────────────────

def parse_grid(s: str) -> tuple[int, int]:
    """Parse 'HxW' string."""
    parts = s.lower().split("x")
    return (int(parts[0]), int(parts[1]))


def extract_pattern(tensor: torch.Tensor, threshold: float = 0.01, padding: int = 2) -> torch.Tensor:
    """Crop settled board to bounding box of non-zero region."""
    active = tensor > threshold
    rows, cols = torch.where(active)
    if len(rows) == 0:
        return tensor
    r0 = max(0, rows.min().item() - padding)
    r1 = min(tensor.shape[0], rows.max().item() + padding + 1)
    c0 = max(0, cols.min().item() - padding)
    c1 = min(tensor.shape[1], cols.max().item() + padding + 1)
    return tensor[r0:r1, c0:c1]


def load_env_list(path: str = "environments.json") -> list[str]:
    """Read environment names from environments.json."""
    with open(path) as f:
        data = json.load(f)
    return [e["name"] for e in data["environments"]]


# ── spawn positions (hyphenated keys match canonical env names) ────────

def get_spawn_position(grid_h, grid_w, ph, pw):
    """Always center the creature in the environment."""
    return (grid_h // 2 - ph // 2, grid_w // 2 - pw // 2)


# ── orbit / initialization loaders ─────────────────────────────────────

def load_orbit_data(code: str, scale: int) -> dict:
    """Load orbit summary (c_bar, d_max, m)."""
    orbit_path = Path(f"orbits/{code}/s{scale}/{code}_s{scale}_orbit.pt")
    if not orbit_path.exists():
        raise FileNotFoundError(f"Orbit file not found: {orbit_path}")
    return torch.load(orbit_path, weights_only=False)


def load_initializations(code: str, scale: int, num_orientations: int | None = None) -> list[dict]:
    """Load pre-settled initializations from the dense all_orientations.pt file.

    When *num_orientations* is set, subsample evenly-spaced orientations.
    """
    init_dir = Path(f"initializations/{code}/s{scale}")
    dense_path = init_dir / f"{code}_s{scale}_all_orientations.pt"

    if not dense_path.exists():
        raise FileNotFoundError(f"Dense orientations file not found: {dense_path}")

    data = torch.load(dense_path, weights_only=False)
    tensor = data['tensor']            # (N, H, W)
    headings = data['headings']        # list of heading angles
    N = tensor.shape[0]

    # subsample if requested
    if num_orientations is not None and num_orientations < N:
        stride = N / num_orientations
        indices = [int(round(i * stride)) for i in range(num_orientations)]
    else:
        indices = list(range(N))

    inits = []
    for idx in indices:
        inits.append({
            'tensor': tensor[idx],     # (H, W)
            'code': code,
            'scale': scale,
            'angle': headings[idx],
            'sit_idx': idx,
        })
    return inits


# ── single run (kept for backward compat) ─────────────────────────────

def run_one(
    code: str,
    animal,
    env_name: str,
    base_grid: tuple[int, int],
    scale: int,
    init_dict: dict,
    orbit_data: dict,
    lam: float,
    device: torch.device,
    gif_path: Path | None = None,
) -> dict:
    """Run one creature × environment × orientation. Returns scalar metrics."""
    grid_h, grid_w = base_grid[0] * scale, base_grid[1] * scale
    shape = (grid_h, grid_w)

    cfg = Config.from_animal(animal, base_grid=base_grid, scale=scale)

    T = cfg.timescale_T
    steps = min(int(N_PERIODS * T), 100_000)

    # ~1000 metric frames
    metric_stride = max(1, steps // 1000)
    n_metric_frames = (steps + metric_stride - 1) // metric_stride

    # orbit parameters
    c_bar = orbit_data['c_bar'].to(device)
    m = orbit_data['m']
    d_max = orbit_data['d_max']
    competency_threshold = COMPETENCY_LAMBDA_MULT * lam * d_max

    pattern = extract_pattern(init_dict['tensor'])
    ph, pw = pattern.shape

    # create simulation
    sim = Simulation(cfg)
    if env_name == "open_field":
        mask = None
    else:
        mask = load_env(env_name, cfg.device, cfg.dtype)
        sim.set_barrier(mask)

    # spawn creature
    pos = get_spawn_position(grid_h, grid_w, ph, pw)
    pos = (max(0, min(pos[0], grid_h - ph)),
           max(0, min(pos[1], grid_w - pw)))
    sim.add_animal(pattern.numpy(), position=pos, wrap=False)

    # pre-allocate metric tensors
    distances = torch.empty(n_metric_frames, device=device, dtype=torch.float32)
    mass_ts = torch.empty(n_metric_frames, device=device, dtype=torch.float32)
    centroids = torch.empty(n_metric_frames, 2, device=device, dtype=torch.float32)
    occ_weighted_ts = torch.full((n_metric_frames,), float('nan'), device=device)
    occ_uniform_ts = torch.full((n_metric_frames,), float('nan'), device=device)
    occ_max_ts = torch.full((n_metric_frames,), float('nan'), device=device)

    current = sim.board.tensor.detach().clone()
    automaton = sim.lenia.automaton
    barrier = mask
    metric_idx = 0
    initial_mass = current.sum().item()

    # precompute trig arrays for centroid (avoids realloc every frame)
    _row_angles = torch.arange(grid_h, device=device, dtype=torch.float32) * (2 * np.pi / grid_h)
    _col_angles = torch.arange(grid_w, device=device, dtype=torch.float32) * (2 * np.pi / grid_w)
    _sin_r, _cos_r = torch.sin(_row_angles), torch.cos(_row_angles)
    _sin_c, _cos_c = torch.sin(_col_angles), torch.cos(_col_angles)
    _TWO_PI = 2 * np.pi

    # last step where creature was within 1*d_max of orbit
    last_return_step = 0

    # GIF frames: target ~500 max
    gif_stride = max(1, steps // 500) if gif_path else 0
    gif_frames = [] if gif_path else None
    termination = "completed"

    with torch.no_grad():
        for step in range(steps):
            if step % metric_stride == 0 and metric_idx < n_metric_frames:
                p = prepare_profile(current.unsqueeze(0), m)  # [1, m]
                dist = (p - c_bar).abs().mean()
                distances[metric_idx] = dist
                total = current.sum()
                mass = total.item()
                mass_ts[metric_idx] = mass
                # inline centroid using precomputed trig arrays
                if mass > 1e-6:
                    row_mass = current.sum(dim=1)  # [H]
                    col_mass = current.sum(dim=0)  # [W]
                    sr = (row_mass * _sin_r).sum() / total
                    cr = (row_mass * _cos_r).sum() / total
                    ang_r = torch.atan2(sr, cr)
                    if ang_r < 0: ang_r += _TWO_PI
                    sc = (col_mass * _sin_c).sum() / total
                    cc = (col_mass * _cos_c).sum() / total
                    ang_c = torch.atan2(sc, cc)
                    if ang_c < 0: ang_c += _TWO_PI
                    centroids[metric_idx, 0] = ang_r * grid_h / _TWO_PI
                    centroids[metric_idx, 1] = ang_c * grid_w / _TWO_PI
                else:
                    centroids[metric_idx, 0] = grid_h / 2.0
                    centroids[metric_idx, 1] = grid_w / 2.0

                # perceptive occlusion: how much kernel-weighted sensory field is blocked
                if barrier is not None and mass > 1e-6:
                    kfft = automaton._rebuild_kernel_fft(grid_h, grid_w)
                    occ_map = 1.0 - irfft2(rfft2(1.0 - barrier) * kfft, s=(grid_h, grid_w))
                    occ_map = occ_map.clamp(0.0, 1.0)
                    occ_weighted_ts[metric_idx] = (occ_map * current).sum() / max(mass, 1e-6)
                    creature_pixels = current > 0.01
                    if creature_pixels.any():
                        occ_uniform_ts[metric_idx] = occ_map[creature_pixels].mean()
                        occ_max_ts[metric_idx] = occ_map[creature_pixels].max()
                    else:
                        occ_uniform_ts[metric_idx] = 0.0
                        occ_max_ts[metric_idx] = 0.0
                else:
                    occ_weighted_ts[metric_idx] = 0.0
                    occ_uniform_ts[metric_idx] = 0.0
                    occ_max_ts[metric_idx] = 0.0

                metric_idx += 1

                # track last return to orbit (within 1 * d_max)
                if dist.item() <= d_max:
                    last_return_step = step

                # early termination on death or explosion
                if mass < DEATH_THRESHOLD * initial_mass:
                    termination = "death"
                    break
                if mass > EXPLOSION_THRESHOLD * initial_mass:
                    termination = "explosion"
                    break

            if gif_frames is not None and step % gif_stride == 0:
                gif_frames.append(current.detach().clone())

            current = automaton.step_batched(
                current.unsqueeze(0),
                blind_masks=barrier,
            ).squeeze(0)

    # trim to actual collected frames
    distances = distances[:metric_idx].unsqueeze(0)  # [1, T_m]
    mass_ts = mass_ts[:metric_idx].unsqueeze(0)
    initial_mass_t = mass_ts[:, 0]

    result = orbit_residence_fraction(
        distances, mass_ts,
        competency_threshold=competency_threshold,
        death_threshold=DEATH_THRESHOLD,
        explosion_threshold=EXPLOSION_THRESHOLD,
        initial_mass=initial_mass_t,
        d_max=d_max,
    )

    if gif_frames:
        write_gif(gif_frames, gif_path, fps=30, barrier_mask=barrier)

    return {
        'M': result['M'].item(),
        'V': result['V'].item(),
        'F': result['F'].item(),
        'D_peak': result['D_peak'].item(),
        'steps': step + 1,
        'last_return': last_return_step,
        'termination': termination,
        'metric_frames': metric_idx,
        'competency_threshold': competency_threshold,
        'centroids': centroids[:metric_idx].cpu().numpy(),  # [T_m, 2]
        'mass': mass_ts.squeeze(0).cpu().numpy(),  # [T_m]
        'distances': distances[:metric_idx].cpu().numpy(),  # [T_m] W1 from c_bar
        'occlusion_weighted': occ_weighted_ts[:metric_idx].cpu().numpy(),  # [T_m]
        'occlusion_uniform': occ_uniform_ts[:metric_idx].cpu().numpy(),  # [T_m]
        'occlusion_max': occ_max_ts[:metric_idx].cpu().numpy(),  # [T_m]
    }


# ── batched run: all orientations for one env ─────────────────────────

def run_batch(
    code: str,
    animal,
    env_name: str,
    base_grid: tuple[int, int],
    scale: int,
    init_dicts: list[dict],
    orbit_data: dict,
    lam: float,
    device: torch.device,
    gif_ori_indices: set,
    chunk_offset: int,
    out_dir: Path,
) -> list[dict]:
    """Run B orientations in parallel for one environment.

    Batches all orientations into a single [B, H, W] state tensor and
    steps them together, vectorizing metric computation.
    """
    B = len(init_dicts)
    grid_h, grid_w = base_grid[0] * scale, base_grid[1] * scale

    cfg = Config.from_animal(animal, base_grid=base_grid, scale=scale)
    T_period = cfg.timescale_T
    steps = min(int(N_PERIODS * T_period), 100_000)

    metric_stride = max(1, steps // 1000)
    n_metric_frames = (steps + metric_stride - 1) // metric_stride

    # orbit parameters
    c_bar = orbit_data['c_bar'].to(device)   # [m]
    m = orbit_data['m']
    d_max = orbit_data['d_max']
    competency_threshold = COMPETENCY_LAMBDA_MULT * lam * d_max

    # ── build [B, H, W] state tensor ────────────────────────────
    states = torch.zeros(B, grid_h, grid_w, device=device, dtype=cfg.dtype)
    for i, init_dict in enumerate(init_dicts):
        pattern = extract_pattern(init_dict['tensor']).to(device=device, dtype=cfg.dtype)
        ph, pw = pattern.shape
        pos = get_spawn_position(grid_h, grid_w, ph, pw)
        pos = (max(0, min(pos[0], grid_h - ph)),
               max(0, min(pos[1], grid_w - pw)))
        states[i, pos[0]:pos[0]+ph, pos[1]:pos[1]+pw] = pattern

    # automaton + barrier
    sim = Simulation(cfg)
    automaton = sim.lenia.automaton
    if env_name == "open_field":
        barrier = None
    else:
        barrier = load_env(env_name, cfg.device, cfg.dtype)

    # ── pre-allocate metric storage [B, T_m] ────────────────────
    distances_all = torch.empty(B, n_metric_frames, device=device)
    mass_all = torch.empty(B, n_metric_frames, device=device)
    centroids_all = torch.empty(B, n_metric_frames, 2, device=device)
    occ_w_all = torch.full((B, n_metric_frames), float('nan'), device=device)
    occ_u_all = torch.full((B, n_metric_frames), float('nan'), device=device)
    occ_m_all = torch.full((B, n_metric_frames), float('nan'), device=device)

    initial_mass = states.sum(dim=(1, 2))   # [B]
    alive = torch.ones(B, dtype=torch.bool, device=device)
    termination = ["completed"] * B
    term_step = [steps] * B
    last_return = torch.zeros(B, dtype=torch.long, device=device)
    metric_idx = 0

    # precompute trig arrays for vectorized centroid
    _TWO_PI = 2 * np.pi
    _sin_r = torch.sin(torch.arange(grid_h, device=device, dtype=torch.float32) * (_TWO_PI / grid_h))
    _cos_r = torch.cos(torch.arange(grid_h, device=device, dtype=torch.float32) * (_TWO_PI / grid_h))
    _sin_c = torch.sin(torch.arange(grid_w, device=device, dtype=torch.float32) * (_TWO_PI / grid_w))
    _cos_c = torch.cos(torch.arange(grid_w, device=device, dtype=torch.float32) * (_TWO_PI / grid_w))

    # precompute occlusion map once (depends only on barrier, not creature state)
    occ_map = None
    if barrier is not None:
        kfft = automaton._rebuild_kernel_fft(grid_h, grid_w)
        occ_map = (1.0 - irfft2(rfft2(1.0 - barrier) * kfft, s=(grid_h, grid_w))).clamp(0.0, 1.0)

    # GIF: map global ori indices → batch-local positions
    gif_local = {}   # local_idx → global_ori_idx
    for i in range(B):
        g = chunk_offset + i
        if g in gif_ori_indices:
            gif_local[i] = g
    gif_stride = max(1, steps // 500) if gif_local else 0
    gif_frames = {i: [] for i in gif_local}

    # sentinel for -inf masking in occ_max
    _NEG_INF = torch.tensor(-float('inf'), device=device)

    # ── simulation loop ─────────────────────────────────────────
    with torch.no_grad():
        for step in range(steps):
            if step % metric_stride == 0 and metric_idx < n_metric_frames:
                # distance to orbit: [B, m] → [B]
                profiles = prepare_profile(states, m)
                dists = (profiles - c_bar.unsqueeze(0)).abs().mean(dim=1)
                distances_all[:, metric_idx] = dists

                total = states.sum(dim=(1, 2))   # [B]
                mass_all[:, metric_idx] = total

                # vectorized centroid (circular mean on torus)
                safe_total = total.clamp(min=1e-6)
                row_m = states.sum(dim=2)                       # [B, H]
                col_m = states.sum(dim=1)                       # [B, W]
                sr = (row_m * _sin_r).sum(dim=1) / safe_total   # [B]
                cr = (row_m * _cos_r).sum(dim=1) / safe_total
                sc = (col_m * _sin_c).sum(dim=1) / safe_total
                cc = (col_m * _cos_c).sum(dim=1) / safe_total
                centroids_all[:, metric_idx, 0] = (torch.atan2(sr, cr) % _TWO_PI) * grid_h / _TWO_PI
                centroids_all[:, metric_idx, 1] = (torch.atan2(sc, cc) % _TWO_PI) * grid_w / _TWO_PI
                # fallback for dead/empty creatures
                dead_now = total < 1e-6
                centroids_all[dead_now, metric_idx, 0] = grid_h / 2.0
                centroids_all[dead_now, metric_idx, 1] = grid_w / 2.0

                # occlusion metrics (vectorized across B)
                if occ_map is not None:
                    # weighted occlusion
                    occ_w_all[:, metric_idx] = (occ_map * states).sum(dim=(1, 2)) / safe_total
                    # uniform occlusion: mean occ over active pixels per orientation
                    active = (states > 0.01).float()                         # [B, H, W]
                    act_cnt = active.sum(dim=(1, 2)).clamp(min=1)            # [B]
                    occ_u_all[:, metric_idx] = (occ_map * active).sum(dim=(1, 2)) / act_cnt
                    # max occlusion: mask inactive → -inf, then amax
                    occ_exp = occ_map.unsqueeze(0).expand_as(states)
                    masked = torch.where(active.bool(), occ_exp, _NEG_INF)
                    mx = masked.amax(dim=(1, 2))                             # [B]
                    has_act = active.sum(dim=(1, 2)) > 0
                    occ_m_all[:, metric_idx] = torch.where(has_act, mx, torch.zeros_like(mx))
                    # zero occlusion for dead creatures
                    occ_w_all[dead_now, metric_idx] = 0.0
                    occ_u_all[dead_now, metric_idx] = 0.0
                    occ_m_all[dead_now, metric_idx] = 0.0
                else:
                    occ_w_all[:, metric_idx] = 0.0
                    occ_u_all[:, metric_idx] = 0.0
                    occ_m_all[:, metric_idx] = 0.0

                metric_idx += 1

                # update last-return per orientation
                returned = (dists <= d_max) & alive
                last_return = torch.where(
                    returned,
                    torch.tensor(step, device=device, dtype=torch.long),
                    last_return,
                )

                # per-orientation death/explosion detection
                just_died = alive & (total < DEATH_THRESHOLD * initial_mass)
                just_exploded = alive & (total > EXPLOSION_THRESHOLD * initial_mass)
                for b in torch.where(just_died)[0].tolist():
                    termination[b] = "death"
                    term_step[b] = step + 1
                for b in torch.where(just_exploded)[0].tolist():
                    termination[b] = "explosion"
                    term_step[b] = step + 1
                alive = alive & ~just_died & ~just_exploded
                states[~alive] = 0.0          # zero dead states (no compute cost)

                if not alive.any():
                    break

            # GIF capture for selected orientations
            if gif_local and gif_stride > 0 and step % gif_stride == 0:
                for li in gif_local:
                    gif_frames[li].append(states[li].detach().clone())

            # step all B orientations forward in parallel
            states = automaton.step_batched(states, blind_masks=barrier)
            states[~alive] = 0.0

    # ── write GIFs ──────────────────────────────────────────────
    for li in gif_local:
        if gif_frames[li]:
            hdeg = int(round(init_dicts[li]['angle']))
            env_subdir = out_dir / env_name
            env_subdir.mkdir(parents=True, exist_ok=True)
            write_gif(gif_frames[li], env_subdir / f"{code}_{env_name}_{hdeg}deg.gif",
                      fps=30, barrier_mask=barrier)

    # ── competency via orbit_residence_fraction [B, T_m] ────────
    d_trim = distances_all[:, :metric_idx]
    m_trim = mass_all[:, :metric_idx]
    result = orbit_residence_fraction(
        d_trim, m_trim,
        competency_threshold=competency_threshold,
        death_threshold=DEATH_THRESHOLD,
        explosion_threshold=EXPLOSION_THRESHOLD,
        initial_mass=m_trim[:, 0],
        d_max=d_max,
    )

    # ── unpack per-orientation results ──────────────────────────
    out = []
    for b in range(B):
        out.append({
            'M': result['M'][b].item(),
            'V': result['V'][b].item(),
            'F': result['F'][b].item(),
            'D_peak': result['D_peak'][b].item(),
            'steps': term_step[b],
            'last_return': last_return[b].item(),
            'termination': termination[b],
            'metric_frames': metric_idx,
            'competency_threshold': competency_threshold,
            'centroids': centroids_all[b, :metric_idx].cpu().numpy(),
            'mass': m_trim[b].cpu().numpy(),
            'distances': d_trim[b].cpu().numpy(),
            'occlusion_weighted': occ_w_all[b, :metric_idx].cpu().numpy(),
            'occlusion_uniform': occ_u_all[b, :metric_idx].cpu().numpy(),
            'occlusion_max': occ_m_all[b, :metric_idx].cpu().numpy(),
        })
    return out


# ── main: one creature, all envs × orientations ───────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score one creature's competency across barrier environments")
    parser.add_argument("--code", required=True, help="Animal code")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor (default: 4)")
    parser.add_argument("--grid", default="128x256", help="Base grid as HxW (default: 128x256)")
    parser.add_argument("--envs", nargs="+", default=None,
                        help="Environments to test (default: all from environments.json)")
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0,
                        help="Lambda multiplier for last-return threshold (default: 1.0)")
    parser.add_argument("--output", default="results/env_competency",
                        help="Output root directory")
    parser.add_argument("--num-orientations", "-n", type=int, default=None,
                        help="Number of orientations to sample (default: all available)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Max orientations per GPU batch (default: 64)")
    parser.add_argument("--gif-headings", type=int, default=4,
                        help="Number of evenly-spaced headings to GIF (4=cardinal, 8=semicardinal, 0=none)")
    parser.add_argument("--catalog", default="animals.json", help="Animal catalog path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    code = args.code
    base_grid = parse_grid(args.grid)
    batch_size = args.batch_size
    verbose = not args.quiet
    device = _auto_device()

    # resolve environment list
    if args.envs:
        env_names = args.envs
    else:
        env_names = load_env_list()

    # validate env names
    unknown = [e for e in env_names if e not in ENVIRONMENTS and e != "open_field"]
    if unknown:
        print(f"ERROR: Unknown environments: {unknown}")
        print(f"Available: {sorted(ENVIRONMENTS.keys()) + ['open_field']}")
        sys.exit(1)

    # load animal
    animals = load_animals(Path(args.catalog), codes=[code])
    if not animals:
        print(f"ERROR: Animal '{code}' not found in {args.catalog}")
        sys.exit(1)
    animal = animals[0]
    T = animal.params.get("T", 10)

    # load orbit + initializations
    orbit_data = load_orbit_data(code, args.scale)
    inits = load_initializations(code, args.scale, num_orientations=args.num_orientations)

    d_max = orbit_data['d_max']
    c_hat = orbit_data['c_hat']
    m = orbit_data['m']
    competency_threshold = COMPETENCY_LAMBDA_MULT * args.lam * d_max

    if verbose:
        print(f"\n{'='*60}")
        print(f"COMPETENCY: {code}")
        print(f"{'='*60}")
        print(f"  Scale: {args.scale}, Grid: {base_grid[0]*args.scale}x{base_grid[1]*args.scale}")
        print(f"  T={T}, steps={int(N_PERIODS * T)} ({N_PERIODS} T-periods, early stop on death/explosion)")
        print(f"  Orbit: m={m}, d_max={d_max:.6f}, c_hat={c_hat:.6f}")
        print(f"  Threshold: {COMPETENCY_LAMBDA_MULT}*{args.lam}*d_max = {competency_threshold:.6f}")
        print(f"  Orientations: {len(inits)}, batch_size: {batch_size}")
        print(f"  Environments: {', '.join(env_names)}")
        print()

    results = {
        'code': code,
        'scale': args.scale,
        'T': T,
        'n_periods': N_PERIODS,
        'lambda': args.lam,
        'd_max': d_max,
        'c_hat': c_hat,
        'm': m,
        'competency_threshold': competency_threshold,
        'n_orientations': len(inits),
        'environments': {},
    }

    out_dir = Path(args.output) / code
    out_dir.mkdir(parents=True, exist_ok=True)

    # figure out which orientations get GIFs
    gif_ori_indices = set()
    if args.gif_headings > 0:
        target_headings = [i * 360.0 / args.gif_headings for i in range(args.gif_headings)]
        all_headings = [init['angle'] for init in inits]
        for target in target_headings:
            # find closest orientation by angular distance
            best = min(range(len(all_headings)),
                       key=lambda i: min(abs(all_headings[i] - target),
                                         360 - abs(all_headings[i] - target)))
            gif_ori_indices.add(best)

    if verbose and gif_ori_indices:
        gif_degs = sorted(inits[i]['angle'] for i in gif_ori_indices)
        print(f"  GIF headings: {[f'{d:.0f}°' for d in gif_degs]}")
        print()

    total_steps = min(int(N_PERIODS * T), 100_000)
    n_chunks = (len(inits) + batch_size - 1) // batch_size

    for env_name in env_names:
        t0 = time.time()
        all_metrics = []

        for ci, chunk_start in enumerate(range(0, len(inits), batch_size)):
            chunk = inits[chunk_start:chunk_start + batch_size]
            chunk_results = run_batch(
                code, animal, env_name, base_grid, args.scale,
                chunk, orbit_data, args.lam, device,
                gif_ori_indices=gif_ori_indices,
                chunk_offset=chunk_start,
                out_dir=out_dir,
            )
            all_metrics.extend(chunk_results)
            if verbose and n_chunks > 1:
                print(f"    {env_name} chunk {ci+1}/{n_chunks} done ({len(chunk)} orientations)")

        M_list = [met['M'] for met in all_metrics]
        V_list = [met['V'] for met in all_metrics]
        F_list = [met['F'] for met in all_metrics]
        D_list = [met['D_peak'] for met in all_metrics]
        lr_list = [met['last_return'] for met in all_metrics]
        centroid_list = [met['centroids'] for met in all_metrics]
        distance_list = [met['distances'] for met in all_metrics]
        occ_weighted_list = [met['occlusion_weighted'] for met in all_metrics]
        occ_uniform_list = [met['occlusion_uniform'] for met in all_metrics]
        occ_max_list = [met['occlusion_max'] for met in all_metrics]
        mass_list = [met['mass'] for met in all_metrics]

        agg = aggregate_competency(
            torch.tensor(M_list), torch.tensor(V_list),
            torch.tensor(F_list), torch.tensor(D_list),
        )
        lr_arr = np.array(lr_list)
        agg['last_return_per_ori'] = lr_list
        agg['last_return_mean'] = float(lr_arr.mean())
        agg['last_return_std'] = float(lr_arr.std())
        agg['centroids_per_ori'] = centroid_list  # list of [T_m, 2] arrays
        agg['distances_per_ori'] = distance_list  # list of [T_m] arrays — W1 from c_bar
        agg['occ_weighted_per_ori'] = occ_weighted_list   # list of [T_m] arrays
        agg['occ_uniform_per_ori'] = occ_uniform_list
        agg['occ_max_per_ori'] = occ_max_list
        agg['mass_per_ori'] = mass_list
        results['environments'][env_name] = agg
        elapsed = time.time() - t0

        lr_frac = lr_arr.mean() / total_steps
        if verbose:
            print(f"  {env_name:24s}  last_return={lr_arr.mean():.0f}±{lr_arr.std():.0f} "
                  f"({lr_frac:.1%} of {total_steps})  "
                  f"M={agg['M_mean']:.3f}  ({elapsed:.1f}s)")

    # save JSON (strip non-serializable centroid arrays)
    json_results = dict(results)
    json_results['environments'] = {}
    for e, env_data in results['environments'].items():
        json_results['environments'][e] = {
            k: v for k, v in env_data.items()
            if k not in ('centroids_per_ori', 'distances_per_ori', 'occ_weighted_per_ori', 'occ_uniform_per_ori', 'occ_max_per_ori', 'mass_per_ori')
        }

    json_path = out_dir / f"{code}_competency.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    # save NPZ
    env_names_done = list(results['environments'].keys())
    M_matrix = np.array([results['environments'][e]['M_per_ori'] for e in env_names_done])
    V_matrix = np.array([results['environments'][e]['V_per_ori'] for e in env_names_done])
    F_matrix = np.array([results['environments'][e]['F_per_ori'] for e in env_names_done])
    LR_matrix = np.array([results['environments'][e]['last_return_per_ori'] for e in env_names_done])

    npz_path = out_dir / f"{code}_competency.npz"
    np.savez(npz_path, env_names=env_names_done,
             M=M_matrix, V=V_matrix, F=F_matrix, last_return=LR_matrix,
             code=code, scale=args.scale, total_steps=total_steps)

    # save centroid traces [n_envs, n_orientations, n_metric_frames, 2]
    # n_metric_frames may vary per orientation (early termination), so pad with NaN
    max_frames = max(
        c.shape[0]
        for env_data in results['environments'].values()
        for c in env_data['centroids_per_ori']
    )
    n_ori = len(inits)
    n_envs = len(env_names_done)
    centroid_matrix = np.full((n_envs, n_ori, max_frames, 2), np.nan, dtype=np.float32)
    distance_matrix = np.full((n_envs, n_ori, max_frames), np.nan, dtype=np.float32)
    occ_weighted_matrix = np.full((n_envs, n_ori, max_frames), np.nan, dtype=np.float32)
    occ_uniform_matrix = np.full((n_envs, n_ori, max_frames), np.nan, dtype=np.float32)
    occ_max_matrix = np.full((n_envs, n_ori, max_frames), np.nan, dtype=np.float32)
    mass_matrix = np.full((n_envs, n_ori, max_frames), np.nan, dtype=np.float32)
    for ei, e in enumerate(env_names_done):
        env_data = results['environments'][e]
        for oi, c in enumerate(env_data['centroids_per_ori']):
            T = c.shape[0]
            centroid_matrix[ei, oi, :T, :] = c
            distance_matrix[ei, oi, :T] = env_data['distances_per_ori'][oi]
            occ_weighted_matrix[ei, oi, :T] = env_data['occ_weighted_per_ori'][oi]
            occ_uniform_matrix[ei, oi, :T] = env_data['occ_uniform_per_ori'][oi]
            occ_max_matrix[ei, oi, :T] = env_data['occ_max_per_ori'][oi]
            mass_matrix[ei, oi, :T] = env_data['mass_per_ori'][oi]

    centroid_npz_path = out_dir / f"{code}_centroids.npz"
    np.savez(centroid_npz_path, centroids=centroid_matrix,
             distance=distance_matrix,
             occlusion_weighted=occ_weighted_matrix,
             occlusion_uniform=occ_uniform_matrix,
             occlusion_max=occ_max_matrix,
             mass=mass_matrix,
             env_names=env_names_done, code=code, scale=args.scale)

    if verbose:
        print(f"\n  Saved: {json_path}")
        print(f"  Saved: {npz_path}")
        print(f"  Saved: {centroid_npz_path}  {centroid_matrix.shape}")
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
