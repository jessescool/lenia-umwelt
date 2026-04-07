"""Measure orbit residence across barrier environments."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from substrate import Simulation, load_animals, Config
from substrate.lenia import _auto_device
from environments import ENVIRONMENTS, load_env
from experiments.run_env_batch import extract_pattern, get_spawn_position, parse_grid
from metrics_and_machinery.distance_metrics import prepare_profile
from metrics_and_machinery.competency import orbit_residence_fraction, aggregate_competency
from viz.gif import write_gif


# Default environments for the initial survey
DEFAULT_ENVS = [
    "pegs", "chips", "shuriken",
    "guidelines", "membrane-1px", "membrane-3px",
    "box", "capsule", "ring",
    "corridor", "funnel", "noise",
]

# Competency threshold multiplier (wider than recovery lambda)
COMPETENCY_LAMBDA_MULT = 3.0

# T-periods to simulate
N_PERIODS = 60

# Minimum steps (ensures even T=10 creatures get enough frames)
MIN_STEPS = 600


def load_orbit_data(code: str, scale: int) -> dict:
    """Load orbit summary (c_bar, d_max, m)."""
    orbit_path = Path(f"orbits/{code}/s{scale}/{code}_s{scale}_orbit.pt")
    if not orbit_path.exists():
        raise FileNotFoundError(f"Orbit file not found: {orbit_path}")
    return torch.load(orbit_path, weights_only=False)


def load_initializations(code: str, scale: int) -> list[dict]:
    """Load pre-settled .pt initialization files."""
    init_dir = Path(f"initializations/{code}/s{scale}")
    pt_files = sorted(init_dir.glob(f"{code}_s{scale}_o*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No initialization .pt files in {init_dir}")
    return [torch.load(f, weights_only=False) for f in pt_files]


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
    verbose: bool = True,
    gif_path: Path | None = None,
) -> dict:
    """Run one creature × environment × orientation and compute M, V, F, D_peak.

    Returns dict with scalar metrics.
    """
    grid_h, grid_w = base_grid[0] * scale, base_grid[1] * scale
    shape = (grid_h, grid_w)

    cfg = Config.from_animal(animal, base_grid=base_grid, scale=scale)

    T = cfg.timescale_T
    steps = max(MIN_STEPS, int(N_PERIODS * T))

    # Metric stride: target ~1000 metric frames
    metric_stride = max(1, steps // 1000)
    n_metric_frames = (steps + metric_stride - 1) // metric_stride

    # Orbit parameters
    c_bar = orbit_data['c_bar'].to(device)
    m = orbit_data['m']
    d_max = orbit_data['d_max']
    competency_threshold = COMPETENCY_LAMBDA_MULT * lam * d_max

    pattern = extract_pattern(init_dict['tensor'])
    ph, pw = pattern.shape

    # Create simulation with barrier (open_field = no barrier, for sanity checks)
    sim = Simulation(cfg)
    if env_name == "open_field":
        mask = None
    else:
        mask = load_env(env_name, cfg.device, cfg.dtype)
        sim.set_barrier(mask)

    # Spawn creature
    pos = get_spawn_position(env_name, grid_h, grid_w, ph, pw)
    pos = (max(0, min(pos[0], grid_h - ph)),
           max(0, min(pos[1], grid_w - pw)))
    sim.add_animal(pattern.numpy(), position=pos, wrap=False)

    # Pre-allocate metric tensors
    distances = torch.empty(n_metric_frames, device=device, dtype=torch.float32)
    mass_ts = torch.empty(n_metric_frames, device=device, dtype=torch.float32)
    centroids = torch.empty(n_metric_frames, 2, device=device, dtype=torch.float32)

    # Run simulation, computing metrics online
    current = sim.board.tensor.detach().clone()
    automaton = sim.lenia.automaton
    barrier = mask  # [H, W] persistent barrier mask
    metric_idx = 0

    # precompute trig arrays for centroid (avoids realloc every frame)
    _row_angles = torch.arange(grid_h, device=device, dtype=torch.float32) * (2 * np.pi / grid_h)
    _col_angles = torch.arange(grid_w, device=device, dtype=torch.float32) * (2 * np.pi / grid_w)
    _sin_r, _cos_r = torch.sin(_row_angles), torch.cos(_row_angles)
    _sin_c, _cos_c = torch.sin(_col_angles), torch.cos(_col_angles)
    _TWO_PI = 2 * np.pi

    # GIF frame collection: target ~500 frames max
    gif_stride = max(1, steps // 500) if gif_path else 0
    gif_frames = [] if gif_path else None

    with torch.no_grad():
        for step in range(steps):
            if step % metric_stride == 0 and metric_idx < n_metric_frames:
                # Profile distance to c_bar
                p = prepare_profile(current.unsqueeze(0), m)  # [1, m]
                distances[metric_idx] = (p - c_bar).abs().mean()
                total = current.sum()
                mass_ts[metric_idx] = total
                # inline centroid using precomputed trig arrays
                if total.item() > 1e-6:
                    row_mass = current.sum(dim=1)
                    col_mass = current.sum(dim=0)
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
                metric_idx += 1

            if gif_frames is not None and step % gif_stride == 0:
                gif_frames.append(current.detach().clone())

            # Step simulation with renormalized convolution around barriers
            current = automaton.step_batched(
                current.unsqueeze(0),
                blind_masks=barrier,  # [H, W] broadcast to [1, H, W]
            ).squeeze(0)

    # Trim to actual collected frames
    distances = distances[:metric_idx].unsqueeze(0)  # [1, T_m]
    mass_ts = mass_ts[:metric_idx].unsqueeze(0)  # [1, T_m]
    initial_mass = mass_ts[:, 0]  # [1]

    # Compute competency metrics
    result = orbit_residence_fraction(
        distances, mass_ts,
        competency_threshold=competency_threshold,
        death_threshold=0.01,
        explosion_threshold=3.0,
        initial_mass=initial_mass,
        d_max=d_max,
    )

    # Write GIF if requested
    if gif_frames:
        write_gif(gif_frames, gif_path, fps=30, barrier_mask=barrier)

    return {
        'M': result['M'].item(),
        'V': result['V'].item(),
        'F': result['F'].item(),
        'D_peak': result['D_peak'].item(),
        'steps': steps,
        'metric_frames': metric_idx,
        'competency_threshold': competency_threshold,
        'centroids': centroids[:metric_idx].cpu().numpy(),  # [T_m, 2]
    }


def run_creature(
    code: str,
    scale: int,
    base_grid: tuple[int, int],
    env_names: list[str],
    lam: float,
    catalog: str = "animals.json",
    verbose: bool = True,
    gif_dir: Path | None = None,
) -> dict:
    """Run all environments × orientations for one creature. Returns results dict."""
    device = _auto_device()

    # Load animal
    animals = load_animals(Path(catalog), codes=[code])
    if not animals:
        raise ValueError(f"Animal '{code}' not found in {catalog}")
    animal = animals[0]
    T = animal.params.get("T", 10)

    # Load orbit and initializations
    orbit_data = load_orbit_data(code, scale)
    inits = load_initializations(code, scale)

    d_max = orbit_data['d_max']
    c_hat = orbit_data['c_hat']
    m = orbit_data['m']
    competency_threshold = COMPETENCY_LAMBDA_MULT * lam * d_max

    steps = max(MIN_STEPS, int(N_PERIODS * T))

    if verbose:
        print(f"\n{'='*60}")
        print(f"COMPETENCY SWEEP: {code}")
        print(f"{'='*60}")
        print(f"  Scale: {scale}, Grid: {base_grid[0]*scale}x{base_grid[1]*scale}")
        print(f"  T={T}, steps={steps} ({steps/T:.0f} T-periods)")
        print(f"  Orbit: m={m}, d_max={d_max:.6f}, c_hat={c_hat:.6f}")
        print(f"  Competency threshold: {COMPETENCY_LAMBDA_MULT}*{lam}*d_max = {competency_threshold:.6f}")
        print(f"  Orientations: {len(inits)}")
        print(f"  Environments: {', '.join(env_names)}")
        print()

    results = {
        'code': code,
        'scale': scale,
        'T': T,
        'steps': steps,
        'n_periods': N_PERIODS,
        'lambda': lam,
        'd_max': d_max,
        'c_hat': c_hat,
        'm': m,
        'competency_threshold': competency_threshold,
        'n_orientations': len(inits),
        'environments': {},
    }

    for env_name in env_names:
        t0 = time.time()
        M_list, V_list, F_list, D_list = [], [], [], []
        centroid_list = []

        for ori_idx, init_dict in enumerate(inits):
            # GIF path: results/env/{env_name}/{code}/{code}_{env_name}_o{ori}.gif
            gif_path = None
            if gif_dir is not None:
                gif_path = gif_dir / env_name / code / f"{code}_{env_name}_o{ori_idx}.gif"

            metrics = run_one(
                code, animal, env_name, base_grid, scale,
                init_dict, orbit_data, lam, device, verbose=False,
                gif_path=gif_path,
            )
            M_list.append(metrics['M'])
            V_list.append(metrics['V'])
            F_list.append(metrics['F'])
            D_list.append(metrics['D_peak'])
            centroid_list.append(metrics['centroids'])

        # Aggregate across orientations
        M_t = torch.tensor(M_list)
        V_t = torch.tensor(V_list)
        F_t = torch.tensor(F_list)
        D_t = torch.tensor(D_list)
        agg = aggregate_competency(M_t, V_t, F_t, D_t)

        agg['centroids_per_ori'] = centroid_list  # list of [T_m, 2] arrays
        results['environments'][env_name] = agg
        elapsed = time.time() - t0

        if verbose:
            print(f"  {env_name:24s}  M={agg['M_mean']:.3f} ± {agg['sigma_M']:.3f}  "
                  f"V={agg['V_mean']:.3f}  F={agg['F_mean']:.3f}  "
                  f"D_peak={agg['D_peak_mean']:.2f}  ({elapsed:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Environment competency sweep")
    parser.add_argument("--code", default=None, help="Single animal code")
    parser.add_argument("--all", action="store_true",
                        help="Run all animals from animals_to_run.json")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor (default: 4)")
    parser.add_argument("--grid", default="128x256",
                        help="Base grid as HxW (default: 128x256)")
    parser.add_argument("--envs", nargs="+", default=None,
                        help=f"Environments to test (default: {DEFAULT_ENVS})")
    parser.add_argument("--output", default="results/env_competency",
                        help="Output root directory")
    parser.add_argument("--gif-dir", default="results/env",
                        help="GIF output root (default: results/env)")
    parser.add_argument("--no-gif", action="store_true",
                        help="Skip GIF generation")
    parser.add_argument("--manifest", default="animals_to_run.json",
                        help="Animals manifest file")
    parser.add_argument("--catalog", default="animals.json",
                        help="Animal catalog path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    base_grid = parse_grid(args.grid)
    env_names = args.envs or DEFAULT_ENVS
    verbose = not args.quiet

    # Validate environments (open_field is a virtual env with no barriers)
    unknown = [e for e in env_names if e not in ENVIRONMENTS and e != "open_field"]
    if unknown:
        print(f"ERROR: Unknown environments: {unknown}")
        print(f"Available: {sorted(ENVIRONMENTS.keys()) + ['open_field']}")
        sys.exit(1)

    # Build work list
    if args.all:
        with open(args.manifest) as f:
            manifest = json.load(f)
        work = [(entry["code"], entry.get("lambda", 1.0)) for entry in manifest["animals"]]
    elif args.code:
        # Look up lambda from manifest
        lam = 1.0
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            for entry in manifest["animals"]:
                if entry["code"] == args.code:
                    lam = entry.get("lambda", 1.0)
                    break
        work = [(args.code, lam)]
    else:
        parser.error("Specify --code or --all")

    output_root = Path(args.output)
    gif_dir = None if args.no_gif else Path(args.gif_dir)

    for code, lam in work:
        try:
            results = run_creature(
                code, args.scale, base_grid, env_names, lam,
                catalog=args.catalog, verbose=verbose,
                gif_dir=gif_dir,
            )
        except FileNotFoundError as e:
            print(f"SKIP {code}: {e}")
            continue

        # Save results
        out_dir = output_root / code
        out_dir.mkdir(parents=True, exist_ok=True)

        # Strip non-serializable centroid arrays before JSON dump
        json_results = dict(results)
        json_results['environments'] = {}
        for e, env_data in results['environments'].items():
            json_results['environments'][e] = {
                k: v for k, v in env_data.items() if k != 'centroids_per_ori'
            }

        json_path = out_dir / f"{code}_competency.json"
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        # Also save as npz for easy numpy loading
        env_names_done = list(results['environments'].keys())
        M_matrix = np.array([results['environments'][e]['M_per_ori'] for e in env_names_done])
        V_matrix = np.array([results['environments'][e]['V_per_ori'] for e in env_names_done])
        F_matrix = np.array([results['environments'][e]['F_per_ori'] for e in env_names_done])

        npz_path = out_dir / f"{code}_competency.npz"
        np.savez(
            npz_path,
            env_names=env_names_done,
            M=M_matrix,  # [n_envs, n_orientations]
            V=V_matrix,
            F=F_matrix,
            code=code,
            scale=args.scale,
        )

        # Save centroid traces as separate NPZ [n_envs, n_orientations, n_metric_frames, 2]
        # Pad orientations to same length (n_metric_frames is deterministic per creature)
        centroid_arrays = []
        for e in env_names_done:
            ori_traces = results['environments'][e]['centroids_per_ori']
            centroid_arrays.append(np.stack(ori_traces, axis=0))  # [n_ori, T_m, 2]
        centroid_matrix = np.stack(centroid_arrays, axis=0)  # [n_envs, n_ori, T_m, 2]

        centroid_npz_path = out_dir / f"{code}_centroids.npz"
        np.savez(
            centroid_npz_path,
            centroids=centroid_matrix,
            env_names=env_names_done,
            code=code,
            scale=args.scale,
        )

        if verbose:
            print(f"\n  Saved: {json_path}")
            print(f"  Saved: {npz_path}")
            print(f"  Saved: {centroid_npz_path}  {centroid_matrix.shape}")

    print(f"\nOutput: {output_root}")


if __name__ == "__main__":
    main()
