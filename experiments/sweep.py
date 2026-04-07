"""Exhaustive perturbation sweep over all grid positions."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from substrate import load_animals, Config, Simulation
from substrate.lenia import Automaton
from config import RECOVERY_TIMEOUT_MULTIPLIER
from metrics_and_machinery import (
    Intervention,
    make_intervention,
)
from metrics_and_machinery.distance_metrics import detect_recovery
from viz.gif import write_gif, render_convergence_gif
from utils.batched import (
    rollout_batched_with_ctrl,
    rollout_online_metrics,
    build_blind_masks,
    apply_interventions_batched,
    estimate_batch_size,
)

from viz.maps import (
    plot_recovery_status_map,
    plot_max_distance_map,
    plot_relative_heading,
    plot_summary,
)


# Outcome code -> label mapping (used in batch loop)
OUTCOME_MAP = {0: "died", 1: "recovered", 2: "never"}


@dataclass
class GridSearchResult:
    recovery_map: np.ndarray              # [H, W] recovery times (normalized by T)
    erased_map: np.ndarray                # [H, W] mass erased at each position
    creature_state: np.ndarray            # [H, W] creature state before perturbation
    positions: list                       # List of (x, y, time, mass, outcome) tuples
    recovery_status_map: np.ndarray       # [H, W] int8: -1=empty, 0=died, 1=recovered, 2=never
    # Trajectory analysis maps
    heading_change_map: np.ndarray        # [H, W]
    heading_vec_relative_map: np.ndarray  # [H, W, 2] heading relative to control
    max_distance_map: np.ndarray          # [H, W] peak Wasserstein distance after perturbation
    centroid_data: dict = field(default_factory=dict)

    # Map names for serialization (maps field name -> npy file stem)
    _SAVE_FIELDS = {
        'recovery_map': 'recovery_map',
        'erased_map': 'erased_map',
        'recovery_status_map': 'recovery_status_map',
        'heading_change_map': 'heading_change',
        'heading_vec_relative_map': 'heading_vec_relative',
        'max_distance_map': 'max_distance',
    }

    def save_npy(self, analysis_dir: Path, prefix: str) -> None:
        """Save all map arrays as .npy files with prefix."""
        for attr, stem in self._SAVE_FIELDS.items():
            np.save(analysis_dir / f"{prefix}_{stem}.npy", getattr(self, attr))


def _compute_trajectory_stats_batched(
    ctrl_centroids: np.ndarray,
    test_centroids: np.ndarray,
    outcomes: torch.Tensor,
    recovery_threshold: float,
    timescale_T: float = 10.0,
    precomputed_distances: torch.Tensor = None,
    grid_shape: tuple = None,
) -> dict:
    """Compute trajectory-level statistics from precomputed centroids.

    Fully vectorized -- no Python loops or .item() sync points.
    Operates on lightweight centroid arrays, not raw frames.
    """
    B, T = test_centroids.shape[:2]

    # Use at least T frames for heading average to avoid phase aliasing
    n_heading_avg = max(5, int(timescale_T))

    # Default outputs
    heading_diff = np.zeros(B, dtype=np.float32)
    heading_vec_rel = np.zeros((B, 2), dtype=np.float32)
    max_distance = np.zeros(B, dtype=np.float32)

    recovered_mask = outcomes == 1  # only recovered; exclude died (0) and never_recovered (2)
    alive_indices = recovered_mask.nonzero(as_tuple=True)[0]
    B_alive = len(alive_indices)

    if B_alive == 0:
        return {
            'heading_change': heading_diff,
            'heading_vec_relative': heading_vec_rel,
            'max_distance': max_distance,
        }

    alive_idx_np = alive_indices.cpu().numpy()

    if T >= n_heading_avg + 1:
        n = n_heading_avg
        from viz._helpers import (
            heading_from_centroids, rolling_circular_mean,
            heading_deflection as _heading_deflection, speed_from_centroids,
        )
        assert grid_shape is not None, "grid_shape required for heading computation"

        # Centroids already precomputed -- just slice alive
        ctrl_cents = ctrl_centroids                              # [T, 2]
        test_cents = test_centroids[alive_idx_np]                # [B_alive, T, 2]

        test_h = rolling_circular_mean(
            heading_from_centroids(test_cents, grid_shape), n)       # [B_alive, T-1]
        ctrl_h = rolling_circular_mean(
            heading_from_centroids(ctrl_cents[np.newaxis], grid_shape), n)[0]  # [T-1]

        defl = _heading_deflection(test_h, ctrl_h)                   # [B_alive, T-1]

        # Per-frame speed for stationary guard (OR -- mask if EITHER is slow)
        test_spd = speed_from_centroids(test_cents, grid_shape)      # [B_alive, T-1]
        ctrl_spd = speed_from_centroids(ctrl_cents[np.newaxis], grid_shape)[0]  # [T-1]
        speed_threshold = 0.1  # pixels/frame
        stationary = (ctrl_spd[np.newaxis] < speed_threshold) | (test_spd < speed_threshold)
        defl[stationary] = 0.0

        # Peak deflection and heading vectors at peak time
        abs_defl = np.abs(defl)
        peak_idx = np.argmax(abs_defl, axis=1)                      # [B_alive]
        arange = np.arange(B_alive)

        heading_diff[alive_idx_np] = defl[arange, peak_idx]         # signed peak

        # Unit heading vectors at peak frame (for quiver plot)
        peak_test_h = test_h[arange, peak_idx]
        peak_ctrl_h = ctrl_h[peak_idx]
        test_units = np.stack([np.sin(peak_test_h), np.cos(peak_test_h)], axis=1)
        ctrl_units = np.stack([np.sin(peak_ctrl_h), np.cos(peak_ctrl_h)], axis=1)

        peak_stat = stationary[arange, peak_idx]
        test_units[peak_stat] = 0.0

        heading_vec_rel[alive_idx_np] = test_units - ctrl_units
        heading_vec_rel[alive_idx_np[peak_stat]] = 0.0

    d = precomputed_distances[alive_indices].cpu().numpy()  # [B_alive, T]
    max_distance[alive_idx_np] = np.max(d, axis=1)

    return {
        'heading_change': heading_diff,
        'heading_vec_relative': heading_vec_rel,
        'max_distance': max_distance,
    }


def sweep(
    lenia: "Lenia",
    lenia_cfg: Config,
    intervention: Intervention,
    warmup: int,
    window: int,
    orientation_idx: int = 0,
    verbose: bool = True,
    batch_size: int = 256,
    stability_window: int = 20,
    death_threshold: float = 0.01,
    timescale_T: float = 10.0,
    recovery_threshold: float = 0.002201,
    shortcut: bool = False,
    orbit_data: dict = None,
    blind_duration: int | None = None,
) -> GridSearchResult:
    """
    Test every grid position using GPU batching.

    Returns GridSearchResult with recovery_map, erased_map, creature_state, positions,
    recovery_status_map, and trajectory analysis maps (heading_change,
    heading_vec_relative, max_distance).
    """
    H, W = lenia_cfg.grid_shape
    device = lenia_cfg.device

    initial_state = lenia.board.tensor.detach().clone()
    automaton = lenia.automaton
    creature_state = initial_state.cpu().numpy()

    # Results storage: original maps
    recovery_map = np.zeros((H, W), dtype=np.float32)
    erased_map = np.zeros((H, W), dtype=np.float32)
    recovery_status_map = np.full((H, W), -1, dtype=np.int8)
    positions_list = []

    # Results storage: trajectory analysis maps
    heading_change_map = np.zeros((H, W), dtype=np.float32)
    heading_vec_relative_map = np.zeros((H, W, 2), dtype=np.float32)
    max_distance_map = np.zeros((H, W), dtype=np.float32)

    # Gather all positions to test
    if shortcut:
        # Only test positions with non-zero mass
        nonzero_mask = initial_state.cpu().numpy() > 0.001
        all_positions = [(x, y) for y in range(H) for x in range(W) if nonzero_mask[y, x]]
    else:
        all_positions = [(x, y) for y in range(H) for x in range(W)]
    total_positions = len(all_positions)
    total_steps = warmup + window

    # Accumulators for per-frame centroid trajectories (convergence GIF)
    centroid_trajectories = []  # list of [N_alive_batch, T, 2] numpy arrays
    centroid_outcomes = []      # list of [N_alive_batch] numpy arrays
    centroid_positions = []     # list of (x, y) tuples
    saved_ctrl_frames = None    # saved from first batch for convergence GIF

    # no gradients needed
    n_batches = (total_positions + batch_size - 1) // batch_size
    pbar = tqdm(
        range(0, total_positions, batch_size),
        desc="    Sweep",
        total=n_batches,
        disable=not verbose,
        unit="batch",
        dynamic_ncols=True,
        leave=True,
        position=0,
    )
    with torch.no_grad():
      for batch_start in pbar:
        batch_end = min(batch_start + batch_size, total_positions)
        batch_positions = all_positions[batch_start:batch_end]
        B = len(batch_positions)

        # Apply interventions to create batch
        test_states, affected = apply_interventions_batched(
            initial_state, batch_positions, intervention
        )

        # Build blind masks (None for plain erase/additive)
        blind = build_blind_masks(
            intervention, (H, W), batch_positions,
            device=device, dtype=initial_state.dtype,
        )

        # OOM retry: halve batch
        rollout_kwargs = dict(
            ctrl_state=initial_state,
            automaton=automaton,
            n_steps=total_steps,
            orbit_c_bar=orbit_data['c_bar'],
            orbit_m=orbit_data['m'],
            blind_duration=blind_duration,
        )
        try:
            metrics = rollout_online_metrics(
                test_states, **rollout_kwargs,
                blind_masks=blind,
            )
        except (torch.cuda.OutOfMemoryError if hasattr(torch.cuda, 'OutOfMemoryError') else RuntimeError):
            # OOM: halve the batch, process in two parts
            torch.cuda.empty_cache()
            mid = B // 2
            b1 = blind[:mid] if blind is not None else None
            b2 = blind[mid:] if blind is not None else None
            m1 = rollout_online_metrics(
                test_states[:mid], **rollout_kwargs,
                blind_masks=b1,
            )
            m2 = rollout_online_metrics(
                test_states[mid:], **rollout_kwargs,
                blind_masks=b2,
            )
            # Merge: distances/mass on GPU, centroids/ctrl_frames on CPU
            metrics = {
                'distances': torch.cat([m1['distances'], m2['distances']], dim=0),
                'centroids': torch.cat([
                    m1['centroids'][:-1],            # first half test centroids
                    m2['centroids'][:-1],            # second half test centroids
                    m1['centroids'][-1:],            # ctrl centroid (identical in both)
                ], dim=0),                           # [B+1, T, 2]
                'mass': torch.cat([m1['mass'], m2['mass']], dim=0),
                'ctrl_frames': m1['ctrl_frames'],    # ctrl is identical
            }
            del m1, m2, b1, b2

        batch_distances = metrics['distances']  # [B, T] GPU

        # Detect recovery from precomputed distances + mass
        recovery_times, outcomes = detect_recovery(
            batch_distances,
            metrics['mass'],
            recovery_threshold=recovery_threshold,
            stability_window=stability_window,
            death_threshold=death_threshold,
            warmup_frames=warmup,
        )

        # Extract centroids as numpy for trajectory stats
        all_centroids = metrics['centroids'].numpy()       # [B+1, T, 2]
        ctrl_cents_np = all_centroids[-1]                  # [T, 2]
        test_cents_np = all_centroids[:B]                  # [B, T, 2]

        # Compute trajectory statistics from precomputed centroids
        traj_stats = _compute_trajectory_stats_batched(
            ctrl_cents_np, test_cents_np, outcomes,
            recovery_threshold=recovery_threshold,
            timescale_T=timescale_T,
            precomputed_distances=batch_distances,
            grid_shape=(H, W),
        )

        # Save control frames from first batch (already on CPU)
        if saved_ctrl_frames is None:
            saved_ctrl_frames = metrics['ctrl_frames']

        # Collect per-frame centroids for alive sims (for convergence GIF)
        # Already computed -- just slice from online metrics
        alive_mask_centroid = outcomes != 0
        if alive_mask_centroid.any():
            alive_idx_c = alive_mask_centroid.nonzero(as_tuple=True)[0]
            cents = test_cents_np[alive_idx_c.cpu().numpy()]  # [N_alive, T, 2]
            centroid_trajectories.append(cents)
            centroid_outcomes.append(outcomes[alive_idx_c].cpu().numpy())
            centroid_positions.extend(
                [batch_positions[j] for j in alive_idx_c.cpu().tolist()]
            )

        # Normalize recovery times by creature timescale
        recovery_times_normalized = recovery_times / timescale_T

        # Convert to numpy
        recovery_times_np = recovery_times_normalized.cpu().numpy()
        outcomes_np = outcomes.cpu().numpy()
        affected_np = affected.cpu().numpy()

        # Vectorized scatter: batch results into [H, W] maps
        ys = np.array([p[1] for p in batch_positions])
        xs = np.array([p[0] for p in batch_positions])

        erased_map[ys, xs] = affected_np
        active = affected_np >= 0.001
        recovery_map[ys, xs] = np.where(active, recovery_times_np, 0.0)
        recovery_status_map[ys, xs] = np.where(active, outcomes_np, -1).astype(np.int8)

        # Build positions_list entries for active positions
        active_idx = np.where(active)[0]
        for i in active_idx:
            outcome_str = OUTCOME_MAP[outcomes_np[i]]
            positions_list.append((xs[i], ys[i], recovery_times_np[i], affected_np[i], outcome_str))

        # Scatter trajectory maps (all positions, including inactive)
        heading_change_map[ys, xs] = traj_stats['heading_change']
        heading_vec_relative_map[ys, xs] = traj_stats['heading_vec_relative']
        max_distance_map[ys, xs] = traj_stats['max_distance']

        # Free batch GPU tensors before next iteration
        del metrics, batch_distances
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        pbar.set_postfix({"positions": len(positions_list)}, refresh=False)

    pbar.close()

    # Build centroid data for convergence GIF
    # ctrl_centroids computed from saved_ctrl_frames (render_convergence_gif expects torch tensor)
    from metrics_and_machinery.trajectory_metrics import centroid_batched
    ctrl_cents = (
        centroid_batched(saved_ctrl_frames)
        if saved_ctrl_frames is not None else None
    )
    centroid_data = {
        'ctrl_frames': saved_ctrl_frames,
        'ctrl_centroids': ctrl_cents,
        'centroids': (
            np.concatenate(centroid_trajectories, axis=0)
            if centroid_trajectories
            else np.empty((0, total_steps, 2))
        ),
        'outcomes': (
            np.concatenate(centroid_outcomes, axis=0)
            if centroid_outcomes
            else np.empty((0,))
        ),
        'positions': centroid_positions,
    }

    return GridSearchResult(
        recovery_map=recovery_map,
        erased_map=erased_map,
        creature_state=creature_state,
        positions=positions_list,
        recovery_status_map=recovery_status_map,
        heading_change_map=heading_change_map,
        heading_vec_relative_map=heading_vec_relative_map,
        max_distance_map=max_distance_map,
        centroid_data=centroid_data,
    )


def _categorize_positions(
    result: GridSearchResult,
    n_slowest: int = 8,
    n_deaths: int = 8,
    n_never: int = 8,
    n_deflected: int = 8,
) -> dict:
    """Categorize positions by outcome for GIF selection."""
    recovered, never, died = [], [], []
    deflected = []

    for pos in result.positions:
        x, y, value, affected, outcome = pos
        entry = (x, y, value, affected, outcome)
        if outcome == "recovered":
            recovered.append(entry)
        elif outcome == "never":
            never.append(entry)
        elif outcome == "died":
            died.append(entry)

    # Biggest heading deflections: alive positions sorted by |heading_change|
    hc = result.heading_change_map
    status = result.recovery_status_map
    alive = status > 0
    if alive.any():
        abs_hc = np.abs(hc)
        ys, xs = np.where(alive)
        for y, x in zip(ys, xs):
            outcome_str = OUTCOME_MAP[status[y, x]]
            deflected.append((x, y, float(abs_hc[y, x]), float(result.erased_map[y, x]), outcome_str))

    recovered.sort(key=lambda p: p[2], reverse=True)
    never.sort(key=lambda p: p[2], reverse=True)
    deflected.sort(key=lambda p: p[2], reverse=True)

    return {
        'slowest_recoveries': recovered[:n_slowest],
        'never_recovered': never[:n_never],
        'deaths': died[:n_deaths],
        'furthest_recoveries': deflected[:n_deflected],
    }


def _generate_top_k_gifs(
    ranked: List[Tuple[int, int, float, float, str]],
    lenia_cfg: Config,
    intervention: Intervention,
    warmup: int,
    window: int,
    output_dir: Path,
    subdir: str = "top_k_gifs",
    situation_tensor: torch.Tensor = None,
    upscale: int = 4,
    fft: bool = True,
    prefix: str = "",
    blind_duration: int | None = None,
) -> None:
    """Re-run top-K positions and generate single-panel test GIFs."""
    gifs_dir = output_dir / subdir
    gifs_dir.mkdir(exist_ok=True)
    total_steps = warmup + window

    print(f"  Generating {len(ranked)} GIFs → {subdir}/")

    positions = [(x, y) for x, y, *_ in ranked]
    test_states, _ = apply_interventions_batched(situation_tensor, positions, intervention)

    # Build blind masks (None for plain erase/additive)
    blind = build_blind_masks(
        intervention, situation_tensor.shape, positions,
        device=lenia_cfg.device, dtype=lenia_cfg.dtype,
    )

    # Batched rollout
    test_frames, ctrl_frames = rollout_batched_with_ctrl(
        test_states, situation_tensor, Automaton(lenia_cfg, fft=fft), total_steps,
        blind_masks=blind,
        blind_duration=blind_duration,
    )
    # Convert to CPU, free GPU
    test_cpu = test_frames.cpu()
    del test_frames, ctrl_frames, test_states
    if lenia_cfg.device.type == 'cuda':
        torch.cuda.empty_cache()

    for i, (x, y, value, affected, outcome) in enumerate(ranked):
        test_traj = list(test_cpu[i])
        gif_path = gifs_dir / f"{prefix}_r{y}_c{x}.gif" if prefix else gifs_dir / f"r{y}_c{x}.gif"
        half = intervention.size // 2
        write_gif(test_traj, gif_path, fps=15, upscale=upscale,
                  marker_rect=(y - half, x - half, intervention.size, intervention.size))

    print(f"  Saved {len(ranked)} GIFs to {gifs_dir}/")


def _load_orbit_data(code: str, scale: int) -> dict:
    """Load orbit summary data for a creature at the given scale."""
    orbit_path = Path(f"orbits/{code}/s{scale}/{code}_s{scale}_orbit.pt")
    if not orbit_path.exists():
        raise FileNotFoundError(
            f"Orbit file not found: {orbit_path}\n"
            f"Run:  python orbits/orbits.py orbit ..."
        )
    orbit_data = torch.load(orbit_path, weights_only=False)
    return orbit_data


def main():
    parser = argparse.ArgumentParser(
        description="Exhaustive grid search for recovery time mapping"
    )
    parser.add_argument("--init", type=Path, required=True,
                        help="Path to an initialization .pt file (from generate_initializations.py)")
    parser.add_argument("--code", default="O2u", help="Animal code (default: O2u)")
    parser.add_argument("--size", type=int, default=2, help="Intervention size NxN at base resolution (default: 2)")
    parser.add_argument("--grid", type=int, default=128, help="Base grid size (default: 128)")
    parser.add_argument("--intervention-type", choices=["erase", "additive", "blind_erase", "blind"], default="erase",
                        help="Intervention type (default: erase)")
    parser.add_argument("--intensity", type=float, default=0.3,
                        help="Intensity for additive intervention (default: 0.3)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    parser.add_argument("--batch-size", type=int, default=None,
                        help="Positions per GPU batch (default: auto-detect)")
    parser.add_argument("--scale", type=int, default=1,
                        help="Upscale factor (default: 1)")
    parser.add_argument("--shortcut", action="store_true",
                        help="Only test non-zero pixels (faster)")
    parser.add_argument("--no-gifs", action="store_true",
                        help="Skip GIF generation (saves time for large grids)")
    parser.add_argument("--conv", action="store_true",
                        help="Use spatial conv2d instead of FFT (default: FFT)")
    parser.add_argument("--crop", type=int, default=None,
                        help="Crop size for heatmap plots (base px, before scaling). "
                             "Defaults to --grid value. Use creature's cropped_grid to zoom in.")
    parser.add_argument("--duration", type=int, default=None,
                        help="How many steps blind masks are active. "
                             "-1 = persistent (all steps). Omit = use intervention default.")
    parser.add_argument("--recovery-lambda", type=float, default=1.0,
                        help="Multiplier on orbit d_max for recovery threshold (default: 1.0)")

    args = parser.parse_args()
    verbose = not args.quiet

    if not args.init.exists():
        raise SystemExit(f"Initialization file not found: {args.init}")
    sit = torch.load(args.init, weights_only=False)
    sit_tensor = sit['tensor']       # (H, W) CPU
    sit_idx = sit['sit_idx']
    sit_angle = sit['angle']
    sit_code = sit['code']
    sit_name = sit.get('name', sit_code)

    # Load creature (needed for Config physics params)
    creatures = load_animals(Path("animals.json"), codes=[args.code])
    if not creatures:
        raise SystemExit(f"No animal found with code '{args.code}'")
    creature = creatures[0]

    actual_grid = args.grid * args.scale
    actual_size = args.size * args.scale

    print(f"{'='*60}")
    print(f"GRID SEARCH: {sit_name} ({args.code}) orientation {sit_idx} ({sit_angle:.1f}°)")
    if args.scale > 1:
        print(f"Base: {args.grid}x{args.grid} grid, {args.size}x{args.size} intervention")
        print(f"Scale: {args.scale}x → {actual_grid}x{actual_grid} grid, {actual_size}x{actual_size} intervention")
    print(f"{'='*60}")

    lenia_cfg = Config.from_animal(creature, base_grid=args.grid, scale=args.scale)

    warmup = int(5.0 * lenia_cfg.timescale_T)
    window = int(RECOVERY_TIMEOUT_MULTIPLIER * lenia_cfg.timescale_T)

    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = estimate_batch_size(actual_grid, warmup + window, lenia_cfg.device, online=True)

    orbit_data = _load_orbit_data(args.code, args.scale)
    recovery_threshold = orbit_data['d_max'] * args.recovery_lambda

    intervention = make_intervention(args.intervention_type, actual_size, intensity=args.intensity)

    # Resolve blind_duration: CLI override > intervention default
    # -1 = persistent (None internally), omit = intervention default
    blind_duration = (
        None if args.duration == -1
        else args.duration if args.duration is not None
        else intervention.default_blind_duration
    )

    if args.output_dir is None:
        output_dir = Path(f"results/sweep/{args.code}/{args.code}_x{args.scale}/{args.code}_x{args.scale}_i{args.size}")
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # File prefix: encodes code, scale, size, and orientation
    prefix = f"{args.code}_x{args.scale}_i{args.size}_o{sit_idx}"

    # Crop
    H, W = lenia_cfg.grid_shape
    scale = lenia_cfg.scale
    base_grid = lenia_cfg.base_grid or H
    crop_size = args.crop * scale if args.crop is not None else base_grid * scale
    crop_base = crop_size // scale

    if verbose:
        print(f"Grid: {H}x{W} = {H*W} positions" + (" (shortcut: non-zero only)" if args.shortcut else ""))
        print(f"Intervention: {actual_size}x{actual_size} {args.intervention_type}")
        print(f"Timing: warmup={warmup}, window={window}")
        d_max = orbit_data['d_max']
        m = orbit_data['m']
        print(f"Recovery: orbit-based (m={m}, d_max={d_max:.6f})")
        print(f"Init: {args.init} (orientation {sit_idx}, {sit_angle:.1f}°)")
        print(f"Batch size: {batch_size}")
        print(f"Device: {lenia_cfg.device}")
        if scale > 1:
            print(f"Scale: {scale}x (base grid: {base_grid})")
        print()

    sim = Simulation(lenia_cfg, fft=not args.conv)
    sim.board.tensor.copy_(sit_tensor.to(lenia_cfg.device))

    result = sweep(
        sim.lenia, lenia_cfg, intervention, warmup, window,
        orientation_idx=sit_idx,
        verbose=verbose,
        batch_size=batch_size,
        stability_window=20,
        death_threshold=0.01,
        timescale_T=lenia_cfg.timescale_T,
        recovery_threshold=recovery_threshold,
        shortcut=args.shortcut,
        orbit_data=orbit_data,
        blind_duration=blind_duration,
    )

    # Move centroid GPU tensors to CPU
    cd = result.centroid_data
    if cd and cd.get('ctrl_frames') is not None:
        cd['ctrl_frames'] = cd['ctrl_frames'].cpu()
        cd['ctrl_centroids'] = cd['ctrl_centroids'].cpu()

    n_positions = len(result.positions)
    creature_pixels = np.sum(result.erased_map > 0.01)
    if verbose:
        print(f"  → {n_positions} positions ({100*n_positions/max(creature_pixels,1):.1f}% of creature)")

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    result.save_npy(analysis_dir, prefix)

    situation_tensor_gpu = sit_tensor.to(lenia_cfg.device)
    gif_upscale = max(1, 4 // scale)

    if not args.no_gifs:
        categories = _categorize_positions(result)
        for cat_name, subdir in [
            ('slowest_recoveries', 'slowest_recoveries'),
            ('never_recovered', 'never_recovered'),
            ('deaths', 'death'),
            ('furthest_recoveries', 'furthest_recoveries'),
        ]:
            if categories[cat_name]:
                _generate_top_k_gifs(
                    categories[cat_name],
                    lenia_cfg, intervention,
                    warmup, window, output_dir,
                    subdir=subdir,
                    situation_tensor=situation_tensor_gpu,
                    upscale=gif_upscale,
                    fft=not args.conv,
                    prefix=prefix,
                    blind_duration=blind_duration,
                )

    subtitle = f"size={args.size}  scale={scale}  intervention={args.intervention_type}  ori={sit_idx}"
    plot_recovery_status_map(
        result.recovery_status_map, result.recovery_map,
        sit_name, 'Recovery Time Map',
        output_dir / f'{prefix}_recovery_time_map.png',
        subtitle=subtitle, crop_size=crop_size, crop_grid=crop_base,
    )
    print(f"  Saved recovery time map to {output_dir}/{prefix}_recovery_time_map.png")

    plot_max_distance_map(
        result.max_distance_map, result.recovery_status_map,
        sit_name,
        output_dir / f'{prefix}_max_distance_map.png',
        subtitle=subtitle, crop_size=crop_size, crop_grid=crop_base,
    )

    plot_relative_heading([result], sit_name, output_dir,
                          subtitle=subtitle, crop_size=crop_size, crop_grid=crop_base,
                          filename=f'{prefix}_relative_heading_map.png')

    plot_summary(output_dir, prefix=prefix)

    if not args.no_gifs:
        centroid_data = result.centroid_data
        if (centroid_data
                and centroid_data.get('ctrl_frames') is not None
                and len(centroid_data.get('centroids', [])) > 0):
            convergence_path = output_dir / f"{prefix}_all_centroids.gif"
            convergence_upscale = max(1, 4 // scale)
            render_convergence_gif(
                centroid_data['ctrl_frames'],
                centroid_data['ctrl_centroids'],
                centroid_data,
                convergence_path,
                upscale=convergence_upscale,
                flash_frames=int(4 * lenia_cfg.timescale_T),
            )
            np.savez(
                analysis_dir / f"{prefix}_centroid_trajectories.npz",
                centroids=centroid_data['centroids'],
                outcomes=centroid_data['outcomes'],
                ctrl_centroids=centroid_data['ctrl_centroids'].cpu().numpy(),
            )

    if verbose:
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Creature: {sit_name} ({args.code}), orientation {sit_idx} ({sit_angle:.1f}°)")
        print(f"Creature size: {creature_pixels:.0f} pixels")
        print(f"Positions tested: {n_positions}")
        print(f"Results saved to: {output_dir}/")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
