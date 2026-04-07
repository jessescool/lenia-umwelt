# experiments/recovery_test.py — Jesse Cool (jessescool)
"""Single-creature recovery test with side-by-side GIF output."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RECOVERY_TIMEOUT_MULTIPLIER
from metrics_and_machinery import make_intervention
from metrics_and_machinery.distance_metrics import prepare_profile, wasserstein
from substrate import Config, Simulation, load_animals, prepare_scaled_simulation
from substrate.lenia import Automaton, _auto_device
from utils.core import rotate_tensor
from utils.batched import (
    apply_interventions_batched,
    build_blind_masks,
    estimate_batch_size,
    rollout_batched_with_ctrl,
)


def compute_profile_distances(
    frames: torch.Tensor,
    c_bar: torch.Tensor,
    m: int,
) -> torch.Tensor:
    """Compute d(profile(frame), c̄) for each frame in a batch of trajectories."""
    B, T, H, W = frames.shape
    device = frames.device

    distances = torch.empty(B, T, device=device, dtype=frames.dtype)
    for t in range(T):
        # [B, H, W] → [B, m] profiles
        profiles = prepare_profile(frames[:, t], m)
        # L1 distance to c_bar: (profile - c_bar).abs().mean(dim=1) → [B]
        distances[:, t] = (profiles - c_bar.to(device)).abs().mean(dim=1)

    return distances


def run_recovery_test(
    creature,
    lenia_cfg: Config,
    orbit_data: dict,
    intervention_type: str = "erase",
    intervention_size: int = 2,
    intensity: float = 0.3,
    shortcut: bool = False,
    batch_size: int = 256,
    window_mult: float = RECOVERY_TIMEOUT_MULTIPLIER,
    verbose: bool = True,
) -> dict:
    """Run perturbation experiment and collect d(x, c̄) timeseries."""
    H, W = lenia_cfg.grid_shape
    device = lenia_cfg.device
    T_creature = lenia_cfg.timescale_T
    scale = lenia_cfg.scale

    # Orbit data
    c_bar = orbit_data["c_bar"]            # (m,)
    c_hat = orbit_data["c_hat"]            # float
    sigma = orbit_data["sigma"]            # float
    recovery_threshold = c_hat + 3 * sigma
    m = orbit_data["m"]                    # int

    # Timing
    window = int(window_mult * T_creature)

    # Intervention
    intervention = make_intervention(intervention_type, intervention_size, intensity=intensity)

    if verbose:
        print(f"Grid: {H}x{W}, scale={scale}")
        print(f"Orbit: m={m}, ĉ={c_hat:.6f}, σ={sigma:.6f}, threshold=ĉ+2σ={recovery_threshold:.6f}")
        print(f"Intervention: {intervention_size}x{intervention_size} {intervention_type}")
        print(f"Rollout: {window} steps ({window_mult}*T={T_creature})")
        print(f"Batch size: {batch_size}")
        print(f"Device: {device}")
        print()

    # Settle creature
    settle_steps = int(30 * T_creature)
    sim = prepare_scaled_simulation(
        creature, base_grid=lenia_cfg.base_grid or (H // max(scale, 1)),
        scale=scale, settle_steps=settle_steps, recenter=True,
    )
    initial_state = sim.board.tensor.detach().clone()
    automaton = sim.lenia.automaton

    if verbose:
        mass = initial_state.sum().item()
        print(f"Settled creature ({settle_steps} steps), mass={mass:.1f}")

    baseline_profile = prepare_profile(initial_state, m)
    baseline_dist = float((baseline_profile - c_bar.to(device)).abs().mean())
    if verbose:
        print(f"Baseline d(settled, c̄) = {baseline_dist:.6f}")

    # Gather positions
    if shortcut:
        nonzero_mask = initial_state.cpu().numpy() > 0.001
        all_positions = [(x, y) for y in range(H) for x in range(W) if nonzero_mask[y, x]]
    else:
        all_positions = [(x, y) for y in range(H) for x in range(W)]
    total_positions = len(all_positions)

    if verbose:
        print(f"Positions: {total_positions}" + (" (shortcut)" if shortcut else ""))
        print()

    # Storage
    all_distances = []      # list of [B, T] arrays
    all_masses = []         # list of [B, T] arrays — total mass per frame
    all_positions_used = []
    ctrl_distances = None   # [T] control trajectory distances to c̄

    n_batches = (total_positions + batch_size - 1) // batch_size
    pbar = tqdm(
        range(0, total_positions, batch_size),
        desc="Recovery test",
        total=n_batches,
        disable=not verbose,
        unit="batch",
    )

    with torch.no_grad():
        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, total_positions)
            batch_positions = all_positions[batch_start:batch_end]
            B = len(batch_positions)

            # Apply interventions
            test_states, affected = apply_interventions_batched(
                initial_state, batch_positions, intervention
            )

            # Skip positions with no effect (empty space)
            active_mask = affected > 0.001
            if not active_mask.any():
                del test_states, affected
                continue

            active_idx = active_mask.nonzero(as_tuple=True)[0]
            test_states_active = test_states[active_idx]
            active_positions = [batch_positions[i] for i in active_idx.cpu().tolist()]

            # Build blind masks (None for plain erase/additive)
            blind = build_blind_masks(
                intervention, (H, W), active_positions,
                device=device, dtype=initial_state.dtype,
            )
            blind_dur = intervention.default_blind_duration

            # Rollout
            try:
                test_frames, ctrl_frames = rollout_batched_with_ctrl(
                    test_states_active, initial_state, automaton, window,
                    blind_masks=blind,
                    blind_duration=blind_dur,
                )
            except (torch.cuda.OutOfMemoryError if hasattr(torch.cuda, 'OutOfMemoryError') else RuntimeError):
                torch.cuda.empty_cache()
                mid = test_states_active.shape[0] // 2
                b1 = blind[:mid] if blind is not None else None
                b2 = blind[mid:] if blind is not None else None
                tf1, cf1 = rollout_batched_with_ctrl(
                    test_states_active[:mid], initial_state, automaton, window,
                    blind_masks=b1, blind_duration=blind_dur,
                )
                tf2, cf2 = rollout_batched_with_ctrl(
                    test_states_active[mid:], initial_state, automaton, window,
                    blind_masks=b2, blind_duration=blind_dur,
                )
                test_frames = torch.cat([tf1, tf2], dim=0)
                ctrl_frames = cf1
                del tf1, tf2, cf1, cf2

            # Compute profile distances to c̄
            batch_dists = compute_profile_distances(test_frames, c_bar, m)
            all_distances.append(batch_dists.cpu().numpy())

            # Track mass per frame: sum over spatial dims → [B, T]
            batch_mass = test_frames.sum(dim=(2, 3))
            all_masses.append(batch_mass.cpu().numpy())

            all_positions_used.extend(active_positions)

            # Control trajectory distances (compute once)
            if ctrl_distances is None:
                ctrl_dists = compute_profile_distances(
                    ctrl_frames.unsqueeze(0), c_bar, m
                )
                ctrl_distances = ctrl_dists.squeeze(0).cpu().numpy()  # [T]

            # Cleanup
            del test_frames, ctrl_frames, test_states, affected, batch_dists
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            pbar.set_postfix({"positions": len(all_positions_used)}, refresh=False)

    pbar.close()

    if not all_distances:
        print("No active positions found!")
        return {}

    # Concatenate all batches
    distances = np.concatenate(all_distances, axis=0)  # [N, T]
    masses = np.concatenate(all_masses, axis=0)        # [N, T]

    DEATH_THRESHOLD = 0.01
    final_mass = masses[:, -1]
    died = final_mass < DEATH_THRESHOLD
    # Recovered: alive AND final distance within orbit radius ĉ
    alive = ~died
    recovered = alive & (distances[:, -1] < c_hat)
    never = alive & ~recovered
    # Pack: 0=died, 1=recovered, 2=never-recovered
    outcomes = np.where(died, 0, np.where(recovered, 1, 2))

    n_died = int(died.sum())
    n_recovered = int(recovered.sum())
    n_never = int(never.sum())

    # Check which alive trajectories never dip below ĉ at ANY timestep
    ever_below = (distances < c_hat).any(axis=1)  # [N]
    never_below = alive & ~ever_below
    n_never_below = int(never_below.sum())

    if verbose:
        print(f"\nResults: {distances.shape[0]} positions × {distances.shape[1]} timesteps")
        print(f"  Outcomes: {n_recovered} recovered, {n_never} never, {n_died} died")
        print(f"  Never below ĉ: {n_never_below} / {int(alive.sum())} alive")
        if n_recovered > 0:
            print(f"  Recovered mean final d: {distances[recovered, -1].mean():.6f}")
        if n_never > 0:
            print(f"  Never mean final d:     {distances[never, -1].mean():.6f}")
        print(f"  Orbit band:     ĉ={c_hat:.6f} ± σ={sigma:.6f}")

    return {
        "distances": distances,                       # [N, T]
        "masses": masses,                             # [N, T]
        "outcomes": outcomes,                         # [N] int: 0=died 1=recovered 2=never
        "never_below_chat": never_below,              # [N] bool: never dipped below ĉ
        "ctrl_distances": ctrl_distances,             # [T]
        "positions": all_positions_used,              # list of (x,y)
        "c_bar": c_bar.cpu().numpy(),                 # (m,)
        "c_hat": c_hat,
        "sigma": sigma,
        "recovery_threshold": recovery_threshold,
        "m": m,
        "baseline_dist": baseline_dist,
        "code": creature.code,
        "scale": scale,
        "grid": H,
        "intervention_type": intervention_type,
        "intervention_size": intervention_size,
        "window": window,
        "T": T_creature,
    }


def _outcome_counts(outcomes):
    """Return (n_died, n_recovered, n_never) from outcomes array."""
    return int((outcomes == 0).sum()), int((outcomes == 1).sum()), int((outcomes == 2).sum())


def _trailing_mean(arr, window=20):
    """Trailing rolling mean along last axis. arr: (..., T) → (..., T).

    For t < window, averages over frames 0..t (expanding window at the start).
    """
    cs = np.cumsum(arr, axis=-1)
    out = np.empty_like(arr)
    for t in range(arr.shape[-1]):
        t0 = max(0, t - window + 1)
        if t0 == 0:
            out[..., t] = cs[..., t] / (t + 1)
        else:
            out[..., t] = (cs[..., t] - cs[..., t0 - 1]) / window
    return out


def _first_crossing(trace, threshold):
    """Index of first timestep where trace drops below threshold, or -1."""
    below = np.where(trace < threshold)[0]
    return int(below[0]) if len(below) > 0 else -1


SMOOTH_WINDOW = 5
CROSSING_SKIP = 5   # ignore first N plot-frames when detecting mean crossing


def plot_recovery(results: dict, output_dir: Path):
    """Plot d(x, c̄) timeseries: mean + envelope, orbit band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_distances = results["distances"]      # [N, T]
    ctrl_dist = results["ctrl_distances"]     # [T]
    outcomes = results["outcomes"]            # [N]
    c_hat = results["c_hat"]
    sigma = results["sigma"]
    n_died, n_rec, n_never = _outcome_counts(outcomes)

    # Never-below-ĉ count (alive trajectories that never dip below ĉ)
    never_below = results.get("never_below_chat")
    if never_below is not None:
        n_never_below = int(never_below.sum())
    else:
        # Recompute from distances if loaded from older results
        alive_all = outcomes != 0
        n_never_below = int((alive_all & ~(all_distances < c_hat).any(axis=1)).sum())

    # Filter out dead trajectories so they don't contaminate mean/envelope
    alive_mask = outcomes != 0
    raw_distances = all_distances[alive_mask]
    N, T = raw_distances.shape
    timesteps = np.arange(T)

    # Smooth: trailing rolling mean over SMOOTH_WINDOW frames, trim warmup
    distances = _trailing_mean(raw_distances, SMOOTH_WINDOW)[:, SMOOTH_WINDOW:]
    ctrl_smooth = _trailing_mean(ctrl_dist[np.newaxis, :], SMOOTH_WINDOW)[0, SMOOTH_WINDOW:]
    T_plot = distances.shape[1]
    timesteps = np.arange(SMOOTH_WINDOW, SMOOTH_WINDOW + T_plot)

    recovery_threshold = results.get("recovery_threshold", c_hat + 3 * sigma)

    fig, ax = plt.subplots(figsize=(10, 5))

    band_top = recovery_threshold

    # Per-curve first-crossing times (after CROSSING_SKIP) — computed before
    # subsampling so first_idx/last_idx can be pinned into indices below
    band_t = np.full(N, -1, dtype=int)   # first crossing of ĉ+2σ per curve
    for j in range(N):
        tc = _first_crossing(distances[j, CROSSING_SKIP:], band_top)
        if tc >= 0:
            band_t[j] = tc + CROSSING_SKIP

    # Select first & last curves to hit ĉ+2σ
    crossed_band = band_t >= 0
    if crossed_band.any():
        crossed_indices = np.where(crossed_band)[0]
        first_idx = crossed_indices[band_t[crossed_band].argmin()]
        last_idx = crossed_indices[band_t[crossed_band].argmax()]
    else:
        first_idx, last_idx = -1, -1

    # Individual traces (subsample if too many), pinning first/last crossing curves
    max_traces = 50
    if N > max_traces:
        indices = np.random.default_rng(42).choice(N, max_traces, replace=False)
        for pin in (first_idx, last_idx):
            if pin >= 0 and pin not in indices:
                indices = np.append(indices, pin)
    else:
        indices = np.arange(N)
    for i in indices:
        ax.plot(timesteps, distances[i], color="steelblue", alpha=0.08, linewidth=0.5)

    # Mean + 95% CI of mean (of smoothed traces)
    mean_d = distances.mean(axis=0)
    se = distances.std(axis=0) / np.sqrt(N)
    ci95_lo = mean_d - 1.96 * se
    ci95_hi = mean_d + 1.96 * se

    ax.fill_between(timesteps, ci95_lo, ci95_hi, alpha=0.35, color="navy", label="95% CI of mean")
    ax.plot(timesteps, mean_d, color="navy", linewidth=2, label=f"mean (N={N})")

    # Mean curve crossings — dots sit ON the blue line
    mean_band_t = _first_crossing(mean_d[CROSSING_SKIP:], band_top)
    if mean_band_t >= 0:
        mean_band_t += CROSSING_SKIP

    def _plot_crossing_dots(a):
        """Plot crossing dots: first/last curve + mean curve × ĉ+2σ threshold."""
        # First curve to hit ĉ+2σ
        if first_idx >= 0:
            tb = band_t[first_idx]
            a.plot(timesteps[tb], band_top, 'o', color="gold", markersize=7, zorder=6,
                   markeredgecolor="black", markeredgewidth=0.5,
                   label=f"first ĉ+2σ (t={timesteps[tb]})")
        # Mean curve — y sits on the actual mean line
        if mean_band_t >= 0:
            a.plot(timesteps[mean_band_t], mean_d[mean_band_t], 'o', color="gold",
                   markersize=9, zorder=7, markeredgecolor="black", markeredgewidth=1.5,
                   label=f"mean ĉ+2σ (t={timesteps[mean_band_t]})")
        # Last curve to hit ĉ+2σ
        if last_idx >= 0 and last_idx != first_idx:
            tb = band_t[last_idx]
            a.plot(timesteps[tb], band_top, 'o', color="gold", markersize=7, zorder=6,
                   markeredgecolor="black", markeredgewidth=0.5,
                   label=f"last ĉ+2σ (t={timesteps[tb]})")

    _plot_crossing_dots(ax)

    # Control trajectory
    ax.plot(timesteps, ctrl_smooth, color="green", linewidth=1.5, linestyle="--",
            label="control (unperturbed)")

    # Orbit band: 0 → ĉ+2σ (recovery threshold)
    ax.axhline(c_hat, color="orange", linewidth=1.5, linestyle="-", label=f"ĉ = {c_hat:.5f}")
    ax.axhspan(0, recovery_threshold, alpha=0.15, color="orange",
               label=f"orbit (0 → ĉ+2σ={recovery_threshold:.5f})")

    # Baseline
    ax.axhline(results["baseline_dist"], color="gray", linewidth=1, linestyle=":",
               label=f"baseline d = {results['baseline_dist']:.5f}")

    code = results["code"]
    scale = results["scale"]
    itype = results["intervention_type"]
    isize = results["intervention_size"]

    ax.set_xlabel("Timestep")
    ax.set_ylabel("d(profile, c_bar)")
    ax.set_title(
        f"Recovery: {code} s{scale} -- {isize}x{isize} {itype}\n"
        f"({n_rec} recovered, {n_never} never, {n_died} died, "
        f"{n_never_below} never below ĉ)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, SMOOTH_WINDOW + T_plot - 1)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(output_dir / "recovery_timeseries.png", dpi=150)
    plt.close(fig)

    # Also plot a zoomed version focusing on the orbit band region
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    for i in indices:
        ax2.plot(timesteps, distances[i], color="steelblue", alpha=0.08, linewidth=0.5)

    ax2.fill_between(timesteps, ci95_lo, ci95_hi, alpha=0.35, color="navy", label="95% CI of mean")
    ax2.plot(timesteps, mean_d, color="navy", linewidth=2, label=f"mean (N={N})")
    _plot_crossing_dots(ax2)
    ax2.plot(timesteps, ctrl_smooth, color="green", linewidth=1.5, linestyle="--",
             label="control")
    ax2.axhline(c_hat, color="orange", linewidth=1.5, label=f"ĉ = {c_hat:.5f}")
    ax2.axhspan(0, recovery_threshold, alpha=0.15, color="orange",
                label=f"orbit (0 → ĉ+2σ={recovery_threshold:.5f})")
    ax2.axhline(results["baseline_dist"], color="gray", linewidth=1, linestyle=":",
                label=f"baseline d = {results['baseline_dist']:.5f}")

    # Zoom to ~3x orbit band
    y_top = max(recovery_threshold * 3, mean_d[:min(T_plot, 30)].max() * 1.2)
    ax2.set_ylim(0, y_top)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("d(profile, c_bar)")
    ax2.set_title(
        f"Recovery (zoomed): {code} s{scale}\n"
        f"({n_never_below} / {N} alive never below ĉ)"
    )
    ax2.set_xlim(0, SMOOTH_WINDOW + T_plot - 1)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig2.savefig(output_dir / "recovery_timeseries_zoomed.png", dpi=150)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(
        description="Recovery validation: d(x, c̄) over time after perturbation"
    )
    parser.add_argument("--code", default="O2u", help="Animal code (default: O2u)")
    parser.add_argument("--scale", type=int, default=2, help="Scale factor (default: 2)")
    parser.add_argument("--grid", type=int, default=64, help="Base grid size (default: 64)")
    parser.add_argument("--orbit", type=Path, required=True,
                        help="Path to orbit .pt file (e.g., orbits/O2u/s2/O2u_s2_orbit.pt)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: results/recovery_test/<code>_s<scale>)")
    parser.add_argument("--intervention-type", choices=["erase", "blind_erase", "additive"],
                        default="erase", help="Intervention type (default: erase)")
    parser.add_argument("--size", type=int, default=2,
                        help="Intervention size at base resolution (default: 2)")
    parser.add_argument("--intensity", type=float, default=0.3,
                        help="Intensity for additive intervention (default: 0.3)")
    parser.add_argument("--shortcut", action="store_true",
                        help="Only test non-zero pixel positions")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="GPU batch size (default: auto)")
    parser.add_argument("--window-mult", type=float, default=RECOVERY_TIMEOUT_MULTIPLIER,
                        help=f"Rollout = T * this (default: {RECOVERY_TIMEOUT_MULTIPLIER})")
    parser.add_argument("--rotation", type=float, default=0.0,
                        help="Rotate creature by this angle (degrees) before settling (default: 0)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    # Load creature
    creatures = load_animals(Path("animals.json"), codes=[args.code])
    if not creatures:
        raise SystemExit(f"No animal found with code '{args.code}'")
    creature = creatures[0]

    # Rotate creature cells if requested
    if args.rotation != 0.0:
        creature = copy.deepcopy(creature)
        device = _auto_device()
        cells = torch.as_tensor(creature.cells, device=device, dtype=torch.float32)
        creature.cells = rotate_tensor(cells, args.rotation, device).cpu().numpy()

    # Load orbit data
    if not args.orbit.exists():
        raise SystemExit(f"Orbit file not found: {args.orbit}")
    orbit_data = torch.load(args.orbit, weights_only=False)

    # Config
    lenia_cfg = Config.from_animal(creature, base_grid=args.grid, scale=args.scale)
    actual_size = args.size * args.scale

    # Auto batch size
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        window = int(args.window_mult * lenia_cfg.timescale_T)
        actual_grid = args.grid * args.scale
        batch_size = estimate_batch_size(actual_grid, window, lenia_cfg.device)

    # Output dir
    if args.output_dir is None:
        output_dir = Path(f"results/recovery_test/{args.code}_s{args.scale}")
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet

    if verbose:
        print(f"{'='*60}")
        rot_str = f" rotation={args.rotation}°" if args.rotation != 0.0 else ""
        print(f"RECOVERY TEST: {creature.name} ({creature.code}){rot_str}")
        print(f"{'='*60}")

    t0 = time.time()

    results = run_recovery_test(
        creature, lenia_cfg, orbit_data,
        intervention_type=args.intervention_type,
        intervention_size=actual_size,
        intensity=args.intensity,
        shortcut=args.shortcut,
        batch_size=batch_size,
        window_mult=args.window_mult,
        verbose=verbose,
    )

    if not results:
        return

    elapsed = time.time() - t0

    # Save results
    save_path = output_dir / "recovery_data.npz"
    np.savez(
        save_path,
        distances=results["distances"],
        masses=results["masses"],
        outcomes=results["outcomes"],
        never_below_chat=results["never_below_chat"],
        ctrl_distances=results["ctrl_distances"],
        c_bar=results["c_bar"],
    )
    # JSON sidecar with scalars and metadata
    meta = {k: v for k, v in results.items()
            if isinstance(v, (int, float, str, list))}
    meta["positions"] = [(int(x), int(y)) for x, y in results["positions"]]
    (output_dir / "recovery_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    if verbose:
        print(f"\nSaved {save_path}")
        print(f"Saved {output_dir / 'recovery_meta.json'}")
        print(f"Total time: {elapsed:.1f}s")

    # Plot
    if not args.no_plot:
        plot_recovery(results, output_dir)
        if verbose:
            print(f"Saved {output_dir / 'recovery_timeseries.png'}")
            print(f"Saved {output_dir / 'recovery_timeseries_zoomed.png'}")


if __name__ == "__main__":
    main()
