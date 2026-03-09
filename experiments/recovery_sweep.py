"""
Recovery sweep: run recovery_test at many orientations of the same creature.

Tests whether recovery from perturbation (goal-directedness) depends on the
creature's orientation. The orbit c_bar (sorted activation profile barycenter)
is rotation-invariant by construction, so the same orbit file works for all
angles. A baseline stability plot verifies this empirically.

Usage:
    python experiments/recovery_sweep.py \
        --code O2u --scale 2 --grid 64 --shortcut \
        --orbit orbits/O2u/s2/O2u_s2_orbit.pt \
        --output-dir results/recovery_sweep/O2u_s2 \
        --no-per-angle-plots
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RECOVERY_TIMEOUT_MULTIPLIER
from experiments.recovery_test import run_recovery_test, plot_recovery
from substrate import Config, load_animals
from substrate.lenia import _auto_device
from utils.batched import estimate_batch_size
from utils.core import rotate_tensor


def plot_sweep_summary(summary: dict, output_dir: Path):
    """Generate 5 summary plots from the aggregated sweep data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    angles = np.array(summary["angles"])
    recovery_rates = np.array(summary["recovery_rates"])
    death_rates = np.array(summary["death_rates"])
    never_rates = np.array(summary["never_rates"])
    mean_final_dists = np.array(summary["mean_final_dists"])
    baseline_dists = np.array(summary["baseline_dists"])
    n_positions = np.array(summary["n_positions"])

    code = summary["code"]
    scale = summary["scale"]

    angles_rad = np.deg2rad(angles)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    # Close the loop for plotting
    a = np.append(angles_rad, angles_rad[0])
    r = np.append(recovery_rates, recovery_rates[0])
    ax.plot(a, r, color="navy", linewidth=1.5)
    ax.fill(a, r, alpha=0.2, color="navy")
    # Mean overlay
    mean_rate = recovery_rates.mean()
    ax.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, mean_rate),
            color="orange", linewidth=1.5, linestyle="--", label=f"mean={mean_rate:.1%}")
    ax.set_title(f"Recovery rate vs orientation\n{code} s{scale}", pad=20)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "recovery_polar.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(angles, recovery_rates, "o-", color="green", markersize=3,
            linewidth=1, label="recovered")
    ax.plot(angles, death_rates, "o-", color="red", markersize=3,
            linewidth=1, label="died")
    ax.plot(angles, never_rates, "o-", color="gray", markersize=3,
            linewidth=1, label="never recovered")
    # Mean overlays
    ax.axhline(recovery_rates.mean(), color="green", linewidth=1.5,
               linestyle="--", alpha=0.7)
    ax.axhline(death_rates.mean(), color="red", linewidth=1.5,
               linestyle="--", alpha=0.7)
    ax.set_xlabel("Orientation (degrees)")
    ax.set_ylabel("Rate")
    ax.set_title(f"Outcome rates vs orientation -- {code} s{scale}")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "rates_vs_angle.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    n_rec = (recovery_rates * n_positions).astype(int)
    n_died = (death_rates * n_positions).astype(int)
    n_never = n_positions - n_rec - n_died
    bar_width = angles[1] - angles[0] if len(angles) > 1 else 2
    ax.bar(angles, n_rec, width=bar_width * 0.9, color="green", label="recovered")
    ax.bar(angles, n_died, width=bar_width * 0.9, bottom=n_rec,
           color="red", label="died")
    ax.bar(angles, n_never, width=bar_width * 0.9, bottom=n_rec + n_died,
           color="gray", label="never recovered")
    ax.set_xlabel("Orientation (degrees)")
    ax.set_ylabel("Count")
    ax.set_title(f"Outcome counts vs orientation -- {code} s{scale}")
    ax.set_xlim(-bar_width, 360)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "outcome_bars.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    # Mask NaN entries (angles with 0 positions)
    valid = ~np.isnan(mean_final_dists)
    ax.plot(angles[valid], mean_final_dists[valid], "o-", color="steelblue",
            markersize=3, linewidth=1)
    if valid.any():
        ax.axhline(mean_final_dists[valid].mean(), color="navy",
                    linewidth=1.5, linestyle="--",
                    label=f"mean={mean_final_dists[valid].mean():.6f}")
    c_hat = summary.get("c_hat")
    if c_hat is not None:
        ax.axhline(c_hat, color="orange", linewidth=1.5, linestyle="-",
                    label=f"c_hat={c_hat:.6f}")
    ax.set_xlabel("Orientation (degrees)")
    ax.set_ylabel("Mean final d(x, c_bar)")
    ax.set_title(f"Recovery quality vs orientation -- {code} s{scale}")
    ax.set_xlim(0, 360)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "mean_final_dist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(angles, baseline_dists, "o-", color="purple", markersize=3,
            linewidth=1, label="d(settled, c_bar)")
    mean_bl = np.mean(baseline_dists)
    std_bl = np.std(baseline_dists)
    ax.axhline(mean_bl, color="purple", linewidth=1.5, linestyle="--",
               label=f"mean={mean_bl:.6f}")
    ax.fill_between([0, 360], mean_bl - std_bl, mean_bl + std_bl,
                    alpha=0.15, color="purple",
                    label=f"std={std_bl:.6f}")
    if c_hat is not None:
        ax.axhline(c_hat, color="orange", linewidth=1.5, linestyle="-",
                    label=f"c_hat={c_hat:.6f}")
    ax.set_xlabel("Orientation (degrees)")
    ax.set_ylabel("Baseline d(settled, c_bar)")
    ax.set_title(
        f"Baseline stability (rotation invariance check) -- {code} s{scale}\n"
        f"Expect near-constant if sorted profile is rotation-invariant"
    )
    ax.set_xlim(0, 360)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "baseline_stability.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Recovery sweep: test recovery at many creature orientations"
    )
    # Same args as recovery_test.py
    parser.add_argument("--code", default="O2u", help="Animal code (default: O2u)")
    parser.add_argument("--scale", type=int, default=2, help="Scale factor (default: 2)")
    parser.add_argument("--grid", type=int, default=64, help="Base grid size (default: 64)")
    parser.add_argument("--orbit", type=Path, required=True,
                        help="Path to orbit .pt file")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: results/recovery_sweep/<code>_s<scale>)")
    parser.add_argument("--intervention-type", choices=["erase", "blind_erase", "additive"],
                        default="erase")
    parser.add_argument("--size", type=int, default=2,
                        help="Intervention size at base resolution (default: 2)")
    parser.add_argument("--intensity", type=float, default=0.3)
    parser.add_argument("--shortcut", action="store_true",
                        help="Only test non-zero pixel positions")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="GPU batch size (default: auto)")
    parser.add_argument("--window-mult", type=float, default=RECOVERY_TIMEOUT_MULTIPLIER)
    # Sweep-specific args
    parser.add_argument("--angle-step", type=int, default=2,
                        help="Degrees between orientations (default: 2)")
    parser.add_argument("--no-per-angle-plots", action="store_true",
                        help="Skip per-angle recovery PNGs (saves time + disk)")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    # Load creature once
    creatures = load_animals(Path("animals.json"), codes=[args.code])
    if not creatures:
        raise SystemExit(f"No animal found with code '{args.code}'")
    creature_base = creatures[0]

    # Load orbit data once
    if not args.orbit.exists():
        raise SystemExit(f"Orbit file not found: {args.orbit}")
    orbit_data = torch.load(args.orbit, weights_only=False)

    # Config from base creature (orientation doesn't change config)
    lenia_cfg = Config.from_animal(creature_base, base_grid=args.grid, scale=args.scale)
    actual_size = args.size * args.scale
    device = lenia_cfg.device

    # Auto batch size
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        window = int(args.window_mult * lenia_cfg.timescale_T)
        actual_grid = args.grid * args.scale
        batch_size = estimate_batch_size(actual_grid, window, device)

    # Output dir
    if args.output_dir is None:
        output_dir = Path(f"results/recovery_sweep/{args.code}_s{args.scale}")
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet
    angles = list(range(0, 360, args.angle_step))
    n_angles = len(angles)

    if verbose:
        print(f"{'='*60}")
        print(f"RECOVERY SWEEP: {creature_base.name} ({creature_base.code})")
        print(f"  {n_angles} orientations, step={args.angle_step}deg")
        print(f"  grid={args.grid}, scale={args.scale}, batch={batch_size}")
        print(f"  orbit: {args.orbit}")
        print(f"  output: {output_dir}")
        print(f"{'='*60}\n")

    # Per-angle aggregation storage
    sweep_data = {
        "angles": [],
        "recovery_rates": [],
        "death_rates": [],
        "never_rates": [],
        "mean_final_dists": [],
        "baseline_dists": [],
        "n_positions": [],
    }

    t0_sweep = time.time()

    for i, angle in enumerate(angles):
        t0_angle = time.time()

        if verbose:
            elapsed = time.time() - t0_sweep
            if i > 0:
                per_angle = elapsed / i
                eta = per_angle * (n_angles - i)
                eta_str = f"  ETA: {eta/60:.0f}min"
            else:
                eta_str = ""
            print(f"\n{'─'*60}")
            print(f"[{i+1}/{n_angles}] angle={angle}°{eta_str}")
            print(f"{'─'*60}")

        creature = copy.deepcopy(creature_base)
        if angle != 0:
            cells = torch.as_tensor(creature.cells, device=device, dtype=torch.float32)
            creature.cells = rotate_tensor(cells, float(angle), device).cpu().numpy()

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
            # No active positions — record NaNs
            sweep_data["angles"].append(angle)
            sweep_data["recovery_rates"].append(0.0)
            sweep_data["death_rates"].append(0.0)
            sweep_data["never_rates"].append(0.0)
            sweep_data["mean_final_dists"].append(float("nan"))
            sweep_data["baseline_dists"].append(float("nan"))
            sweep_data["n_positions"].append(0)
            continue

        outcomes = results["outcomes"]
        N = len(outcomes)
        n_rec = int((outcomes == 1).sum())
        n_died = int((outcomes == 0).sum())
        n_never = int((outcomes == 2).sum())

        sweep_data["angles"].append(angle)
        sweep_data["recovery_rates"].append(n_rec / N if N > 0 else 0.0)
        sweep_data["death_rates"].append(n_died / N if N > 0 else 0.0)
        sweep_data["never_rates"].append(n_never / N if N > 0 else 0.0)
        sweep_data["mean_final_dists"].append(
            float(results["distances"][:, -1].mean()) if N > 0 else float("nan")
        )
        sweep_data["baseline_dists"].append(results["baseline_dist"])
        sweep_data["n_positions"].append(N)

        # Save per-angle data
        angle_dir = output_dir / f"angle_{angle:03d}"
        angle_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            angle_dir / "recovery_data.npz",
            distances=results["distances"],
            masses=results["masses"],
            outcomes=results["outcomes"],
            ctrl_distances=results["ctrl_distances"],
            c_bar=results["c_bar"],
        )

        # Per-angle plot (optional)
        if not args.no_per_angle_plots:
            plot_recovery(results, angle_dir)

        elapsed_angle = time.time() - t0_angle
        if verbose:
            print(f"  → {n_rec}/{N} recovered ({n_rec/N:.1%}), "
                  f"{n_died} died, {n_never} never  [{elapsed_angle:.1f}s]")

        # Free memory between angles
        del results
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_time = time.time() - t0_sweep

    # Add metadata to summary
    sweep_data["code"] = args.code
    sweep_data["scale"] = args.scale
    sweep_data["grid"] = args.grid
    sweep_data["angle_step"] = args.angle_step
    sweep_data["c_hat"] = float(orbit_data["c_hat"])
    sweep_data["sigma"] = float(orbit_data["sigma"])
    sweep_data["m"] = int(orbit_data["m"])
    sweep_data["total_time_s"] = total_time

    # Save aggregated data
    np.savez(
        output_dir / "sweep_aggregated.npz",
        angles=np.array(sweep_data["angles"]),
        recovery_rates=np.array(sweep_data["recovery_rates"]),
        death_rates=np.array(sweep_data["death_rates"]),
        never_rates=np.array(sweep_data["never_rates"]),
        mean_final_dists=np.array(sweep_data["mean_final_dists"]),
        baseline_dists=np.array(sweep_data["baseline_dists"]),
        n_positions=np.array(sweep_data["n_positions"]),
    )

    # JSON summary (convert numpy types for serialization)
    json_summary = {}
    for k, v in sweep_data.items():
        if isinstance(v, list):
            json_summary[k] = [float(x) if isinstance(x, (np.floating, float)) else int(x)
                               if isinstance(x, (np.integer, int)) else x for x in v]
        else:
            json_summary[k] = v
    (output_dir / "sweep_summary.json").write_text(json.dumps(json_summary, indent=2) + "\n")

    if verbose:
        print(f"\n{'='*60}")
        print(f"SWEEP COMPLETE: {n_angles} orientations in {total_time/60:.1f}min")
        rates = np.array(sweep_data["recovery_rates"])
        print(f"  Recovery rate: {rates.mean():.1%} mean, "
              f"{rates.std():.1%} std, [{rates.min():.1%}, {rates.max():.1%}] range")
        bl = np.array(sweep_data["baseline_dists"])
        valid_bl = bl[~np.isnan(bl)]
        if len(valid_bl) > 0:
            print(f"  Baseline dist: {valid_bl.mean():.6f} mean, "
                  f"{valid_bl.std():.6f} std (rotation invariance check)")
        print(f"{'='*60}")

    # Summary plots
    plot_sweep_summary(sweep_data, output_dir)
    if verbose:
        print(f"\nSaved summary plots to {output_dir}/")
        for name in ["recovery_polar.png", "rates_vs_angle.png",
                      "outcome_bars.png", "mean_final_dist.png",
                      "baseline_stability.png"]:
            print(f"  {name}")
        print(f"Saved sweep_summary.json")
        print(f"Saved sweep_aggregated.npz")
        print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
