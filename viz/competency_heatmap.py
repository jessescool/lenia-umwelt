# viz/competency_heatmap.py — Jesse Cool (jessescool)
"""Environment competency heatmaps and comparison figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Ordered creature codes for consistent display
DEFAULT_ORDER = ["O2u", "S1s", "P4al", "K6s", "K4s", "O2v"]


def load_results(result_dir: Path, codes: list[str] | None = None) -> dict:
    """Load competency JSON files from result directory.

    Returns dict: code -> {env_name -> {M_mean, V_mean, F_mean, sigma_M, ...}}
    """
    results = {}
    for json_file in sorted(result_dir.glob("*/*_competency.json")):
        with open(json_file) as f:
            data = json.load(f)
        code = data['code']
        if codes is not None and code not in codes:
            continue
        results[code] = data
    return results


def _build_matrix(results: dict, metric: str, creature_order: list[str],
                  env_order: list[str]) -> np.ndarray:
    """Build [n_creatures, n_envs] matrix for a metric."""
    matrix = np.full((len(creature_order), len(env_order)), np.nan)
    for i, code in enumerate(creature_order):
        if code not in results:
            continue
        envs = results[code]['environments']
        for j, env in enumerate(env_order):
            if env in envs:
                matrix[i, j] = envs[env][metric]
    return matrix


def plot_competency_heatmap(
    results: dict,
    output_path: Path,
    creature_order: list[str] | None = None,
    env_order: list[str] | None = None,
):
    """Main heatmap: creatures × environments, colored by M."""
    if creature_order is None:
        creature_order = [c for c in DEFAULT_ORDER if c in results]
    if env_order is None:
        # Collect all environments across all creatures
        all_envs = set()
        for data in results.values():
            all_envs.update(data['environments'].keys())
        env_order = sorted(all_envs)

    M = _build_matrix(results, 'M_mean', creature_order, env_order)

    fig, ax = plt.subplots(figsize=(max(6, len(env_order) * 1.2), max(3, len(creature_order) * 0.8)))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    # Annotate cells
    for i in range(len(creature_order)):
        for j in range(len(env_order)):
            val = M[i, j]
            if np.isnan(val):
                ax.text(j, i, '—', ha='center', va='center', fontsize=9, color='gray')
            else:
                color = 'white' if val < 0.4 or val > 0.85 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=10,
                        fontweight='bold', color=color)

    ax.set_xticks(range(len(env_order)))
    ax.set_xticklabels(env_order, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(creature_order)))
    ax.set_yticklabels(creature_order, fontsize=10)
    ax.set_title('Competency Score (M): Orbit Residence Fraction', fontsize=12, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label('M (0 = dead/deformed, 1 = fully competent)', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_decomposition(
    results: dict,
    output_path: Path,
    creature_order: list[str] | None = None,
    env_order: list[str] | None = None,
):
    """Side-by-side heatmaps for V (viability) and F (fidelity)."""
    if creature_order is None:
        creature_order = [c for c in DEFAULT_ORDER if c in results]
    if env_order is None:
        all_envs = set()
        for data in results.values():
            all_envs.update(data['environments'].keys())
        env_order = sorted(all_envs)

    V = _build_matrix(results, 'V_mean', creature_order, env_order)
    F = _build_matrix(results, 'F_mean', creature_order, env_order)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(env_order) * 2.2),
                                                    max(3, len(creature_order) * 0.8)))

    for ax, data, title, label in [
        (ax1, V, 'Viability (V)', 'V: frames alive / total'),
        (ax2, F, 'Fidelity (F)', 'F: frames in-orbit / alive'),
    ]:
        im = ax.imshow(data, cmap=plt.cm.RdYlGn, vmin=0, vmax=1, aspect='auto')
        for i in range(len(creature_order)):
            for j in range(len(env_order)):
                val = data[i, j]
                if np.isnan(val):
                    ax.text(j, i, '—', ha='center', va='center', fontsize=9, color='gray')
                else:
                    color = 'white' if val < 0.4 or val > 0.85 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)
        ax.set_xticks(range(len(env_order)))
        ax.set_xticklabels(env_order, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(creature_order)))
        ax.set_yticklabels(creature_order, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
        cbar.set_label(label, fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_sigma_heatmap(
    results: dict,
    output_path: Path,
    creature_order: list[str] | None = None,
    env_order: list[str] | None = None,
):
    """Heatmap of orientation sensitivity (sigma_M)."""
    if creature_order is None:
        creature_order = [c for c in DEFAULT_ORDER if c in results]
    if env_order is None:
        all_envs = set()
        for data in results.values():
            all_envs.update(data['environments'].keys())
        env_order = sorted(all_envs)

    sigma = _build_matrix(results, 'sigma_M', creature_order, env_order)

    fig, ax = plt.subplots(figsize=(max(6, len(env_order) * 1.2), max(3, len(creature_order) * 0.8)))

    im = ax.imshow(sigma, cmap='YlOrRd', vmin=0, vmax=0.3, aspect='auto')

    for i in range(len(creature_order)):
        for j in range(len(env_order)):
            val = sigma[i, j]
            if np.isnan(val):
                ax.text(j, i, '—', ha='center', va='center', fontsize=9, color='gray')
            else:
                color = 'white' if val > 0.2 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9, color=color)

    ax.set_xticks(range(len(env_order)))
    ax.set_xticklabels(env_order, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(creature_order)))
    ax.set_yticklabels(creature_order, fontsize=10)
    ax.set_title('Orientation Sensitivity (σ_M)', fontsize=12, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label('σ_M (higher = more heading-dependent)', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_radar(
    results: dict,
    output_path: Path,
    creature_order: list[str] | None = None,
    env_order: list[str] | None = None,
):
    """Per-creature radar plots of M across environments."""
    if creature_order is None:
        creature_order = [c for c in DEFAULT_ORDER if c in results]
    if env_order is None:
        all_envs = set()
        for data in results.values():
            all_envs.update(data['environments'].keys())
        env_order = sorted(all_envs)

    n_creatures = len(creature_order)
    ncols = min(3, n_creatures)
    nrows = (n_creatures + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                              subplot_kw=dict(polar=True))
    if n_creatures == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    n_envs = len(env_order)
    angles = np.linspace(0, 2 * np.pi, n_envs, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    colors = plt.cm.Set2(np.linspace(0, 1, n_creatures))

    for idx, code in enumerate(creature_order):
        ax = axes[idx]
        if code not in results:
            ax.set_visible(False)
            continue

        envs = results[code]['environments']
        values = [envs.get(e, {}).get('M_mean', 0) for e in env_order]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(env_order, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(code, fontsize=12, fontweight='bold', pad=15)

    for idx in range(n_creatures, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate competency heatmaps")
    parser.add_argument("result_dir", type=Path,
                        help="Root results directory (e.g., results/env_competency)")
    parser.add_argument("--codes", nargs="+", default=None,
                        help="Creature codes to include (default: all found)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: result_dir)")
    args = parser.parse_args()

    results = load_results(args.result_dir, codes=args.codes)
    if not results:
        print(f"No competency results found in {args.result_dir}")
        sys.exit(1)

    out_dir = args.output or args.result_dir
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    creature_order = [c for c in DEFAULT_ORDER if c in results]
    # Add any codes not in default order
    for c in sorted(results.keys()):
        if c not in creature_order:
            creature_order.append(c)

    print(f"Found results for: {', '.join(creature_order)}")

    plot_competency_heatmap(results, out_dir / "competency_heatmap.png",
                            creature_order=creature_order)
    plot_decomposition(results, out_dir / "competency_decomposition.png",
                       creature_order=creature_order)
    plot_sigma_heatmap(results, out_dir / "competency_sigma.png",
                       creature_order=creature_order)
    plot_radar(results, out_dir / "competency_radar.png",
               creature_order=creature_order)

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
