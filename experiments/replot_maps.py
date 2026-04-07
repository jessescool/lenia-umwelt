#!/usr/bin/env python3
# experiments/replot_maps.py — Jesse Cool (jessescool)
"""Re-render map PNGs from saved analysis arrays.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from viz.maps import (
    plot_recovery_status_map,
    plot_max_distance_map,
    plot_relative_heading,
    plot_summary,
)

# Map from .npy stem suffix → GridSearchResult field name
_NPY_TO_FIELD = {
    'recovery_map': 'recovery_map',
    'recovery_status_map': 'recovery_status_map',
    'heading_change': 'heading_change_map',
    'heading_vec_relative': 'heading_vec_relative_map',
    'max_distance': 'max_distance_map',
    'erased_map': 'erased_map',
}

# Parse orientation dir name: {CODE}_x{SCALE}_i{SIZE}_o{ORI}
_ORI_RE = re.compile(r'^(.+?)_x(\d+)_i(\d+)_o(\d+)$')

# Per-creature crop sizes from animals_to_run.json
def _load_crop_map() -> dict[str, int]:
    """Load code → base_grid mapping for crop sizes."""
    manifest = ROOT / 'animals_to_run.json'
    if manifest.exists():
        with open(manifest) as f:
            data = json.load(f)
        return {a['code']: a['crop'] for a in data['animals']}
    return {}

_CROP_MAP = _load_crop_map()


def load_result(analysis_dir: Path, prefix: str) -> SimpleNamespace:
    """Load .npy files into a namespace matching GridSearchResult fields."""
    ns = SimpleNamespace()
    for stem_suffix, field_name in _NPY_TO_FIELD.items():
        npy_path = analysis_dir / f'{prefix}_{stem_suffix}.npy'
        if npy_path.exists():
            setattr(ns, field_name, np.load(npy_path))
        else:
            setattr(ns, field_name, None)
    return ns


def replot_orientation(ori_dir: Path, dry_run: bool = False) -> bool:
    """Replot all maps for one orientation directory. Returns True if plotted."""
    analysis_dir = ori_dir / 'analysis'
    if not analysis_dir.exists():
        return False

    m = _ORI_RE.match(ori_dir.name)
    if not m:
        print(f"  Skipping {ori_dir.name} — can't parse name")
        return False

    code, scale_s, size_s, ori_s = m.groups()
    scale = int(scale_s)
    prefix = ori_dir.name  # e.g. O2u_x4_i1_o0

    if dry_run:
        print(f"  Would replot: {ori_dir}")
        return True

    result = load_result(analysis_dir, prefix)

    # Check we have the minimum required arrays
    if result.recovery_status_map is None:
        print(f"  Skipping {prefix} — missing recovery_status_map")
        return False

    # Crop to creature's natural grid (from animals_to_run.json)
    crop_base = _CROP_MAP.get(code)
    if crop_base is not None:
        crop_size = crop_base * scale
    else:
        # Fallback: no crop info, use full map
        crop_size = None
        H, W = result.recovery_status_map.shape
        crop_base = H // scale if scale > 0 else H

    creature_name = code
    subtitle = f"size={size_s}  scale={scale}  ori={ori_s}"

    # Recovery time map
    if result.recovery_map is not None:
        plot_recovery_status_map(
            result.recovery_status_map, result.recovery_map,
            creature_name, 'Recovery Time Map',
            ori_dir / f'{prefix}_recovery_time_map.png',
            subtitle=subtitle, crop_size=crop_size, crop_grid=crop_base,
        )
        print(f"  Saved {prefix}_recovery_time_map.png")

    # Max distance map
    if result.max_distance_map is not None:
        plot_max_distance_map(
            result.max_distance_map, result.recovery_status_map,
            creature_name,
            ori_dir / f'{prefix}_max_distance_map.png',
            subtitle=subtitle, crop_size=crop_size, crop_grid=crop_base,
        )

    # Heading quiver plot
    if result.heading_vec_relative_map is not None:
        plot_relative_heading(
            [result], creature_name, ori_dir,
            subtitle=subtitle, crop_size=crop_size, crop_grid=crop_base,
            filename=f'{prefix}_relative_heading_map.png',
        )

    # Summary composite
    plot_summary(ori_dir, prefix=prefix)

    return True


def main():
    parser = argparse.ArgumentParser(description='Replot maps from analysis/ .npy files')
    parser.add_argument('--code', type=str, default=None, help='Only replot this creature code')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be replotted')
    args = parser.parse_args()

    sweep_root = ROOT / 'results' / 'sweep'
    if not sweep_root.exists():
        print(f"No sweep results at {sweep_root}")
        return

    # Walk: sweep_root / {CODE} / {CODE}_x{SCALE} / {CODE}_x{SCALE}_i{SIZE} / {CODE}_x{SCALE}_i{SIZE}_o{ORI}
    creature_dirs = sorted(sweep_root.iterdir())
    if args.code:
        creature_dirs = [d for d in creature_dirs if d.name == args.code]

    count = 0
    for creature_dir in creature_dirs:
        if not creature_dir.is_dir() or creature_dir.name.startswith('.'):
            continue
        print(f"\n{'='*60}")
        print(f"Creature: {creature_dir.name}")
        print(f"{'='*60}")

        # Walk all orientation dirs under this creature
        for ori_dir in sorted(creature_dir.rglob('*_o[0-9]*')):
            if not ori_dir.is_dir():
                continue
            if replot_orientation(ori_dir, dry_run=args.dry_run):
                count += 1

    print(f"\n{'='*60}")
    verb = "Would replot" if args.dry_run else "Replotted"
    print(f"{verb} {count} orientation(s)")


if __name__ == '__main__':
    main()
