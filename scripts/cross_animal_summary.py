#!/usr/bin/env python3
"""Cross-animal comparison grid: stack per-creature summary PNGs vertically.

Each creature already has a 3-panel summary image (recovery, max distance,
heading) with its own colorbars and scales. This script just tiles them
into one tall image with creature code labels.

Usage:
    python scripts/cross_animal_summary.py --scale 4 --size 1 --ori 0
    python scripts/cross_animal_summary.py                         # all combos at default scale=4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent


def _load_animals() -> list[dict]:
    with open(ROOT / 'animals_to_run.json') as f:
        return json.load(f)['animals']


def _get_font(size: int):
    """Try to load a nice font, fall back to default."""
    for name in ['Helvetica', 'Arial', 'DejaVuSans']:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_grid(scale: int, size: int, ori: int, animals: list[dict]) -> Path | None:
    """Stack existing summary PNGs into one tall comparison image."""
    sweep_root = ROOT / 'results' / 'sweep'
    panels = []  # (code, Image)

    for animal in animals:
        code = animal['code']
        prefix = f'{code}_x{scale}_i{size}_o{ori}'
        summary_path = (sweep_root / code / f'{code}_x{scale}' /
                        f'{code}_x{scale}_i{size}' / prefix /
                        f'{prefix}_summary.png')
        if not summary_path.exists():
            print(f"  Skipping {code} — no summary at {summary_path}")
            continue
        panels.append((code, Image.open(summary_path)))

    if not panels:
        print(f"  No summaries found for scale={scale} size={size} ori={ori}")
        return None

    # Normalize widths to the widest panel
    label_width = 80  # pixels for creature code label on the left
    max_w = max(img.width for _, img in panels)
    gap = 15  # vertical gap between rows

    resized = []
    for code, img in panels:
        if img.width != max_w:
            new_h = int(img.height * max_w / img.width)
            img = img.resize((max_w, new_h), Image.LANCZOS)
        resized.append((code, img))

    total_w = label_width + max_w
    total_h = sum(img.height for _, img in resized) + gap * (len(resized) - 1)

    composite = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(composite)
    font = _get_font(28)

    y = 0
    for code, img in resized:
        # Draw creature label centered vertically in the row
        bbox = draw.textbbox((0, 0), code, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = (label_width - text_w) // 2
        text_y = y + (img.height - text_h) // 2
        draw.text((text_x, text_y), code, fill='black', font=font)

        composite.paste(img, (label_width, y))
        y += img.height + gap

    out_path = sweep_root / f'cross_animal_x{scale}_i{size}_o{ori}.png'
    composite.save(out_path, dpi=(150, 150))
    print(f"  Saved {out_path}")
    return out_path


def _discover_combos(animals: list[dict], scale: int) -> list[tuple[int, int]]:
    """Find all (size, ori) combos that exist on disk for any animal at this scale."""
    sweep_root = ROOT / 'results' / 'sweep'
    combos = set()
    for animal in animals:
        code = animal['code']
        scale_dir = sweep_root / code / f'{code}_x{scale}'
        if not scale_dir.exists():
            continue
        for size_dir in sorted(scale_dir.iterdir()):
            if not size_dir.is_dir():
                continue
            for ori_dir in sorted(size_dir.iterdir()):
                if not ori_dir.is_dir():
                    continue
                parts = ori_dir.name.split('_')
                try:
                    size_val = int([p for p in parts if p.startswith('i')][0][1:])
                    ori_val = int([p for p in parts if p.startswith('o')][0][1:])
                    combos.add((size_val, ori_val))
                except (IndexError, ValueError):
                    continue
    return sorted(combos)


def main():
    parser = argparse.ArgumentParser(
        description='Cross-animal comparison grid from existing summary PNGs')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--size', type=int, default=None,
                        help='Intervention size (omit to loop all)')
    parser.add_argument('--ori', type=int, default=None,
                        help='Orientation index (omit to loop all)')
    args = parser.parse_args()

    animals = _load_animals()

    if args.size is not None and args.ori is not None:
        build_grid(args.scale, args.size, args.ori, animals)
    else:
        combos = _discover_combos(animals, args.scale)
        if args.size is not None:
            combos = [(s, o) for s, o in combos if s == args.size]
        if args.ori is not None:
            combos = [(s, o) for s, o in combos if o == args.ori]

        if not combos:
            print(f"No results found for scale={args.scale}")
            return

        print(f"Found {len(combos)} (size, ori) combo(s) at scale={args.scale}")
        for size_val, ori_val in combos:
            print(f"\n--- size={size_val}, ori={ori_val} ---")
            build_grid(args.scale, size_val, ori_val, animals)


if __name__ == '__main__':
    main()
