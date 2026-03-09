"""Barrier environment generators for constrained Lenia simulations.

Each generator creates a binary mask tensor where:
    1.0 = barrier (impassable)
    0.0 = open space

All environments are designed for 128x256 grids by default.
At scale=2 the actual grid is 256x512; creature ~40x40px, kernel radius ~26px.
Barriers thinner than ~26px are semi-permeable (tunneling possible).

When scaled=True, pixel dimensions (wall thickness, gap width, etc.) are
multiplied by a factor relative to REFERENCE_SHAPE so features stay
proportional on larger grids.
"""

import math
from pathlib import Path
from typing import Tuple

import torch


# Reference grid for which hardcoded pixel values were tuned
REFERENCE_SHAPE = (128, 256)


def compute_scale_factor(shape: Tuple[int, int]) -> float:
    """Scale factor relative to reference 128x256 grid."""
    return min(shape[0] / REFERENCE_SHAPE[0], shape[1] / REFERENCE_SHAPE[1])


def _s(value: float, sf: float) -> int:
    """Scale a pixel dimension and round to int (at least 1)."""
    return max(1, int(round(value * sf)))


def _meshgrid(shape, device, dtype):
    """Return (yy, xx) coordinate grids."""
    h, w = shape
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    return torch.meshgrid(y, x, indexing='ij')


def _polar(shape, device, dtype, cy=None, cx=None):
    """Return (dist, theta, yy, xx) relative to center."""
    h, w = shape
    if cy is None:
        cy = h / 2
    if cx is None:
        cx = w / 2
    yy, xx = _meshgrid(shape, device, dtype)
    dy = yy - cy
    dx = xx - cx
    dist = torch.sqrt(dy ** 2 + dx ** 2)
    theta = torch.atan2(dy, dx)
    return dist, theta, yy, xx


def make_box(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    border_width: int = 8,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Rectangular perimeter wall."""
    h, w = shape
    bw = _s(border_width, scale_factor)
    mask = torch.zeros(shape, dtype=dtype, device=device)
    mask[:bw, :] = 1.0
    mask[-bw:, :] = 1.0
    mask[:, :bw] = 1.0
    mask[:, -bw:] = 1.0
    return mask


def make_pegs(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    radius: int = 4,
    spacing: int = 40,
    jitter: float = 0.4,
    seed: int = 42,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Grid of small circular obstacles with randomized positions."""
    h, w = shape
    r = _s(radius, scale_factor)
    sp = _s(spacing, scale_factor)
    mask = torch.zeros(shape, dtype=dtype, device=device)
    yy, xx = _meshgrid(shape, device, dtype)
    rng = torch.Generator().manual_seed(seed)
    max_offset = int(sp * jitter)

    for cy in range(sp // 2, h, sp):
        for cx in range(sp // 2, w, sp):
            jy = torch.randint(-max_offset, max_offset + 1, (1,), generator=rng).item()
            jx = torch.randint(-max_offset, max_offset + 1, (1,), generator=rng).item()
            dist_sq = (yy - (cy + jy)) ** 2 + (xx - (cx + jx)) ** 2
            mask = torch.where(dist_sq <= r ** 2, torch.ones_like(mask), mask)
    return mask


def make_maze(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    num_walls: int = 3,
    wall_thickness: int = 4,
    gap_width: int = 25,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Horizontal barriers with staggered gaps."""
    h, w = shape
    wt = _s(wall_thickness, scale_factor)
    gw = _s(gap_width, scale_factor)
    mask = torch.zeros(shape, dtype=dtype, device=device)
    wall_spacing = h // (num_walls + 1)

    for i in range(num_walls):
        wall_y = wall_spacing * (i + 1)
        y_start = max(0, wall_y - wt // 2)
        y_end = min(h, wall_y + wt // 2 + wt % 2)
        mask[y_start:y_end, :] = 1.0

        if i % 3 == 0:
            gap_center = w // 4
        elif i % 3 == 1:
            gap_center = w // 2
        else:
            gap_center = 3 * w // 4

        gap_start = max(0, gap_center - gw // 2)
        gap_end = min(w, gap_center + gw // 2)
        mask[y_start:y_end, gap_start:gap_end] = 0.0
    return mask


def make_membrane_wall(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    thickness: int = 1,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Single vertical wall at x=W/3 with specified thickness.

    Systematic thickness sweep to probe the tunneling threshold:
    - 1px: trivially permeable
    - 3px: sub-kernel, should tunnel easily
    - 5px: still sub-kernel but thicker
    """
    h, w = shape
    t = _s(thickness, scale_factor)
    mask = torch.zeros(shape, dtype=dtype, device=device)
    wall_x = w // 3
    x_start = max(0, wall_x - t // 2)
    x_end = min(w, wall_x + (t + 1) // 2)
    mask[:, x_start:x_end] = 1.0
    return mask


def make_funnel(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    mouth_width: int = 200,
    exit_width: int = 35,
    wall_thickness: int = 15,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Diagonal walls converging L-to-R. Wide mouth, narrow exit."""
    h, w = shape
    mw = mouth_width * scale_factor
    ew = exit_width * scale_factor
    wt = wall_thickness * scale_factor
    mask = torch.zeros(shape, dtype=dtype, device=device)
    yy, xx = _meshgrid(shape, device, dtype)

    cy = h / 2
    frac = xx / w  # 0 at left, 1 at right
    half_gap_top = (mw / 2) * (1 - frac) + (ew / 2) * frac
    half_gap_bot = half_gap_top

    top_wall_inner = cy - half_gap_top
    top_wall_outer = top_wall_inner - wt
    bot_wall_inner = cy + half_gap_bot
    bot_wall_outer = bot_wall_inner + wt

    mask = torch.where((yy >= top_wall_outer) & (yy <= top_wall_inner),
                       torch.ones_like(mask), mask)
    mask = torch.where((yy >= bot_wall_inner) & (yy <= bot_wall_outer),
                       torch.ones_like(mask), mask)
    return mask


def make_corridor(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    channel_height: int = 50,
    wall_thickness: int = 15,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Two horizontal walls creating a channel at mid-height."""
    h, w = shape
    ch = _s(channel_height, scale_factor)
    wt = _s(wall_thickness, scale_factor)
    mask = torch.zeros(shape, dtype=dtype, device=device)
    cy = h // 2
    half_ch = ch // 2

    # Upper wall
    mask[cy - half_ch - wt:cy - half_ch, :] = 1.0
    # Lower wall
    mask[cy + half_ch:cy + half_ch + wt, :] = 1.0
    return mask


def make_circular_enclosure(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    radius: int = 60,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Open circle in an otherwise filled grid. Extreme confinement."""
    r = _s(radius, scale_factor)
    dist, _, _, _ = _polar(shape, device, dtype)
    return (dist > r).to(dtype)


ENVIRONMENTS = {
    "pegs": make_pegs,
    "funnel": make_funnel,
    "membrane_1px": lambda s, d, t, scale_factor=1.0: make_membrane_wall(s, d, t, thickness=1, scale_factor=scale_factor),
    "membrane_3px": lambda s, d, t, scale_factor=1.0: make_membrane_wall(s, d, t, thickness=3, scale_factor=scale_factor),
    "membrane_5px": lambda s, d, t, scale_factor=1.0: make_membrane_wall(s, d, t, thickness=5, scale_factor=scale_factor),
    "maze": make_maze,
    "corridor": make_corridor,
    "box": make_box,
    "circular_enclosure": make_circular_enclosure,
}


def make_env(
    name: str,
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    scaled: bool = False,
) -> torch.Tensor:
    """Build environment mask by name.

    If scaled=True, pixel dimensions adapt proportionally to grid size
    relative to the 128x256 reference.
    """
    sf = compute_scale_factor(shape) if scaled else 1.0
    return ENVIRONMENTS[name](shape, device, dtype, scale_factor=sf)


def generate_all_masks(
    shape: Tuple[int, int] = (128, 256),
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    output_dir: Path = None,
    scaled: bool = False,
) -> dict[str, torch.Tensor]:
    """Generate all environment masks and optionally save to disk."""
    if device is None:
        device = torch.device("cpu")

    masks = {}
    for name in ENVIRONMENTS:
        masks[name] = make_env(name, shape, device, dtype, scaled=scaled)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, mask in masks.items():
            torch.save(mask.cpu(), output_dir / f"{name}.pt")
            print(f"Saved {name}.pt")

    return masks


def visualize_masks(masks: dict[str, torch.Tensor], save_path: Path = None):
    """Create a multi-column visualization grid of all masks."""
    import matplotlib.pyplot as plt

    n = len(masks)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 2.5 * nrows))
    axes = axes.flatten()

    for i, (name, mask) in enumerate(masks.items()):
        axes[i].imshow(1.0 - mask.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(name, fontsize=10, fontweight='bold')
        axes[i].set_aspect('equal')
        axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.close()


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    env_dir = script_dir / "environments"

    print(f"Generating {len(ENVIRONMENTS)} barrier environment masks (128x256)...")
    masks = generate_all_masks(
        shape=(128, 256),
        output_dir=env_dir,
    )

    visualize_masks(masks, save_path=env_dir / "preview.png")
    print(f"\nDone! {len(masks)} environments saved to: {env_dir.resolve()}")
