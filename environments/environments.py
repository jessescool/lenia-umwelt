# environments/environments.py — Jesse Cool (jessescool)
"""Barrier environment generators for constrained Lenia simulations."""

from __future__ import annotations

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
    clearance: int = 30,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Grid of small circular obstacles with randomized positions.

    Pegs within `clearance` px of grid center are skipped so creatures
    can be spawned there without overlap.
    """
    h, w = shape
    r = _s(radius, scale_factor)
    sp = _s(spacing, scale_factor)
    cl = _s(clearance, scale_factor)
    mask = torch.zeros(shape, dtype=dtype, device=device)
    yy, xx = _meshgrid(shape, device, dtype)
    rng = torch.Generator().manual_seed(seed)
    max_offset = int(sp * jitter)
    center_y, center_x = h / 2, w / 2

    for cy in range(sp // 2, h, sp):
        for cx in range(sp // 2, w, sp):
            jy = torch.randint(-max_offset, max_offset + 1, (1,), generator=rng).item()
            jx = torch.randint(-max_offset, max_offset + 1, (1,), generator=rng).item()
            py, px = cy + jy, cx + jx
            if abs(py - center_y) < cl and abs(px - center_x) < cl:
                continue
            dist_sq = (yy - py) ** 2 + (xx - px) ** 2
            mask = torch.where(dist_sq <= r ** 2, torch.ones_like(mask), mask)
    return mask


def make_membrane_wall(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    thickness: int = 1,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Single vertical wall at x=2W/3 with specified thickness.

    Systematic thickness sweep to probe the tunneling threshold:
    - 1px: trivially permeable
    - 3px: sub-kernel, should tunnel easily
    - 5px: still sub-kernel but thicker
    """
    h, w = shape
    t = _s(thickness, scale_factor)
    mask = torch.zeros(shape, dtype=dtype, device=device)
    wall_x = 2 * w // 3
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
    channel_height: int = 38,
    wall_thickness: int = 15,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Horizontal channel at mid-height — everything above and below is barrier."""
    h, w = shape
    ch = _s(channel_height, scale_factor)
    mask = torch.ones(shape, dtype=dtype, device=device)
    cy = h // 2
    half_ch = ch // 2

    mask[cy - half_ch:cy + half_ch, :] = 0.0
    return mask


def make_chips(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    base: int = 8,
    height: int = 16,
    spacing: int = 45,
    jitter: float = 0.3,
    seed: int = 77,
    clearance: int = 30,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Grid-placed sharp isosceles triangles, well-spaced and non-overlapping."""
    h, w = shape
    b = _s(base, scale_factor)
    ht = _s(height, scale_factor)
    sp = _s(spacing, scale_factor)
    cl = _s(clearance, scale_factor)
    mask = torch.zeros(shape, dtype=dtype, device=device)
    yy, xx = _meshgrid(shape, device, dtype)
    rng = torch.Generator().manual_seed(seed)
    center_y, center_x = h / 2, w / 2
    max_offset = int(sp * jitter)

    for cy in range(sp // 2, h, sp):
        for cx in range(sp // 2, w, sp):
            jy = torch.randint(-max_offset, max_offset + 1, (1,), generator=rng).item()
            jx = torch.randint(-max_offset, max_offset + 1, (1,), generator=rng).item()
            py, px = cy + jy, cx + jx
            if abs(py - center_y) < cl and abs(px - center_x) < cl:
                continue
            angle = torch.rand(1, generator=rng).item() * 2 * math.pi
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            dy = yy - py
            dx = xx - px
            local_x = dx * cos_a + dy * sin_a
            local_y = -dx * sin_a + dy * cos_a
            frac = (local_x + ht / 2) / ht  # 0 at base, 1 at tip
            half_w = (b / 2) * (1 - frac)
            in_tri = (frac >= 0) & (frac <= 1) & (local_y.abs() <= half_w)
            mask = torch.where(in_tri, torch.ones_like(mask), mask)
    return mask


def make_shuriken(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    n_stars: int = 7,
    radius: int = 25,
    valley: float = 0.18,
    seed: int = 55,
    clearance: int = 18,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """5-pointed shuriken shapes: triangular blades radiating from a hub.

    Each star is a small central disk (hub_frac * radius) plus 5 triangular
    blades that taper to sharp points at full radius.  The concave gaps
    between blades are the spaces where no triangle exists.

    valley controls the hub size (fraction of radius).
    """
    h, w = shape
    r = _s(radius, scale_factor)
    cl = _s(clearance, scale_factor)
    mask = torch.zeros(shape, dtype=dtype, device=device)
    yy, xx = _meshgrid(shape, device, dtype)
    rng = torch.Generator().manual_seed(seed)
    center_y, center_x = h / 2, w / 2
    n_verts = 5
    sector = 2 * math.pi / n_verts       # 72° between blades
    blade_half = sector * 0.35            # angular half-width of blade at hub

    placed = []  # (cy, cx) of placed stars
    min_sep = 2 * r + 4  # minimum center-to-center distance (no overlap)
    attempts = 0
    while len(placed) < n_stars and attempts < 200:
        attempts += 1
        cy = torch.randint(r, h - r, (1,), generator=rng).item()
        cx = torch.randint(r, w - r, (1,), generator=rng).item()
        # Keep center open
        if math.hypot(cy - center_y, cx - center_x) < cl + r:
            continue
        if any(math.hypot(cy - py, cx - px) < min_sep for py, px in placed):
            continue
        placed.append((cy, cx))
        spin = torch.rand(1, generator=rng).item() * 2 * math.pi
        dy = yy - cy
        dx = xx - cx
        dist = torch.sqrt(dy ** 2 + dx ** 2)
        theta = torch.atan2(dy, dx) - spin

        # Central hub
        in_star = dist <= r * valley

        # 5 triangular blades
        for i in range(n_verts):
            blade_angle = i * sector
            # Signed angular distance from blade center, wrapped to [-π, π]
            delta = (theta - blade_angle + math.pi) % (2 * math.pi) - math.pi
            # Blade tapers linearly: full half-width at center, zero at tip
            taper = blade_half * (1 - dist / r).clamp(min=0)
            in_blade = (delta.abs() <= taper) & (dist <= r)
            in_star = in_star | in_blade

        mask = torch.where(in_star, torch.ones_like(mask), mask)
    return mask


_GUIDELINES_PNG = Path(__file__).parent / "guidelines_src.png"


def make_guidelines(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    threshold: float = 0.5,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Load guidelines bitmap from PNG, threshold to binary mask."""
    from PIL import Image

    img = Image.open(_GUIDELINES_PNG).convert("L")
    # Crop inward to remove the border frame
    iw, ih = img.size
    margin = int(0.05 * min(iw, ih))
    img = img.crop((margin, margin, iw - margin, ih - margin))
    img = img.resize((shape[1], shape[0]), Image.LANCZOS)
    arr = torch.tensor(list(img.getdata()), dtype=dtype).reshape(shape) / 255.0
    # Dark pixels → barrier (1.0), light → open (0.0)
    mask = (arr < threshold).to(dtype=dtype, device=device)
    return mask


def make_containment_capsule(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    radius_y: int = 55,
    radius_x: int = 110,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Elliptical capsule enclosure — outside is barrier, inside is open."""
    h, w = shape
    ry = _s(radius_y, scale_factor)
    rx = _s(radius_x, scale_factor)
    yy, xx = _meshgrid(shape, device, dtype)
    cy, cx = h / 2, w / 2
    # Ellipse equation: (y/ry)^2 + (x/rx)^2 <= 1
    ellipse = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    return (ellipse > 1.0).to(dtype)


def make_capsule(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    radius_y: int = 45,
    radius_x: int = 80,
    wall_thickness: float = 1.5,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Thin elliptical ring — just the capsule wall, open inside and outside."""
    h, w = shape
    ry = _s(radius_y, scale_factor)
    rx = _s(radius_x, scale_factor)
    wt = _s(wall_thickness, scale_factor)
    yy, xx = _meshgrid(shape, device, dtype)
    cy, cx = h / 2, w / 2
    # Normalized distance from ellipse boundary
    ellipse = torch.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    # Ring where ellipse ≈ 1.0
    # Wall spans from (1 - half) to (1 + half) in normalized units
    half = wt / (2 * min(ry, rx))
    in_wall = (ellipse >= 1.0 - half) & (ellipse <= 1.0 + half)
    return in_wall.to(dtype)


def make_noise(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    density: float = 0.02,
    seed: int = 314,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Random barrier pixels at given density."""
    gen = torch.Generator(device='cpu').manual_seed(seed)
    mask = (torch.rand(shape, generator=gen) < density).to(dtype=dtype, device=device)
    return mask


ENVIRONMENTS = {
    # Row 1: scattered obstacles
    "pegs": lambda s, d, t, scale_factor=1.0: make_pegs(s, d, t, seed=19, scale_factor=scale_factor),
    "chips": make_chips,
    "shuriken": make_shuriken,
    # Row 2: disruption / membranes
    "guidelines": make_guidelines,
    "membrane-1px": lambda s, d, t, scale_factor=1.0: make_membrane_wall(s, d, t, thickness=1, scale_factor=scale_factor),
    "membrane-3px": lambda s, d, t, scale_factor=1.0: make_membrane_wall(s, d, t, thickness=3, scale_factor=scale_factor),
    # Row 3: enclosures
    "box": make_box,
    "capsule": make_containment_capsule,
    "ring": make_capsule,
    # Row 4: channels / noise
    "corridor": make_corridor,
    "funnel": make_funnel,
    "noise": lambda s, d, t, scale_factor=1.0: make_noise(s, d, t, density=0.30, seed=271, scale_factor=scale_factor),
}


def barrier_to_salience(mask: torch.Tensor) -> torch.Tensor:
    """Convert binary barrier mask (1=blind) to salience field (W = 1 - M)."""
    return 1.0 - mask


# Salience environment generators
# Each returns an [H, W] tensor with W=1.0 as background.
# W=0 is blind, W>1 is amplified sensory weighting (excited region).

def make_salience_wall(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    amplitude: float = 2.0,
    thickness: int = 8,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Vertical strip of amplified salience at x=2W/3."""
    h, w = shape
    t = _s(thickness, scale_factor)
    sal = torch.ones(shape, dtype=dtype, device=device)
    wall_x = 2 * w // 3
    x_start = max(0, wall_x - t // 2)
    x_end = min(w, wall_x + (t + 1) // 2)
    sal[:, x_start:x_end] = amplitude
    return sal


def make_salience_box(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    amplitude: float = 2.0,
    border_width: int = 8,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Rectangular perimeter of amplified salience."""
    h, w = shape
    bw = _s(border_width, scale_factor)
    sal = torch.ones(shape, dtype=dtype, device=device)
    sal[:bw, :] = amplitude
    sal[-bw:, :] = amplitude
    sal[:, :bw] = amplitude
    sal[:, -bw:] = amplitude
    return sal


def make_salience_corridor(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    amplitude: float = 2.0,
    channel_height: int = 38,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Horizontal channel of amplified salience at mid-height."""
    h, w = shape
    ch = _s(channel_height, scale_factor)
    sal = torch.ones(shape, dtype=dtype, device=device)
    cy = h // 2
    half_ch = ch // 2
    sal[cy - half_ch:cy + half_ch, :] = amplitude
    return sal


def make_salience_gradient(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    w_low: float = 0.5,
    w_high: float = 2.0,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Linear salience ramp from w_low (left) to w_high (right)."""
    h, w = shape
    # Linearly interpolate across columns
    ramp = torch.linspace(w_low, w_high, w, dtype=dtype, device=device)
    sal = ramp.unsqueeze(0).expand(h, w).contiguous()
    return sal


SALIENCE_ENVIRONMENTS = {
    "salience-wall": make_salience_wall,
    "salience-box": make_salience_box,
    "salience-corridor": make_salience_corridor,
    "salience-gradient": make_salience_gradient,
}


def make_salience_env(
    name: str,
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    scaled: bool = False,
) -> torch.Tensor:
    """Build salience environment by name.

    If scaled=True, pixel dimensions adapt proportionally to grid size
    relative to the 128x256 reference.
    """
    sf = compute_scale_factor(shape) if scaled else 1.0
    return SALIENCE_ENVIRONMENTS[name](shape, device, dtype, scale_factor=sf)


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


def load_env(
    name: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    env_dir: str = "environments/",
) -> torch.Tensor:
    """Load precomputed environment mask from .pt file."""
    path = Path(env_dir) / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Env mask not found: {path}. Run: python environments/make_envs.py"
        )
    return torch.load(path, weights_only=False).to(device=device, dtype=dtype)
