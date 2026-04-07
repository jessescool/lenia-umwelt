# substrate/scaling.py — Jesse Cool (jessescool)
"""Scaling utilities for resolution-independent creature preparation."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from substrate.animals import Animal
from substrate.lenia import Config


def interpolate_pattern(
    pattern: np.ndarray | torch.Tensor,
    scale: int,
    mode: str = 'nearest',
) -> np.ndarray:
    """Upscale a 2D pattern by integer factor."""
    if scale <= 1:
        # No scaling needed
        if isinstance(pattern, torch.Tensor):
            return pattern.detach().cpu().numpy()
        return np.asarray(pattern)

    # Handle both numpy and torch inputs
    if isinstance(pattern, torch.Tensor):
        t = pattern.detach().cpu().float()
    else:
        t = torch.from_numpy(np.asarray(pattern)).float()

    # Interpolate: expects (N, C, H, W)
    t = t.unsqueeze(0).unsqueeze(0)
    t_up = F.interpolate(t, scale_factor=float(scale), mode=mode)
    return t_up.squeeze().numpy()


def compute_centroid(field: torch.Tensor) -> Tuple[float, float]:
    """Compute center of mass of activations."""
    H, W = field.shape
    total_mass = field.sum()

    if total_mass < 1e-9:
        # No mass - return center
        return H / 2.0, W / 2.0

    row_coords = torch.arange(H, device=field.device, dtype=field.dtype)
    col_coords = torch.arange(W, device=field.device, dtype=field.dtype)

    row_centroid = (field.sum(dim=1) * row_coords).sum() / total_mass
    col_centroid = (field.sum(dim=0) * col_coords).sum() / total_mass

    return float(row_centroid), float(col_centroid)


def compute_median_center(field: torch.Tensor, threshold: float = 0.01) -> Tuple[float, float]:
    """Compute median coordinates of active pixels (robust to bright outliers)."""
    H, W = field.shape
    active = field > threshold

    if not active.any():
        return H / 2.0, W / 2.0

    # Get coordinates of all active pixels
    rows, cols = torch.where(active)

    row_median = float(torch.median(rows.float()).item())
    col_median = float(torch.median(cols.float()).item())

    return row_median, col_median


def compute_toroidal_center(field: torch.Tensor, threshold: float = 0.01) -> Tuple[float, float]:
    """Compute center using circular statistics (handles wrap-around)."""
    def circular_mean(coords: torch.Tensor, size: int) -> float:
        """Circular mean of coordinates on a torus of given size."""
        angles = coords.float() * (2 * np.pi / size)
        mean_angle = torch.atan2(torch.sin(angles).mean(), torch.cos(angles).mean()).item()
        if mean_angle < 0:
            mean_angle += 2 * np.pi
        return mean_angle * size / (2 * np.pi)

    H, W = field.shape
    active = field > threshold

    if not active.any():
        return H / 2.0, W / 2.0

    rows, cols = torch.where(active)
    return circular_mean(rows, H), circular_mean(cols, W)


def recenter_field(
    field: torch.Tensor,
    method: str = "toroidal",
    threshold: float = 0.01,
) -> torch.Tensor:
    """Roll field so creature is at grid center (toroidal)."""
    H, W = field.shape

    if method == "toroidal":
        row_c, col_c = compute_toroidal_center(field, threshold)
    elif method == "median":
        row_c, col_c = compute_median_center(field, threshold)
    else:
        row_c, col_c = compute_centroid(field)

    target_row = H / 2.0
    target_col = W / 2.0

    shift_row = int(round(target_row - row_c))
    shift_col = int(round(target_col - col_c))

    return torch.roll(field, shifts=(shift_row, shift_col), dims=(0, 1))


def prepare_scaled_simulation(
    animal: Animal,
    base_grid: int,
    scale: int = 1,
    settle_steps: int | None = None,
    recenter: bool = True,
):
    """Full upscale-settle-recenter workflow."""
    # Avoid circular import by importing here
    from substrate.simulation import Simulation

    cfg = Config.from_animal(animal, base_grid=base_grid, scale=scale)

    # Create simulation and place creature (place_animal handles pattern scaling)
    sim = Simulation(cfg)
    sim.place_animal(animal, center=True)

    # Default settle steps: 5 * timescale (enough to reach attractor)
    if settle_steps is None:
        settle_steps = int(5 * cfg.timescale_T)

    for _ in range(settle_steps):
        sim.lenia.step()

    if recenter:
        sim.board.replace_tensor(recenter_field(sim.board.tensor))

    return sim
