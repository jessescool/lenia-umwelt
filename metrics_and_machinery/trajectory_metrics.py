"""Trajectory-level metrics for characterizing Lenia recovery dynamics."""

import torch
import numpy as np
from typing import Sequence, Tuple


def centroid(grid: torch.Tensor, threshold: float = 1e-6) -> Tuple[float, float]:
    """Mass-weighted toroidal centroid for trajectory analysis.

    cf. scaling.py::compute_toroidal_center which uses unweighted active-pixel stats
    (more robust for recentering; this variant is needed for intensity-aware tracking).
    """
    H, W = grid.shape
    total = grid.sum()
    if total < threshold:
        return H / 2.0, W / 2.0

    # Circular statistics for toroidal grids
    row_coords = torch.arange(H, device=grid.device, dtype=grid.dtype)
    col_coords = torch.arange(W, device=grid.device, dtype=grid.dtype)

    # Convert to angles on circle [0, 2pi)
    row_angles = row_coords * (2 * np.pi / H)
    col_angles = col_coords * (2 * np.pi / W)

    # Mass-weighted circular mean for rows
    row_mass = grid.sum(dim=1)  # [H]
    sin_r = (row_mass * torch.sin(row_angles)).sum() / total
    cos_r = (row_mass * torch.cos(row_angles)).sum() / total
    mean_row_angle = torch.atan2(sin_r, cos_r).item()
    if mean_row_angle < 0:
        mean_row_angle += 2 * np.pi
    row_c = mean_row_angle * H / (2 * np.pi)

    # Mass-weighted circular mean for cols
    col_mass = grid.sum(dim=0)  # [W]
    sin_c = (col_mass * torch.sin(col_angles)).sum() / total
    cos_c = (col_mass * torch.cos(col_angles)).sum() / total
    mean_col_angle = torch.atan2(sin_c, cos_c).item()
    if mean_col_angle < 0:
        mean_col_angle += 2 * np.pi
    col_c = mean_col_angle * W / (2 * np.pi)

    return row_c, col_c


def centroid_displacement(
    ctrl_final: torch.Tensor,
    test_final: torch.Tensor,
) -> Tuple[float, float]:
    """Toroidal displacement between control and test centroids."""
    H, W = ctrl_final.shape
    r_ctrl, c_ctrl = centroid(ctrl_final)
    r_test, c_test = centroid(test_final)

    # Toroidal displacement: shortest path on torus
    dr = r_test - r_ctrl
    dc = c_test - c_ctrl

    # Wrap to [-H/2, H/2] and [-W/2, W/2]
    if dr > H / 2:
        dr -= H
    elif dr < -H / 2:
        dr += H
    if dc > W / 2:
        dc -= W
    elif dc < -W / 2:
        dc += W

    return dr, dc


def centroid_batched(grids: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
    """Compute centroids for B grids using circular statistics."""
    B, H, W = grids.shape
    device = grids.device
    dtype = grids.dtype

    total_mass = grids.sum(dim=(1, 2))  # [B]
    safe_mass = total_mass.clamp(min=1e-9)

    # Row coordinates
    row_coords = torch.arange(H, device=device, dtype=dtype)
    row_angles = row_coords * (2 * np.pi / H)

    row_mass = grids.sum(dim=2)  # [B, H]

    # Circular mean for rows
    sin_r = (row_mass * torch.sin(row_angles).unsqueeze(0)).sum(dim=1) / safe_mass  # [B]
    cos_r = (row_mass * torch.cos(row_angles).unsqueeze(0)).sum(dim=1) / safe_mass
    mean_row_angle = torch.atan2(sin_r, cos_r)
    mean_row_angle = torch.where(mean_row_angle < 0, mean_row_angle + 2 * np.pi, mean_row_angle)
    row_c = mean_row_angle * H / (2 * np.pi)

    # Col coordinates
    col_coords = torch.arange(W, device=device, dtype=dtype)
    col_angles = col_coords * (2 * np.pi / W)

    col_mass = grids.sum(dim=1)  # [B, W]

    # Circular mean for cols
    sin_c = (col_mass * torch.sin(col_angles).unsqueeze(0)).sum(dim=1) / safe_mass
    cos_c = (col_mass * torch.cos(col_angles).unsqueeze(0)).sum(dim=1) / safe_mass
    mean_col_angle = torch.atan2(sin_c, cos_c)
    mean_col_angle = torch.where(mean_col_angle < 0, mean_col_angle + 2 * np.pi, mean_col_angle)
    col_c = mean_col_angle * W / (2 * np.pi)

    # Handle dead grids: return grid center
    dead = total_mass < threshold
    row_c = torch.where(dead, torch.tensor(H / 2.0, device=device, dtype=dtype), row_c)
    col_c = torch.where(dead, torch.tensor(W / 2.0, device=device, dtype=dtype), col_c)

    return torch.stack([row_c, col_c], dim=1)  # [B, 2]


def toroidal_displacement_batched(
    centroids_a: torch.Tensor,
    centroids_b: torch.Tensor,
    grid_shape: Tuple[int, int],
) -> torch.Tensor:
    """Toroidal displacement vectors (b - a) for B centroid pairs."""
    H, W = grid_shape
    diff = centroids_b - centroids_a  # [B, 2]

    # Wrap row displacement to [-H/2, H/2]
    diff[:, 0] = torch.where(diff[:, 0] > H / 2, diff[:, 0] - H, diff[:, 0])
    diff[:, 0] = torch.where(diff[:, 0] < -H / 2, diff[:, 0] + H, diff[:, 0])

    # Wrap col displacement to [-W/2, W/2]
    diff[:, 1] = torch.where(diff[:, 1] > W / 2, diff[:, 1] - W, diff[:, 1])
    diff[:, 1] = torch.where(diff[:, 1] < -W / 2, diff[:, 1] + W, diff[:, 1])

    return diff


def heading_angle(
    grid_prev: torch.Tensor,
    grid_curr: torch.Tensor,
) -> float:
    """Heading direction from two consecutive grids (radians, CCW from col-axis)."""
    H, W = grid_curr.shape
    r0, c0 = centroid(grid_prev)
    r1, c1 = centroid(grid_curr)

    # Toroidal displacement
    dr = r1 - r0
    dc = c1 - c0
    if dr > H / 2:
        dr -= H
    elif dr < -H / 2:
        dr += H
    if dc > W / 2:
        dc -= W
    elif dc < -W / 2:
        dc += W

    if abs(dr) < 1e-8 and abs(dc) < 1e-8:
        return 0.0

    return float(np.arctan2(dr, dc))


def speed(
    grid_prev: torch.Tensor,
    grid_curr: torch.Tensor,
) -> float:
    """Toroidal distance traveled between consecutive grid centroids."""
    H, W = grid_curr.shape
    r0, c0 = centroid(grid_prev)
    r1, c1 = centroid(grid_curr)

    dr = r1 - r0
    dc = c1 - c0
    if dr > H / 2:
        dr -= H
    elif dr < -H / 2:
        dr += H
    if dc > W / 2:
        dc -= W
    elif dc < -W / 2:
        dc += W

    return float(np.sqrt(dr**2 + dc**2))


def heading_change(
    ctrl_frames: Sequence[torch.Tensor],
    test_frames: Sequence[torch.Tensor],
    n_avg: int = 5,
) -> float:
    """Heading deflection between test and control at end of recovery."""
    def avg_heading(frames, n):
        """Average heading over last n frames."""
        if len(frames) < n + 1:
            n = len(frames) - 1
        if n < 1:
            return 0.0
        angles = []
        for i in range(-n, 0):
            angles.append(heading_angle(frames[i - 1], frames[i]))
        # Circular mean of angles
        sin_sum = sum(np.sin(a) for a in angles)
        cos_sum = sum(np.cos(a) for a in angles)
        return float(np.arctan2(sin_sum, cos_sum))

    h_ctrl = avg_heading(ctrl_frames, n_avg)
    h_test = avg_heading(test_frames, n_avg)

    # Angular difference in [-pi, pi]
    diff = h_test - h_ctrl
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi

    return diff


def speed_change(
    ctrl_frames: Sequence[torch.Tensor],
    test_frames: Sequence[torch.Tensor],
    n_avg: int = 5,
) -> float:
    """Speed difference between test and control at end of recovery."""
    def avg_speed(frames, n):
        if len(frames) < n + 1:
            n = len(frames) - 1
        if n < 1:
            return 0.0
        speeds = []
        for i in range(-n, 0):
            speeds.append(speed(frames[i - 1], frames[i]))
        return sum(speeds) / len(speeds)

    return avg_speed(test_frames, n_avg) - avg_speed(ctrl_frames, n_avg)


def recovery_directness(distances: Sequence[float]) -> float:
    """How directly does the creature return to its attractor?

    RD = d_initial / integral(|d'(t)|)
    RD ~ 1 means monotonic decay (passive resilience).
    RD ~ 0 means lots of back-and-forth (active resilience).
    """
    if len(distances) < 2:
        return 1.0

    d_initial = distances[0]
    if d_initial < 1e-10:
        return 1.0

    # Total variation of the distance curve = integral of |d'(t)|
    total_path = sum(abs(distances[i+1] - distances[i]) for i in range(len(distances) - 1))

    if total_path < 1e-10:
        return 1.0

    return min(1.0, d_initial / total_path)


def count_peaks(distances: Sequence[float], min_prominence: float = 0.0) -> int:
    """Count local maxima in a distance time series."""
    if len(distances) < 3:
        return 0

    n_peaks = 0
    for i in range(1, len(distances) - 1):
        if distances[i] > distances[i-1] and distances[i] > distances[i+1]:
            if min_prominence > 0:
                prominence = min(distances[i] - distances[i-1], distances[i] - distances[i+1])
                if prominence >= min_prominence:
                    n_peaks += 1
            else:
                n_peaks += 1

    return n_peaks


def time_above_threshold(
    distances: Sequence[float],
    threshold: float,
) -> float:
    """Count timesteps where distance exceeds threshold."""
    return sum(1.0 for d in distances if d > threshold)


def time_above_threshold_integral(
    distances: Sequence[float],
    threshold: float,
) -> float:
    """Area above threshold (trapezoidal integration)."""
    if len(distances) < 2:
        return max(0, distances[0] - threshold) if distances else 0.0

    total = 0.0
    for i in range(len(distances) - 1):
        excess_i = max(0, distances[i] - threshold)
        excess_j = max(0, distances[i+1] - threshold)
        total += (excess_i + excess_j) / 2.0  # trapezoidal rule, dt=1

    return total
