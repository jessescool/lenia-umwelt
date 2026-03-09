"""
Shared trajectory computation helpers for visualization.

All functions operate on numpy arrays. No torch dependency.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def toroidal_diff(a: np.ndarray, b: np.ndarray, period: float) -> np.ndarray:
    """Shortest signed displacement b - a on a torus of given period."""
    d = b - a
    return d - period * np.round(d / period)


def heading_from_centroids(
    centroids: np.ndarray, grid_shape: tuple,
) -> np.ndarray:
    """Compute heading angles from centroid trajectories."""
    H, W = grid_shape
    dr = toroidal_diff(centroids[:, :-1, 0], centroids[:, 1:, 0], H)
    dc = toroidal_diff(centroids[:, :-1, 1], centroids[:, 1:, 1], W)
    return np.arctan2(dr, dc)


def speed_from_centroids(
    centroids: np.ndarray, grid_shape: tuple,
) -> np.ndarray:
    """Compute speed (pixels/frame) from centroid trajectories."""
    H, W = grid_shape
    dr = toroidal_diff(centroids[:, :-1, 0], centroids[:, 1:, 0], H)
    dc = toroidal_diff(centroids[:, :-1, 1], centroids[:, 1:, 1], W)
    return np.sqrt(dr**2 + dc**2)


def rolling_circular_mean(angles: np.ndarray, window: int) -> np.ndarray:
    """Rolling circular mean via sin/cos decomposition."""
    sin_avg = uniform_filter1d(np.sin(angles), size=window, axis=1, mode='nearest')
    cos_avg = uniform_filter1d(np.cos(angles), size=window, axis=1, mode='nearest')
    return np.arctan2(sin_avg, cos_avg)


def heading_deflection(
    test_heading: np.ndarray, ctrl_heading: np.ndarray,
) -> np.ndarray:
    """Wrapped angular difference (test - ctrl) in [-pi, pi]."""
    diff = test_heading - ctrl_heading[np.newaxis, :]
    return diff - 2 * np.pi * np.round(diff / (2 * np.pi))


def displacement_from_ctrl(
    test_centroids: np.ndarray, ctrl_centroids: np.ndarray, grid_shape: tuple,
) -> np.ndarray:
    """Toroidal Euclidean distance from ctrl centroid per frame."""
    H, W = grid_shape
    dr = toroidal_diff(ctrl_centroids[np.newaxis, :, 0], test_centroids[:, :, 0], H)
    dc = toroidal_diff(ctrl_centroids[np.newaxis, :, 1], test_centroids[:, :, 1], W)
    return np.sqrt(dr**2 + dc**2)


