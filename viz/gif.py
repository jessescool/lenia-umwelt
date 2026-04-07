# viz/gif.py — Jesse Cool (jessescool)
"""GIF rendering for Lenia simulations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

if TYPE_CHECKING:
    from utils.analysis import Rollout


def _build_colormap_lut(colormap_name: str) -> np.ndarray:
    """Pre-compute a (256, 3) uint8 lookup table from a matplotlib colormap."""
    cmap = colormaps[colormap_name]
    # Sample colormap at 256 evenly-spaced points in [0, 1]
    indices = np.linspace(0.0, 1.0, 256)
    # cmap() returns (256, 4) RGBA float in [0,1]; take RGB, convert to uint8
    return (cmap(indices)[:, :3] * 255).astype(np.uint8)


def _quantize_and_upscale(rgb_frame: np.ndarray, factor: int) -> Image.Image:
    """Quantize to palette at native res, then upscale."""
    img = Image.fromarray(rgb_frame, mode="RGB")
    # Quantize to 256-color palette at native (small) resolution
    img = img.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
    if factor > 1:
        new_size = (img.width * factor, img.height * factor)
        img = img.resize(new_size, Image.Resampling.NEAREST)
    return img


def _to_rgb_batch(
    frames: "Sequence[torch.Tensor] | torch.Tensor | np.ndarray",
    colormap: str = "magma",
) -> np.ndarray:
    """Convert frames to (T, H, W, 3) uint8 via colormap LUT.

    Accepts Sequence[Tensor], Tensor, ndarray, or Rollout (anything torch.stack
    or np.clip can handle).
    """
    lut = _build_colormap_lut(colormap)
    if isinstance(frames, np.ndarray):
        arr = np.clip(frames, 0.0, 1.0)
    else:
        # Sequence[Tensor] / Tensor / Rollout -- stack if needed
        if isinstance(frames, torch.Tensor):
            arr = frames.clamp(0.0, 1.0).cpu().numpy()
        else:
            arr = torch.stack(list(frames)).clamp(0.0, 1.0).cpu().numpy()
    indices = (arr * 255).astype(np.uint8)  # (T, H, W)
    return lut[indices]  # (T, H, W, 3)


def _upscale_nn(rgb: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbor upscale a single (H, W, 3) frame."""
    if factor <= 1:
        return rgb
    return np.repeat(np.repeat(rgb, factor, axis=0), factor, axis=1)


def _encode_gif(
    rgb_frames: "Sequence[np.ndarray] | np.ndarray",
    path: Path,
    *,
    fps: int = 30,
    upscale: int = 1,
) -> Path:
    """Quantize, optionally upscale, and save an animated GIF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / max(fps, 1))

    # Subsample long trajectories to cap GIF length
    max_frames = 500
    if isinstance(rgb_frames, np.ndarray) and rgb_frames.ndim == 4:
        n = len(rgb_frames)
        stride = max(1, (n + max_frames - 1) // max_frames)  # ceil division
        pil_frames = [_quantize_and_upscale(rgb_frames[i], upscale) for i in range(0, n, stride)]
    else:
        frames_list = list(rgb_frames)
        n = len(frames_list)
        stride = max(1, (n + max_frames - 1) // max_frames)
        pil_frames = [_quantize_and_upscale(f, upscale) for f in frames_list[::stride]]

    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return path


def draw_dot(image: np.ndarray, row: int, col: int, radius: int, color: np.ndarray):
    """Draw a filled circle on an RGB image, toroidal-aware."""
    H, W, _ = image.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                py = int(row + dy) % H
                px = int(col + dx) % W
                image[py, px] = color


def _apply_barrier_overlay_batch(rgb_batch: np.ndarray, barrier_np: np.ndarray) -> np.ndarray:
    """Apply barrier overlay to a (N, H, W, 3) batch in-place and return it."""
    mask = barrier_np > 0.5  # (H, W) bool
    rgb_batch[:, mask, 0] = 200  # R
    rgb_batch[:, mask, 1] = 200  # G
    rgb_batch[:, mask, 2] = 200  # B
    return rgb_batch


def write_gif(
    frames: Sequence[torch.Tensor] | Rollout,
    path: Path,
    *,
    fps: int = 30,
    colormap: str = "magma",
    upscale: int = 1,
    barrier_mask: torch.Tensor | None = None,
    marker_rect: tuple | None = None,
) -> Path:
    """Write a single-panel GIF from (H, W) tensors."""
    rgb_batch = _to_rgb_batch(frames, colormap)

    if barrier_mask is not None:
        barrier_np = barrier_mask.clamp(0.0, 1.0).cpu().numpy()
        rgb_batch = _apply_barrier_overlay_batch(rgb_batch, barrier_np)

    if marker_rect is not None:
        r, c, h, w = marker_rect
        rgb_batch[0, r:r+h, c:c+w] = [255, 40, 40]

    return _encode_gif(rgb_batch, path, fps=fps, upscale=upscale)


def render_centroid_gif(
    frames: np.ndarray,
    test_centroids: np.ndarray,
    ctrl_centroids: np.ndarray,
    path: Path,
    *,
    fps: int = 15,
    upscale: int = 1,
    colormap: str = "magma",
    flash_frames: int = 0,
) -> Path:
    """Single-trajectory GIF with centroid dot overlays."""
    rgb_batch = _to_rgb_batch(frames, colormap)
    T = rgb_batch.shape[0]

    red = np.array([255, 40, 40], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)
    cyan = np.array([0, 255, 255], dtype=np.uint8)
    dot_radius = max(2, upscale)
    ctrl_radius = max(3, upscale + 1)

    composed = []
    for t in range(T):
        rgb = _upscale_nn(rgb_batch[t].copy(), upscale)

        # Test centroid: red during flash, then white
        tr, tc = test_centroids[t]
        test_color = red if t < flash_frames else white
        draw_dot(rgb, int(tr * upscale), int(tc * upscale), dot_radius, test_color)

        # Control centroid (cyan)
        cr, cc = ctrl_centroids[t]
        draw_dot(rgb, int(cr * upscale), int(cc * upscale), ctrl_radius, cyan)
        composed.append(rgb)

    return _encode_gif(composed, path, fps=fps, upscale=1)


def render_convergence_gif(
    ctrl_frames: torch.Tensor | np.ndarray,
    ctrl_centroids: torch.Tensor | np.ndarray,
    recovery_data: dict,
    path: Path,
    fps: int = 8,
    upscale: int = 4,
    flash_frames: int = 0,
) -> Path:
    """Centroid convergence GIF -- dots over control trajectory."""
    rgb_batch = _to_rgb_batch(ctrl_frames, "magma")
    T = rgb_batch.shape[0]

    centroids_np = ctrl_centroids.cpu().numpy() if isinstance(ctrl_centroids, torch.Tensor) else ctrl_centroids
    sim_centroids = recovery_data['centroids']   # [N, T, 2]
    sim_outcomes = recovery_data['outcomes']       # [N]
    N = sim_centroids.shape[0]

    white = np.array([255, 255, 255], dtype=np.uint8)
    orange = np.array([255, 165, 0], dtype=np.uint8)
    red = np.array([255, 40, 40], dtype=np.uint8)
    cyan = np.array([0, 255, 255], dtype=np.uint8)
    dot_radius = max(2, upscale)
    ctrl_radius = max(3, upscale + 1)

    composed = []
    for t in range(T):
        rgb = _upscale_nn(rgb_batch[t].copy(), upscale)

        # Draw sim centroid dots
        for i in range(N):
            r, c = sim_centroids[i, t]
            if t < flash_frames:
                color = red
            else:
                color = white if sim_outcomes[i] == 1 else orange
            draw_dot(rgb, int(r * upscale), int(c * upscale), dot_radius, color)

        # Control centroid (cyan, on top)
        cr, cc = centroids_np[t]
        draw_dot(rgb, int(cr * upscale), int(cc * upscale), ctrl_radius, cyan)
        composed.append(rgb)

    return _encode_gif(composed, path, fps=fps, upscale=1)


def write_side_by_side_gif(
    test_frames: Sequence[torch.Tensor] | Rollout,
    ctrl_frames: Sequence[torch.Tensor] | Rollout,
    kernel: torch.Tensor | None,
    path: Path,
    *,
    fps: int = 15,
    colormap: str = "magma",
    pre_warmup_frames: int,
    post_warmup_frames: int,
    barrier_mask: torch.Tensor | None = None,
    detection_frame: int | None = None,
    upscale: int = 4,
) -> Path:
    """Side-by-side GIF with phase-colored borders.

    Phase colors: Blue = pre-warmup, Red = post-warmup, Green = detection.
    """
    if detection_frame is None:
        detection_frame = pre_warmup_frames + post_warmup_frames

    n_frames = len(test_frames)

    test_rgb = _to_rgb_batch(test_frames, colormap)
    ctrl_rgb = _to_rgb_batch(ctrl_frames, colormap)

    if kernel is not None:
        k_np = kernel.clamp(0.0, 1.0).cpu().numpy()
        k_start = pre_warmup_frames
        k_end = min(detection_frame, n_frames)
        if k_start < k_end:
            overlay_slice = test_rgb[k_start:k_end].astype(np.float32)
            overlay_slice[..., 0] = np.clip(overlay_slice[..., 0] + k_np * 150, 0, 255)
            overlay_slice[..., 1] *= (1.0 - k_np * 0.6)
            overlay_slice[..., 2] *= (1.0 - k_np * 0.6)
            test_rgb[k_start:k_end] = overlay_slice.astype(np.uint8)

    if barrier_mask is not None:
        barrier_np = barrier_mask.clamp(0.0, 1.0).cpu().numpy()
        test_rgb = _apply_barrier_overlay_batch(test_rgb, barrier_np)
        ctrl_rgb = _apply_barrier_overlay_batch(ctrl_rgb, barrier_np)

    phase_borders = [
        (0, pre_warmup_frames, [50, 100, 255]),              # Phase 1: Blue
        (pre_warmup_frames, detection_frame, [255, 50, 50]), # Phase 2: Red
        (detection_frame, n_frames, [50, 255, 100]),          # Phase 3: Green
    ]

    for start, end, color in phase_borders:
        end = min(end, n_frames)
        if start >= end:
            continue
        c = np.array(color, dtype=np.uint8)
        test_rgb[start:end, 0, :] = c
        test_rgb[start:end, -1, :] = c
        ctrl_rgb[start:end, 0, :] = c
        ctrl_rgb[start:end, -1, :] = c
        test_rgb[start:end, :, 0] = c
        test_rgb[start:end, :, -1] = c
        ctrl_rgb[start:end, :, 0] = c
        ctrl_rgb[start:end, :, -1] = c

    combined = np.concatenate([test_rgb, ctrl_rgb], axis=2)

    return _encode_gif(combined, path, fps=fps, upscale=upscale)
