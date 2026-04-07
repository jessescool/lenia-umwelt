# viz/maps.py — Jesse Cool (jessescool)
"""Spatial heatmaps and vector field plots for Lenia grid search results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import numpy as np

from viz import (
    STATUS_EMPTY, STATUS_DIED, STATUS_RECOVERED, STATUS_NEVER,
    COLOR_GREEN, COLOR_YELLOW, COLOR_ORANGE, COLOR_RED, COLOR_PURPLE, COLOR_BLUE,
    COLOR_BG,
    HEX_GREEN, HEX_YELLOW, HEX_ORANGE, HEX_RED, HEX_PURPLE, HEX_BLUE,
    _FIG_SIZE, _DPI, _CBAR_KWARGS,
)

# Power-law exponent for all maps (0.5 = sqrt)
_GAMMA = 0.5


def _center_crop(arr, crop_size):
    """Crop array to crop_size x crop_size centered box. Returns original if crop_size >= grid."""
    H, W = arr.shape[:2]
    if crop_size is None or crop_size >= min(H, W):
        return arr
    r0 = (H - crop_size) // 2
    c0 = (W - crop_size) // 2
    return arr[r0:r0+crop_size, c0:c0+crop_size]


def _make_norm(vmin, vmax):
    """Build a PowerNorm from the shared _GAMMA setting."""
    return PowerNorm(gamma=_GAMMA, vmin=vmin, vmax=vmax, clip=True)


def _overlay_status(ax, status, H, W):
    """Draw died / never-recovered / empty overlays on top of a gradient imshow."""
    overlay = np.zeros((H, W, 4), dtype=np.float32)

    died = status == STATUS_DIED
    overlay[died] = [*COLOR_RED, 1.0]

    never = status == STATUS_NEVER
    overlay[never] = [*COLOR_PURPLE, 1.0]

    ax.imshow(overlay, interpolation='nearest')


def plot_recovery_status_map(
    recovery_status_map: np.ndarray,
    recovery_map: np.ndarray,
    creature_name: str,
    title_suffix: str,
    save_path: Path,
    subtitle: str = "",
    crop_size: int | None = None,
    crop_grid: int | None = None,
) -> Path:
    """Render a single recovery status map.

    Each pixel is colored by outcome:
      - Green->Yellow gradient: recovered (green=fast, yellow=slow)
      - Red: died
      - Purple: never recovered
      - Transparent: empty (no creature mass)

    Uses percentile clipping + PowerNorm for visible gradients.
    """
    save_path = Path(save_path)
    recovery_status_map = _center_crop(recovery_status_map, crop_size)
    recovery_map = _center_crop(recovery_map, crop_size)
    H, W = recovery_map.shape

    recovered_mask = recovery_status_map == STATUS_RECOVERED
    recovered_values = recovery_map[recovered_mask]

    if len(recovered_values) > 0:
        vmin = np.percentile(recovered_values, 2)
        vmax = np.percentile(recovered_values, 98)
        if vmax <= vmin:
            vmin, vmax = recovered_values.min(), recovered_values.max()
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0, 1

    # Float array: recovered pixels get their value, everything else is NaN
    display = np.full((H, W), np.nan, dtype=np.float32)
    display[recovered_mask] = recovery_map[recovered_mask]

    cmap_gradient = LinearSegmentedColormap.from_list('recovery', [COLOR_GREEN, COLOR_YELLOW])
    cmap_gradient.set_bad(color=COLOR_BG, alpha=0)  # transparent for NaN
    norm = _make_norm(vmin, vmax)

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    ax.set_facecolor(COLOR_BG)
    ax.imshow(display, cmap=cmap_gradient, norm=norm, interpolation='nearest')
    _overlay_status(ax, recovery_status_map, H, W)

    title = f'{creature_name}\n{title_suffix}'
    if subtitle:
        title += f'\n{subtitle}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap=cmap_gradient, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, **_CBAR_KWARGS)
    cbar.set_label('TTR (sim steps / T)', fontsize=10)

    legend_elements = [
        mpatches.Patch(facecolor=HEX_GREEN, edgecolor='black', label='Fast recovery'),
        mpatches.Patch(facecolor=HEX_YELLOW, edgecolor='black', label='Slow recovery'),
        mpatches.Patch(facecolor=HEX_PURPLE, edgecolor='black', label='Never recovered'),
        mpatches.Patch(facecolor=HEX_RED, edgecolor='black', label='Died'),
        mpatches.Patch(facecolor=COLOR_BG, edgecolor='black', label='Empty (no mass)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

    footer = f'p2/p98 clip, gamma={_GAMMA}'
    if crop_grid is not None:
        footer += f'  |  creature grid: {crop_grid}\u00d7{crop_grid}'
    fig.text(0.5, 0.01, footer, ha='center', fontsize=7, color='gray')
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(save_path, dpi=_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    return save_path


def plot_max_distance_map(
    max_distance_map: np.ndarray,
    recovery_status_map: np.ndarray,
    creature_name: str,
    save_path: Path,
    subtitle: str = "",
    crop_size: int | None = None,
    crop_grid: int | None = None,
) -> Path:
    """Render peak Wasserstein distance reached after perturbation.

    Recovered pixels colored by max distance (viridis), red=died,
    purple=never recovered, transparent=empty.

    Uses percentile clipping + PowerNorm for visible gradients.
    """
    save_path = Path(save_path)
    max_distance_map = _center_crop(max_distance_map, crop_size)
    recovery_status_map = _center_crop(recovery_status_map, crop_size)
    H, W = max_distance_map.shape

    recovered_mask = recovery_status_map == STATUS_RECOVERED
    recovered_vals = max_distance_map[recovered_mask]

    if len(recovered_vals) > 0:
        vmin = np.percentile(recovered_vals, 2)
        vmax = np.percentile(recovered_vals, 98)
        if vmax <= vmin:
            vmin, vmax = recovered_vals.min(), recovered_vals.max()
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0, 1

    display = np.full((H, W), np.nan, dtype=np.float32)
    display[recovered_mask] = max_distance_map[recovered_mask]

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color=COLOR_BG, alpha=0)
    norm = _make_norm(vmin, vmax)

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    ax.set_facecolor(COLOR_BG)
    ax.imshow(display, cmap=cmap, norm=norm, interpolation='nearest')
    _overlay_status(ax, recovery_status_map, H, W)

    title = f'{creature_name}\nMax Distance Map'
    if subtitle:
        title += f'\n{subtitle}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, **_CBAR_KWARGS)
    cbar.set_label('Peak Wasserstein distance', fontsize=10)

    legend_elements = [
        mpatches.Patch(facecolor=HEX_PURPLE, edgecolor='black', label='Never recovered'),
        mpatches.Patch(facecolor=HEX_RED, edgecolor='black', label='Died'),
        mpatches.Patch(facecolor=COLOR_BG, edgecolor='black', label='Empty (no mass)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

    footer = f'p2/p98 clip, gamma={_GAMMA}'
    if crop_grid is not None:
        footer += f'  |  creature grid: {crop_grid}\u00d7{crop_grid}'
    fig.text(0.5, 0.01, footer, ha='center', fontsize=7, color='gray')
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(save_path, dpi=_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved max distance map to {save_path}")

    return save_path


def plot_relative_heading(
    all_results: list,
    creature_name: str,
    output_dir: Path,
    subtitle: str = "",
    crop_size: int | None = None,
    crop_grid: int | None = None,
    filename: str = "relative_heading_map.png",
) -> None:
    """Quiver plot of heading change relative to control.

    Arrows scaled by deflection magnitude, colored by deflection angle.
    Uses percentile clipping + PowerNorm for visible color gradients.
    Output: relative_heading.png.
    """
    result = all_results[0]
    status = _center_crop(result.recovery_status_map, crop_size)
    H, W = status.shape

    alive_mask = status == STATUS_RECOVERED
    if not alive_mask.any():
        return

    ys, xs = np.where(alive_mask)
    vec_map = _center_crop(result.heading_vec_relative_map, crop_size)

    vr = vec_map[alive_mask, 0]  # row component
    vc = vec_map[alive_mask, 1]  # col component
    magnitudes = np.sqrt(vr**2 + vc**2)

    if magnitudes.max() < 1e-6:
        return

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    ax.set_facecolor(COLOR_BG)

    # Creature silhouette
    tested = status >= 0
    silhouette = np.full((H, W, 4), 0.0, dtype=np.float32)
    silhouette[tested, :3] = 0.9
    silhouette[tested, 3] = 0.3
    ax.imshow(silhouette, interpolation='nearest')

    # Normalize arrows to unit length
    safe_mag = np.where(magnitudes > 1e-8, magnitudes, 1.0)
    vr_norm = vr / safe_mag
    vc_norm = vc / safe_mag
    zero_mask = magnitudes < 1e-8
    vr_norm[zero_mask] = 0.0
    vc_norm[zero_mask] = 0.0

    # |u1 - u2| = 2*sin(theta/2) => theta = 2*arcsin(mag/2)
    deflection_deg = np.degrees(2 * np.arcsin(np.clip(magnitudes, 0, 2) / 2))

    # Fixed 0–180° scale for cross-creature comparability
    max_deg = 180.0
    clipped_deg = np.clip(deflection_deg, 0, max_deg)

    # Color encodes deflection magnitude; arrows are uniform size (direction only)
    norm = PowerNorm(gamma=0.3, vmin=0.5, vmax=180.0, clip=True)

    _heading_cmap = plt.cm.turbo
    colors = _heading_cmap(norm(clipped_deg))

    rel_arrow_scale = max(H, W) / 1.5 / 0.65

    ax.quiver(
        xs, ys, vc_norm, -vr_norm,
        color=colors, scale=rel_arrow_scale, scale_units='width',
        width=0.003, headwidth=3, headlength=4, alpha=0.8,
    )

    sm = plt.cm.ScalarMappable(cmap=_heading_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, **_CBAR_KWARGS)
    cbar.set_label('\u0394 heading (degrees)', fontsize=10)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    title = f'{creature_name}\nHeading Change (vs control)'
    if subtitle:
        title += f'\n{subtitle}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    actual_max = deflection_deg.max() if len(deflection_deg) > 0 else 0.0
    footer = f'actual max={actual_max:.1f}\u00b0, scale=0\u2013180\u00b0, gamma=0.3'
    if crop_grid is not None:
        footer += f'  |  creature grid: {crop_grid}\u00d7{crop_grid}'
    fig.text(0.5, 0.01, footer, ha='center', fontsize=7, color='gray')
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(output_dir / filename, dpi=_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved {filename} to {output_dir}/{filename}")


def plot_summary(output_dir: Path, prefix: str = "", dpi: int = _DPI) -> Path | None:
    """Tile the 3 sweep maps into a single horizontal composite image."""
    from PIL import Image

    output_dir = Path(output_dir)
    suffixes = ['recovery_time_map.png', 'max_distance_map.png', 'relative_heading_map.png']
    names = [f"{prefix}_{s}" if prefix else s for s in suffixes]
    panels = []
    for name in names:
        p = output_dir / name
        if not p.exists():
            print(f"  plot_summary: skipping, missing {name}")
            return None
        panels.append(Image.open(p))

    # normalise heights to the tallest panel
    max_h = max(img.height for img in panels)
    resized = []
    for img in panels:
        if img.height != max_h:
            new_w = int(img.width * max_h / img.height)
            img = img.resize((new_w, max_h), Image.LANCZOS)
        resized.append(img)

    gap = 10  # pixels between panels
    total_w = sum(img.width for img in resized) + gap * (len(resized) - 1)
    composite = Image.new('RGB', (total_w, max_h), (255, 255, 255))

    x = 0
    for img in resized:
        composite.paste(img, (x, 0))
        x += img.width + gap

    out_name = f"{prefix}_summary.png" if prefix else "summary.png"
    out_path = output_dir / out_name
    composite.save(out_path, dpi=(dpi, dpi))
    print(f"  Saved {out_name} to {out_path}")
    return out_path
