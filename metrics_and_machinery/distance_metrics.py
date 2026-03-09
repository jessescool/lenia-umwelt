"""Distance metrics for comparing Lenia grids."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Sequence, Tuple


def prepare_profile(grid: torch.Tensor, m: int, threshold: float = 0.0) -> torch.Tensor:
    """Convert grid(s) to fixed-length sorted activation profiles.

    Single: (H, W) -> (m,)
    Batch:  (B, H, W) -> (B, m)  -- fully vectorized
    """
    if grid.ndim == 3:
        return _prepare_profile_batched(grid, m, threshold)

    flat = grid.flatten()
    nonzero = flat[flat > threshold]
    n = nonzero.numel()
    if n == 0:
        return torch.zeros(m, dtype=grid.dtype, device=grid.device)
    sorted_vals = torch.sort(nonzero, descending=True).values
    if n > m:
        return sorted_vals[:m]                # trim smallest
    # n <= m: zero-pad to length m
    pad = torch.zeros(m - n, dtype=grid.dtype, device=grid.device)
    return torch.cat([sorted_vals, pad])


def _prepare_profile_batched(grids: torch.Tensor, m: int, threshold: float = 0.0) -> torch.Tensor:
    """Vectorized batch path for prepare_profile."""
    B, H, W = grids.shape
    N = H * W
    flat = grids.reshape(B, N)
    # Mask sub-threshold values to zero so they sort to the tail
    flat = flat.clone()
    flat[flat <= threshold] = 0.0
    sorted_desc, _ = torch.sort(flat, dim=1, descending=True)  # nonzeros first
    return sorted_desc[:, :m].contiguous()  # compact copy; slice shares full (B, H*W) storage


def _to_profiles(x: torch.Tensor, m: int, threshold: float) -> torch.Tensor:
    """Coerce input to prepared profiles, handling single or batch.

    Dispatch by ndim:
      1D (m,)      -> single profile, returned as-is
      2D (H, W)    -> single grid   -> prepare_profile -> (m,)
      2D (B, m)    -> batch of profiles, returned as-is (detected when dim-1 == m)
      3D (B, H, W) -> batch of grids -> vectorized prepare_profile -> (B, m)
    """
    if x.ndim == 1:
        return prepare_profile(x, m, threshold)
    if x.ndim == 2 and x.shape[-1] == m:
        return x
    if x.ndim == 2:
        return prepare_profile(x, m, threshold)
    # (B, H, W) batch of grids -- vectorized
    return _prepare_profile_batched(x, m, threshold)


def wasserstein(a: torch.Tensor, b: torch.Tensor, m: int,
                threshold: float = 0.0):
    """W1 distance between grids or profiles via CDF integral.

    Single: (H,W) or (m,)   -> float
    Batch:  (B,H,W) or (B,m) -> (B,) tensor
    """
    pa = _to_profiles(a, m, threshold)
    pb = _to_profiles(b, m, threshold)
    if pa.ndim == 1:
        return float((pa - pb).abs().mean())
    # Batched: (B, m) -> (B,)
    return (pa - pb).abs().mean(dim=1)


def mass_distance(grid_a: torch.Tensor, grid_b: torch.Tensor) -> float:
    """Absolute difference in total mass."""
    return abs(float(grid_a.sum() - grid_b.sum()))


def total_variation(grid_a: torch.Tensor, grid_b: torch.Tensor, eps: float = 1e-10) -> float:
    """TV(P,Q) = 0.5 * sum(|P_i - Q_i|) where P,Q are normalized to distributions."""
    a = grid_a.flatten()
    b = grid_b.flatten()
    p = a / (a.sum() + eps)
    q = b / (b.sum() + eps)
    return 0.5 * float((p - q).abs().sum())


def wasserstein_nonzero(grid_a: torch.Tensor, grid_b: torch.Tensor, threshold: float = 1e-6) -> float:
    """Wasserstein distance over non-zero pixels (grid-size independent)."""
    from scipy.stats import wasserstein_distance
    a = grid_a.detach().cpu().flatten().numpy()
    b = grid_b.detach().cpu().flatten().numpy()
    a_nonzero = a[a > threshold]
    b_nonzero = b[b > threshold]

    if len(a_nonzero) == 0 or len(b_nonzero) == 0:
        return 0.0  # Edge case: one or both grids empty

    return wasserstein_distance(a_nonzero, b_nonzero)


def wasserstein_1d_torch_nonzero(
    a: torch.Tensor,
    b: torch.Tensor,
    threshold: float = 1e-6,
    n_bins: int = 2048,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Batched 1D Wasserstein over non-zero pixels (GPU-friendly, histogram/CDF)."""
    B, N = a.shape
    bin_width = 1.0 / n_bins
    results = torch.empty(B, device=a.device, dtype=a.dtype)

    # Auto-scale chunk size: ~4GB peak memory budget for intermediates
    # Main cost per row: a_bins (long=8B) + b_bins + ones + a_hist + b_hist + active masks
    # ~ 2*N*8 + N*4 + 2*n_bins*4 ~ 20*N bytes per row (dominant term)
    mem_budget = 4e9  # 4GB
    chunk_size = max(64, min(chunk_size, int(mem_budget / (20 * N + 1))))

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        a_chunk = a[start:end]
        b_chunk = b[start:end]
        C = end - start

        # Bin assignment: clamp to [0, n_bins-1], cast to long for scatter
        a_bins = (a_chunk / bin_width).clamp(0, n_bins - 1).long()  # [C, N]
        b_bins = (b_chunk / bin_width).clamp(0, n_bins - 1).long()

        # Batched histogram via scatter_add: [C, K]
        ones = torch.ones(C, N, device=a.device, dtype=a.dtype)
        a_hist = torch.zeros(C, n_bins, device=a.device, dtype=a.dtype)
        b_hist = torch.zeros(C, n_bins, device=a.device, dtype=a.dtype)
        a_hist.scatter_add_(1, a_bins, ones)
        b_hist.scatter_add_(1, b_bins, ones)

        # W1 via CDF integral: sum |CDF_a - CDF_b| * bin_width
        cdf_diff = torch.cumsum(a_hist - b_hist, dim=1)
        raw_dist = bin_width * cdf_diff.abs().sum(dim=1)  # [C]

        # Nonzero normalization (same convention as before)
        n_active_a = (a_chunk > threshold).sum(dim=-1).float()
        n_active_b = (b_chunk > threshold).sum(dim=-1).float()
        avg_active = (n_active_a + n_active_b) / 2 + 1e-10

        results[start:end] = raw_dist / avg_active

        # Free intermediates for memory-constrained devices
        del a_bins, b_bins, a_hist, b_hist, ones, cdf_diff

    return results


def batched_frame_distances(
    frames_a: Sequence[torch.Tensor],
    frames_b: Sequence[torch.Tensor],
    threshold: float = 1e-6,
) -> torch.Tensor:
    """Compute Wasserstein distances for all frame pairs at once."""
    if len(frames_a) == 0:
        return torch.tensor([], dtype=torch.float32)

    # Stack frames: (T, H, W) -> flatten to (T, H*W)
    stacked_a = torch.stack(list(frames_a)).flatten(start_dim=1)
    stacked_b = torch.stack(list(frames_b)).flatten(start_dim=1)

    return wasserstein_1d_torch_nonzero(stacked_a, stacked_b, threshold=threshold)


def time_to_orbit_recovery_batched(
    test_frames: torch.Tensor,
    c_bar: torch.Tensor,
    m: int,
    recovery_threshold: float,
    stability_window: int = 20,
    death_threshold: float = 0.01,
    warmup_frames: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Batched orbit-based recovery: is profile distance within orbit threshold?

    Recovery = d(profile(x), c_bar) < recovery_threshold for stability_window
    consecutive frames. Returns (recovery_times [B], outcomes [B], distances [B,T]).
    """
    B, T, H, W = test_frames.shape
    device = test_frames.device
    dtype = test_frames.dtype

    recovery_times = torch.full((B,), T, device=device, dtype=torch.long)
    outcomes = torch.full((B,), 2, device=device, dtype=torch.long)  # default: "never"
    all_distances = torch.zeros(B, T, device=device, dtype=dtype)

    # Step 1: Death detection
    mass_per_frame = test_frames.sum(dim=(2, 3))  # [B, T]
    died_mask_per_frame = mass_per_frame < death_threshold
    died_any = died_mask_per_frame.any(dim=1)
    death_frame = died_mask_per_frame.int().argmax(dim=1)

    outcomes[died_any] = 0
    recovery_times[died_any] = death_frame[died_any]

    alive_mask = ~died_any
    if not alive_mask.any():
        return recovery_times.float(), outcomes, None

    # Step 2: Compute profile distances to c_bar for alive trajectories
    alive_indices = alive_mask.nonzero(as_tuple=True)[0]
    B_alive = len(alive_indices)
    c_bar_dev = c_bar.to(device)

    distances = torch.empty(B_alive, T, device=device, dtype=dtype)
    for t in range(T):
        profiles = prepare_profile(test_frames[alive_indices, t], m)  # [B_alive, H, W] -> [B_alive, m]
        distances[:, t] = (profiles - c_bar_dev).abs().mean(dim=1)    # L1 distance to c_bar: [B_alive]

    all_distances[alive_indices] = distances

    # Step 3: Threshold crossing with stability window
    if warmup_frames > 0 and warmup_frames < T:
        detect_distances = distances[:, warmup_frames:]
    else:
        detect_distances = distances

    T_detect = detect_distances.shape[1]
    below_threshold = detect_distances < recovery_threshold

    if T_detect >= stability_window:
        below_float = below_threshold.float().unsqueeze(1)  # [B_alive, 1, T_detect]
        kernel = torch.ones(1, 1, stability_window, device=device, dtype=dtype)
        conv_result = F.conv1d(below_float, kernel).squeeze(1)  # [B_alive, T_detect-sw+1]

        stable_found = conv_result >= stability_window
        any_stable = stable_found.any(dim=1)
        first_stable_idx = stable_found.int().argmax(dim=1)

        recovery_frame = first_stable_idx + warmup_frames

        stable_alive = alive_indices[any_stable]
        outcomes[stable_alive] = 1
        recovery_times[stable_alive] = recovery_frame[any_stable]

    return recovery_times.float(), outcomes, all_distances


def detect_recovery(
    distances: torch.Tensor,
    mass: torch.Tensor,
    recovery_threshold: float,
    stability_window: int = 20,
    death_threshold: float = 0.01,
    warmup_frames: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Detect recovery from precomputed distance and mass timeseries."""
    B, T = distances.shape
    device = distances.device
    dtype = distances.dtype

    recovery_times = torch.full((B,), T, device=device, dtype=torch.long)
    outcomes = torch.full((B,), 2, device=device, dtype=torch.long)  # default: "never"

    # Step 1: Death detection from mass timeseries
    mass_dev = mass.to(device) if mass.device != device else mass
    died_mask_per_frame = mass_dev < death_threshold
    died_any = died_mask_per_frame.any(dim=1)
    death_frame = died_mask_per_frame.int().argmax(dim=1)

    outcomes[died_any] = 0
    recovery_times[died_any] = death_frame[died_any]

    alive_mask = ~died_any
    if not alive_mask.any():
        return recovery_times.float(), outcomes

    # Step 2: Stability window via 1D convolution on distances
    alive_indices = alive_mask.nonzero(as_tuple=True)[0]
    alive_dists = distances[alive_indices]  # [B_alive, T]

    if warmup_frames > 0 and warmup_frames < T:
        detect_distances = alive_dists[:, warmup_frames:]
    else:
        detect_distances = alive_dists

    T_detect = detect_distances.shape[1]
    below_threshold = detect_distances < recovery_threshold

    if T_detect >= stability_window:
        below_float = below_threshold.float().unsqueeze(1)  # [B_alive, 1, T_detect]
        kernel = torch.ones(1, 1, stability_window, device=device, dtype=torch.float32)
        conv_result = F.conv1d(below_float, kernel).squeeze(1)  # [B_alive, T_detect-sw+1]

        stable_found = conv_result >= stability_window
        any_stable = stable_found.any(dim=1)
        first_stable_idx = stable_found.int().argmax(dim=1)

        recovery_frame = first_stable_idx + warmup_frames

        stable_alive = alive_indices[any_stable]
        outcomes[stable_alive] = 1
        recovery_times[stable_alive] = recovery_frame[any_stable]

    return recovery_times.float(), outcomes
