"""Batched simulation utilities for GPU-parallel rollout and intervention."""

from __future__ import annotations

from typing import List, Tuple

import torch

from substrate.lenia import Automaton
from metrics_and_machinery import Intervention
from metrics_and_machinery.distance_metrics import prepare_profile
from metrics_and_machinery.trajectory_metrics import centroid_batched


def rollout_batched(
    states: torch.Tensor,
    automaton: Automaton,
    n_steps: int,
    collect_every: int = 1,
    *,
    blind_masks: torch.Tensor | None = None,
    transient_blind_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run B simulations for n_steps, collecting frames."""
    B, H, W = states.shape
    n_collected = (n_steps + collect_every - 1) // collect_every

    # Pre-allocate frame storage
    frames = torch.empty(B, n_collected, H, W, device=states.device, dtype=states.dtype)

    current = states  # Caller already clones; avoid double-clone
    frame_idx = 0

    # Frame collection with off-by-one handling:
    # - We collect at step 0, collect_every, 2*collect_every, ...
    # - If n_steps isn't divisible by collect_every, the final state
    #   after all steps may not have been collected, so we grab it at the end.
    for step in range(n_steps):
        if step % collect_every == 0:
            frames[:, frame_idx] = current
            frame_idx += 1
        # Transient blind: merge with persistent masks on step 0 only
        if step == 0 and transient_blind_masks is not None:
            step_masks = (
                torch.maximum(blind_masks, transient_blind_masks)
                if blind_masks is not None
                else transient_blind_masks
            )
        else:
            step_masks = blind_masks
        current = automaton.step_batched(current, blind_masks=step_masks)

    # Collect final frame if loop didn't land on it
    if frame_idx < n_collected:
        frames[:, frame_idx] = current

    return frames


def rollout_batched_with_ctrl(
    test_states: torch.Tensor,
    ctrl_state: torch.Tensor,
    automaton: Automaton,
    n_steps: int,
    *,
    transient_blind_masks: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run B test + 1 control simulation as a combined batch."""
    B, H, W = test_states.shape

    # Combine test and ctrl into single batch: [B+1, H, W]
    combined = torch.cat([test_states, ctrl_state.unsqueeze(0)], dim=0)

    # Pad transient masks with a zero row for the ctrl slot
    combined_transient = None
    if transient_blind_masks is not None:
        ctrl_pad = torch.zeros(1, H, W, device=transient_blind_masks.device,
                               dtype=transient_blind_masks.dtype)
        combined_transient = torch.cat([transient_blind_masks, ctrl_pad], dim=0)

    combined_frames = rollout_batched(
        combined, automaton, n_steps,
        transient_blind_masks=combined_transient,
    )  # [B+1, T, H, W]

    test_frames = combined_frames[:-1]  # [B, T, H, W]
    ctrl_frames = combined_frames[-1]   # [T, H, W]

    return test_frames, ctrl_frames


def rollout_online_metrics(
    test_states: torch.Tensor,
    ctrl_state: torch.Tensor,
    automaton: Automaton,
    n_steps: int,
    *,
    orbit_c_bar: torch.Tensor,
    orbit_m: int,
    transient_blind_masks: torch.Tensor | None = None,
) -> dict:
    """Fused simulation + per-frame metrics -- no frame storage.

    Memory: ~5x [B+1, H, W] GPU for state + FFT, plus lightweight CPU
    tensors for summaries. Eliminates the [B, T, H, W] frame tensor.
    """
    B, H, W = test_states.shape
    device = test_states.device
    dtype = test_states.dtype

    # Combine test and ctrl into single batch: [B+1, H, W]
    current = torch.cat([test_states, ctrl_state.unsqueeze(0)], dim=0)

    # Pad transient masks with a zero row for the ctrl slot
    combined_transient = None
    if transient_blind_masks is not None:
        ctrl_pad = torch.zeros(1, H, W, device=device, dtype=dtype)
        combined_transient = torch.cat([transient_blind_masks, ctrl_pad], dim=0)

    # Move orbit barycenter to device once
    c_bar_dev = orbit_c_bar.to(device)

    # Pre-allocate all outputs on GPU -- single bulk .cpu() after the loop
    distances = torch.empty(B, n_steps, device=device, dtype=dtype)
    centroids_gpu = torch.empty(B + 1, n_steps, 2, device=device, dtype=torch.float32)
    mass_gpu = torch.empty(B, n_steps, device=device, dtype=dtype)
    ctrl_frames_gpu = torch.empty(n_steps, H, W, device=device, dtype=dtype)

    for step in range(n_steps):
        # Compute per-frame metrics BEFORE stepping (frame t = state before step t)

        # Profile distance: test sims only -> [B, m] profiles -> L1 to c_bar
        profiles = prepare_profile(current[:B], orbit_m)  # [B, m]
        distances[:, step] = (profiles - c_bar_dev).abs().mean(dim=1)

        # Centroids: all B+1 sims -> [B+1, 2]
        centroids_gpu[:, step, :] = centroid_batched(current)

        # Mass: test sims only -> [B]
        mass_gpu[:, step] = current[:B].sum(dim=(1, 2))

        # Ctrl frame: single control trajectory
        ctrl_frames_gpu[step] = current[-1]

        # Advance simulation
        if step == 0 and combined_transient is not None:
            step_masks = combined_transient
        else:
            step_masks = None
        current = automaton.step_batched(current, blind_masks=step_masks)

    # Single bulk GPU -> CPU transfer (replaces ~900 per-step sync points)
    centroids = centroids_gpu.cpu()
    mass = mass_gpu.cpu().float()
    ctrl_frames = ctrl_frames_gpu.cpu()

    return {
        'distances': distances,      # [B, T] GPU
        'centroids': centroids,      # [B+1, T, 2] CPU
        'mass': mass,                # [B, T] CPU
        'ctrl_frames': ctrl_frames,  # [T, H, W] CPU
    }


def build_transient_blind_masks(
    intervention: Intervention,
    shape: Tuple[int, int],
    positions: list[Tuple[int, int]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Build [B, H, W] transient blind masks if the intervention supports it."""
    # Quick probe: does this intervention produce transient barriers?
    probe = intervention.transient_barrier(
        shape, {'x': 0, 'y': 0}, device=device, dtype=dtype,
    )
    if probe is None:
        return None

    # BlindEraseIntervention inherits masks_batched from SquareEraseIntervention
    return intervention.masks_batched(shape, positions, device=device, dtype=dtype)


def apply_interventions_batched(
    initial_state: torch.Tensor,
    positions: List[Tuple[int, int]],
    intervention: Intervention,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply interventions at multiple positions, creating a batch."""
    # Fast path: vectorized batch for SquareEraseIntervention
    if hasattr(intervention, 'apply_batched'):
        return intervention.apply_batched(initial_state, positions)

    # Fallback: sequential loop for other intervention types
    B = len(positions)
    device = initial_state.device
    dtype = initial_state.dtype

    states = initial_state.unsqueeze(0).expand(B, -1, -1).clone()
    affected = torch.zeros(B, device=device, dtype=dtype)

    for i, (x, y) in enumerate(positions):
        action = {'x': x, 'y': y}
        states[i] = intervention.apply(states[i], action, clamp=True)
        affected[i] = (states[i] - initial_state).abs().sum()

    return states, affected


def estimate_batch_size(grid_size: int, n_steps: int, device: torch.device,
                        online: bool = False) -> int:
    """Estimate batch size from available GPU memory."""
    if device.type == "cpu":
        return 64
    else:  # CUDA
        props = torch.cuda.get_device_properties(device)
        total_mem = props.total_memory  # bytes

        if online:
            # Online mode: only sim state + FFT workspace live on GPU.
            # Empirical: ~8x state for FFT/conv workspace across grid sizes.
            element_cost = 8 * 4 * grid_size * grid_size
            budget = int(total_mem * 0.70)
            batch = (budget // max(element_cost, 1)) // 64 * 64
            # Cap to avoid per-step CPU overhead dominating at small grids
            cap = 8192 if total_mem > 80e9 else 4096
            return min(max(batch, 64), cap)
        else:
            # Original: frame storage on GPU
            frame_bytes = 2 * 4 * n_steps * grid_size * grid_size  # 2x for test+ctrl
            frames_budget = int(total_mem * 0.50)
            batch = (frames_budget // max(frame_bytes, 1)) // 64 * 64
            return max(batch, 1)
