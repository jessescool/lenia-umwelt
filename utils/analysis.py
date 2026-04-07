# utils/analysis.py — Jesse Cool (jessescool)
"""Rollout and analysis utilities for Lenia simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Tuple, overload

import numpy as np
import torch

from substrate import Automaton, Board, Config

if TYPE_CHECKING:
    from substrate import Simulation


@dataclass
class Rollout:
    """Frame sequence with optional blind-mask metadata.

    Drop-in replacement for List[torch.Tensor] — supports len(), indexing,
    slicing, iteration, concatenation, and torch.stack().

    blind_log is keyed by step index; only populated when a mask was active.
    No entry = no mask = no cost.
    """
    frames: List[torch.Tensor]
    blind_log: dict[int, torch.Tensor] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    @overload
    def __getitem__(self, idx: int) -> torch.Tensor: ...
    @overload
    def __getitem__(self, idx: slice) -> Rollout: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.frames[idx]
        # slice -> new Rollout with remapped blind_log
        sliced = self.frames[idx]
        indices = range(*idx.indices(len(self.frames)))
        new_log = {}
        for new_i, old_i in enumerate(indices):
            if old_i in self.blind_log:
                new_log[new_i] = self.blind_log[old_i]
        return Rollout(sliced, new_log)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.frames)

    def __add__(self, other) -> Rollout:
        if isinstance(other, Rollout):
            offset = len(self.frames)
            merged = dict(self.blind_log)
            for k, v in other.blind_log.items():
                merged[k + offset] = v
            return Rollout(self.frames + other.frames, merged)
        if isinstance(other, list):
            return Rollout(self.frames + other, dict(self.blind_log))
        return NotImplemented

    def __radd__(self, other) -> Rollout:
        if isinstance(other, list):
            offset = len(other)
            shifted = {k + offset: v for k, v in self.blind_log.items()}
            return Rollout(other + self.frames, shifted)
        return NotImplemented

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, blind_log: dict | None = None) -> Rollout:
        """Convert [T, H, W] stacked tensor -> Rollout of T individual frames."""
        return cls(list(tensor), blind_log or {})

    def to_tensor(self) -> torch.Tensor:
        """Stack frames -> [T, H, W] tensor."""
        return torch.stack(self.frames)


def rollout_parallel(
    sim: Simulation,
    control_state: torch.Tensor,
    cfg: Config,
    *,
    warmup: int = 16,
    window: int = 16,
    blind_masks: torch.Tensor | None = None,
    blind_duration: int | None = None,
) -> Tuple[Rollout, Rollout]:
    """Roll out perturbed + unperturbed simulations in parallel.

    blind_masks:    2-D [H, W] blind mask for the test sim (or None).
    blind_duration: None = persistent (all steps), N = steps 0..N-1.
    """
    ctrl_board = Board(cfg)
    ctrl_board.tensor.copy_(control_state)
    ctrl_auto = Automaton(cfg, fft=sim.automaton.use_fft)

    barrier_mask = getattr(sim, 'barrier_mask', None)

    frames_test: List[torch.Tensor] = []
    frames_ctrl: List[torch.Tensor] = []
    blind_log: dict[int, torch.Tensor] = {}

    total_steps = warmup + window

    for step in range(total_steps):
        # Test sim: apply blind_masks for steps 0..duration-1 (or all if None)
        if blind_masks is not None and (blind_duration is None or step < blind_duration):
            if barrier_mask is not None:
                test_blind = torch.maximum(barrier_mask, blind_masks)
            else:
                test_blind = blind_masks
            blind_log[step] = blind_masks
        else:
            test_blind = barrier_mask

        sim.lenia.step(blind_mask=test_blind)
        ctrl_auto.step(ctrl_board, blind_mask=barrier_mask)

        frames_test.append(sim.board.tensor.detach().clone())
        frames_ctrl.append(ctrl_board.tensor.detach().clone())

    return Rollout(frames_test, blind_log), Rollout(frames_ctrl)


def warmup_rollout(sim: Simulation, steps: int, *, capture: bool = False) -> Rollout | None:
    """Step simulation forward, optionally capturing frames."""
    frames = [] if capture else None

    barrier_mask = getattr(sim, 'barrier_mask', None)

    for _ in range(steps):
        sim.lenia.step(blind_mask=barrier_mask)
        if capture:
            frames.append(sim.board.tensor.detach().clone())

    return Rollout(frames) if frames is not None else None


def load_grid_search_results(input_dir: Path, ori: int = 0) -> dict:
    """Load all .npy maps from a grid search result directory.

    Checks the analysis/ subdirectory first, then the root directory,
    for backward compatibility with older result layouts.
    """
    results = {}

    map_names = [
        'recovery_map', 'erased_map', 'recovery_status_map',
        'heading_change', 'heading_vec_relative',
        'max_distance',
    ]

    search_dirs = [input_dir / "analysis", input_dir]

    for name in map_names:
        for d in search_dirs:
            path = d / f"{name}_ori{ori}.npy"
            if path.exists():
                results[name] = np.load(path)
                break

    return results


_STATUS_DIED = 0
_STATUS_RECOVERED = 1
_STATUS_NEVER = 2


def compute_creature_profile(results: dict) -> dict:
    """Compute summary statistics from grid search results.

    Returns empty dict if status map is missing or creature has no pixels.
    """
    status = results.get('recovery_status_map')
    recovery = results.get('recovery_map')
    erased = results.get('erased_map')

    if status is None:
        return {}

    creature_mask = (erased > 0.01) if erased is not None else (status >= 0)
    n_creature = creature_mask.sum()

    if n_creature == 0:
        return {}

    died = (status == _STATUS_DIED) & creature_mask
    recovered = (status == _STATUS_RECOVERED) & creature_mask
    never = (status == _STATUS_NEVER) & creature_mask

    n_died = died.sum()
    n_recovered = recovered.sum()
    n_never = never.sum()
    n_tested = n_died + n_recovered + n_never

    profile = {
        'n_creature_pixels': int(n_creature),
        'n_tested': int(n_tested),
        'n_died': int(n_died),
        'n_recovered': int(n_recovered),
        'n_never': int(n_never),
        'survivable_fraction': float((n_recovered + n_never) / max(n_tested, 1)),
        'critical_pixel_fraction': float(n_died / max(n_tested, 1)),
    }

    if recovery is not None and n_recovered > 0:
        profile['mean_recovery_time'] = float(recovery[recovered].mean())
        profile['median_recovery_time'] = float(np.median(recovery[recovered]))
    else:
        profile['mean_recovery_time'] = float('nan')
        profile['median_recovery_time'] = float('nan')

    # Asymmetry index: coefficient of variation of recovery times
    if recovery is not None and creature_mask.any():
        rec_vals = recovery[creature_mask]
        if rec_vals.std() > 0:
            profile['asymmetry_index'] = float(rec_vals.std() / max(rec_vals.mean(), 1e-6))
        else:
            profile['asymmetry_index'] = 0.0
    else:
        profile['asymmetry_index'] = 0.0

    return profile
