"""Intervention strategies for Lenia perturbations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class ActionParam:
    name: str
    type: str  # 'discrete' or 'continuous'
    size: int = 0  # For discrete params, the number of options
    description: str = ""


class Intervention(ABC):
    """Abstract base class for perturbation interventions."""

    @abstractmethod
    def kernel(
        self,
        shape: Tuple[int, int],
        action: Dict,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generate a perturbation kernel given action parameters."""
        raise NotImplementedError

    def action_space(self) -> List[ActionParam]:
        return [
            ActionParam("x", "discrete", 0, "Column coordinate (size = grid width)"),
            ActionParam("y", "discrete", 0, "Row coordinate (size = grid height)"),
        ]

    def apply(self, state: torch.Tensor, action: Dict, *, clamp: bool = True) -> torch.Tensor:
        """Apply intervention to state. Default: S_new = clip(S - kernel, 0, 1)."""
        kernel = self.kernel(
            state.shape,
            action,
            device=state.device,
            dtype=state.dtype,
        )
        new_state = state - kernel
        if clamp:
            new_state = torch.clamp(new_state, 0.0, 1.0)
        return new_state

    def barrier(
        self,
        shape: Tuple[int, int],
        action: Dict,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Return a blind mask for this intervention, or None.

        How long the mask is applied is controlled by blind_duration at the
        rollout level (None = persistent, 1 = step-0 only, N = steps 0..N-1).
        """
        return None

    @property
    def default_blind_duration(self) -> int | None:
        """Default blind_duration for this intervention type.

        None = persistent (all steps), 1 = step-0 only, N = steps 0..N-1.
        Base class returns None (moot since barrier() returns None).
        """
        return None


class SquareEraseIntervention(Intervention):
    """Binary square erase: set an NxN square of pixels to zero."""

    def __init__(self, size: int = 5) -> None:
        self.size = size

    def kernel(
        self,
        shape: Tuple[int, int],
        action: Dict,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Binary NxN square kernel centered at (x, y), toroidal wrap."""
        H, W = shape
        x = int(action['x']) % W
        y = int(action['y']) % H

        kernel = torch.zeros((H, W), device=device, dtype=dtype)
        half = self.size // 2
        offsets = torch.arange(-half, self.size - half, device=device)
        rows = (y + offsets) % H
        cols = (x + offsets) % W
        kernel[rows[:, None], cols[None, :]] = 1.0
        return kernel

    def masks_batched(
        self,
        shape: Tuple[int, int],
        positions: List[Tuple[int, int]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build [B, H, W] binary erase masks for B positions via advanced indexing."""
        H, W = shape
        B = len(positions)
        half = self.size // 2

        offsets = torch.arange(-half, self.size - half, device=device)  # [S]
        S = len(offsets)

        xs = torch.tensor([p[0] for p in positions], device=device, dtype=torch.long)
        ys = torch.tensor([p[1] for p in positions], device=device, dtype=torch.long)

        # Compute all row/col indices: [B, S] with toroidal wrap
        row_idx = (ys[:, None] + offsets[None, :]) % H  # [B, S]
        col_idx = (xs[:, None] + offsets[None, :]) % W  # [B, S]

        # Expand to [B, S, S] grid of pixel coordinates
        row_grid = row_idx[:, :, None].expand(B, S, S)  # [B, S, S]
        col_grid = col_idx[:, None, :].expand(B, S, S)  # [B, S, S]

        batch_idx = torch.arange(B, device=device)[:, None, None].expand(B, S, S)

        masks = torch.zeros(B, H, W, device=device, dtype=dtype)
        masks[batch_idx.reshape(-1), row_grid.reshape(-1), col_grid.reshape(-1)] = 1.0

        return masks

    def apply_batched(
        self,
        initial_state: torch.Tensor,
        positions: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply erase at B positions in one vectorized op."""
        masks = self.masks_batched(
            initial_state.shape, positions,
            device=initial_state.device, dtype=initial_state.dtype,
        )
        B = len(positions)

        states = initial_state.unsqueeze(0).expand(B, -1, -1).clone()
        states.mul_(1.0 - masks)
        states.clamp_(0.0, 1.0)

        # Affected mass = how much was actually removed at each position
        affected = (masks * initial_state.unsqueeze(0)).sum(dim=(1, 2))

        return states, affected

    def action_space(self) -> List[ActionParam]:
        return [
            ActionParam("x", "discrete", 0, "Column coordinate"),
            ActionParam("y", "discrete", 0, "Row coordinate"),
        ]


class BlindEraseIntervention(SquareEraseIntervention):
    """Erase NxN square AND produce a one-step blind mask for the same region."""

    def barrier(
        self,
        shape: Tuple[int, int],
        action: Dict,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the same NxN binary mask as the erase kernel."""
        return self.kernel(shape, action, device=device, dtype=dtype)

    @property
    def default_blind_duration(self) -> int | None:
        return 1


class SquareBlindIntervention(SquareEraseIntervention):
    """Persistent NxN blindness: creature pixels untouched, sensory field blocked."""

    def apply_batched(
        self,
        initial_state: torch.Tensor,
        positions: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """No pixel erasure — clone state unchanged. Affected = mass under blind square."""
        masks = self.masks_batched(
            initial_state.shape, positions,
            device=initial_state.device, dtype=initial_state.dtype,
        )
        B = len(positions)
        states = initial_state.unsqueeze(0).expand(B, -1, -1).clone()
        # Affected mass = creature mass hidden under the blind square
        affected = (masks * initial_state.unsqueeze(0)).sum(dim=(1, 2))
        return states, affected

    def barrier(
        self,
        shape: Tuple[int, int],
        action: Dict,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the NxN binary mask as a blind region."""
        return self.kernel(shape, action, device=device, dtype=dtype)


class SquareAdditiveIntervention(Intervention):
    """Additive square intervention: add/subtract a fixed intensity to an NxN region."""

    def __init__(self, size: int = 5, intensity: float = 0.3) -> None:
        self.size = size
        self.intensity = intensity

    def kernel(
        self,
        shape: Tuple[int, int],
        action: Dict,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Binary NxN square mask kernel centered at (x, y), toroidal wrap."""
        H, W = shape
        x = int(action['x']) % W
        y = int(action['y']) % H

        kernel = torch.zeros((H, W), device=device, dtype=dtype)
        half = self.size // 2
        offsets = torch.arange(-half, self.size - half, device=device)
        rows = (y + offsets) % H
        cols = (x + offsets) % W
        kernel[rows[:, None], cols[None, :]] = 1.0
        return kernel

    def apply(self, state: torch.Tensor, action: Dict, *, clamp: bool = True) -> torch.Tensor:
        """Apply additive intervention: S_new = clip(S + mask * intensity, 0, 1)."""
        mask = self.kernel(state.shape, action, device=state.device, dtype=state.dtype)
        new_state = state + mask * self.intensity
        if clamp:
            new_state = torch.clamp(new_state, 0.0, 1.0)
        return new_state

    def action_space(self) -> List[ActionParam]:
        return [
            ActionParam("x", "discrete", 0, "Column coordinate"),
            ActionParam("y", "discrete", 0, "Row coordinate"),
        ]



def make_intervention(intervention_type: str, size: int, *, intensity: float = 0.3) -> Intervention:
    """Create an intervention by type name."""
    if intervention_type == "erase":
        return SquareEraseIntervention(size=size)
    elif intervention_type == "blind_erase":
        return BlindEraseIntervention(size=size)
    elif intervention_type == "additive":
        return SquareAdditiveIntervention(size=size, intensity=intensity)
    elif intervention_type == "blind":
        return SquareBlindIntervention(size=size)
    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}. Valid: erase, blind_erase, additive, blind")
