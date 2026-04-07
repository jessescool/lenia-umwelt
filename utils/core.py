# utils/core.py — Jesse Cool (jessescool)
"""Core creature utilities."""

import copy
import math

import torch

from substrate import Animal


def augment_creature(creature: Animal, device: torch.device) -> Animal:
    """Apply random continuous rotation to a creature's cell pattern."""
    augmented = copy.deepcopy(creature)
    cells = torch.as_tensor(augmented.cells, device=device, dtype=torch.float32).clone()
    angle = torch.rand(1).item() * 360.0
    cells = rotate_tensor(cells, angle, device)
    augmented.cells = cells.cpu()
    return augmented


def rotate_tensor(tensor: torch.Tensor, angle_deg: float, device: torch.device) -> torch.Tensor:
    """Rotate a 2D tensor by an arbitrary angle (bilinear interpolation)."""
    angle_rad = torch.tensor(angle_deg * math.pi / 180.0, device=device)
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    theta = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0]
    ], dtype=torch.float32, device=device).unsqueeze(0)  # (1, 2, 3)

    tensor_4d = tensor.unsqueeze(0).unsqueeze(0)

    grid = torch.nn.functional.affine_grid(
        theta, tensor_4d.size(), align_corners=False
    )

    rotated = torch.nn.functional.grid_sample(
        tensor_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=False
    )

    return rotated.squeeze(0).squeeze(0)
