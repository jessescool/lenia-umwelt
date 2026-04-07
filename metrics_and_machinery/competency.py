# metrics_and_machinery/competency.py — Jesse Cool (jessescool)
"""Competency scoring for environment testing.

Orbit Residence Fraction (M): fraction of the experiment the creature spent
alive AND morphologically intact (within its orbit).

M = V * F where:
  V (viability)  = frames alive / total frames
  F (fidelity)   = frames in-orbit / frames alive
  sigma_M        = std(M) across orientations
  D_peak         = max profile distance / d_max
"""

import torch


def orbit_residence_fraction(
    distances: torch.Tensor,
    mass: torch.Tensor,
    competency_threshold: float,
    death_threshold: float = 0.01,
    explosion_threshold: float | None = None,
    initial_mass: torch.Tensor | None = None,
    d_max: float = 1.0,
) -> dict:
    """Compute M, V, F, D_peak per trajectory from precomputed timeseries."""
    B, T = distances.shape

    alive = mass > death_threshold  # [B, T] bool

    # If explosion_threshold provided, treat exploded frames as not-alive
    if explosion_threshold is not None and initial_mass is not None:
        not_exploded = mass <= explosion_threshold * initial_mass.unsqueeze(1)
        alive = alive & not_exploded

    # Propagate death/explosion: once a creature loses status, it stays lost
    # Find first non-alive frame; all subsequent frames are also non-alive
    not_alive = ~alive
    # cummax propagates the first True forward
    not_alive_propagated = not_alive.cummax(dim=1).values
    alive = ~not_alive_propagated

    in_orbit = distances < competency_threshold  # [B, T] bool

    alive_float = alive.float()
    in_orbit_float = in_orbit.float()

    # V = fraction of frames alive
    V = alive_float.mean(dim=1)  # [B]

    # F = fraction of alive frames that are in-orbit
    alive_count = alive_float.sum(dim=1)  # [B]
    in_orbit_and_alive = (alive & in_orbit).float().sum(dim=1)  # [B]
    F = torch.where(alive_count > 0, in_orbit_and_alive / alive_count, torch.zeros_like(V))

    # M = V * F = (alive AND in_orbit) / total frames
    M = (alive & in_orbit).float().mean(dim=1)

    # D_peak: max distance reached (in d_max units), only for alive frames
    # Set dead-frame distances to 0 so they don't dominate
    masked_dist = distances * alive_float
    D_peak = masked_dist.max(dim=1).values / max(d_max, 1e-10)

    return {'M': M, 'V': V, 'F': F, 'D_peak': D_peak}


def aggregate_competency(
    M_per_ori: torch.Tensor,
    V_per_ori: torch.Tensor,
    F_per_ori: torch.Tensor,
    D_peak_per_ori: torch.Tensor,
) -> dict:
    """Average over orientations, compute sigma_M."""
    return {
        'M_mean': M_per_ori.mean().item(),
        'V_mean': V_per_ori.mean().item(),
        'F_mean': F_per_ori.mean().item(),
        'D_peak_mean': D_peak_per_ori.mean().item(),
        'sigma_M': M_per_ori.std().item() if len(M_per_ori) > 1 else 0.0,
        'M_per_ori': M_per_ori.tolist(),
        'V_per_ori': V_per_ori.tolist(),
        'F_per_ori': F_per_ori.tolist(),
        'D_peak_per_ori': D_peak_per_ori.tolist(),
    }
