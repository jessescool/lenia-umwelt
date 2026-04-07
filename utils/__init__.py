# utils/__init__.py — Jesse Cool (jessescool)
"""
Shared utilities for curriculum RL training and evaluation.

This package contains:
  - core: augment_creature (minimal dependencies)
  - analysis: rollout_parallel, warmup_rollout, load_grid_search_results, compute_creature_profile
  - batched: rollout_batched, apply_interventions_batched (GPU batching)

Visualization functions have moved to the top-level `viz` package.
For backwards compatibility, GIF functions are re-exported here.
"""

from utils.core import augment_creature, rotate_tensor
from utils.analysis import (
    Rollout,
    rollout_parallel,
    warmup_rollout,
    load_grid_search_results,
    compute_creature_profile,
)
from utils.batched import (
    rollout_batched,
    rollout_batched_with_ctrl,
    apply_interventions_batched,
    estimate_batch_size,
)

# Backwards-compat re-exports from viz.gif
from viz.gif import (
    write_side_by_side_gif,
    write_gif,
    draw_dot,
    render_centroid_gif,
    render_convergence_gif,
)


__all__ = [
    "Rollout",
    "augment_creature",
    "rotate_tensor",
    "draw_dot",
    "rollout_parallel",
    "write_side_by_side_gif",
    "write_gif",
    "render_centroid_gif",
    "render_convergence_gif",
    "warmup_rollout",
    "rollout_batched",
    "rollout_batched_with_ctrl",
    "apply_interventions_batched",
    "estimate_batch_size",
    "load_grid_search_results",
    "compute_creature_profile",
]
