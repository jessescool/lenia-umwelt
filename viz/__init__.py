"""Visualization package for Lenia simulations."""

import numpy as np

STATUS_EMPTY = -1      # no creature mass at this position (not tested)
STATUS_DIED = 0        # creature died after perturbation
STATUS_RECOVERED = 1   # creature recovered to within threshold
STATUS_NEVER = 2       # creature survived but never recovered

COLOR_GREEN = np.array([0.18, 0.80, 0.44])   # fast recovery / high directness
COLOR_YELLOW = np.array([1.00, 0.85, 0.00])   # slow recovery
COLOR_ORANGE = np.array([1.00, 0.55, 0.00])   # low directness / active recovery
COLOR_RED = np.array([0.91, 0.30, 0.24])   # died
COLOR_PURPLE = np.array([0.60, 0.40, 0.80])   # never recovered
COLOR_BLUE = np.array([0.20, 0.20, 1.00])   # died (distance heatmap variant)
COLOR_BG = '#f5f5f5'                           # background / empty

HEX_GREEN = '#2ecc71'
HEX_YELLOW = '#f1d900'
HEX_ORANGE = '#ff8c00'
HEX_RED = '#e74c3c'
HEX_PURPLE = '#9966cc'
HEX_BLUE = '#3333ff'

_FIG_SIZE = (10, 10)
_DPI = 300
_CBAR_KWARGS = dict(fraction=0.046, pad=0.04, shrink=0.6)

from viz.gif import (
    write_gif,
    write_side_by_side_gif,
    render_centroid_gif,
    render_convergence_gif,
    draw_dot,
)

from viz.maps import (
    plot_recovery_status_map,
    plot_max_distance_map,
    plot_relative_heading,
    plot_summary,
)


__all__ = [
    "STATUS_EMPTY", "STATUS_DIED", "STATUS_RECOVERED", "STATUS_NEVER",
    "COLOR_GREEN", "COLOR_YELLOW", "COLOR_ORANGE", "COLOR_RED",
    "COLOR_PURPLE", "COLOR_BLUE", "COLOR_BG",
    "HEX_GREEN", "HEX_YELLOW", "HEX_ORANGE", "HEX_RED", "HEX_PURPLE", "HEX_BLUE",
    "_FIG_SIZE", "_DPI", "_CBAR_KWARGS",
    "write_gif", "write_side_by_side_gif",
    "render_centroid_gif", "render_convergence_gif", "draw_dot",
    "plot_recovery_status_map", "plot_max_distance_map", "plot_relative_heading",
    "plot_summary",
]
