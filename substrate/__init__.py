"""Core Lenia substrate: simulation primitives decoupled from RL."""

from substrate.lenia import Config, Board, Automaton, Lenia, build_kernel, ArrayLike
from substrate.animals import Animal, decode_cells, decode_rle, load_animals, iter_animals
from substrate.simulation import Simulation
from substrate.scaling import (
    interpolate_pattern,
    compute_centroid,
    compute_median_center,
    compute_toroidal_center,
    recenter_field,
    prepare_scaled_simulation,
)

__all__ = [
    # lenia.py
    "Config",
    "Board",
    "Automaton",
    "Lenia",
    "build_kernel",
    "ArrayLike",
    # animals.py
    "Animal",
    "decode_cells",
    "decode_rle",
    "load_animals",
    "iter_animals",
    # simulation.py
    "Simulation",
    # scaling.py
    "interpolate_pattern",
    "compute_centroid",
    "compute_median_center",
    "compute_toroidal_center",
    "recenter_field",
    "prepare_scaled_simulation",
]
