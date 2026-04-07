# substrate/simulation.py — Jesse Cool (jessescool)
"""Core Simulation class for Lenia environments."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, Tuple

import torch

from substrate.animals import Animal, decode_cells, load_animals
from substrate.lenia import ArrayLike, Config, Lenia
from substrate.scaling import interpolate_pattern
from utils.analysis import Rollout


class Simulation:
    def __init__(self, cfg: Config | None = None, *, fft: bool = True):
        self.cfg = cfg or Config()
        self.lenia = Lenia.from_config(self.cfg, fft=fft)
        self.board = self.lenia.board
        self.automaton = self.lenia.automaton
        self.barrier_mask: torch.Tensor | None = None
        self.salience_map: torch.Tensor | None = None

    def add_animal(
        self,
        pattern: ArrayLike,
        *,
        position: Tuple[int, int] | None = None,
        wrap: bool = True,
    ) -> None:
        self.board.place(pattern, position=position, wrap=wrap)

    def add_animal_from_rle(self,
        rle: str,
        *,
        position: Tuple[int, int] | None = None,
        wrap: bool = True,
    ) -> None:
        pattern = decode_cells(rle)
        self.add_animal(pattern, position=position, wrap=wrap)

    def load_animals_from_catalog(
        self,
        catalog_path: str | Path = "animals.json",
        *,
        codes: Sequence[str] | None = None,
        wrap: bool = True,
    ) -> list[Animal]:
        animals = load_animals(catalog_path, codes=codes)
        for animal in animals:
            self.add_animal(animal.cells, wrap=wrap)
        return animals

    def set_barrier(self, mask: torch.Tensor) -> None:
        """Set the barrier mask (replaces any existing barrier)."""
        self.barrier_mask = mask.to(self.board.tensor.device, self.board.tensor.dtype)

    def add_barrier(self, mask: torch.Tensor) -> None:
        """Union with existing barriers."""
        if self.barrier_mask is None:
            self.set_barrier(mask)
        else:
            new_mask = mask.to(self.board.tensor.device, self.board.tensor.dtype)
            self.barrier_mask = torch.maximum(self.barrier_mask, new_mask)

    def clear_barrier(self) -> None:
        """Remove all barriers."""
        self.barrier_mask = None

    def set_salience(self, salience: torch.Tensor) -> None:
        """Set the salience map (replaces any existing salience)."""
        self.salience_map = salience.to(self.board.tensor.device, self.board.tensor.dtype)

    def clear_salience(self) -> None:
        """Remove the salience map."""
        self.salience_map = None

    def run(
        self,
        steps: int,
        *,
        callback: Callable[[torch.Tensor, int], None] | None = None,
    ) -> Rollout | None:
        frames: list[torch.Tensor] | None = [] if callback is None else None
        for _ in range(steps):
            state = self.lenia.step(blind_mask=self.barrier_mask, salience_map=self.salience_map)
            if callback is not None:
                callback(state, self.lenia.tick)
            else:
                frames.append(state.detach().clone())
        return Rollout(frames) if frames is not None else None

    def place_animal(self, animal: Animal, *, center: bool = True) -> None:
        """Place an animal on the board, optionally centered."""
        pattern = animal.cells

        if self.cfg.scale > 1:
            pattern = interpolate_pattern(pattern, self.cfg.scale)

        rows, cols = self.board.shape
        ph, pw = pattern.shape

        if center:
            pos = ((rows - ph) // 2, (cols - pw) // 2)
        else:
            # Random placement (truly uniform on torus - includes edge wrapping)
            pos = (
                torch.randint(0, rows, (1,)).item(),
                torch.randint(0, cols, (1,)).item()
            )

        self.add_animal(pattern, position=pos, wrap=True)

    def clone(self) -> "Simulation":
        """Return a deep-ish clone sharing cfg but copying board and barrier state.

        Config is shared (not copied), which is safe as long as it isn't mutated
        after construction.
        """
        copy = Simulation(self.cfg, fft=self.automaton.use_fft)
        copy.board.tensor.copy_(self.board.tensor)
        if self.barrier_mask is not None:
            copy.barrier_mask = self.barrier_mask.clone()
        if self.salience_map is not None:
            copy.salience_map = self.salience_map.clone()
        return copy
