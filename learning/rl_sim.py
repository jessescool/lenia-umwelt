"""RL simulation controller for multi-environment Lenia experiments."""
from __future__ import annotations

from typing import Callable, Protocol, Tuple

import torch

from metrics_and_machinery import Intervention
from substrate import ArrayLike, Config, Simulation


class Agent(Protocol):
    """Protocol for any agent that can act on an RLSim controller.

    This allows RLSim to work with any agent implementation without
    importing specific agent classes, keeping the simulation layer
    decoupled from the RL layer.
    """
    def act(self, controller: "RLSim") -> None:
        """Execute one action step on the given RLSim controller."""
        ...


class RLSim:
    """Simple multi-env controller that lets an agent juggle up to `max_envs` simulations."""

    def __init__(self, *, max_envs: int = 8, agent: Agent | None = None) -> None:
        self.max_envs = max(1, max_envs)
        self.environments: list[Simulation] = []
        self.agent = agent

    def attach_agent(self, agent: Agent) -> None:
        self.agent = agent

    def observations(self, stacked: bool = False) -> list[torch.Tensor] | torch.Tensor:
        """Return current board tensors for all envs.

        If ``stacked`` is True, returns a single tensor of shape (N, H, W).
        """
        boards = [env.board.tensor for env in self.environments]
        return torch.stack(boards, dim=0) if stacked else boards

    def apply_delta(self, index: int, delta: torch.Tensor, *, clamp: bool = True) -> None:
        """Add a delta tensor to an environment board in place."""
        board = self.environments[index].board.tensor
        board.add_(delta.to(board.device, board.dtype))
        if clamp:
            board.clamp_(0.0, 1.0)

    def apply_intervention(
        self,
        index: int,
        intervention: Intervention,
        action: dict,
        *,
        clamp: bool = True,
    ) -> torch.Tensor:
        """Apply an intervention to an environment's board state."""
        board = self.environments[index].board
        old_state = board.tensor
        new_state = intervention.apply(old_state, action, clamp=clamp)
        board.cells = new_state
        return intervention.kernel(
            old_state.shape,
            action,
            device=old_state.device,
            dtype=old_state.dtype,
        )

    def add_env(self, sim: "Simulation") -> int:
        """Add an existing Simulation to the controller."""
        if len(self.environments) >= self.max_envs:
            raise RuntimeError("Maximum environment capacity reached")
        self.environments.append(sim)
        return len(self.environments) - 1

    def spawn(self, cfg: Config | None = None) -> int:
        if len(self.environments) >= self.max_envs:
            raise RuntimeError("Maximum environment capacity reached")
        sim = Simulation(cfg)
        self.environments.append(sim)
        return len(self.environments) - 1

    def remove(self, index: int) -> None:
        self.environments.pop(index)

    def clone_env(self, index: int) -> Simulation:
        """Clone a specific environment without adding it to the pool."""
        return self.environments[index].clone()

    def fork_env(self, index: int) -> int:
        """Clone an environment and append it to the world array."""
        if len(self.environments) >= self.max_envs:
            raise RuntimeError("Maximum environment capacity reached for fork")
        fork = self.environments[index].clone()
        self.environments.append(fork)
        return len(self.environments) - 1

    def add_animal(self, index: int, pattern: ArrayLike, *, position: Tuple[int, int] | None = None, wrap: bool = True) -> None:
        self.environments[index].board.place(pattern, position=position, wrap=wrap)

    def set_barrier(self, index: int, mask: torch.Tensor) -> None:
        """Set barrier mask for a specific environment."""
        self.environments[index].set_barrier(mask)

    def add_barrier(self, index: int, mask: torch.Tensor) -> None:
        """Add barrier to a specific environment (union with existing)."""
        self.environments[index].add_barrier(mask)

    def clear_barrier(self, index: int) -> None:
        """Clear barrier from a specific environment."""
        self.environments[index].clear_barrier()

    def clear_all_barriers(self) -> None:
        """Clear barriers from all environments."""
        for sim in self.environments:
            sim.clear_barrier()

    def step(self, steps: int = 1, *, callback: Callable[[torch.Tensor, int, int], None] | None = None) -> list[list[torch.Tensor]] | None:
        if not self.environments:
            return [] if callback is None else None

        frames: list[list[torch.Tensor]] | None = (
            [[] for _ in self.environments] if callback is None else None
        )

        for _ in range(steps):
            if self.agent is not None:
                self.agent.act(self)
            for idx, sim in enumerate(list(self.environments)):
                state = sim.lenia.step(blind_mask=sim.barrier_mask)
                if callback is not None:
                    callback(state, sim.lenia.tick, idx)
                else:
                    frames[idx].append(state.detach().clone())
        return frames
