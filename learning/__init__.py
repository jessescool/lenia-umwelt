"""Learning components for Lenia perturbation experiments.

This package contains:
- rl_sim: Multi-environment simulation controller (RLSim, Agent protocol)
- perturbot: Policy network and RL agent (PerturBot)
- train: Single-stage training script
- train_curriculum: Curriculum training across decreasing perturbation sizes
- train_slow_recovery: Recovery-objective training
"""

from learning.rl_sim import RLSim, Agent
from learning.perturbot import (
    PerturBot,
    Policy,
    PolicyConfig,
    AgentConfig,
)

__all__ = [
    # rl_sim
    "RLSim",
    "Agent",
    # perturbot
    "PerturBot",
    "Policy",
    "PolicyConfig",
    "AgentConfig",
]
