"""Policy and agent for Lenia RL perturbation learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

if TYPE_CHECKING:
    from learning.rl_sim import RLSim

from metrics_and_machinery import Intervention


@dataclass
class PolicyConfig:
    channels: int = 16
    hidden: int = 64
    value_head: bool = True


@dataclass
class AgentConfig:
    lr: float = 1e-3
    batch_size: int = 32
    entropy_coeff: float = 0.05


class Policy(nn.Module):
    """CNN: grid state -> per-pixel location logits + optional value estimate."""

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, cfg.channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, padding=1)
        self.logits_head = nn.Conv2d(cfg.channels, 1, kernel_size=1)

        self.value_head = nn.Sequential(
            nn.Linear(cfg.channels, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, 1),
        ) if cfg.value_head else None

    def forward(self, board: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass. Returns (logits, value) where logits is (B, H, W)."""
        x = board.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        logits = self.logits_head(x).squeeze(1)  # (B, H, W)

        value = None
        if self.value_head is not None:
            pooled = x.mean(dim=[2, 3])  # Global average pool
            value = self.value_head(pooled).squeeze(-1)  # (B,)

        return logits, value


class PerturBot:
    """RL agent that learns intervention locations for Lenia patterns."""

    name: str

    def __init__(
        self,
        *,
        policy_cfg: PolicyConfig | None = None,
        agent_cfg: AgentConfig | None = None,
        device: torch.device | str | None = None,
        name: str = "PerturBot",
        intervention: Intervention,
    ) -> None:
        self.name = name
        self.policy_cfg = policy_cfg or PolicyConfig()
        self.cfg = agent_cfg or AgentConfig()
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.policy = Policy(self.policy_cfg).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr)

        self.intervention = intervention

        # Buffers for policy gradient computation
        self._logprobs: dict[int, list[torch.Tensor]] = {}
        self._values: dict[int, list[torch.Tensor]] = {}
        self._rewards: dict[int, list[float]] = {}
        self._entropies: dict[int, list[torch.Tensor]] = {}
        self._steps = 0

    def _ensure_buffers(self, env_idx: int) -> None:
        """Initialize buffers for an environment if not present."""
        self._logprobs.setdefault(env_idx, [])
        self._values.setdefault(env_idx, [])
        self._rewards.setdefault(env_idx, [])
        self._entropies.setdefault(env_idx, [])

    def act(self, controller: "RLSim", greedy: bool = False, temperature: float = 1.0) -> torch.Tensor | None:
        """Select intervention location and apply to environment(s).

        Temperature controls exploration: T > 1 flattens the distribution,
        T = 1 is standard categorical, T < 1 sharpens toward greedy.
        """
        boards = controller.observations(stacked=True).to(self.device)
        logits, value = self.policy(boards)
        B, H, W = logits.shape

        flat_logits = logits.view(B, -1)
        if greedy:
            pos_idx = flat_logits.argmax(dim=1)
            pos_logprob = torch.zeros(B, device=boards.device)
        else:
            scaled_logits = flat_logits / temperature
            pos_dist = Categorical(logits=scaled_logits)
            pos_idx = pos_dist.sample()
            pos_logprob = pos_dist.log_prob(pos_idx)
            pos_entropy = pos_dist.entropy()

        rows = (pos_idx // W).to(torch.long)
        cols = (pos_idx % W).to(torch.long)

        kernel_to_return = None

        for b in range(B):
            self._ensure_buffers(b)

            if not greedy:
                self._logprobs[b].append(pos_logprob[b])
                self._entropies[b].append(pos_entropy[b])
                if value is not None:
                    self._values[b].append(value[b])

            action = {
                'x': int(cols[b]),
                'y': int(rows[b]),
            }

            kernel = controller.apply_intervention(b, self.intervention, action, clamp=True)

            if b == 0:
                kernel_to_return = kernel.detach().cpu().clone()

        self._steps += 1
        return kernel_to_return

    def observe_reward(self, env_idx: int, reward: float) -> None:
        """Record reward and trigger policy update if rollout complete."""
        self._ensure_buffers(env_idx)
        self._rewards[env_idx].append(float(reward))

        if self._steps >= self.cfg.batch_size:
            self.update()
            self._steps = 0

    def reset_env(self, env_idx: int, device: torch.device) -> None:
        """Reset buffers for an environment."""
        self._logprobs.pop(env_idx, None)
        self._values.pop(env_idx, None)
        self._rewards.pop(env_idx, None)
        self._entropies.pop(env_idx, None)
        self.device = device
        self.policy.to(device)

    def update(self) -> None:
        """Update policy using collected trajectories (REINFORCE)."""

        if not self._logprobs:
            return

        all_returns: list[torch.Tensor] = []
        all_logprobs: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []
        all_entropies: list[torch.Tensor] = []

        for env_idx, logprobs in self._logprobs.items():
            rewards = self._rewards.get(env_idx, [])
            if not rewards or len(rewards) != len(logprobs):
                continue

            R = torch.tensor(rewards, device=self.device)
            logp = torch.stack(logprobs).to(self.device)

            all_returns.append(R)
            all_logprobs.append(logp)

            if self._values.get(env_idx):
                vals = torch.stack(self._values[env_idx]).to(self.device)
                all_values.append(vals)

            if self._entropies.get(env_idx):
                ent = torch.stack(self._entropies[env_idx]).to(self.device)
                all_entropies.append(ent)

        if not all_returns:
            return

        R_batch = torch.cat(all_returns)
        logp_batch = torch.cat(all_logprobs)

        entropy_bonus = 0.0
        if all_entropies:
            ent_batch = torch.cat(all_entropies)
            entropy_bonus = self.cfg.entropy_coeff * ent_batch.mean()

        if all_values:
            vals_batch = torch.cat(all_values)
            adv = R_batch - vals_batch
            value_loss = 0.5 * adv.pow(2).mean()
        else:
            adv = R_batch - R_batch.mean()
            value_loss = 0.0

        loss = -(logp_batch * adv.detach()).mean() + value_loss - entropy_bonus

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.opt.step()

        self._logprobs.clear()
        self._values.clear()
        self._rewards.clear()
        self._entropies.clear()
