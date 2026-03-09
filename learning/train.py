"""
Train PerturBot on a single intervention size.

Simple training script without curriculum complexity. Train an agent to find
vulnerable spots on Lenia creatures using a fixed perturbation size.

Usage:
    python train.py --size 3 --max-episodes 5000
    python train.py --size 5 --code O2u --intervention erase
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch

from substrate import load_animals, Config, Simulation
from learning import RLSim, AgentConfig, PerturBot, PolicyConfig
from config import (
    DEFAULT_ANIMAL_CODE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CATALOG,
    DEFAULT_GRID_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPISODES,
    PRE_WARMUP_STEPS,
)
from metrics_and_machinery import (
    make_intervention,
    compute_damage,
    compute_detection_frame,
    compute_timing_windows,
)
from utils import augment_creature, rollout_parallel, warmup_rollout, write_side_by_side_gif


def train(
    lenia_cfg: Config,
    creature,
    *,
    size: int = 3,
    intervention_type: str = "erase",
    intensity: float = 0.3,
    max_episodes: int = 10000,
    lr: float = 1e-3,
    batch_size: int = 32,
    warmup: int | None = None,
    window: int | None = None,
    checkpoint_path: Path = Path("checkpoints/perturbot.pt"),
    viz_dir: Path = Path("viz_training"),
    viz_interval: int = 100,
    viz_explore_limit: int = 5,
    temperature_start: float = 2.0,
    temperature_end: float = 1.0,
    convergence_window: int = 100,
    convergence_threshold: float = 0.001,
    verbose: bool = True,
) -> PerturBot:
    """Train PerturBot with a fixed intervention size."""
    # Compute timing windows from creature timescale if not specified
    if warmup is None or window is None:
        computed_warmup, computed_window = compute_timing_windows(lenia_cfg.timescale_T)
        warmup = warmup or computed_warmup
        window = window or computed_window

    if verbose:
        print(f"Training {size}x{size} {intervention_type} intervention")
        print(f"Timing: warmup={warmup}, window={window} (T={lenia_cfg.timescale_T})")

    # Create output directories
    viz_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize intervention and agent
    intervention = make_intervention(intervention_type, size, intensity=intensity)
    agent = PerturBot(
        policy_cfg=PolicyConfig(),
        agent_cfg=AgentConfig(lr=lr, batch_size=batch_size),
        intervention=intervention,
        device=lenia_cfg.device,
    )

    # Training state
    episode = 0
    rewards_window: deque = deque(maxlen=convergence_window)
    viz_explore_count = 0
    cumulative_reward = 0.0

    def compute_temperature(ep: int) -> float:
        """Linear decay from temperature_start to temperature_end."""
        if max_episodes <= 1:
            return temperature_end
        progress = ep / (max_episodes - 1)
        return temperature_start + progress * (temperature_end - temperature_start)

    # Training loop
    while episode < max_episodes:
        # Fresh simulation with augmented creature at random position
        sim = Simulation(lenia_cfg)
        augmented = augment_creature(creature, lenia_cfg.device)
        sim.place_animal(augmented, center=False)

        # Pre-intervention warmup (let creature settle)
        frames_pre = warmup_rollout(sim, steps=PRE_WARMUP_STEPS, capture=True)

        # Setup controller for agent
        controller = RLSim(max_envs=1, agent=agent)
        controller.add_env(sim)
        agent.reset_env(0, sim.board.tensor.device)

        # Snapshot → act → rollout
        temperature = compute_temperature(episode)
        pre_action = sim.board.tensor.detach().clone()
        kernel = agent.act(controller, greedy=False, temperature=temperature)
        post_action = sim.board.tensor.detach().clone()
        erased_mass = (pre_action - post_action).clamp(min=0).sum().item()

        # Parallel rollout of perturbed and control trajectories
        frames_test_post, frames_ctrl_post = rollout_parallel(
            sim, pre_action, lenia_cfg, warmup=warmup, window=window
        )

        # Compute damage on measurement window only
        frames_test_measure = frames_test_post[-window:]
        frames_ctrl_measure = frames_ctrl_post[-window:]
        damage = compute_damage(frames_ctrl_measure, frames_test_measure)

        # Reward: damage normalized by erased mass
        reward = damage / max(erased_mass, 0.01)
        agent.observe_reward(0, reward)

        if verbose:
            print(f"ep {episode}: reward={reward:.6f}, dmg={damage:.4f}, erased={erased_mass:.2f}, temp={temperature:.2f}")

        rewards_window.append(reward)
        cumulative_reward += reward
        episode += 1

        # Visualization: early exploration
        if reward > 1e-3 and viz_explore_count < viz_explore_limit:
            test_frames_full = frames_pre + frames_test_post
            ctrl_frames_full = frames_pre + frames_ctrl_post
            # Compute detection frame for accurate GIF phase coloring
            detection_frame_rel, _ = compute_detection_frame(
                frames_ctrl_post, frames_test_post
            )
            detection_frame = PRE_WARMUP_STEPS + detection_frame_rel

            path = viz_dir / f"explore_{viz_explore_count:02d}_ep{episode:05d}.gif"
            write_side_by_side_gif(
                test_frames_full, ctrl_frames_full, kernel, path,
                fps=15, pre_warmup_frames=PRE_WARMUP_STEPS,
                post_warmup_frames=warmup,
                detection_frame=detection_frame,
            )
            if verbose:
                print(f"  Saved: {path.name}")
            viz_explore_count += 1

        # Visualization: periodic progress
        if episode % viz_interval == 0:
            test_frames_full = frames_pre + frames_test_post
            ctrl_frames_full = frames_pre + frames_ctrl_post
            # Compute detection frame for accurate GIF phase coloring
            detection_frame_rel, _ = compute_detection_frame(
                frames_ctrl_post, frames_test_post
            )
            detection_frame = PRE_WARMUP_STEPS + detection_frame_rel

            path = viz_dir / f"periodic_ep{episode:05d}.gif"
            write_side_by_side_gif(
                test_frames_full, ctrl_frames_full, kernel, path,
                fps=15, pre_warmup_frames=PRE_WARMUP_STEPS,
                post_warmup_frames=warmup,
                detection_frame=detection_frame,
            )

        # Logging
        if verbose and episode % 100 == 0:
            avg = np.mean(rewards_window) if rewards_window else 0.0
            print(f"Episode {episode}/{max_episodes}: avg_reward={avg:.4f}")

        # Convergence check
        if len(rewards_window) == convergence_window:
            mean_r = np.mean(rewards_window)
            if mean_r > 0.001 and np.std(rewards_window) < convergence_threshold:
                if verbose:
                    print(f"Converged at episode {episode} (plateau={mean_r:.4f})")
                break

    # Save policy
    torch.save(agent.policy.state_dict(), checkpoint_path)
    if verbose:
        avg_final = cumulative_reward / max(episode, 1)
        print(f"Training complete: {episode} episodes, avg_reward={avg_final:.4f}")
        print(f"Saved: {checkpoint_path}")

    return agent


def main():
    parser = argparse.ArgumentParser(
        description="Train PerturBot on a single intervention size"
    )
    parser.add_argument("--size", type=int, default=3, help="Intervention size (default: 3 = 3x3)")
    parser.add_argument("--catalog", type=Path, default=Path(DEFAULT_CATALOG))
    parser.add_argument("--code", default=DEFAULT_ANIMAL_CODE, help="Animal code (default: O2u)")
    parser.add_argument("--grid", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--intervention", default="erase", choices=["erase", "additive"])
    parser.add_argument("--intensity", type=float, default=0.3)
    parser.add_argument("--max-episodes", type=int, default=DEFAULT_MAX_EPISODES)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/perturbot.pt"))
    parser.add_argument("--viz-dir", type=Path, default=Path("viz_training"))
    parser.add_argument("--viz-interval", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Load creature
    creatures = load_animals(args.catalog, codes=[args.code])
    if not creatures:
        raise SystemExit(f"No animal found: {args.code}")
    creature = creatures[0]

    print(f"Training on: {creature.name} ({creature.code})")

    # Create Lenia config
    lenia_cfg = Config.from_animal(creature, args.grid)
    print(f"Grid: {lenia_cfg.grid_shape}, Device: {lenia_cfg.device}")

    # Train
    train(
        lenia_cfg,
        creature,
        size=args.size,
        intervention_type=args.intervention,
        intensity=args.intensity,
        max_episodes=args.max_episodes,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup=args.warmup,
        window=args.window,
        checkpoint_path=args.checkpoint,
        viz_dir=args.viz_dir,
        viz_interval=args.viz_interval,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
