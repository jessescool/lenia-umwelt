"""Train PerturBot to find perturbations that create long recovery trajectories.

Reward structure:
    - died:      -1.0  (strong penalty)
    - never:      0.2  (survived but didn't stabilize)
    - recovered:  time / max_time  (normalized recovery time [0, 1])
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List

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
    DEFAULT_STAGES,
    PRE_WARMUP_STEPS,
    RECOVERY_TIMEOUT_MULTIPLIER,
)
from metrics_and_machinery import (
    make_intervention,
    compute_detection_frame,
    WassersteinRecoveryReward,
)
from utils import augment_creature, rollout_parallel, warmup_rollout, write_side_by_side_gif


@dataclass
class StageConfig:
    """Configuration for a single curriculum stage."""
    size: int
    stage_idx: int
    prev_policy_path: Path | None
    max_episodes: int
    viz_dir: Path
    checkpoint_path: Path

    intervention_type: str = "erase"
    intensity: float = 0.3

    lr: float = 1e-3
    batch_size: int = 32
    temperature_start: float = 2.0
    temperature_end: float = 1.0

    observation_steps: int = 300

    wasserstein_threshold: float = 0.00075
    stability_window: int = 20

    convergence_window: int = 100
    convergence_threshold: float = 0.001

    viz_explore_limit: int = 5
    viz_interval: int = 100


def train_stage(
    lenia_cfg: Config,
    creature,
    stage_cfg: StageConfig,
    *,
    verbose: bool = True,
) -> PerturBot:
    """Train PerturBot for one curriculum stage using slow recovery reward."""
    stage_cfg.viz_dir.mkdir(parents=True, exist_ok=True)
    stage_cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    intervention = make_intervention(
        stage_cfg.intervention_type,
        stage_cfg.size,
        intensity=stage_cfg.intensity,
    )

    agent = PerturBot(
        policy_cfg=PolicyConfig(),
        agent_cfg=AgentConfig(
            lr=stage_cfg.lr,
            batch_size=stage_cfg.batch_size,
        ),
        intervention=intervention,
        device=lenia_cfg.device,
    )

    if stage_cfg.prev_policy_path and stage_cfg.prev_policy_path.exists():
        if verbose:
            print(f"  Loading weights from {stage_cfg.prev_policy_path}")
        agent.policy.load_state_dict(
            torch.load(stage_cfg.prev_policy_path, weights_only=True)
        )

    reward_metric = WassersteinRecoveryReward(
        threshold=stage_cfg.wasserstein_threshold,
        stability_window=stage_cfg.stability_window,
    )

    episode = 0
    rewards_window: deque = deque(maxlen=stage_cfg.convergence_window)
    viz_explore_count = 0
    cumulative_reward = 0.0
    outcome_counts = {"died": 0, "never": 0, "recovered": 0}

    def compute_temperature(episode: int, max_episodes: int) -> float:
        if max_episodes <= 1:
            return stage_cfg.temperature_end
        progress = episode / (max_episodes - 1)
        temp = stage_cfg.temperature_start + progress * (stage_cfg.temperature_end - stage_cfg.temperature_start)
        return max(temp, stage_cfg.temperature_end)

    while episode < stage_cfg.max_episodes:
        sim = Simulation(lenia_cfg)
        augmented = augment_creature(creature, lenia_cfg.device)
        sim.place_animal(augmented, center=False)
        frames_pre = warmup_rollout(sim, steps=PRE_WARMUP_STEPS, capture=True)

        controller = RLSim(max_envs=1, agent=agent)
        controller.add_env(sim)
        agent.reset_env(0, sim.board.tensor.device)

        temperature = compute_temperature(episode, stage_cfg.max_episodes)
        pre_action = sim.board.tensor.detach().clone()
        kernel = agent.act(controller, greedy=False, temperature=temperature)
        post_action = sim.board.tensor.detach().clone()
        erased_mass = (pre_action - post_action).clamp(min=0).sum().item()

        # No warmup skipping -- observe the full post-intervention trajectory
        frames_test_post, frames_ctrl_post = rollout_parallel(
            sim,
            pre_action,
            lenia_cfg,
            warmup=0,
            window=stage_cfg.observation_steps,
        )

        test_frames_full = frames_pre + frames_test_post
        ctrl_frames_full = frames_pre + frames_ctrl_post

        reward, outcome = reward_metric.compute_with_status(frames_ctrl_post, frames_test_post)
        outcome_counts[outcome] += 1

        agent.observe_reward(0, reward)

        if verbose:
            print(f"  ep {episode}: reward={reward:.4f}, outcome={outcome}, erased={erased_mass:.2f}, temp={temperature:.2f}")

        rewards_window.append(reward)
        cumulative_reward += reward
        episode += 1

        if reward > 0.3 and viz_explore_count < stage_cfg.viz_explore_limit:
            detection_frame_rel, _ = compute_detection_frame(
                frames_ctrl_post, frames_test_post,
                threshold=stage_cfg.wasserstein_threshold,
                stability_window=stage_cfg.stability_window,
            )
            detection_frame = PRE_WARMUP_STEPS + detection_frame_rel

            path = stage_cfg.viz_dir / f"explore_{viz_explore_count:02d}_ep{episode:05d}_{outcome}.gif"
            write_side_by_side_gif(
                test_frames_full,
                ctrl_frames_full,
                kernel,
                path,
                fps=15,
                pre_warmup_frames=PRE_WARMUP_STEPS,
                post_warmup_frames=0,

                detection_frame=detection_frame,
            )
            if verbose:
                print(f"    Saved exploration GIF: {path.name}")
            viz_explore_count += 1

        if episode % stage_cfg.viz_interval == 0:
            detection_frame_rel, _ = compute_detection_frame(
                frames_ctrl_post, frames_test_post,
                threshold=stage_cfg.wasserstein_threshold,
                stability_window=stage_cfg.stability_window,
            )
            detection_frame = PRE_WARMUP_STEPS + detection_frame_rel

            path = stage_cfg.viz_dir / f"periodic_ep{episode:05d}_{outcome}.gif"
            write_side_by_side_gif(
                test_frames_full,
                ctrl_frames_full,
                kernel,
                path,
                fps=15,
                pre_warmup_frames=PRE_WARMUP_STEPS,
                post_warmup_frames=0,

                detection_frame=detection_frame,
            )

        if verbose and episode % 100 == 0:
            avg_reward = np.mean(rewards_window) if rewards_window else 0.0
            total = sum(outcome_counts.values())
            death_rate = outcome_counts["died"] / total if total > 0 else 0
            recovery_rate = outcome_counts["recovered"] / total if total > 0 else 0
            print(f"  Episode {episode}/{stage_cfg.max_episodes}: avg_reward={avg_reward:.4f}, "
                  f"death_rate={death_rate:.1%}, recovery_rate={recovery_rate:.1%}")

        if len(rewards_window) == stage_cfg.convergence_window:
            mean_reward = np.mean(rewards_window)
            if mean_reward > 0.001 and np.std(rewards_window) < stage_cfg.convergence_threshold:
                if verbose:
                    print(f"  Converged after {episode} episodes (plateau at {mean_reward:.4f})")
                break

    torch.save(agent.policy.state_dict(), stage_cfg.checkpoint_path)
    if verbose:
        avg_final = cumulative_reward / max(episode, 1)
        total = sum(outcome_counts.values())
        print(f"  Stage complete: {episode} episodes, avg_reward={avg_final:.4f}")
        print(f"  Outcomes: died={outcome_counts['died']}/{total} ({outcome_counts['died']/total:.1%}), "
              f"recovered={outcome_counts['recovered']}/{total} ({outcome_counts['recovered']/total:.1%}), "
              f"never={outcome_counts['never']}/{total} ({outcome_counts['never']/total:.1%})")
        print(f"  Saved policy to {stage_cfg.checkpoint_path}")

    return agent


def train_curriculum(
    stages: List[int],
    creature,
    lenia_cfg: Config,
    *,
    max_episodes: int = 5000,
    checkpoint_dir: Path = Path("checkpoints"),
    viz_dir: Path = Path("viz_slow_recovery"),
    run_name: str | None = None,
    intervention_type: str = "erase",
    intensity: float = 0.3,
    verbose: bool = True,
) -> None:
    """Run curriculum training for slow recovery objective."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    observation_steps = int(RECOVERY_TIMEOUT_MULTIPLIER * lenia_cfg.timescale_T)

    if verbose:
        print(f"Observation window (T={lenia_cfg.timescale_T}): {observation_steps} steps")

    for stage_idx, size in enumerate(stages):
        if verbose:
            print(f"\n{'=' * 60}")
            int_desc = f"{intervention_type}" if intervention_type == "erase" else f"{intervention_type} (intensity={intensity})"
            print(f"STAGE {stage_idx + 1}/{len(stages)}: {size}x{size} {int_desc}")
            print(f"{'=' * 60}")

        prev_policy = None
        if stage_idx > 0:
            prev_size = stages[stage_idx - 1]
            prev_policy = checkpoint_dir / f"slow_recovery_{prev_size}x{prev_size}.pt"

        if run_name:
            stage_viz_dir = viz_dir / run_name
        else:
            stage_viz_dir = viz_dir / f"slow_recovery_{size}x{size}"

        stage_cfg = StageConfig(
            size=size,
            stage_idx=stage_idx,
            prev_policy_path=prev_policy,
            max_episodes=max_episodes,
            viz_dir=stage_viz_dir,
            checkpoint_path=checkpoint_dir / f"slow_recovery_{size}x{size}.pt",
            intervention_type=intervention_type,
            intensity=intensity,
            observation_steps=observation_steps,
        )

        train_stage(lenia_cfg, creature, stage_cfg, verbose=verbose)

    if verbose:
        print(f"\nSlow recovery training complete!")
        print(f"  Policies saved to: {checkpoint_dir}/")
        print(f"  Training GIFs in: {viz_dir}/")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PerturBot for slow recovery (long arcs, no death)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        default=DEFAULT_STAGES,
        help=f"Square sizes for curriculum stages (default: {' '.join(map(str, DEFAULT_STAGES))})",
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(DEFAULT_CATALOG),
        help=f"Path to animals JSON catalog (default: {DEFAULT_CATALOG})",
    )
    parser.add_argument(
        "--code",
        default=DEFAULT_ANIMAL_CODE,
        help=f"Animal code to train on (default: {DEFAULT_ANIMAL_CODE})",
    )

    parser.add_argument(
        "--grid",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f"Grid size (default: {DEFAULT_GRID_SIZE})",
    )

    parser.add_argument(
        "--intervention",
        type=str,
        default="erase",
        choices=["erase", "additive"],
        help="Intervention type (default: erase)",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=0.3,
        help="Intensity for additive intervention (default: 0.3)",
    )

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=DEFAULT_MAX_EPISODES,
        help=f"Maximum episodes per stage (default: {DEFAULT_MAX_EPISODES})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Episodes per policy update (default: {DEFAULT_BATCH_SIZE})",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for policy checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=Path("viz_slow_recovery"),
        help="Directory for visualization GIFs (default: viz_slow_recovery)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this training run",
    )

    parser.add_argument(
        "--viz-interval",
        type=int,
        default=100,
        help="Save periodic GIF every N episodes (default: 100)",
    )
    parser.add_argument(
        "--viz-explore",
        type=int,
        default=5,
        help="Save GIFs for first N high-reward episodes (default: 5)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    creatures = load_animals(args.catalog, codes=[args.code])
    if not creatures:
        raise SystemExit(f"No animals found with code '{args.code}' in {args.catalog}")

    creature = creatures[0]
    print(f"Training on: {creature.name} ({creature.code})")

    lenia_cfg = Config.from_animal(creature, args.grid)
    print(f"Grid: {lenia_cfg.grid_shape}, Device: {lenia_cfg.device}")

    train_curriculum(
        stages=args.stages,
        creature=creature,
        lenia_cfg=lenia_cfg,
        max_episodes=args.max_episodes,
        checkpoint_dir=args.checkpoint_dir,
        viz_dir=args.viz_dir,
        run_name=args.run_name,
        intervention_type=args.intervention,
        intensity=args.intensity,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
