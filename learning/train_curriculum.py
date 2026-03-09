"""Curriculum training for PerturBot across decreasing perturbation sizes."""

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
    DEFAULT_WARMUP,
    DEFAULT_WINDOW,
    PRE_WARMUP_STEPS,
    RECOVERY_TIMEOUT_MULTIPLIER,
)
from metrics_and_machinery import (
    make_intervention,
    compute_damage,
    compute_detection_frame,
    compute_timing_windows,
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

    warmup: int = 50
    window: int = 50

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
    """Train PerturBot for one curriculum stage."""
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

    # Transfer learning from previous stage
    if stage_cfg.prev_policy_path and stage_cfg.prev_policy_path.exists():
        if verbose:
            print(f"  Loading weights from {stage_cfg.prev_policy_path}")
        agent.policy.load_state_dict(
            torch.load(stage_cfg.prev_policy_path, weights_only=True)
        )

    episode = 0
    rewards_window: deque = deque(maxlen=stage_cfg.convergence_window)
    viz_explore_count = 0
    cumulative_reward = 0.0

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

        frames_test_post, frames_ctrl_post = rollout_parallel(
            sim,
            pre_action,
            lenia_cfg,
            warmup=stage_cfg.warmup,
            window=stage_cfg.window,
        )

        test_frames_full = frames_pre + frames_test_post
        ctrl_frames_full = frames_pre + frames_ctrl_post

        # Damage measured only on the measurement window (last N frames)
        measurement_window = stage_cfg.window
        frames_test_measure = frames_test_post[-measurement_window:]
        frames_ctrl_measure = frames_ctrl_post[-measurement_window:]
        damage = compute_damage(frames_ctrl_measure, frames_test_measure)

        # Normalized reward: incentivizes vulnerable spots over bright spots
        reward = damage / max(erased_mass, 0.01)

        agent.observe_reward(0, reward)

        if verbose:
            print(f"  ep {episode}: reward={reward:.6f}, dmg={damage:.4f}, erased={erased_mass:.2f}, temp={temperature:.2f}")

        rewards_window.append(reward)
        cumulative_reward += reward
        episode += 1

        if reward > 1e-3 and viz_explore_count < stage_cfg.viz_explore_limit:
            detection_frame_rel, _ = compute_detection_frame(
                frames_ctrl_post, frames_test_post
            )
            detection_frame = PRE_WARMUP_STEPS + detection_frame_rel

            path = stage_cfg.viz_dir / f"explore_{viz_explore_count:02d}_ep{episode:05d}.gif"
            write_side_by_side_gif(
                test_frames_full,
                ctrl_frames_full,
                kernel,
                path,
                fps=15,
                pre_warmup_frames=PRE_WARMUP_STEPS,
                post_warmup_frames=stage_cfg.warmup,

                detection_frame=detection_frame,
            )
            if verbose:
                print(f"    Saved exploration GIF: {path.name} (reward={reward:.4f})")
            viz_explore_count += 1

        if episode % stage_cfg.viz_interval == 0:
            detection_frame_rel, _ = compute_detection_frame(
                frames_ctrl_post, frames_test_post
            )
            detection_frame = PRE_WARMUP_STEPS + detection_frame_rel

            path = stage_cfg.viz_dir / f"periodic_ep{episode:05d}.gif"
            write_side_by_side_gif(
                test_frames_full,
                ctrl_frames_full,
                kernel,
                path,
                fps=15,
                pre_warmup_frames=PRE_WARMUP_STEPS,
                post_warmup_frames=stage_cfg.warmup,

                detection_frame=detection_frame,
            )

        if verbose and episode % 100 == 0:
            avg_reward = np.mean(rewards_window) if rewards_window else 0.0
            print(f"  Episode {episode}/{stage_cfg.max_episodes}: avg_reward={avg_reward:.4f}")

        if len(rewards_window) == stage_cfg.convergence_window:
            mean_reward = np.mean(rewards_window)
            if mean_reward > 0.001 and np.std(rewards_window) < stage_cfg.convergence_threshold:
                if verbose:
                    print(f"  Converged after {episode} episodes (plateau at {mean_reward:.4f})")
                break

    torch.save(agent.policy.state_dict(), stage_cfg.checkpoint_path)
    if verbose:
        avg_final = cumulative_reward / max(episode, 1)
        print(f"  Stage complete: {episode} episodes, avg_reward={avg_final:.4f}")
        print(f"  Saved policy to {stage_cfg.checkpoint_path}")

    return agent


def train_curriculum(
    stages: List[int],
    creature,
    lenia_cfg: Config,
    *,
    max_episodes: int = 5000,
    checkpoint_dir: Path = Path("checkpoints"),
    viz_dir: Path = Path("viz_training"),
    run_name: str | None = None,
    intervention_type: str = "erase",
    intensity: float = 0.3,
    objective: str = "damage",
    verbose: bool = True,
) -> None:
    """Run full curriculum training across all stages."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    if objective == "recovery_time":
        warmup = int(5.0 * lenia_cfg.timescale_T)
        window = int(RECOVERY_TIMEOUT_MULTIPLIER * lenia_cfg.timescale_T)
        if verbose:
            print(f"Objective: recovery_time")
    else:
        warmup, window = compute_timing_windows(lenia_cfg.timescale_T)

    if verbose:
        print(f"Timing windows (T={lenia_cfg.timescale_T}): warmup={warmup}, window={window}")

    for stage_idx, size in enumerate(stages):
        if verbose:
            print(f"\n{'=' * 60}")
            int_desc = f"{intervention_type}" if intervention_type == "erase" else f"{intervention_type} (intensity={intensity})"
            print(f"STAGE {stage_idx + 1}/{len(stages)}: {size}x{size} {int_desc}")
            print(f"{'=' * 60}")

        prev_policy = None
        if stage_idx > 0:
            prev_size = stages[stage_idx - 1]
            prev_policy = checkpoint_dir / f"curriculum_{prev_size}x{prev_size}.pt"

        if run_name:
            stage_viz_dir = viz_dir / run_name
        else:
            stage_viz_dir = viz_dir / f"curriculum_{size}x{size}"

        stage_cfg = StageConfig(
            size=size,
            stage_idx=stage_idx,
            prev_policy_path=prev_policy,
            max_episodes=max_episodes,
            viz_dir=stage_viz_dir,
            checkpoint_path=checkpoint_dir / f"curriculum_{size}x{size}.pt",
            intervention_type=intervention_type,
            intensity=intensity,
            warmup=warmup,
            window=window,
        )

        train_stage(lenia_cfg, creature, stage_cfg, verbose=verbose)

    if verbose:
        print(f"\nCurriculum training complete!")
        print(f"  Policies saved to: {checkpoint_dir}/")
        print(f"  Training GIFs in: {viz_dir}/")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for curriculum training."""
    parser = argparse.ArgumentParser(
        description="Curriculum training for PerturBot across perturbation radii",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full curriculum training
  python train_curriculum.py --stages 5 4 3 2 1 --max-episodes 5000

  # Quick test with just the first stage
  python train_curriculum.py --stages 5 --max-episodes 100

  # Resume from stage 3 (assumes stage 0,1,2 checkpoints exist)
  python train_curriculum.py --stages 2 1 --checkpoint-dir checkpoints
        """,
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        default=DEFAULT_STAGES,
        help=f"Square sizes for curriculum stages, e.g. 5=5x5 (default: {' '.join(map(str, DEFAULT_STAGES))})",
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
        help=f"Animal code to train on (default: {DEFAULT_ANIMAL_CODE} = Orbium)",
    )

    parser.add_argument(
        "--grid",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f"Grid size for Lenia simulation (default: {DEFAULT_GRID_SIZE})",
    )

    parser.add_argument(
        "--intervention",
        type=str,
        default="erase",
        choices=["erase", "additive"],
        help="Intervention type: erase (set to 0) or additive (add/subtract intensity). Default: erase",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=0.3,
        help="Intensity for additive intervention (positive=add, negative=subtract). Default: 0.3",
    )

    parser.add_argument(
        "--objective",
        type=str,
        default="damage",
        choices=["damage", "recovery_time"],
        help="Objective function: 'damage' (steady-state divergence) or 'recovery_time' (time to return to attractor). Default: damage",
    )

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=DEFAULT_MAX_EPISODES,
        help=f"Maximum episodes per curriculum stage (default: {DEFAULT_MAX_EPISODES})",
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
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Warmup steps before measuring damage (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"Measurement window steps (default: {DEFAULT_WINDOW})",
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
        default=Path("viz_training"),
        help="Directory for visualization GIFs (default: viz_training)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this training run (creates viz_training/<run-name>/ instead of curriculum_NxN)",
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
        help="Save GIFs for first N non-zero rewards (default: 5)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for curriculum training."""
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
        objective=args.objective,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
