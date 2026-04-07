# metrics_and_machinery/reward.py — Jesse Cool (jessescool)
"""Reward functions for Lenia RL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence, Tuple

import numpy as np
import torch

from metrics_and_machinery.distance_metrics import (
    mass_distance,
    total_variation,
    wasserstein_nonzero,
    batched_frame_distances,
)


WARMUP_MULTIPLIER: float = 5.0   # warmup_steps = T * WARMUP_MULTIPLIER
WINDOW_MULTIPLIER: float = 5.0   # window_steps = T * WINDOW_MULTIPLIER
MIN_WARMUP_STEPS: int = 10       # Floor for very fast creatures (T=1)
MIN_WINDOW_STEPS: int = 20       # Floor for very fast creatures (T=1)

DEFAULT_DISTANCE: MetricFn = wasserstein_nonzero # Default grid-to-grid distance function


def compute_timing_windows(T: float) -> Tuple[int, int]:
    """Compute warmup and measurement windows from creature timescale T."""
    warmup = max(int(WARMUP_MULTIPLIER * T), MIN_WARMUP_STEPS)
    window = max(int(WINDOW_MULTIPLIER * T), MIN_WINDOW_STEPS)
    return warmup, window


# Type alias for grid-to-grid metric functions
MetricFn = Callable[[torch.Tensor, torch.Tensor], float]


def is_dead(frame: torch.Tensor, threshold: float = 0.01) -> bool:
    """Check if creature has died (total mass below threshold)."""
    return frame.sum().item() < threshold


def is_stable(
    frames: Sequence[torch.Tensor],
    metric_fn: MetricFn,
    window: int = 5,
    threshold: float = 0.01,
) -> bool:
    """Have the last `window` frames settled (consecutive distances < threshold)?"""
    if len(frames) < window + 1:
        return False

    for i in range(-window, 0):
        dist = metric_fn(frames[i - 1], frames[i])
        if dist >= threshold:
            return False

    return True


def has_recovered(
    ctrl_frames: Sequence[torch.Tensor],
    test_frames: Sequence[torch.Tensor],
    metric_fn: MetricFn,
    window: int = 5,
    threshold: float = 0.05,
) -> bool:
    """Are the last `window` test frames within `threshold` of corresponding ctrl frames?"""
    if len(test_frames) < window or len(ctrl_frames) < window:
        return False

    for i in range(-window, 0):
        dist = metric_fn(ctrl_frames[i], test_frames[i])
        if dist >= threshold:
            return False

    return True


def time_to_recovery(
    ctrl_frames: Sequence[torch.Tensor],
    test_frames: Sequence[torch.Tensor],
    metric_fn: MetricFn = DEFAULT_DISTANCE,
    threshold: float | None = None,
    stability_window: int = 20,
) -> int | None:
    """Count steps until test trajectory stays within threshold of control for consecutive frames."""
    if threshold is None:
        threshold = 0.00075

    n_frames = min(len(ctrl_frames), len(test_frames))
    if n_frames < stability_window:
        return None

    if metric_fn is wasserstein_nonzero:
        distances = batched_frame_distances(
            ctrl_frames[:n_frames], test_frames[:n_frames]
        ).cpu().numpy()
    else:
        distances = np.array([metric_fn(ctrl_frames[i], test_frames[i])
                              for i in range(n_frames)])

    # Find first window of consecutive frames below threshold
    consecutive_within = 0
    for i, dist in enumerate(distances):
        if dist < threshold:
            consecutive_within += 1
            if consecutive_within >= stability_window:
                return i - stability_window + 1
        else:
            consecutive_within = 0

    return None


class DamageMetric(ABC):
    """Base class for computing damage from ctrl/test trajectories."""

    @abstractmethod
    def compute(
        self,
        ctrl_frames: Sequence[torch.Tensor],
        test_frames: Sequence[torch.Tensor],
    ) -> float:
        """Compute damage between control and test trajectories."""
        raise NotImplementedError


class WindowedDamage(DamageMetric):
    """Skip warmup frames, then compare framewise in window."""

    def __init__(
        self,
        metric_fn: MetricFn = DEFAULT_DISTANCE,
        warmup: int = 10,
        window: int = 20,
    ) -> None:
        self.metric_fn = metric_fn
        self.warmup = warmup
        self.window = window

    def compute(
        self,
        ctrl_frames: Sequence[torch.Tensor],
        test_frames: Sequence[torch.Tensor],
    ) -> float:
        """Return mean distance over window frames."""
        start = self.warmup
        end = self.warmup + self.window

        ctrl_window = ctrl_frames[start:end]
        test_window = test_frames[start:end]

        if self.metric_fn is wasserstein_nonzero:
            distances = batched_frame_distances(ctrl_window, test_window)
            return float(distances.mean().item())
        else:
            distances = [self.metric_fn(c, t) for c, t in zip(ctrl_window, test_window)]
            return sum(distances) / len(distances)


class WassersteinRecoveryReward(DamageMetric):
    """
    Reward long recovery trajectories using Wasserstein distance.

    Reward structure:
        - died:      -1.0  (strong penalty)
        - never:      0.2  (survived but didn't recover)
        - recovered:  time / max_time  (normalized recovery time [0, 1])
    """

    def __init__(
        self,
        threshold: float = 0.00075,
        stability_window: int = 20,
        death_threshold: float = 0.01,
    ) -> None:
        self.threshold = threshold
        self.stability_window = stability_window
        self.death_threshold = death_threshold

    def compute(
        self,
        ctrl_frames: Sequence[torch.Tensor],
        test_frames: Sequence[torch.Tensor],
    ) -> float:
        """Compute reward based on recovery time and survival."""
        reward, _ = self.compute_with_status(ctrl_frames, test_frames)
        return reward

    def compute_with_status(
        self,
        ctrl_frames: Sequence[torch.Tensor],
        test_frames: Sequence[torch.Tensor],
    ) -> Tuple[float, str]:
        """Compute reward and return outcome status."""
        n_frames = min(len(ctrl_frames), len(test_frames))
        max_time = n_frames

        for i, frame in enumerate(test_frames):
            if is_dead(frame, self.death_threshold):
                return -1.0, "died"

        recovery_time = time_to_recovery(
            ctrl_frames,
            test_frames,
            metric_fn=wasserstein_nonzero,
            threshold=self.threshold,
            stability_window=self.stability_window,
        )

        if recovery_time is None:
            return 0.2, "never"
        else:
            return recovery_time / max_time, "recovered"


def compute_detection_frame(
    ctrl_frames: Sequence[torch.Tensor],
    test_frames: Sequence[torch.Tensor],
    *,
    metric_fn: MetricFn = None,
    threshold: float = 0.00075,
    stability_window: int = 20,
    death_threshold: float = 0.01,
) -> Tuple[int, str]:
    """Compute when recovery/death is detected for GIF phase coloring.
    Returns (detection_frame, outcome) where outcome is "died", "recovered", or "never".
    """
    n_frames = min(len(ctrl_frames), len(test_frames))

    for i, frame in enumerate(test_frames):
        if is_dead(frame, death_threshold):
            return (i, "died")

    if metric_fn is None:
        metric_fn = wasserstein_nonzero
    recovery_time = time_to_recovery(
        ctrl_frames,
        test_frames,
        metric_fn=metric_fn,
        threshold=threshold,
        stability_window=stability_window,
    )

    if recovery_time is not None:
        return (recovery_time, "recovered")
    else:
        return (n_frames, "never")


def compute_damage(
    ctrl_frames: Sequence[torch.Tensor],
    test_frames: Sequence[torch.Tensor],
    metric: DamageMetric | None = None,
) -> float:
    """Compute damage between control and test trajectories."""
    if metric is None:
        metric = WassersteinRecoveryReward()
    return metric.compute(ctrl_frames, test_frames)
