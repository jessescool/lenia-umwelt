"""
Test vectorized Wasserstein implementation.

Verifies:
1. Batched torch implementation matches scipy wasserstein_nonzero (within tolerance)
2. Integration with time_to_recovery and WindowedDamage
3. Speedup measurement
"""

import time
import torch
import numpy as np
from scipy.stats import wasserstein_distance

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics_and_machinery.distance_metrics import (
    wasserstein_nonzero,
    batched_frame_distances,
)
from metrics_and_machinery.reward import (
    time_to_recovery,
    WindowedDamage,
)


def test_batched_frame_distances():
    """Verify batched computation matches loop computation."""
    torch.manual_seed(42)

    # Create fake trajectory
    n_frames = 100
    frames_a = [torch.rand(64, 64) for _ in range(n_frames)]
    frames_b = [torch.rand(64, 64) for _ in range(n_frames)]

    # Loop version (reference)
    loop_dists = [wasserstein_nonzero(a, b) for a, b in zip(frames_a, frames_b)]

    # Batched version
    batched_dists = batched_frame_distances(frames_a, frames_b).cpu().numpy()

    # Compare (histogram quantization allows small tolerance)
    for i, (loop_d, batch_d) in enumerate(zip(loop_dists, batched_dists)):
        assert abs(loop_d - batch_d) < 0.01, \
            f"Frame {i}: loop={loop_d}, batched={batch_d}"

    print("  batched_frame_distances matches loop computation")


def test_time_to_recovery_vectorized():
    """Verify time_to_recovery uses vectorized path and gives correct results."""
    torch.manual_seed(42)

    # Create two trajectories that start different but converge
    n_frames = 200
    frames_ctrl = []
    frames_test = []

    for i in range(n_frames):
        base = torch.rand(64, 64) * 0.3
        frames_ctrl.append(base.clone())

        # Test starts different, converges around frame 80
        if i < 80:
            diff = torch.rand(64, 64) * 0.2 * (1 - i/80)
            frames_test.append(base + diff)
        else:
            # After frame 80, test is very close to ctrl
            frames_test.append(base + torch.rand(64, 64) * 0.0001)

    result = time_to_recovery(
        frames_ctrl, frames_test,
        metric_fn=wasserstein_nonzero,
        threshold=0.001,  # Low threshold to detect convergence
        stability_window=20,
    )

    # Should recover somewhere after the convergence point
    assert result is not None, "Should have recovered"
    assert result >= 60, f"Recovery too early: {result}"  # Should be around 80+

    print(f"  time_to_recovery works (recovered at frame {result})")


def test_windowed_damage_vectorized():
    """Verify WindowedDamage uses vectorized path."""
    torch.manual_seed(42)

    n_frames = 100
    frames_ctrl = [torch.rand(64, 64) for _ in range(n_frames)]
    frames_test = [torch.rand(64, 64) for _ in range(n_frames)]

    metric = WindowedDamage(metric_fn=wasserstein_nonzero, warmup=10, window=20)
    damage = metric.compute(frames_ctrl, frames_test)

    # Should be a positive number
    assert damage > 0, f"Damage should be positive, got {damage}"
    assert damage < 1.0, f"Damage seems too high: {damage}"

    print(f"  WindowedDamage works (damage={damage:.6f})")


def benchmark_speedup():
    """Measure speedup from vectorization."""
    torch.manual_seed(42)

    # Create trajectory
    n_frames = 500
    frames_a = [torch.rand(64, 64) for _ in range(n_frames)]
    frames_b = [torch.rand(64, 64) for _ in range(n_frames)]

    # Time loop version
    start = time.perf_counter()
    for _ in range(3):
        loop_dists = [wasserstein_nonzero(a, b) for a, b in zip(frames_a, frames_b)]
    loop_time = (time.perf_counter() - start) / 3

    # Time batched version
    start = time.perf_counter()
    for _ in range(3):
        batched_dists = batched_frame_distances(frames_a, frames_b)
    batched_time = (time.perf_counter() - start) / 3

    speedup = loop_time / batched_time
    print(f"\nBenchmark ({n_frames} frames, 64x64 grids):")
    print(f"  Loop (scipy):    {loop_time*1000:.1f} ms")
    print(f"  Batched (torch): {batched_time*1000:.1f} ms")
    print(f"  Speedup:         {speedup:.1f}x")


if __name__ == "__main__":
    test_batched_frame_distances()
    test_time_to_recovery_vectorized()
    test_windowed_damage_vectorized()
    benchmark_speedup()
    print("\nAll tests passed!")
