"""
Numerical validation: histogram CDF W1 vs sort-based W1.

Compares the new histogram approach against the exact sort-based formula
on random Lenia-like data to verify accuracy.
"""
import torch
import numpy as np


def wasserstein_sort_nonzero(a, b, threshold=1e-6):
    """Reference sort-based W1 (the old implementation)."""
    a_sorted = torch.sort(a, dim=-1).values
    b_sorted = torch.sort(b, dim=-1).values
    raw_dist = (a_sorted - b_sorted).abs().sum(dim=-1)
    n_active_a = (a > threshold).sum(dim=-1).float()
    n_active_b = (b > threshold).sum(dim=-1).float()
    avg_active = (n_active_a + n_active_b) / 2 + 1e-10
    return raw_dist / avg_active


def test_nonzero_variant():
    """Test wasserstein_1d_torch_nonzero: histogram vs sort."""
    from metrics_and_machinery.distance_metrics import wasserstein_1d_torch_nonzero

    torch.manual_seed(42)
    B, N = 200, 64 * 64  # 200 pairs of 64x64 grids

    # Sparse Lenia-like data: ~5% nonzero, values in [0, 1]
    a = torch.zeros(B, N)
    b = torch.zeros(B, N)
    mask_a = torch.rand(B, N) < 0.05
    mask_b = torch.rand(B, N) < 0.05
    a[mask_a] = torch.rand(mask_a.sum())
    b[mask_b] = torch.rand(mask_b.sum())

    # Histogram version (new)
    hist_result = wasserstein_1d_torch_nonzero(a, b)

    # Sort version (reference)
    sort_result = wasserstein_sort_nonzero(a, b)

    # Compare: relative error should be < 1% for K=8192
    rel_error = ((hist_result - sort_result).abs() / (sort_result.abs() + 1e-10))

    mean_rel = rel_error.mean().item()
    max_rel = rel_error.max().item()

    print(f"wasserstein_1d_torch_nonzero (B={B}, N={N}):")
    print(f"  Mean relative error: {mean_rel:.6f} ({mean_rel*100:.4f}%)")
    print(f"  Max  relative error: {max_rel:.6f} ({max_rel*100:.4f}%)")
    print(f"  Sort mean: {sort_result.mean():.6f}")
    print(f"  Hist mean: {hist_result.mean():.6f}")

    assert mean_rel < 0.01, f"Mean relative error {mean_rel:.4f} exceeds 1%"
    assert max_rel < 0.05, f"Max relative error {max_rel:.4f} exceeds 5%"
    print("  PASSED")


def test_multi_chunk():
    """Test that chunking loop works when B > chunk_size (2048)."""
    from metrics_and_machinery.distance_metrics import wasserstein_1d_torch_nonzero

    torch.manual_seed(99)
    B, N = 3000, 64 * 64  # B > chunk_size=2048, exercises the loop

    a = torch.zeros(B, N)
    b = torch.zeros(B, N)
    mask_a = torch.rand(B, N) < 0.05
    mask_b = torch.rand(B, N) < 0.05
    a[mask_a] = torch.rand(mask_a.sum())
    b[mask_b] = torch.rand(mask_b.sum())

    # Histogram version
    hist_result = wasserstein_1d_torch_nonzero(a, b)

    # Sort version (reference)
    sort_result = wasserstein_sort_nonzero(a, b)

    rel_error = ((hist_result - sort_result).abs() / (sort_result.abs() + 1e-10))
    mean_rel = rel_error.mean().item()
    max_rel = rel_error.max().item()

    print(f"\nMulti-chunk test (B={B} > chunk_size=2048):")
    print(f"  Mean relative error: {mean_rel:.6f} ({mean_rel*100:.4f}%)")
    print(f"  Max  relative error: {max_rel:.6f} ({max_rel*100:.4f}%)")

    assert hist_result.shape == (B,), f"Expected shape ({B},), got {hist_result.shape}"
    assert mean_rel < 0.01, f"Mean relative error {mean_rel:.4f} exceeds 1%"
    assert max_rel < 0.05, f"Max relative error {max_rel:.4f} exceeds 5%"
    print("  PASSED")


def test_all_zero():
    """Test that all-zero inputs return 0 distance."""
    from metrics_and_machinery.distance_metrics import wasserstein_1d_torch_nonzero

    B, N = 10, 64 * 64
    a = torch.zeros(B, N)
    b = torch.zeros(B, N)

    result = wasserstein_1d_torch_nonzero(a, b)

    print(f"\nAll-zero test (B={B}):")
    print(f"  Result: {result}")

    assert result.shape == (B,), f"Expected shape ({B},), got {result.shape}"
    assert (result == 0).all(), f"Expected all zeros, got {result}"
    print("  PASSED")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    test_nonzero_variant()
    test_multi_chunk()
    test_all_zero()
    print("\nAll tests passed!")
