## Profiling: grid → sorted activation profile

A Lenia grid is an H×W matrix of pixel intensities. Most pixels are zero (background). The **profile** extracts just the signal — the top-m pixel values, sorted descending.

**`profile(grid, m)`** in `distance_metrics.py`:

```python
def profile(grid: torch.Tensor, m: int) -> torch.Tensor:
    """(B, H, W) → (B, m). Top-m values via topk."""
    B, H, W = grid.shape
    top_vals, _ = torch.topk(grid.reshape(B, H * W), m, dim=1, sorted=True)
    return top_vals
```

Batch-only: always takes `(B, H, W)` and returns `(B, m)`. For a single grid, use `.unsqueeze(0)` / `.squeeze(0)`.

This gives you an `(m,)` vector per grid that captures the creature's "activation signature" — its distribution of pixel intensities — without any spatial information.

### How `m` is chosen

In `build_profiles` (`orbits.py:190`), `m` is computed from the orbit's own statistics:

```python
nnz = (frames > 0).sum(dim=(2, 3)).flatten().float()
m = round((nnz.mean() + 2 * nnz.std()).item())
```

So `m = μ + 2σ` of the nonzero pixel count across all orbit frames. This means the profile length is tuned per-creature: big enough to capture the mass in ~97.7% of frames without trimming, small enough to avoid bloating profiles with trailing zeros.

## Why sorted profiles give you W1

The sorted descending profile is the **quantile function** (inverse CDF) of the creature's pixel intensity distribution. The L1 distance between two quantile functions is exactly the **Wasserstein-1 (earth mover's) distance** between the underlying distributions. So:

```
d(a, b) = (1/m) Σⱼ |aⱼ - bⱼ|
```

is the W1 distance. No histogramming, no binning — just L1 between sorted vectors.

The `wasserstein()` primitive computes exactly this:

```python
def wasserstein(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """(B, m) profiles → (B,) W1 distances."""
    return (p - q).abs().mean(dim=1)
```

## Computing c̄: the Fréchet median

`compute_c_bar` (`orbits.py:281`) takes all the orbit profiles — across rotations and frames — and computes the **componentwise median**:

```python
flat = profiles.reshape(-1, profiles.shape[-1])  # (N, m)
return flat.median(dim=0).values                  # (m,)
```

Why median and not mean? Because the distance we're minimizing is W1 (which is L1 between profiles). The point that minimizes the sum of L1 distances is the **median**, not the mean:

```
argmin_c  Σᵢ d(c, pᵢ)
= argmin_c  Σᵢ (1/m) Σⱼ |cⱼ - pᵢⱼ|
```

This decouples per coordinate `j`, and each coordinate's minimizer is `median(p₁ⱼ, ..., pₙⱼ)`. This is the **Fréchet median under W1** — the point in profile space that minimizes the sum of (unsquared) W1 distances to all orbit samples.

## Then: ĉ and d_max

`compute_c_hat` measures the orbit's **radius** around c̄: the mean and max of L1 distances from each orbit profile to c̄. `d_max` is the hard boundary of the orbit — the maximum distance any natural state reaches. The recovery threshold (`λ * d_max`, default λ=1.0) defines the boundary: if a perturbed creature's profile falls within this distance of c̄, it's "back in the orbit."
