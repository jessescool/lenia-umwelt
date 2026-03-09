# Distance Pipeline: From Orbits to Recovery

A technical writeup of how we measure distance and decide recovery in our Lenia perturbation experiments.

---

## 1. Orbits: Characterizing the Attractor

An **orbit** is our empirical fingerprint of a creature's stable behavior. It answers: *what does this creature normally look like, across time and orientation?*

**Building a raw orbit** (`orbits/orbits.py:101-185`):

For a given creature at a given scale, we:
1. **Rotate** the creature's initial pattern to 15 orientations (0deg through 84deg at 6deg intervals, covering one quadrant)
2. **Settle** each orientation for `30 * T` warmup steps (T = creature's intrinsic timescale), letting transients die out
3. **Recenter** the field after settling
4. **Record** 64 consecutive frames per orientation

This produces a `(15, 64, H, W)` tensor of raw frames — 960 snapshots of the creature doing its thing from many angles.

---

## 2. Profiling: Frames to Sorted Activation Profiles

Raw frames are big and orientation-dependent. We compress each frame into a **sorted activation profile** — a compact, rotation-invariant signature.

**The conversion** (`metrics_and_machinery/distance_metrics.py:profile()`):

```python
def profile(grid: torch.Tensor, m: int) -> torch.Tensor:
    """(B, H, W) → (B, m). Top-m values via topk."""
    B, H, W = grid.shape
    top_vals, _ = torch.topk(grid.reshape(B, H * W), m, dim=1, sorted=True)
    return top_vals
```

For each `(H, W)` frame: flatten all pixel values and grab the top-m via `topk`. The result is an `(m,)` vector of pixel intensities sorted descending — the creature's "activation signature" without spatial information.

**Why sorted profiles?**
- **Rotation invariant**: Sorting discards spatial arrangement
- **Fixed dimensionality**: Always `m` floats regardless of creature position
- **Mathematical grounding**: Sorted profiles are quantile functions; L1 between them is exactly the Wasserstein-1 distance

After profiling, the orbit compresses from `(15, 64, H, W)` raw frames to `(15, 64, m)` profiles — orders of magnitude smaller.

---

## 3. Wasserstein-1: Comparing Two Profiles

We use the **Wasserstein-1 (Earth Mover's) distance** between sorted activation profiles.

**The formula** (`distance_metrics.py:wasserstein()`):

```python
def wasserstein(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """(B, m) profiles → (B,) W1 distances."""
    return (p - q).abs().mean(dim=1)
```

The sorted descending profile is the **quantile function** (inverse CDF) of the pixel intensity distribution. The L1 distance between two quantile functions is exactly the W1 distance. No CDF computation needed — just L1 between sorted vectors.

**Convenience wrapper** (`distance_metrics.py:distance()`):

```python
def distance(a: torch.Tensor, b: torch.Tensor, m: int) -> torch.Tensor:
    """(B, H, W) grids → (B,) W1 distances. Profiles then wasserstein."""
    return wasserstein(profile(a, m), profile(b, m))
```

All three primitives are batch-only — inputs are always `(B, ...)`.

---

## 4. The Orbit Summary: Barycenter and Radius

From the 960 profiles, we distill two numbers that define the orbit's geometry (`orbits/orbits.py:281-343`):

- **Barycenter `c_bar`** (`orbits.py:281-293`): The componentwise median of all 960 profiles. This is the "center" of the orbit — the typical profile.

- **Orbit radius `c_hat`** (`orbits.py:296-312`): The mean W1 distance from each of the 960 profiles to `c_bar`. This measures how far the creature naturally wanders from center.

- **Max orbit distance `d_max`**: The maximum W1 distance any orbit frame reaches from `c_bar`. This is the hard boundary — no natural state ever exceeds it.

Together, `c_hat` and `d_max` define the orbit geometry. A frame whose distance to `c_bar` is less than `c_hat` is squarely *inside* the orbit. `d_max` defines the recovery threshold: if a perturbed creature's distance falls below `λ * d_max` (default λ=1.0), it's back in the orbit.

**Example** (real creature 3R4s at scale 2):
```
c_hat = 0.00158   (orbit radius)
d_max = 0.00248   (max orbit distance → recovery threshold at λ=1)
```

---

## 5. Recovery: Did the Creature Come Back?

After perturbing a creature (erasing a patch, injecting noise, etc.), we roll out the simulation and ask: *did it return to its attractor?*

### Step 1: Compute Distance Over Time

For each frame of the perturbed trajectory, compute:
```
d(t) = wasserstein(profile(frame_t), c_bar)
```
This gives a `[B, T]` timeseries of distances to the orbit barycenter — one per perturbation site, per timestep.

In the sweep pipeline (`utils/batched.py:133-226`), this is done **online** during simulation: each frame is profiled and compared to `c_bar` immediately, then discarded. No need to store the full `[B, T, H, W]` frame tensor.

### Step 2: Check for Death

A creature is **dead** if its total mass (sum of all pixel values) drops below `0.01` at any frame.

### Step 3: Check for Recovery

Among surviving creatures, recovery requires **sustained return to the orbit basin**:

- The distance `d(t)` must fall below `recovery_threshold` (λ * d_max, default λ=1.0)
- And **stay below it** for `stability_window` consecutive frames (default: 20)

The stability window prevents false positives from noise or transient coincidences. A single frame dipping below threshold doesn't count — the creature must demonstrably settle back.

**Implementation** (`distance_metrics.py:time_to_orbit_recovery()`): The batched recovery detector uses a clever 1D convolution — a kernel of `stability_window` ones convolved over the below-threshold boolean tensor. Where the convolution output equals the window size, we've found enough consecutive frames. The first such position is the recovery time.

### Step 4: Classify Outcome

Each perturbation site gets one of three outcomes:

| Outcome | Code | Meaning |
|---------|------|---------|
| **Died** | 0 | Mass dropped below 0.01 |
| **Recovered** | 1 | Distance returned to orbit basin and stayed for 20+ frames |
| **Never** | 2 | Survived but never returned to orbit basin |

These outcomes are assembled into spatial **recovery maps** — `(H, W)` arrays where each pixel records what happened when that position was perturbed. Recovery time maps record *how long* it took.

---

## 6. The Full Chain

```
Creature
  |
  |  [build_orbit]  15 rotations x 64 frames, after 30*T warmup
  v
Raw Frames  (15, 64, H, W)
  |
  |  [profile]  topk → sorted descending
  v
Sorted Profiles  (15, 64, m)
  |
  |  [compute_c_bar, compute_c_hat]  median profile, mean/std distance
  v
Orbit Summary: c_bar (m,), c_hat, d_max
  |
  |                         Perturbed Simulation
  |                               |
  |  [wasserstein per frame]      |  profile each frame, L1 to c_bar
  |                               v
  |                     Distance Timeseries  d(t) for each site
  |                               |
  |  [detect_recovery]            |  death check, threshold + stability window
  |                               v
  +----> Recovery Outcome: died / recovered / never  (per site)
```

---

## Key Constants

| Parameter | Value | Location |
|-----------|-------|----------|
| Orbit rotations | 15 (0-84deg) | `orbits.py:36` |
| Orbit frames | 64 per rotation | `orbits.py:37` |
| Warmup | 30 * T steps | `orbits.py:38` |
| Death threshold | 0.01 (mass) | `distance_metrics.py` |
| Stability window | 20 frames | `distance_metrics.py` |
| Recovery threshold | λ * d_max (default λ=1.0) | `sweep.py`, `config.py` |
