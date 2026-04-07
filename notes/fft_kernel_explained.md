# The Lenia Kernel: What It Is and How It Works

This note explains the core computation that drives Lenia — the kernel convolution — from first principles. Every term is defined before it's used. The companion note `convolution_equivalence.md` gives a rigorous mathematical proof that the two implementations are equivalent; this note focuses on *what* is happening and *why*.

---

## 1. The Big Picture

Lenia is a continuous cellular automaton. The world is a 2D grid of cells, and each cell holds a value between 0 and 1 — think of it as the cell's "activation" or "density." At every timestep, every cell does three things:

1. **Sense**: look at the neighborhood around it and compute a weighted average of nearby cells. This is called the **excitation**.
2. **React**: feed that excitation through a **growth function** that outputs a number between −1 and +1. Positive means "grow," negative means "decay."
3. **Update**: nudge its value up or down by a small amount (the growth scaled by a timestep `dt`), then clamp the result back to [0, 1].

The **kernel** defines step 1: *how* to look at the neighborhood. It's the lens through which each cell perceives its surroundings.


## 2. The Kernel: What It Is

### The Shape

A Lenia kernel is a 2D array of non-negative weights arranged in a ring (donut) shape. Imagine drawing concentric circles out from a cell — cells at a particular distance get high weight, while cells that are too close or too far get low weight. This ring structure is what allows Lenia creatures to sense *around* themselves rather than just under themselves.

The kernel is stored in a square array with side length `2R + 1`, where `R` is the **kernel radius** — the farthest distance a cell can "see." But the non-zero region is **circular**, not square. The distance from the center is computed as Euclidean distance (`sqrt(dx² + dy²)`), and everything beyond radius `R` is zero. Cells in the corners of the square array get no weight. For Orbium (code `O2u`), the base radius is `R = 13`, so the kernel lives in a 27×27 array but only the inscribed disk carries weight.

### Normalization

The kernel weights are normalized to sum to 1. This means the output of the convolution is a *weighted average* of the neighborhood, and its value stays between 0 and 1 (assuming the state is in [0, 1]). Without normalization, the excitation could be arbitrarily large, making the growth function meaningless.

In code, this normalization happens in `_kernel_to_conv_weight` (`substrate/lenia.py:529`):
```python
tensor = tensor / tensor.sum()
```

### The Parameters

A kernel is defined by a few parameters from the creature's specification:

| Parameter | Meaning | Orbium example |
|-----------|---------|----------------|
| `R` | Radius — how far the cell can see | 13 |
| `kn` | Core function type — the shape of the radial profile | 1 (polynomial bump) |
| `b` | Shell amplitudes — how loud each concentric ring is | `"1"` (single shell, full amplitude) |
| `r` | Relative radius cutoff (0 to 1) | 1.0 (default) |

### The Core Functions

The parameter `kn` selects which mathematical function shapes the kernel's radial profile. The input to the core function is a value in (0, 1) representing position within a shell (0 = inner edge, 1 = outer edge). The output is a weight between 0 and 1.

**kn=0 — Smooth bump (exponential compact support)**
Formula: `exp(4 - 1/(r(1-r)))` for `0 < r < 1`, else 0.
This is a very smooth bell that goes exactly to zero at the edges. Infinitely differentiable. Rarely used in practice.

**kn=1 — Polynomial bump** (most common)
Formula: `(4r(1-r))^4` for `0 < r < 1`, else 0.
A smooth hump that peaks in the middle of each shell. This is what Orbium and most standard creatures use. The quartic power makes the peak sharper than a simple parabola.

**kn=3 — Flat-top step**
Formula: 1 for `0.25 ≤ r ≤ 0.75`, else 0.
A plateau — every cell in the middle half of the shell gets equal weight. Produces creatures with different dynamics than the smooth variants.

**kn=4 — Gaussian peak**
Formula: `exp(-((r - 0.5)/0.15)^2 / 2)` for `0 < r < 1`, else 0.
A narrow bell curve centered at `r = 0.5`. Tighter than the polynomial bump, giving a more sharply defined ring.

### Concrete Example: Orbium

Orbium (`O2u`) has `R=13`, `kn=1`, `b="1"`. This means:
- The kernel is 27×27 pixels.
- It has a single shell spanning the full radius.
- The radial profile follows the polynomial bump `(4r(1-r))^4`.
- The result looks like a soft donut: zero at the center, rising to a peak around Euclidean distance 6–7, then falling back to zero at distance 13. The corners of the 27×27 array are zero — the kernel's true footprint is a disk, not a square.


## 3. Multi-Shell Kernels

Some creatures have multiple concentric rings, each with its own amplitude. The `b` parameter controls this.

**How shells work:** The radius is divided into `B` equal-width concentric bands, where `B` is the number of entries in the `b` list. Each band gets its own amplitude multiplier.

For example, `b="1/2,1"` means:
- **Shell 0** (inner ring, distance 0 to R/2): amplitude 0.5
- **Shell 1** (outer ring, distance R/2 to R): amplitude 1.0

The core function `kn` is applied independently within each shell. `Br` maps the distance to a shell index, and `arg` maps position within a shell to the (0, 1) input range for the core function. From `build_kernel` (`substrate/lenia.py:478–497`):

```python
Br = B * dist / r          # distance → shell index (fractional)
idx = floor(Br)             # which shell am I in?
b = weights[idx]            # amplitude for this shell
arg = Br % 1.0              # position within the shell (0 to 1)
kernel = shell * kfunc(arg) * b
```

The `shell` mask ensures everything beyond radius `r` is zero. The final kernel is the product of the core function profile, the per-shell amplitude, and this cutoff mask — then normalized to sum to 1.


## 4. Convolution: What It Does

**Convolution** is the operation that turns the kernel into a sensing mechanism. For every cell on the grid, you:

1. Center the kernel on that cell.
2. Multiply each kernel weight by the state value of the cell it overlaps.
3. Sum all those products.

The result is a single number — the **excitation** — representing what that cell "perceives" about its neighborhood. Do this for every cell and you get the **excitation field**: a new grid the same size as the original, where each entry is the local weighted average.

### Boundary Conditions

Lenia uses **toroidal** (periodic/wrapping) boundary conditions. The grid wraps around: the top edge connects to the bottom, and the left edge connects to the right. So a kernel centered near the edge "sees" cells on the opposite side. Imagine the grid printed on the surface of a donut.

### Why Normalization Matters

Because the kernel sums to 1, the excitation at any cell is a genuine weighted average of the surrounding state values. If the state lives in [0, 1], the excitation also lives in [0, 1]. This is critical because the growth function (Section 8) is designed to interpret excitation values in that range — a value near `μ` means "just right," and values far from `μ` mean "too much" or "too little."


## 5. The Problem: Convolution Is Slow

The direct ("spatial") approach is straightforward but expensive. For each of the `H × W` cells, you sum over all entries in the `(2R+1) × (2R+1)` kernel array — though only the ~πR² entries inside the circular footprint are non-zero. The cost is:

**O(H · W · R²)** — quadratic in the kernel radius.

At base resolution with `R = 13`, that's roughly `π·13² ≈ 531` non-zero multiplications per cell — manageable. But at **scale 4** (where the kernel radius quadruples to `R = 52`), each cell requires roughly `π·52² ≈ 8,495` multiplications. For a 512×512 grid, that's over **2 billion** multiply-adds per timestep.


## 6. The FFT Trick

### The Key Insight

There's a deep mathematical result called the **convolution theorem**:

> Convolution in space is equivalent to multiplication in the frequency domain.

Instead of sliding the kernel over every cell, you can:
1. Transform both the state and the kernel into the **frequency domain**.
2. **Multiply** them pointwise (element by element).
3. Transform the result back to the **spatial domain**.

The output is identical to the direct convolution (up to tiny floating-point rounding differences).

### What the FFT Does

The **Discrete Fourier Transform (DFT)** decomposes a 2D grid into a sum of 2D sine and cosine waves at different frequencies. Think of it as expressing a photograph as a recipe: "this much of the horizontal-stripe pattern, plus this much of the diagonal-stripe pattern, plus…" The **Fast Fourier Transform (FFT)** is an algorithm that computes the DFT in `O(N log N)` operations instead of `O(N²)`.

The **inverse FFT (IFFT)** reassembles the spatial grid from its frequency components.

### The Three-Step Recipe

```
1.  State_freq  = FFT(state)           — transform the grid to frequency domain
2.  Result_freq = State_freq × K_freq  — multiply pointwise (element by element)
3.  Excitation  = IFFT(Result_freq)    — transform back to spatial domain
```

### The Cost

**O(H · W · log(H · W))** — completely independent of the kernel radius.

For a 512×512 grid, that's roughly `262,144 × 18 ≈ 4.7 million` operations, regardless of whether `R = 13` or `R = 52`. Compare to the 3 billion operations for spatial convolution at `R = 52` — the FFT is over **600× faster**.

### Real FFT Optimization

Since the state grid contains only real numbers (no imaginary part), its frequency representation has a symmetry (called **Hermitian symmetry**): the second half of the spectrum is the complex conjugate of the first half. PyTorch's `rfft2` exploits this by computing only the first `W/2 + 1` frequency columns instead of all `W`, halving memory and compute. The inverse is `irfft2`.

In code (`substrate/lenia.py:267`):
```python
excitation = irfft2(rfft2(current) * kfft, s=(H, W))
```


## 7. The Roll: Why We Shift the Kernel

This is the subtlest part of the FFT approach and a common source of confusion.

### The Problem

The DFT's convolution theorem assumes the kernel's "center" (the point corresponding to zero displacement) sits at index `(0, 0)` — the top-left corner of the array. But the kernel, as built by `build_kernel`, is centered at index `(R, R)` — the middle of the `(2R+1) × (2R+1)` array.

If you just zero-pad the kernel into a full-size grid and FFT it, the output will be **shifted** by `(R, R)` relative to the correct result. Every cell's excitation would be attributed to a cell `R` rows down and `R` columns to the right of where it should be.

### The Fix

Before taking the FFT, use `torch.roll` to circularly shift the kernel so its center lands at `(0, 0)`. Circular shifting means values that "fall off" one edge wrap around to the other — which is exactly the toroidal boundary condition.

From the `Automaton.__init__` method (`substrate/lenia.py:188–197`):

```python
H, W = cfg.grid_shape
K = weight.shape[-1]                       # kernel side length (2R+1)
kernel_2d = weight[0, 0]                   # extract the 2D kernel

# Step 1: embed into full-size grid (top-left corner)
padded_k = torch.zeros(H, W)
padded_k[:K, :K] = kernel_2d

# Step 2: roll so kernel center goes from (R, R) to (0, 0)
center = K // 2                            # = R
padded_k = torch.roll(padded_k, shifts=(-center, -center), dims=(0, 1))

# Step 3: FFT (precomputed, reused every step)
kernel_fft = rfft2(padded_k)
```

### What the Rolled Kernel Looks Like

After the roll, the kernel occupies four corners of the full grid:

| Grid region | Contains |
|---|---|
| Top-left `[0..R] × [0..R]` | Bottom-right quadrant of the original kernel (positive displacements) |
| Top-right `[0..R] × [W-R..W-1]` | Bottom-left of kernel (positive vertical, negative horizontal) |
| Bottom-left `[H-R..H-1] × [0..R]` | Top-right of kernel (negative vertical, positive horizontal) |
| Bottom-right `[H-R..H-1] × [W-R..W-1]` | Top-left of kernel (negative displacements — the "wrapped" corner) |

This four-corner layout is exactly how the DFT represents a centered filter on a discrete torus. Everything in between is zero.


## 8. Precomputation

The kernel doesn't change between timesteps — only the state changes. So the expensive FFT of the kernel is computed **once** during `Automaton.__init__` and stored as `self._kernel_fft`.

Each timestep only needs:
1. One forward FFT of the current state → `rfft2(current)`
2. One pointwise multiply → `× self._kernel_fft`
3. One inverse FFT → `irfft2(…)`

### Cache Invalidation

If the grid size changes (e.g., during rescaling experiments), the kernel FFT must be recomputed because the zero-padded, rolled kernel must match the new grid dimensions. The method `_rebuild_kernel_fft` (`substrate/lenia.py:204–218`) handles this lazily: it checks if the current grid shape matches the cached one, and only recomputes if they differ.


## 9. The Growth Function

The excitation field tells each cell what it senses. The **growth function** decides how to react: should the cell grow, decay, or stay the same?

### Parameters

| Parameter | Meaning | Orbium example |
|-----------|---------|----------------|
| `μ` (`m`) | Preferred excitation — the "ideal" neighborhood density | 0.15 |
| `σ` (`s`) | Tolerance — how far from ideal is still okay | 0.015 |
| `gn` | Growth function type | 1 (polynomial) |

### Interpretation

A cell "wants" its excitation to be near `μ`. When excitation equals `μ`, the growth function returns its maximum (+1), meaning maximal growth. As excitation deviates from `μ`, the growth decreases, eventually reaching −1 (maximal decay). The parameter `σ` controls how tolerant the cell is of deviations.

For Orbium: `μ = 0.15`, `σ = 0.015`. The cell is happy when about 15% of its weighted neighborhood is active, and unhappy when the density strays more than ~1.5 percentage points from that target. This narrow tolerance is what creates the delicate balance that sustains the creature.

### The Three Variants

All three variants map excitation → a value in [−1, +1].

**Gaussian (gn=2, default):**
```
G(u) = 2·exp(−(u − μ)² / (2σ²)) − 1
```
A smooth bell curve centered at `μ`. Output is +1 at `u = μ`, and approaches −1 as `u` moves far from `μ`. Most common in Lenia.

**Polynomial (gn=1):**
```
G(u) = 2·max(0, 1 − (u − μ)² / (9σ²))⁴ − 1
```
Similar shape but with **compact support**: the growth goes to exactly −1 beyond `3σ` from `μ`, rather than asymptotically approaching it. The quartic power sharpens the peak. Orbium uses this variant.

**Step (gn=3):**
```
G(u) = +1  if |u − μ| ≤ σ
G(u) = −1  otherwise
```
Binary: either you're in the sweet spot or you're not. Produces very different dynamics — more digital, less analog.


## 10. The Full Update Rule

Putting it all together, one timestep of Lenia:

```
1.  excitation = IFFT( FFT(state) × FFT(kernel) )    — what each cell senses
2.  growth     = G(excitation)                         — how each cell reacts
3.  new_state  = clamp(state + dt × growth, 0, 1)     — update and bound
```

The timestep `dt = 1/T`, where `T` is the creature's **timescale** parameter. For Orbium, `T = 10`, so `dt = 0.1`. This means the state can change by at most ±0.1 per step (since growth is in [−1, +1]). Larger `T` means smoother, more continuous dynamics; smaller `T` means chunkier updates.

The `clamp` at the end keeps cell values in [0, 1]. Without it, growth could push cells below 0 or above 1, which would break the assumption that the excitation (a weighted average of states) also stays in [0, 1].

In code (`substrate/lenia.py:319`):
```python
updated = torch.clamp(current + cfg.dt * growth, 0.0, 1.0)
```


## 11. Renormalized Convolution (Blind Masks)

In our perturbation experiments, we sometimes mask out regions of the grid as "blind" — cells in those regions are invisible to the kernel. This is implemented via a **blind mask**: a grid of 0s and 1s where 1 means "this cell is blind/invisible."

### The Problem with Simple Masking

If you just zero out the blind cells before convolving, any kernel that overlaps a blind region will have some of its weights land on zeros. The weighted sum will be smaller than it should be, because the kernel is "seeing" artificial zeros rather than actual cell values. The excitation is biased downward near blind regions.

### The Fix: Renormalized Convolution

Instead of a single convolution, compute two:

1. **Numerator**: convolve the masked state (blind cells set to 0).
2. **Denominator**: convolve the visibility mask itself (1 where visible, 0 where blind).

Then divide: `excitation = numerator / denominator`.

The denominator tells you what fraction of the kernel's total weight landed on visible cells. Dividing by it rescales the average to account for the missing neighbors. If a kernel is half-covered by a blind region, the denominator at that position is ~0.5, and dividing by it doubles the numerator — restoring the average to what it would be if only the visible cells existed.

In code (`substrate/lenia.py:299–308`):
```python
visible = 1 - blind_mask
exc_fft = rfft2(current * visible) * kfft
vis_fft = rfft2(visible) * kfft
excitation = irfft2(exc_fft, s=(H, W)) / irfft2(vis_fft, s=(H, W)).clamp(min=1e-6)
```

The `.clamp(min=1e-6)` prevents division by zero at positions where the kernel sees no visible cells at all.

When there are no blind cells (visibility is 1 everywhere), the denominator is 1 everywhere, and the formula reduces to the standard convolution. The renormalized version is strictly more general.


## 12. Summary: From Parameters to Dynamics

Starting from a creature's parameter set (e.g., Orbium: `R=13, T=10, kn=1, b="1", m=0.15, s=0.015`):

1. **Build the kernel**: a 27×27 ring of weights shaped by the polynomial bump function, normalized to sum to 1.
2. **Embed and roll**: zero-pad into the full grid, roll the center to (0, 0), take the FFT once.
3. **Each timestep**: FFT the state, multiply by the kernel FFT, inverse FFT → excitation field.
4. **Growth**: pass excitation through the polynomial growth function centered at μ=0.15 with tolerance σ=0.015.
5. **Update**: add `dt × growth` to the state, clamp to [0, 1].

This loop, repeated hundreds or thousands of times, produces the self-organizing patterns we call Lenia creatures — gliders, oscillators, and everything in between. The creature's "personality" emerges entirely from its parameter set, filtered through this kernel→excitation→growth→update pipeline.

---

*Source: `substrate/lenia.py`. See `convolution_equivalence.md` for a formal proof that the FFT and spatial convolution paths produce identical results.*
