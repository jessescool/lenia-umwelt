# Convolution in Lenia: Spatial vs. Frequency Domain

## 1. The Lenia Update Rule

We consider a single-channel Lenia automaton on the discrete torus $\mathbb{T}^2_{H,W} = (\mathbb{Z}/H\mathbb{Z}) \times (\mathbb{Z}/W\mathbb{Z})$. The state at time $t$ is a field $A^t : \mathbb{T}^2_{H,W} \to [0, 1]$. The dynamics are:

$$A^{t+1}(\mathbf{x}) = \operatorname{clamp}\!\Big(A^t(\mathbf{x}) + \Delta t \cdot G\big(U^t(\mathbf{x})\big),\; 0,\; 1\Big)$$

where $\Delta t = 1/T$ is the timestep ($T$ being the creature's characteristic timescale), $G : \mathbb{R} \to [-1, 1]$ is the growth function, and $U^t$ is the excitation field defined below.


## 2. The Excitation Field

The excitation at position $\mathbf{x} \in \mathbb{T}^2_{H,W}$ is the **circular convolution** of the state with the kernel:

$$U^t(\mathbf{x}) = (K \circledast A^t)(\mathbf{x}) \;=\; \sum_{\mathbf{y} \in \mathbb{T}^2_{H,W}} K(\mathbf{x} - \mathbf{y}) \, A^t(\mathbf{y})$$

where subtraction is taken modulo $(H, W)$, i.e., with periodic (torus) boundary conditions. The kernel $K$ has compact support: $K(\mathbf{z}) = 0$ for $\|\mathbf{z}\| > R$, where $R$ is the kernel radius. Its spatial footprint is a $(2R+1) \times (2R+1)$ patch, and $K$ is normalized so $\sum_\mathbf{z} K(\mathbf{z}) = 1$.

We write $\circledast$ for circular convolution to distinguish it from linear convolution $*$.


## 3. Path 1: Spatial Convolution with Circular Padding

The spatial implementation directly evaluates the sum above using `torch.nn.functional.conv2d`, which computes **linear** (non-periodic) cross-correlation. To recover circular convolution from linear convolution, we pre-pad the state periodically.

### 3.1 Setup

Let $A$ denote the $H \times W$ state array and $K$ the $(2R+1) \times (2R+1)$ kernel. We define $\tilde{A}$ as the **circularly padded** state: an $(H + 2R) \times (W + 2R)$ array obtained by wrapping $R$ rows/columns from each edge:

$$\tilde{A}[i, j] = A\big[\,(i - R) \bmod H,\; (j - R) \bmod W\,\big] \qquad \text{for } i \in \{0, \ldots, H + 2R - 1\},\; j \in \{0, \ldots, W + 2R - 1\}$$

This is implemented as `F.pad(state, (R, R, R, R), mode="circular")`.

### 3.2 Convolution vs. cross-correlation

A subtlety: `F.conv2d` actually computes **cross-correlation**, not convolution. The output at position $(i, j)$ is:

$$(\text{F.conv2d})[\,i, j\,] = \sum_{m=0}^{2R} \sum_{n=0}^{2R} K[m, n] \; \tilde{A}[i + m, \; j + n]$$

For a standard convolution, the kernel is flipped: $K[m, n] \to K[2R - m, 2R - n]$. However, our kernel $K$ is **radially symmetric** --- it satisfies $K[m, n] = K[2R - m, 2R - n]$ for all $m, n$ --- so cross-correlation and convolution coincide. This is not a coincidence; it is a structural feature of Lenia's kernel construction.

### 3.3 Evaluating the output

After `F.conv2d` on the padded input, we obtain a "valid-mode" output of shape $H \times W$. At each position $(i, j)$:

$$U[i, j] = \sum_{m=0}^{2R} \sum_{n=0}^{2R} K[m, n] \; \tilde{A}[i + m,\; j + n]$$

Substituting the definition of $\tilde{A}$:

$$= \sum_{m=0}^{2R} \sum_{n=0}^{2R} K[m, n] \; A\big[\,(i + m - R) \bmod H,\; (j + n - R) \bmod W\,\big]$$

With the substitution $m' = m - R$, $n' = n - R$ (ranging over $\{-R, \ldots, R\}$):

$$= \sum_{m'=-R}^{R} \sum_{n'=-R}^{R} K[m' + R, \; n' + R] \; A\big[\,(i + m') \bmod H,\; (j + n') \bmod W\,\big]$$

Identifying $K[m' + R, n' + R]$ with the kernel value at displacement $(m', n')$ and noting that $K$ vanishes for displacements beyond $R$, this is precisely the circular convolution $(K \circledast A)[i, j]$.


## 4. Path 2: FFT Convolution

The frequency-domain implementation exploits the **circular convolution theorem** for the discrete Fourier transform (DFT). This approach has no dependence on kernel size.

### 4.1 The Circular Convolution Theorem

For two functions $f, g : \mathbb{T}^2_{H,W} \to \mathbb{R}$, define the 2D DFT:

$$\hat{f}[\mathbf{k}] = \sum_{\mathbf{x} \in \mathbb{T}^2_{H,W}} f(\mathbf{x}) \, e^{-2\pi i \left(\frac{k_1 x_1}{H} + \frac{k_2 x_2}{W}\right)}$$

Then the DFT of the circular convolution is the pointwise product of the DFTs:

$$\widehat{f \circledast g}[\mathbf{k}] = \hat{f}[\mathbf{k}] \cdot \hat{g}[\mathbf{k}]$$

Therefore: $f \circledast g = \mathcal{F}^{-1}\!\big(\hat{f} \cdot \hat{g}\big)$.

### 4.2 Embedding the kernel

The kernel $K$ is defined on a $(2R+1) \times (2R+1)$ support, centered at position $(R, R)$. To apply the convolution theorem, we need $K$ as a function on the full torus $\mathbb{T}^2_{H,W}$.

**Step 1. Zero-pad.** Place the $(2R+1) \times (2R+1)$ kernel into the top-left corner of an $H \times W$ array, with zeros elsewhere:

$$P[i, j] = \begin{cases} K[i, j] & \text{if } 0 \le i \le 2R \text{ and } 0 \le j \le 2R \\ 0 & \text{otherwise} \end{cases}$$

**Step 2. Roll the center to the origin.** Apply a circular shift by $(-R, -R)$:

$$\bar{K}[i, j] = P\big[\,(i + R) \bmod H,\; (j + R) \bmod W\,\big]$$

This places the kernel center (which was at index $(R, R)$ in the compact support) at index $(0, 0)$ in the torus array.

In code: `torch.roll(padded_k, shifts=(-R, -R), dims=(0, 1))`.

### 4.3 Why the roll is necessary

The convolution theorem computes $(f \circledast g)[\mathbf{x}] = \sum_{\mathbf{y}} f(\mathbf{x} - \mathbf{y}) \, g(\mathbf{y})$, which requires $f$ to be a function on the torus with the correct origin alignment. The kernel's "center" --- the displacement-zero point --- must sit at index $(0, 0)$.

Without the roll, the kernel center sits at $(R, R)$, and the resulting convolution would be **shifted** by $(R, R)$ relative to the correct output. The circular roll corrects this by moving the origin to where the DFT expects it.

Concretely, after rolling, the torus-embedded kernel has the structure:

| Index range | Content |
|---|---|
| $[0, R] \times [0, R]$ | Bottom-right quadrant of original kernel (positive displacements) |
| $[H{-}R, H{-}1] \times [0, R]$ | Top-right quadrant (negative vertical, positive horizontal) |
| $[0, R] \times [W{-}R, W{-}1]$ | Bottom-left quadrant (positive vertical, negative horizontal) |
| $[H{-}R, H{-}1] \times [W{-}R, W{-}1]$ | Top-left quadrant (negative displacements, "wrapped" corner) |

This is the canonical representation of a centered filter on a discrete torus.

### 4.4 The computation

With $\bar{K}$ defined on the torus:

$$U = \mathcal{F}^{-1}\!\Big(\,\hat{\bar{K}} \cdot \hat{A}\,\Big)$$

In code, using the real-valued FFT (exploiting Hermitian symmetry of real inputs):

```python
kernel_fft = rfft2(rolled_kernel)          # precomputed once
excitation = irfft2(rfft2(state) * kernel_fft, s=(H, W))
```

The `rfft2` computes only the non-redundant half of the spectrum (shape $H \times (W/2 + 1)$), halving memory and compute relative to a full complex FFT.


## 5. Equivalence

**Claim.** For any $H \times W$ state $A$ and any radially symmetric kernel $K$ with radius $R < \min(H, W)/2$, the spatial and FFT computations produce identical excitation fields (up to floating-point arithmetic).

**Proof sketch.**

1. The spatial path computes $(K \circledast A)[\mathbf{x}]$ directly from the definition, as shown in Section 3.3. The circular padding ensures that the linear `conv2d` operation reproduces the modular arithmetic of the torus.

2. The FFT path computes $\mathcal{F}^{-1}(\hat{\bar{K}} \cdot \hat{A})$. By the circular convolution theorem, this equals $(\bar{K} \circledast A)[\mathbf{x}]$.

3. It remains to show that $\bar{K}$, viewed as a function on the torus, represents the same kernel as the compact-support $K$ used in the spatial path. By construction:
   - $\bar{K}[i, j] = K[i + R, j + R]$ for $(i, j) \in \{-R, \ldots, R\}^2 \pmod{(H, W)}$
   - $\bar{K}[i, j] = 0$ otherwise (since $K$ has support only within radius $R$)

   This is precisely the torus-embedding of the displacement kernel: $\bar{K}[\mathbf{d}]$ equals the kernel weight at displacement $\mathbf{d}$, with negative displacements wrapping around via modular arithmetic.

4. Therefore both paths compute the same circular convolution. $\square$

**Remark.** The condition $R < \min(H, W)/2$ ensures that the kernel's support does not "wrap around and overlap itself" on the torus. If this condition fails, the torus is too small for the kernel, and the physical meaning of the convolution degenerates. In practice, Lenia grids are always much larger than the kernel diameter.

**Remark on floating-point discrepancy.** The two paths are mathematically identical but numerically distinct. The spatial path accumulates $O((2R+1)^2)$ multiply-adds per output pixel. The FFT path performs $O(HW \log(HW))$ operations globally, with different rounding behavior. Typical relative errors are $O(10^{-6})$ in float32 --- negligible for Lenia dynamics, which are inherently robust to small perturbations in the excitation field.


## 6. Complexity

| | Spatial | FFT |
|---|---|---|
| **Time** | $O(HW \cdot (2R+1)^2)$ | $O(HW \log(HW))$ |
| **Space** | $O((H+2R)(W+2R))$ for padded input | $O(HW)$ for spectrum |
| **Kernel-size dependence** | Quadratic in $R$ | None |
| **GPU efficiency** | Excellent for small $R$ (hardware conv2d) | Excellent for large $R$ |

The crossover point depends on the hardware and grid size. For the kernel radii typical of Lenia creatures ($R = 10\text{--}30$ at base resolution), spatial convolution via cuDNN is often faster on GPU. At scale factor 2--4 ($R = 20\text{--}120$), the FFT path dominates.

The codebase defaults to spatial (`fft=False`) and provides opt-in FFT (`fft=True`) at the `Automaton` constructor. When a blind mask is provided, the FFT path is always used (see Section 7).


## 7. Renormalized Convolution (Blind Mask Variant)

When a subset of cells is designated as "blind" (invisible to the kernel), the standard convolution would underestimate excitation near blind regions, since the kernel's weighted average would include zero-valued blind cells. The fix is **renormalized convolution**.

Let $V : \mathbb{T}^2_{H,W} \to \{0, 1\}$ be the visibility mask, with $V(\mathbf{x}) = 0$ for blind cells and $V(\mathbf{x}) = 1$ for visible cells. The renormalized excitation is:

$$U_{\text{blind}}(\mathbf{x}) = \frac{(K \circledast (A \cdot V))(\mathbf{x})}{(K \circledast V)(\mathbf{x})}$$

The numerator convolves only the visible portion of the state. The denominator measures what fraction of the kernel's weight falls on visible cells at each position. Dividing renormalizes the average to account for the missing neighbors.

This requires **two** convolutions and a pointwise division, making it roughly twice as expensive as the standard computation. The FFT path is always used here (regardless of the `fft` flag) because:
1. Two FFTs share the same precomputed $\hat{\bar{K}}$, so the marginal cost of the second convolution is one forward FFT plus one inverse FFT.
2. The renormalized variant is used primarily as a diagnostic tool (blind erase experiments), where the slight overhead is acceptable.

In code:

```python
visible = 1 - blind_mask
numerator = irfft2(rfft2(state * visible) * kernel_fft, s=(H, W))
denominator = irfft2(rfft2(visible) * kernel_fft, s=(H, W))
excitation = numerator / denominator.clamp(min=1e-6)
```

The `clamp` prevents division by zero at positions where the kernel sees no visible cells at all.

**Interpretation.** Renormalized convolution preserves the excitation that a cell *would* compute if it could only see its visible neighbors. A fully visible field ($V \equiv 1$) recovers the standard convolution exactly. This makes it the natural choice for experiments that restrict a creature's "cognitive light cone" by masking out portions of the field.


## 8. Growth Functions

For completeness, the growth function $G : \mathbb{R} \to [-1, 1]$ maps excitation to update magnitude. Three variants are implemented, parameterized by center $\mu$ and width $\sigma$:

**Gaussian** (default, `gtype=2`):
$$G(u) = 2\exp\!\left(-\frac{(u - \mu)^2}{2\sigma^2}\right) - 1$$

**Polynomial** (`gtype=1`, compact support):
$$G(u) = 2\left[\max\!\left(0,\; 1 - \frac{(u-\mu)^2}{9\sigma^2}\right)\right]^4 - 1$$

**Step** (`gtype=3`):
$$G(u) = \begin{cases} +1 & \text{if } |u - \mu| \le \sigma \\ -1 & \text{otherwise} \end{cases}$$

All three produce values in $[-1, 1]$. The mapping $G$ is applied pointwise to the excitation field, then scaled by $\Delta t$ and added to the current state. The final clamp to $[0, 1]$ ensures the state remains in its valid range.

---

*Note: This document describes the discrete Lenia formulation as implemented in `substrate/lenia.py`. The continuous-limit Lenia (Lenia as a PDE on $\mathbb{R}^2$) replaces the sum with an integral and the torus with $\mathbb{R}^2$ or a continuous torus; the convolution theorem extends to that setting via the continuous Fourier transform.*
