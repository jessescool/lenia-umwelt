# Excited Regions: Inverse of Kernel Blindness

## The Idea

Kernel blindness masks regions so they contribute **nothing** to convolution — sensory deprivation. The natural inverse: regions that contribute **more than normal** — sensory amplification. Both are special cases of a single **salience field** W(x,y).

## Salience Field

| W value | Regime | Interpretation |
|---------|--------|----------------|
| 0 | Blind | Current blind_mask implementation |
| (0, 1) | Partial blindness | Attenuated sensory input |
| 1 | Normal | Standard Lenia |
| > 1 | **Excited** | Amplified sensory input |

The existing `blind_mask` code (`lenia.py:279-288`) sets `visible = 1 - blind_mask`, which is the special case W = 1 - M where M is binary. Replacing `visible` with an arbitrary salience map W generalizes the entire system.

## Equations

### Eq. 1 — Standard Lenia Excitation (no mask)

The excitation at cell $\mathbf{x}$ is a kernel-weighted average of the state field:

$$E(\mathbf{x}) = (K * C)(\mathbf{x}) = \sum_{\mathbf{y}} K(\mathbf{y} - \mathbf{x}) \cdot C(\mathbf{y})$$

where $K$ is the normalized kernel ($\sum K = 1$) and $C$ is the state field $C(\mathbf{x}) \in [0, 1]$. In the FFT implementation (`lenia.py:289-291`):

$$E = \mathcal{F}^{-1}\!\Big[\, \mathcal{F}[C] \cdot \mathcal{F}[K] \,\Big]$$

### Eq. 2 — Renormalized Convolution with Blind Mask (current code)

When a binary blind mask $M(\mathbf{x}) \in \{0, 1\}$ is present, the current code (`lenia.py:279-288`) first computes visibility:

$$V(\mathbf{x}) = 1 - M(\mathbf{x})$$

then the renormalized excitation:

$$E(\mathbf{x}) = \frac{(K * (C \cdot V))(\mathbf{x})}{(K * V)(\mathbf{x})} = \frac{\displaystyle\sum_{\mathbf{y}} K(\mathbf{y}-\mathbf{x}) \cdot C(\mathbf{y}) \cdot V(\mathbf{y})}{\displaystyle\sum_{\mathbf{y}} K(\mathbf{y}-\mathbf{x}) \cdot V(\mathbf{y})}$$

This is a **weighted average** of $C$ over visible cells only. The denominator renormalizes so that excitation magnitude stays in the range the growth function expects, regardless of how much of the kernel footprint is masked.

### Eq. 3 — Generalized Salience Field (proposed)

Replace the binary visibility $V$ with a continuous **salience field** $W(\mathbf{x}) \geq 0$:

$$E_W(\mathbf{x}) = \frac{(K * (C \cdot W))(\mathbf{x})}{(K * W)(\mathbf{x})} = \frac{\displaystyle\sum_{\mathbf{y}} K(\mathbf{y}-\mathbf{x}) \cdot C(\mathbf{y}) \cdot W(\mathbf{y})}{\displaystyle\sum_{\mathbf{y}} K(\mathbf{y}-\mathbf{x}) \cdot W(\mathbf{y})}$$

FFT implementation:

$$\hat{K} = \mathcal{F}[K]$$

$$E_W = \frac{\mathcal{F}^{-1}\!\Big[\, \mathcal{F}[C \cdot W] \cdot \hat{K} \,\Big]}{\max\!\Big(\, \mathcal{F}^{-1}\!\Big[\, \mathcal{F}[W] \cdot \hat{K} \,\Big],\; \varepsilon \,\Big)}$$

### Special cases

| Salience $W(\mathbf{x})$ | Reduces to | Code path |
|---|---|---|
| $W = 1$ everywhere | Eq. 1 (standard) | `lenia.py:289-291` — denominator $= \sum K = 1$, cancels |
| $W = 1 - M,\; M \in \{0,1\}$ | Eq. 2 (blind mask) | `lenia.py:279-288` — current implementation |
| $W \in (0, 1)$ | Partial blindness | Attenuated sensory input |
| $W > 1$ | **Excited region** | Amplified sensory weight — new |

### Why renormalization preserves dynamics

The key invariant: when $W$ is spatially uniform ($W = c$ for any constant $c > 0$), the constant cancels:

$$E_W(\mathbf{x}) = \frac{\sum_{\mathbf{y}} K(\mathbf{y}-\mathbf{x}) \cdot C(\mathbf{y}) \cdot c}{\sum_{\mathbf{y}} K(\mathbf{y}-\mathbf{x}) \cdot c} = \sum_{\mathbf{y}} K(\mathbf{y}-\mathbf{x}) \cdot C(\mathbf{y}) = E(\mathbf{x})$$

So uniform salience of any magnitude is identical to standard Lenia. Only **spatial variation** in $W$ affects behavior. This means:
- The growth function $G(E)$ always receives excitation in its native range
- $W$ reshapes **what the creature attends to**, not the magnitude of its response
- No risk of pushing excitation into a foreign dynamical regime

### Growth and update (unchanged)

After excitation, the existing pipeline applies identically:

$$G(E) = 2\exp\!\left(-\frac{(E - \mu)^2}{2\sigma^2}\right) - 1 \qquad \text{(Gaussian growth, gtype=2)}$$

$$C(\mathbf{x},\, t + \Delta t) = \mathrm{clamp}\!\Big(\, C(\mathbf{x}, t) + \Delta t \cdot G\big(E_W(\mathbf{x})\big),\; 0,\; 1 \,\Big)$$

The growth function and update rule are agnostic to how $E$ was computed.

## Interpretation: Attentional Spotlight

Renormalized excitation is best understood as an **attentional bias**. In a region where W = 3:

- Cells there count 3x in the weighted average
- But the denominator also scales by 3x
- Net effect: the creature's sensory field is *pulled toward* excited cells

It's like peripheral vision vs. foveal vision. Blind regions are scotomas (no input). Excited regions are foveal — the creature's kernel "looks harder" at those cells. The creature doesn't receive more energy; it receives **more information** from that region relative to others.

### Biological analogies

| Salience | Analogy |
|----------|---------|
| W = 0 | Scotoma, sensory deprivation |
| W < 1 | Peripheral vision, anesthesia |
| W = 1 | Normal perception |
| W > 1 | Foveal attention, stimulant-enhanced perception |

## Predicted Effects

**Spatial salience gradients** (e.g., W increases toward a point):
- Creature's kernel effectively "reaches toward" the high-salience region
- Growth decisions near the boundary are biased by the excited side
- May create asymmetric movement — drawn toward or repelled from the spotlight

**Excited corridor** (stripe of W = 2 across the grid):
- Creature crossing the corridor: its sensory field suddenly over-weights cells inside the corridor
- If the corridor contains the creature's own mass, self-interaction amplifies
- If it contains empty space, the creature "sees more nothing" from that direction — may deflect

**Excited ring around creature** (annulus of W > 1 at kernel radius):
- Amplifies long-range sensory input relative to nearby cells
- Like giving the creature a telephoto lens — it weighs distant neighbors more
- Could destabilize creatures that rely on local self-interaction for coherence

## Experiment Ideas

1. **Salience wall**: W = 1 everywhere except a vertical strip at W = 2. Does the creature deflect, tunnel through, or get trapped? Compare with the blind membrane experiments.

2. **Attentional gradient**: Linear gradient from W = 0.5 to W = 2.0 across the grid. Does the creature migrate toward or away from high salience?

3. **Spotlight tracking**: Small circular region of W = 3 that follows the creature's centroid. Does the creature become more robust to perturbation (amplified self-sensing) or less?

4. **Complementary pairs**: Same geometry as existing blind environments (box, corridor, funnel) but with W > 1 instead of W = 0. Direct comparison reveals whether blindness and excitation are truly inverses in their behavioral effects.

5. **Mixed salience**: Blind on one side, excited on the other. Does the creature preferentially occupy one region?

## Implementation Notes

The code change is minimal — replace the binary `blind_mask` path with a general `salience_map` path:

```python
if salience_map is not None:
    kfft = self._rebuild_kernel_fft(H, W)
    weighted = current * salience_map
    num = irfft2(rfft2(weighted) * kfft, s=(H, W))
    den = irfft2(rfft2(salience_map) * kfft, s=(H, W))
    excitation = num / den.clamp(min=1e-6)
```

Existing blind environments produce salience maps where W in {0, 1}. New excited environments produce W > 1. Everything composes.

## Implementation Path (TODO)

The existing plumbing is close. `Intervention.barrier()` returns a binary blind mask, `Simulation` stores it as `barrier_mask`, and `Lenia.step()` inverts it (`visible = 1 - mask`) before the renormalized convolution. Three touches needed:

1. **`interventions.py`** — Add a `salience()` method to `Intervention` (returns `None` by default). New `SalienceFocusIntervention` class returns a W-map with values > 1 in a square region. Existing interventions unchanged.

2. **`lenia.py`** — Add a `salience_map` kwarg to `step()` / `_excitation()`. If provided, use it directly as W (no inversion). Falls back to `blind_mask` path if no salience map. ~5 lines.

3. **`simulation.py`** — Add a `salience_map` field parallel to `barrier_mask`, pass it through to `lenia.step()`. Interventions that return a salience map populate this instead of (or in addition to) the barrier.

~30 lines of real code. No changes to existing interventions or experiments.
