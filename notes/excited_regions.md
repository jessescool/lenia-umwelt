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

## Renormalized Convolution with Salience

The general form, directly extending the existing FFT path:

```
K_fft = FFT(kernel)

numerator   = IFFT( FFT(state * W) * K_fft )
denominator = IFFT( FFT(W)         * K_fft )
excitation  = numerator / clamp(denominator, min=eps)
```

This is a **weighted average** of the state, where kernel shape determines *which* neighbors matter and W determines *how much* each neighbor's contribution is scaled before averaging.

### Why renormalization matters

Without the denominator, W > 1 inflates raw excitation values, pushing cells into different growth regimes — the creature's physics change. With renormalization, the denominator compensates: if excited cells contribute 2x to the numerator, they also contribute 2x to the denominator. The *ratio* stays bounded. What changes is the **weighting** — excited cells have disproportionate influence on the average, but the average itself stays in the normal excitation range.

This is the key property: renormalized excitation reshapes **what the creature pays attention to** without breaking its dynamical regime. The creature's growth function still receives excitation values it "knows how to handle."

### Current code as special case

From `lenia.py:279-288`:
```python
visible = 1 - blind_mask          # W = 1 - M
masked_current = current * visible  # state * W
exc_fft = rfft2(masked_current) * kfft
vis_fft = rfft2(visible) * kfft
excitation = irfft2(exc_fft, s=(H, W))
vis_weight = irfft2(vis_fft, s=(H, W))
excitation = excitation / vis_weight.clamp(min=1e-6)
```

To support excited regions, the only change is letting W take values > 1. The math is identical. No new machinery needed.

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
