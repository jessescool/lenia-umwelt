#thesis #review

# Research Approach Review: State Space Search for Goal-Directedness
*Generated 2026-02-05 — Claude research review*

## Diagnosis: Theory is Ahead of Experiments

Your theoretical framework is strong. Basin shape as priority structure, the yellow/orange zone distinction, the jagged basin hypothesis — these are genuine ideas. But the experiments don't yet demonstrate any of them. You have:

- **1 creature** (O2u) studied in depth
- **1 perturbation type** (square erase) at the core
- **1 primary output** (damage/recovery heatmap)
- **0 recovery trajectory analyses** (the theoretical heart of the thesis)
- **0 multi-creature comparisons** (the generalizability claim)

The thesis paper currently has a placeholder abstract, thin results, and conditional language around the key claim ("If we get good sims, we can talk about basin shape"). Sam and Chris want heatmaps for 8 animals, displacement maps, velocity, recovery time comparisons. These are the right asks.

---

## What's Missing: 7 Experimental Campaigns

### 1. Multi-Creature Survey (Sam's #1 request)

Run the exhaustive grid search pipeline on 8 creatures spanning the Lenia taxonomy:

| Code | Why |
|------|-----|
| O2u | Baseline (already done) |
| O2b | Same genus, different morphology |
| O4s | Higher-order symmetry |
| S1s | Different family (Scutiformes) |
| OG2g | Different Orbium variant |
| O2v | Close variant for within-family comparison |
| Something sessile/stable | **Critical null control** — a blob that doesn't move |
| "Noodle" chaos state | **Critical null control** — texture resilience without structure |

The sessile creature and the noodle chaos are essential controls. Your thesis claims asymmetric vulnerability = goal-directedness. A symmetric rock-like creature should have a symmetric basin. A noodle should "recover" in a trivial sense (stays noodly) but show no spatial structure. If those nulls don't behave as predicted, the framework has a problem — and that's important to know.

**Requires:** Running `calibrate_thresholds.py` per creature, then `sweep.py`. Infrastructure exists. Mainly compute time (~4-6 hrs total).

---

### 2. Centroid Displacement + Heading Maps

For each perturbation site, track not just "did it recover" but "where did it end up?"

- **Centroid displacement**: `centroid(test, t_final) - centroid(ctrl, t_final)`
- **Heading change**: `angle(v_test) - angle(v_ctrl)` where v = velocity vector at end of recovery
- **Speed change**: `|v_test| - |v_ctrl|`

This operationalizes the **subgoal hierarchy**. If the creature recovers shape but changes heading, shape > heading in its priority ordering. This is the "orientation is actually part of the basin" insight from [[THESIS MAIN]].

**Visualization**: Arrow field overlaid on creature silhouette. Each arrow shows deflection when perturbed there.

**Requires:** Adding centroid tracking to the grid search pipeline. ~2-3 hrs coding + re-running searches.

---

### 3. Recovery Trajectory Classification (Yellow vs. Orange Zones)

This is the **most important missing piece**. The [[Email to Amahury]] beautifully describes passive vs. active resilience. Now measure it.

For each surviving perturbation, compute the full distance time series `d(t) = wasserstein(ctrl_t, test_t)` and extract:

**Recovery Directness (RD):**
$$RD = \frac{d(0)}{\int |d'(t)| \, dt}$$
RD ~ 1 = monotonic decay (passive, "yellow"). RD ~ 0 = lots of back-and-forth (active, "orange").

**Number of recovery peaks:** Local maxima of d(t) after perturbation. Zero peaks = direct return. Multiple = "the creature is going through something."

**Time-above-threshold integral:** How long disrupted, not just when stable.

This produces three new maps per creature: `directness_map`, `n_peaks_map`, `time_above_threshold_map`. Color-coding by trajectory type (green=passive, orange=active, red=dead) IS the figure from the Amahury email, but with real data.

**Requires:** Modifying the grid search to compute summary statistics on-the-fly during rollout. Save full time series for ~40 "interesting" positions per creature. 1-2 days implementation.

---

### 4. Far-From-Equilibrium Characterization

Lenia's equilibrium is the zero field. A creature maintains non-zero activation against this through its dynamics — a dissipative structure.

Track $FFE(t) = W(A^t, \mathbf{0})$ through the perturbation-recovery arc:
1. Pre-perturbation: FFE oscillates at baseline
2. Perturbation: FFE drops
3. Recovery: FFE returns to baseline
4. Death: FFE collapses to 0

This directly addresses Sam's "FAR FROM EQUILIBRIUM" note. The creature IS far from equilibrium; perturbation pushes it toward equilibrium; recovery = climbing back. Death = giving up and falling to the fixed point.

**Requires:** Half a day. Infrastructure exists.

---

### 5. Cross-Creature Comparison (The Payoff)

Once Experiments 1-4 run, compute a "creature profile" per species:

| Metric | What it captures |
|--------|-----------------|
| Survivable fraction | Basin size |
| Mean recovery time | Return speed |
| Critical pixel fraction | % of locations where erasure kills |
| Asymmetry index | Departure from rotational symmetry |
| Mean displacement | How far deflected on survival |
| Passive/active ratio | Yellow vs. orange zone balance |
| FFE distance | How far from equilibrium |

**Key prediction to test**: Survivable fraction (basin size) and asymmetry index (basin shape) should NOT be strongly correlated. A creature can have a large basin (survives most perturbations) but symmetric (boring, like a rock) or a small basin but highly asymmetric (complex priority structure). Finding this decorrelation would be a genuine result.

---

### 6. Controls and Null Hypotheses

- **Random noise baseline:** Same total energy as 2x2 erase, random location. Does position even matter?
- **Geometric symmetry check:** O2u has bilateral symmetry — does the vulnerability map respect it?
- **Scale invariance:** Compare vulnerability maps at scale 1, 2, 4 (partially done)
- **Perturbation type comparison:** Run additive (intensity=0.1) alongside erase. Locations where erase kills but additive just deflects are "whisper-able" spots.

---

### 7. Perturbation Magnitude Sweep ("Persuadability Spectrum")

For a fixed set of spatial locations (e.g., 20 positions spanning lethal/non-lethal/boundary), vary perturbation intensity continuously:
- Erase: opacity from 0.1 to 1.0 in 10 steps
- Additive: intensity from -0.5 to +0.5 in 10 steps

For each (position, intensity) pair, record: survived?, recovery time, heading change, trajectory directness.

This maps the **transition from whisper to shout** at each location. Some spots might show a sharp threshold (fine until destroyed). Others might show graded response (small nudge -> heading change, medium -> morphology disruption, large -> death). The latter are the "persuadable" spots — the cognitive surface.

**Visualization**: For a few key locations, plot a "dose-response curve" (perturbation intensity vs. behavioral change). Compare curves across locations. The SHAPE of these curves is informative: sigmoid = threshold behavior, linear = graded, non-monotonic = interesting.

This directly operationalizes the "bus vs. psychoanalysis" distinction from [[Adri 12-27]]. Some perturbation sites respond proportionally to intensity (psychoanalysis). Others are all-or-nothing (bus).

---

## Things That Should Be Added to the Theoretical Framing

### The Beer Connection Needs to Be Explicit
You're doing the Lenia analogue of [[PAPER – Beer 2014]]. Beer exhaustively enumerated all $2^{24}$ perturbations to a GoL glider. You can't do that for Lenia (continuous, ~65K pixels), so you use RL + exhaustive spatial search. Frame PerturBot as the scalable version of Beer's enumeration. Beer found ~0.5% survivable perturbations and asymmetric vulnerability. Your thesis should directly parallel this: what fraction of perturbations does each Lenia creature survive, and is the vulnerability symmetric?

### Levin's TAME Framework Maps Directly
- Your exhaustive grid search IS the TAME test applied at every spatial location
- The vulnerability map IS Levin's "computational boundary" — the edge of what the system can handle
- Displacement after recovery reveals the cognitive light cone — the creature "noticed" the perturbation
- Passive vs. active resilience maps to different levels in Levin's competency hierarchy

### The Sameness Problem Needs a Decision
You can't keep deferring this. Options:
1. **Commit to Wasserstein nonzero** with explicit justification (it captures redistribution of mass, is translation-sensitive in the right way, works on non-zero support)
2. **Use multiple metrics and show agreement** — if mass_distance, Wasserstein, and TV all agree on the vulnerability map, the choice doesn't matter much
3. **Argue that for Lenia, the sameness problem is tractable** because creatures have characteristic morphologies — unlike arbitrary graphs

Option 2 is probably the most defensible for a thesis. Show the heatmap under all three metrics. If they agree, great. If they disagree, that's interesting too (different metrics capture different "aspects" of sameness).

---

## What Would Falsify Your Claims

| Claim | Evidence needed | Falsified by |
|-------|----------------|-------------|
| Lenia creatures have non-trivial attractor basins | Recovery maps with mixed outcomes | Every perturbation either kills or is shrugged off |
| Vulnerability profiles are asymmetric | Spatially structured vulnerability maps | Circularly symmetric or uniform maps |
| Asymmetry reflects functional organization | Consistent patterns across perturbation types | Random-looking maps that don't correlate across methods |
| Different creatures have different basin shapes | Qualitative differences in multi-creature comparison | All creatures having the same generic pattern |
| "Active resilience" trajectories exist | Meandering returns in trajectory analysis | All recoveries being monotonic |

---

## Suggested Priority Order (2+ month timeline)

### Weeks 1-2: Infrastructure + Data Collection
- Add centroid/heading tracking to grid search pipeline
- Implement recovery trajectory summary statistics (RD, n_peaks, TAT)
- Calibrate thresholds for 7 new creatures
- Run exhaustive grid search for all 8 creatures with full tracking

### Weeks 3-4: Trajectory Analysis + FFE
- Generate yellow/orange/red zone maps for all creatures
- FFE characterization for all creatures
- Save full time series for ~40 interesting positions per creature
- Generate recovery trajectory visualizations

### Weeks 5-6: Comparison + Whispering
- Cross-creature comparison dashboard (profiles, radar plots)
- Controls (random noise, symmetry, scale invariance)
- Perturbation magnitude sweep (Experiment 7)
- Additive vs. erase comparison
- Dose-response curves

### Weeks 7-8: Write + Polish
- Results section with real data
- Figures (vulnerability maps, displacement arrows, trajectory classification, creature profiles, dose-response curves)
- Abstract
- Restructure Methods (formal, reproducible)
- Discussion organized around findings
- Sameness problem treatment (multi-metric comparison)
- Connection to Levin/TAME and Beer explicitly

---

## Implementation Tasks (Code Changes Needed)

1. **`metrics_and_machinery/distance_metrics.py`** — add `centroid()`, `heading_angle()`, `recovery_directness()`, `count_peaks()`
2. **`experiments/sweep.py`** — extend to save centroid displacement, heading change, trajectory summary stats
3. **Cross-creature comparison** — produce cross-creature profile table/radar plot (via sweep analysis tools)
4. **Recovery trajectory analysis** — classify and visualize recovery trajectories (integrated into sweep pipeline)
5. **`experiments/calibrate_thresholds.py`** — run for each new creature (no code change, just execution)

---

## Summary

The thesis idea is good. The theory is ahead of the data. The critical path is:
1. **Breadth**: 8 creatures, not 1
2. **Depth**: Recovery trajectories, not just binary outcomes
3. **Displacement**: Where does it end up, not just "did it survive"
4. **Controls**: Null hypotheses that could falsify the claims
5. **Sameness resolution**: Pick a metric strategy and defend it
6. **Whispering**: Dose-response curves operationalizing the bus vs. psychoanalysis distinction
