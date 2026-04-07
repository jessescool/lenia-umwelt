# Agnosiophobia in Lenia

Code and data for *Agnosiophobia in a Virtual Agent: behavioral and dynamical architecture in Lenia* (ALIFE 2026).

We introduce regions of sensory occlusion into Lenia environments — areas from which no information reaches the creature's kernel — and find that several creatures avoid them, a behavior we term *agnosiophobia*. Targeted occlusions at every pixel reveal morphological sensitivity maps that trace the geometry of each creature's attractor basin. Heading change, the free variable, absorbs the cost of morphological recovery: navigational capacity lives near the boundary of the basin of attraction.

Four creatures are studied: O2u (Orbium), S1s (Scutium), K4s (Quadrupedium), and K6s (Hexapodium).

## Repository Layout

| Path | Purpose |
| --- | --- |
| `substrate/` | Core Lenia primitives: kernel, automaton, simulation runner, upscaling |
| `metrics_and_machinery/` | Wasserstein-1 distance, trajectory metrics, intervention types |
| `orbits/` | Orbit pipeline: raw frames → activation profiles → W1 distances → orbit summary |
| `initializations/` | Pre-settled creature tensors and heading calibration |
| `environments/` | Barrier environment generators (renormalized-kernel occlusion) |
| `experiments/` | Perturbation sweeps, environment competency, analysis scripts |
| `figure_generation/` | Publication figure scripts |
| `utils/` | Rotation, augmentation, GPU-batched rollouts |
| `viz/` | GIF rendering, sensitivity maps, heatmaps |
| `config.py` | Shared constants |
| `animals.json` | Lenia creature catalog (codes, params, RLE body plans) |

## Setup

Python 3.10+.
```bash
pip install torch numpy scipy matplotlib imageio tqdm
```

## Key Concepts

**Orbit** — The neighborhood of a creature's attractor: a dataset of sorted activation profiles sampled across orientations and time. The barycenter c̄ and radius d_max define the reference against which recovery is measured (Section 2 of the paper).

**Sweep** — Exhaustive grid search placing a persistent NxN occlusion at every nonzero pixel, measuring frames to recovery, max distortion, and heading change. Produces the sensitivity maps in Figure 4 of the paper.

**Environment competency** — Score each creature across barrier environments by orbit residence fraction: the proportion of simulation time the creature remains alive and morphologically intact. Produces Figure 2 of the paper.

## Generating Orbits

Four-stage pipeline, each producing a `.pt` file:

```bash
# 1. Raw frames: settle creature at multiple orientations, record frames
python orbits/orbits.py raw --code O2u --scale 4 --grid 128

# 2. Sorted activation profiles
python orbits/orbits.py profile orbits/O2u/s4/O2u_s4_raw.pt

# 3. Pairwise W1 distance matrix
python orbits/orbits.py distances orbits/O2u/s4/O2u_s4_profile.pt

# 4. Orbit summary: barycenter, d_max
python orbits/orbits.py orbit orbits/O2u/s4/O2u_s4_profile.pt
```

## Running Sweeps

Requires an orbit file at the matching scale.

```bash
# Targeted occlusion sweep (3x3 blind region at every nonzero pixel)
python experiments/sweep.py --code O2u --grid 128 --scale 4 --size 3 \
    --intervention-type blind --shortcut --crop --orientations 4

# Erase sweep (zeroing instead of occlusion)
python experiments/sweep.py --code O2u --grid 128 --scale 4 --size 3 \
    --shortcut --crop --orientations 1
```

Output: `results/sweep/{CODE}/{CODE}_x{SCALE}/{CODE}_x{SCALE}_i{SIZE}/{CODE}_x{SCALE}_i{SIZE}_o{ORI}/`

## Environment Competency

```bash
# Score one creature across all barrier environments
python experiments/run_env_competency.py --code O2u --scale 4

# Score all creatures
python experiments/env_competency_sweep.py --all --scale 4
```

Output: `results/env_competency/{CODE}/{CODE}_competency.json`

## Quick Preview

```bash
python run.py O2u   # saves a 500-frame GIF to results/
```

## Citation

TBD
