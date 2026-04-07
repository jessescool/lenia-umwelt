# Lenia RL

Exploring goal-directedness in Lenia by perturbing creatures and measuring their recovery. Agency is revealed by how a creature responds to intervention: recovery trajectories map attractor basins in the creature's dynamical landscape.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `substrate/` | Core Lenia primitives: `lenia.py` (Config, Board, Automaton, Lenia, build_kernel), `animals.py` (catalog loader), `simulation.py` (Simulation runner), `scaling.py` (upscale helpers) |
| `metrics_and_machinery/` | Distance metrics, trajectory metrics, intervention types, reward functions |
| `orbits/` | Orbit pipeline: raw frames → activation profiles → W1 distances → orbit summary |
| `initializations/` | Pre-settled creature tensors and generation scripts |
| `environments/` | Barrier environment definitions and generators |
| `experiments/` | Sweep, competency analysis, figure scripts |
| `figure_generation/` | ALIFE publication figure scripts |
| `utils/` | Shared utilities: rotation, augmentation, GPU batched rollouts |
| `viz/` | Visualization: GIFs, maps, heatmaps, orbit plots |
| `config.py` | Default constants (grid size, timing, curriculum stages) |
| `animals.json` | Catalog of Lenia creatures (code, metadata, params, RLE body plan) |
| `results/` | All output (sweep results, competency data, figures) |

## Setup

Python 3.10+. Install dependencies:
```bash
pip install torch numpy scipy matplotlib imageio tqdm
```

## Key Concepts

**Orbit**: The set of states a healthy creature visits across rotations and time. Built by running the creature from multiple initial orientations, converting each frame to a sorted activation profile, then computing the barycenter and radius in profile space.

**Sweep**: Exhaustive grid search that erases a small patch at every pixel position and measures recovery. Uses the orbit to detect when (and whether) the creature returns to its normal attractor.

## How To: Generate an Orbit

The orbit pipeline has four stages, each producing a `.pt` file that feeds the next.

```bash
# 1. Collect raw frames: rotate creature through 15 orientations, settle, record 64 frames each
python orbits/orbits.py raw --code O2u --scale 4 --grid 128
# Output: orbits/O2u/s4/O2u_s4_raw.pt

# 2. Convert frames to sorted activation profiles
python orbits/orbits.py profile orbits/O2u/s4/O2u_s4_raw.pt
# Output: orbits/O2u/s4/O2u_s4_profile.pt

# 3. (Optional) Pairwise W1 distance matrix between all profiles
python orbits/orbits.py distances orbits/O2u/s4/O2u_s4_profile.pt
# Output: orbits/O2u/s4/O2u_s4_distances.pt

# 4. Compute orbit summary: barycenter (c_bar), ĉ and σ
python orbits/orbits.py orbit orbits/O2u/s4/O2u_s4_profile.pt
# Output: orbits/O2u/s4/O2u_s4_orbit.pt + .json sidecar
```

Options for the `raw` subcommand:
- `--code`, `-c`: Creature code from `animals.json` (required)
- `--scale`, `-s`: Upscale factor (default: 1)
- `--grid`, `-g`: Base grid size (default: 64)
- `--rotations`, `-r`: Number of orientations (default: 15)
- `--frames`, `-f`: Frames to record per rotation (default: 64)
- `--warmup-multiplier`, `-w`: Settle time = T * this (default: 30.0)

## How To: Run a Sweep

A sweep requires an orbit file for the creature at the matching scale. Generate the orbit first (steps 1, 2, 4 above), then:

```bash
# Basic sweep: size-3 erase, 1 orientation, shortcut mode, cropped output
python experiments/sweep.py --code O2u --grid 128 --scale 4 --size 3 \
    --shortcut --crop --orientations 1
# Auto output: results/sweep/O2u/O2u_x4/O2u_x4_i3/O2u_x4_i3_o0/

# Multi-orientation sweep
python experiments/sweep.py --code O2u --grid 128 --scale 4 --size 3 \
    --shortcut --crop --orientations 4

# Different intervention types
python experiments/sweep.py --code O2u --grid 128 --size 3 \
    --intervention-type blind_erase --shortcut --crop --orientations 1
python experiments/sweep.py --code O2u --grid 128 --size 3 \
    --intervention-type additive --intensity 0.3 --shortcut --crop --orientations 1

# Use pre-settled initializations (skips warmup)
python experiments/sweep.py --code O2u --grid 128 --scale 4 --size 3 \
    --init --shortcut --crop --orientations 1
```

Sweep options:
- `--code`: Creature code (default: O2u)
- `--grid`: Base grid size (default: 128)
- `--scale`: Upscale factor (default: 1)
- `--size`: Intervention size NxN at base resolution (default: 2)
- `--intervention-type`: `erase`, `blind_erase`, `blind`, or `additive` (default: erase)
- `--duration`: Steps blind masks are active; `-1` = persistent; omit = default
- `--intensity`: Intensity for additive intervention (default: 0.3)
- `--orientations`: Number of rotation orientations (default: 1)
- `--shortcut`: Only test pixels with non-zero mass (much faster)
- `--crop`: Crop output maps to creature bounding box
- `--init`: Use pre-settled initializations from `initializations/`
- `--recovery-lambda`: Recovery threshold multiplier (default: 1.0; K4s needs 2.0)
- `--no-gifs`: Skip GIF generation (faster for large grids)
- `--batch-size`: Override auto-detected GPU batch size

Output naming: `results/sweep/{CODE}/{CODE}_x{SCALE}/{CODE}_x{SCALE}_i{SIZE}/{CODE}_x{SCALE}_i{SIZE}_o{ORI}/`

Sweep outputs (per orientation directory):
- `analysis/` — per-orientation `.npy` maps (recovery status, centroid displacement, heading, etc.)
- `*_recovery_status_map.png` — overview figure
- `slowest_recoveries/`, `never_recovered/`, `death/`, `furthest_centroids/` — top-K side-by-side GIFs
