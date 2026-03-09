# Lenia RL

Exploring goal-directedness in Lenia by perturbing creatures and measuring their recovery. Agency is revealed by how a creature responds to intervention: recovery trajectories map attractor basins in the creature's dynamical landscape.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `substrate/` | Core Lenia primitives: `lenia.py` (Config, Board, Automaton, Lenia, build_kernel), `animals.py` (catalog loader), `simulation.py` (Simulation runner), `scaling.py` (upscale helpers) |
| `metrics_and_machinery/` | Distance metrics (`distance_metrics.py`), trajectory metrics (`trajectory_metrics.py`), intervention types (`interventions.py`), reward functions (`reward.py`) |
| `learning/` | RL agent (`perturbot.py`), RL sim wrapper (`rl_sim.py`), training scripts (`train.py`, `train_curriculum.py`, `train_slow_recovery.py`) |
| `experiments/` | Analysis scripts: `orbits.py` (orbit pipeline), `sweep.py` (exhaustive grid search), visualization tools |
| `utils/` | Shared utilities: `core.py` (rotation, augmentation), `analysis.py`, `viz.py` (GIF rendering), `batched.py` (GPU batched rollouts) |
| `config.py` | Default constants (grid size, timing, curriculum stages) |
| `animals.json` | Catalog of Lenia creatures (code, metadata, params, RLE body plan) |
| `results/` | All output (orbits, grid search results, checkpoints) |

## Setup

Python 3.10+. Install dependencies:
```bash
pip install torch numpy scipy matplotlib imageio tqdm
```

## Key Concepts

**Orbit**: The set of states a healthy creature visits across rotations and time. Built by running the creature from multiple initial orientations, converting each frame to a sorted activation profile, then computing the barycenter and radius in profile space.

**Sweep**: Exhaustive grid search that erases a small patch at every pixel position and measures recovery. Uses the orbit to detect when (and whether) the creature returns to its normal attractor.

## How To: Generate an Orbit

The orbit pipeline has four stages, each producing a `.pt` file that feeds the next. All stages run on the cluster via `./dispatch`.

```bash
# 1. Collect raw frames: rotate creature through 15 orientations, settle, record 64 frames each
./dispatch "python orbits/orbits.py raw --code O2u --scale 2 --grid 64"
# Output: results/orbits/O2u/s2/O2u_s2_raw.pt

# 2. Convert frames to sorted activation profiles
./dispatch "python orbits/orbits.py profile results/orbits/O2u/s2/O2u_s2_raw.pt"
# Output: results/orbits/O2u/s2/O2u_s2_profile.pt

# 3. (Optional) Pairwise W1 distance matrix between all profiles
./dispatch "python orbits/orbits.py distances results/orbits/O2u/s2/O2u_s2_profile.pt"
# Output: results/orbits/O2u/s2/O2u_s2_distances.pt

# 4. Compute orbit summary: barycenter (c_bar), ĉ and σ
./dispatch "python orbits/orbits.py orbit results/orbits/O2u/s2/O2u_s2_profile.pt"
# Output: results/orbits/O2u/s2/O2u_s2_orbit.pt + .json sidecar
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
# Basic sweep: 2x2 erase, 1 orientation, shortcut mode (non-zero pixels only)
./dispatch "python experiments/sweep.py --code O2u --grid 64 --scale 2 --size 2 --shortcut --orientations 1 --output-dir results/grid_search/O2u_2x2_g64_scale2"

# Multi-orientation sweep
./dispatch "python experiments/sweep.py --code O2u --grid 64 --scale 2 --size 2 --shortcut --orientations 5 --output-dir results/grid_search/O2u_2x2_g64_scale2_5ori"

# Different intervention types
./dispatch "python experiments/sweep.py --code O2u --grid 64 --size 2 --intervention-type blind_erase --shortcut --orientations 1 --output-dir results/grid_search/O2u_2x2_blind"
./dispatch "python experiments/sweep.py --code O2u --grid 64 --size 2 --intervention-type additive --intensity 0.3 --shortcut --orientations 1 --output-dir results/grid_search/O2u_2x2_additive"
```

Sweep options:
- `--code`: Creature code (default: O2u)
- `--grid`: Base grid size (default: 64)
- `--scale`: Upscale factor (default: 1)
- `--size`: Intervention size NxN at base resolution (default: 2)
- `--intervention-type`: `erase`, `additive`, or `blind_erase` (default: erase)
- `--intensity`: Intensity for additive intervention (default: 0.3)
- `--orientations`: Number of rotation orientations (default: 1)
- `--shortcut`: Only test pixels with non-zero mass (much faster)
- `--no-gifs`: Skip GIF generation (faster for large grids)
- `--batch-size`: Override auto-detected GPU batch size
- `--output-dir`: Output directory (required for reproducible naming)

Sweep outputs (in `--output-dir`):
- `analysis/` -- per-orientation `.npy` maps (recovery status, centroid displacement, heading, etc.)
- `recovery_status_maps.png` -- overview figure
- `centroid_convergence.gif` -- animated centroid trajectories
- `slowest_recoveries/`, `never_recovered/`, `death/`, `furthest_centroids/` -- top-K side-by-side GIFs

## Running on the Cluster

All compute-heavy work runs on the cluster. Never run simulations locally.

```bash
# Submit a job (4-hour default, 1 GPU)
./dispatch "python experiments/sweep.py --code O2u --grid 64 --shortcut"

# Custom wall time and GPU count
./dispatch --time 08:00:00 --gpus 2 "python experiments/sweep.py --code O2u --grid 128 --scale 2"

# Interactive GPU session
./dispatch --interactive
```

Code syncs automatically via Mutagen. Check job logs in `cluster/logs/`.

## RL Training

Train an RL agent (PerturBot) to find vulnerable intervention locations:

```bash
# Curriculum training (decreasing square sizes)
./dispatch "python learning/train_curriculum.py --stages 5 4 3 2 1 --max-episodes 5000"

# Quick test
./dispatch "python learning/train_curriculum.py --stages 5 --max-episodes 100"
```
