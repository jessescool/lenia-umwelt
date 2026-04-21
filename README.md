# Agnosiophobia in Lenia

Code and data for *Agnosiophobia in a Virtual Agent: behavioral and dynamical architecture in Lenia* (ALIFE 2026 submission).

Authors: **Jesse Cool**, Benedikt Hartl, Michael Levin, Samantha Petti

We subject four Lenia creatures to informational perturbations, occlusions that render parts of their sensory apparatus blind, and ask what their response reveals about their agential and goal-directed status. Creatures avoid regions of low information, a behavior we call **agnosiophobia**, trading heading for morphology. This behavior is neither designed nor selected for. The paper argues that this behavior is visible in the shape of each creature's *basin of attraction* and that novel competencies emerge from the coupling of substrate symmetry and environment, not from either alone. We argue that this is the machinery behind patterns with fixed representations demonstrating competence and goal-directedness when subjected to novel environments.

## Lenia

```bash
python run.py O2u
```

Loads creature `O2u` from `animals.json`, Chan's original catalog, and runs a unperturbed simulation on a $128^2$-grid, writing to `results/O2u_preview.gif`. Any creature code from `animals.json` works.

## Pipeline

1. **Initializations** — settle each creature at a known heading
2. **Neighborhoods** — characterize each creature's natural variation
3. **Environments** — place creatures in barrier environments and watch or score them
4. **Targeted perturbation** — map vulneravility along a creature's 'body'

## Initializations

Prepares a settled initialization for each creature and orientation.

```bash
python initializations/generate_initializations.py --code O2u --scale 4
```

Outputs `initializations/{CODENAME}/s{N}/…` (settled `.pt` per orientation)

## Neighborhoods

A creature's canonical morphology achieves a set of states through the simulation's state space. At finite grid resolution the creature's morphology is not perfectly constant and drifts (heading relative to grid axes, small phase shifts) as it moves. We call this set of morphologies the creature's *neighborhood*. In sorted-activation-profile space (a rotation-invariant representation of a state), the neighborhood is the ball $\mathcal{N}(\bar c, d_{\max})$: the set of profiles within $d_{\max}$ of the barycenter $\bar c$. A later run counts as "in the neighborhood" if its profile lies within this ball. This is the baseline that recovery and competency are measured against.

```bash
python neighborhoods/neighborhoods.py raw          --code O2u --scale 4
python neighborhoods/neighborhoods.py profile       neighborhoods/O2u/s4/O2u_s4_raw.pt
python neighborhoods/neighborhoods.py distances     neighborhoods/O2u/s4/O2u_s4_profile.pt
python neighborhoods/neighborhoods.py neighborhood  neighborhoods/O2u/s4/O2u_s4_profile.pt
```

Outputs per creature per scale: `neighborhoods/{CODENAME}/s{N}/{CODENAME}_s{N}_{raw,profile,distances,neighborhood}.pt`.

### Symbols

A state $s \in \mathbb{R}^{H \times W}$ is mapped to its **sorted activation profile** $\pi(s) \in \mathbb{R}^m$: its top-$m$ cell values in descending order, with $m$ fixed per creature. The L1 distance between sorted profiles equals the $W_1$ (Wasserstein-1) distance between their activation measures, so the profile space is rotation- and translation-invariant by construction. We write $d(x, y) = \|\pi(x) - \pi(y)\|_1$.

From a dataset $C$ of unperturbed snapshots (paper: 5400 samples = 90 orientations × 600 frames) we define

- $\bar c$ — componentwise median of $\{\pi(c) : c \in C\}$; a $W_1$ barycenter of the creature's canonical profiles.
- $d_{\max} = \max_{c \in C} \; d(c, \bar c)$ — furthest any canonical snapshot strays from $\bar c$.
- **Neighborhood** $\mathcal{N}(\bar c, d_{\max}) = \{x : d(x, \bar c) \le d_{\max}\}$.
- **Recovered** at time $t$: the mean of $d(s_{t-k+1}, \bar c), \ldots, d(s_t, \bar c)$ is below $d_{\max}$, with $k = 5$ (temporal smoothing window).
- **Dead**: total mass $\sum s_t < 0.01$ at any frame.
- Otherwise **not recovered** (explosion / metamorphosis).

## Environments

Environments are binary mask tensors (`1.0` = barrier, `0.0` = open) that live in `environments/`: `funnel`, `corridor`, `pegs`, `shuriken`, `box`, `capsule`, `chips`, `guidelines`, `membrane`, `noise`, `ring`. To place a creature at a chosen orientation inside one of these and watch what happens run:

```bash
python experiments/run_single_env_gif.py --code O2u --ori 120 --env guidelines --steps 2000
```

Requires an initialization for the creature at the given scale and heading (see above). Writes a GIF and a tensor of grid states to `results/new/`.

## Targeted perturbation

Apply a small occlusion at every non-zero position along a creature's 'body,' then measure whether/how fast it returns to its neighborhood. Yields per-pixel maps of recovery time, centroid displacement, and heading change, i.e. the creature's local landscape of vulnerability.

```bash
python experiments/sweep.py --code O2u --scale 4 --grid 128 --shortcut --init initializations/O2u/s4/O2u_s4_o0.pt
```

Outputs in `results/sweep/{CODENAME}/{CODENAME}_x{SCALE}_i{SIZE}/`.

## Repository layout

```
substrate/               core Lenia update (Config, Board, Automaton, Simulation)
metrics_and_machinery/   distance metrics, interventions, competence scoring
initializations/         settled starting states and heading calibration
neighborhoods/           natural-variation pipeline and outputs
environments/            barrier mask tensors and generators
experiments/             env competency, targeted perturbation, analysis
figure_generation/       paper figures
viz/                     GIFs, heatmaps, overlays
utils/                   shared helpers (rotation, GPU batching, i/o)
run.py                   preview GIF
```

## Setup

Python 3.10+.
```bash
pip install torch numpy scipy matplotlib imageio tqdm
```

The pipeline stages above are GPU-heavy — in the paper they ran on a SLURM cluster. A GPU (Nvidia, since we use CUDA) is recommended for full sweeps; a CPU is fine for `run.py` and small single-creature tests.

## Citation
**coming soon**
