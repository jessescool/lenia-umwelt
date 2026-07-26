# A minimal Umwelt in Lenia

All embodied agents are fundamentally patterns in physiological or other excitable media, blurring the distinction between objects and processes. What competencies do these patterns possess? We equip the creatures from Chan's Lenia with the capacity to sense regions of occlusion, via a biologically inspired modification to the update rule. These occlusions act as obstacles and are received in a variety of ways. When not immediately destroyed, many creatures steer away from occlusions — some immediately, some only after turning toward the obstacle. Occlusions can serve to push a creature into a new dynamical regime, changing its morphology and by proxy, its identity. Occasionally, occlusions provoke the generation of a second, identical creature. Response depends jointly on the occlusion's size, shape, and placement relative to the creature's heading and on the parameters of the creature itself.

## Paper

**Cool, J.**, Hartl, B., Levin, M., & Petti, S. (2026). *To appear at ALIFE 2026*. [arXiv:2605.30708](https://arxiv.org/abs/2605.30708)

## Demo

[Our GitHub Pages](jessescool.github.io/lenia-umwelt)

## Selected References

- Rosenblueth, A., Wiener, N., & Bigelow, J. (1943). Behavior, Purpose and Teleology. *Philosophy of Science* 10(1), 18–24. [doi:10.1086/286788](https://doi.org/10.1086/286788)
- Maturana, H. R., & Varela, F. J. (1980). *Autopoiesis and Cognition: The Realization of the Living*. D. Reidel. [doi:10.1007/978-94-009-8947-4](https://doi.org/10.1007/978-94-009-8947-4)
- Beer, R. D. (2014). The Cognitive Domain of a Glider in the Game of Life. *Artificial Life* 20(2), 183–206. [doi:10.1162/ARTL_a_00125](https://doi.org/10.1162/ARTL_a_00125)
- Chan, B. W.-C. (2019). Lenia: Biology of Artificial Life. *Complex Systems* 28(3), 251–286. [doi:10.25088/ComplexSystems.28.3.251](https://doi.org/10.25088/ComplexSystems.28.3.251)
- Chan, B. W.-C. (2020). Lenia and Expanded Universe. *ALIFE 2020*, 221–229. [doi:10.1162/isal_a_00297](https://doi.org/10.1162/isal_a_00297)
- Levin, M. (2022). Technological Approach to Mind Everywhere (TAME). *Frontiers in Systems Neuroscience* 16. [doi:10.3389/fnsys.2022.768201](https://doi.org/10.3389/fnsys.2022.768201)
- Heylighen, F. (2023). The meaning and origin of goal-directedness: a dynamical systems perspective. *Biological Journal of the Linnean Society* 139(4), 370–387. [doi:10.1093/biolinnean/blac060](https://doi.org/10.1093/biolinnean/blac060)
- Zhang, T., Goldstein, A., & Levin, M. (2024). Classical Sorting Algorithms as a Model of Morphogenesis. *Adaptive Behavior*. [doi:10.1177/10597123241269740](https://doi.org/10.1177/10597123241269740)

## Project Code

### Lenia

```bash
python run.py O2u
```

Loads creature `O2u` from `animals.json`, Chan's original catalog, and runs a unperturbed simulation on a $128^2$-grid, writing to `results/O2u_preview.gif`. Any creature code from `animals.json` works.

### Pipeline

1. **Initializations** — settle each creature at a known heading
2. **Neighborhoods** — characterize each creature's natural variation
3. **Environments** — place creatures in barrier environments and watch or score them
4. **Targeted perturbation** — map vulneravility along a creature's 'body'

### Initializations

Prepares a settled initialization for each creature and orientation.

```bash
python initializations/generate_initializations.py --code O2u --scale 4
```

Outputs `initializations/{CODENAME}/s{N}/…` (settled `.pt` per orientation)

### Neighborhoods

A creature's canonical morphology achieves a set of states through the simulation's state space. At finite grid resolution the creature's morphology is not perfectly constant and drifts (heading relative to grid axes, small phase shifts) as it moves. We call this set of morphologies the creature's *neighborhood*. In sorted-activation-profile space (a rotation-invariant representation of a state), the neighborhood is the ball $\mathcal{N}(\bar c, d_{\max})$: the set of profiles within $d_{\max}$ of the barycenter $\bar c$. A later run counts as "in the neighborhood" if its profile lies within this ball. This is the baseline that recovery and competency are measured against.

```bash
python neighborhoods/neighborhoods.py raw          --code O2u --scale 4
python neighborhoods/neighborhoods.py profile       neighborhoods/O2u/s4/O2u_s4_raw.pt
python neighborhoods/neighborhoods.py distances     neighborhoods/O2u/s4/O2u_s4_profile.pt
python neighborhoods/neighborhoods.py neighborhood  neighborhoods/O2u/s4/O2u_s4_profile.pt
```

Outputs per creature per scale: `neighborhoods/{CODENAME}/s{N}/{CODENAME}_s{N}_{raw,profile,distances,neighborhood}.pt`.

#### Symbols

A state $s \in \mathbb{R}^{H \times W}$ is mapped to its **sorted activation profile** $\pi(s) \in \mathbb{R}^m$: its top-$m$ cell values in descending order, with $m$ fixed per creature. The L1 distance between sorted profiles equals the $W_1$ (Wasserstein-1) distance between their activation measures, so the profile space is rotation- and translation-invariant by construction. We write $d(x, y) = \|\pi(x) - \pi(y)\|_1$.

From a dataset $C$ of unperturbed snapshots (paper: 5400 samples = 90 orientations × 600 frames) we define

- $\bar c$ — componentwise median of $\{\pi(c) : c \in C\}$; a $W_1$ barycenter of the creature's canonical profiles.
- $d_{\max} = \max_{c \in C} \; d(c, \bar c)$ — furthest any canonical snapshot strays from $\bar c$.
- **Neighborhood** $\mathcal{N}(\bar c, d_{\max}) = \{x : d(x, \bar c) \le d_{\max}\}$.
- **Recovered** at time $t$: the mean of $d(s_{t-k+1}, \bar c), \ldots, d(s_t, \bar c)$ is below $d_{\max}$, with $k = 5$ (temporal smoothing window).
- **Dead**: total mass $\sum s_t < 0.01$ at any frame.
- Otherwise **not recovered** (explosion / metamorphosis).

### Environments

Environments are binary mask tensors (`1.0` = barrier, `0.0` = open) that live in `environments/`: `funnel`, `corridor`, `pegs`, `shuriken`, `box`, `capsule`, `chips`, `guidelines`, `membrane`, `noise`, `ring`. To place a creature at a chosen orientation inside one of these and watch what happens run:

```bash
python experiments/run_single_env_gif.py --code O2u --ori 120 --env guidelines --steps 2000
```

Requires an initialization for the creature at the given scale and heading (see above). Writes a GIF and a tensor of grid states to `results/new/`.

### Targeted perturbation

Apply a small occlusion at every non-zero position along a creature's 'body,' then measure whether/how fast it returns to its neighborhood. Yields per-pixel maps of recovery time, centroid displacement, and heading change, i.e. the creature's local landscape of vulnerability.

```bash
python experiments/sweep.py --code O2u --scale 4 --grid 128 --shortcut --init initializations/O2u/s4/O2u_s4_o0.pt
```

Outputs in `results/sweep/{CODENAME}/{CODENAME}_x{SCALE}_i{SIZE}/`.

### Repository layout

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

### Setup

Python 3.10+.
```bash
pip install torch numpy scipy matplotlib imageio tqdm
```

The pipeline stages above are GPU-heavy — in the paper they ran on a SLURM cluster. A GPU (Nvidia, since we use CUDA) is recommended for full sweeps; a CPU is fine for `run.py` and small single-creature tests.

Enjoy!
