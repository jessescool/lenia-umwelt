"""Default constants for grid size, timing, and curriculum stages."""

# Grid configuration
DEFAULT_GRID_SIZE = 64  # Must match between training and eval!

# Training defaults
DEFAULT_MAX_EPISODES = 10000
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 32  # Episodes collected before policy update

# Rollout parameters — legacy fallbacks; prefer compute_timing_windows(T) from reward.py.
DEFAULT_WARMUP = 50  # 5.0 * T=10 (Orbium default)
DEFAULT_WINDOW = 50  # 5.0 * T=10 (Orbium default)

# Recovery time objective: extended window for observing recovery dynamics
RECOVERY_TIMEOUT_MULTIPLIER = 50.0  # recovery_window = T * 50 (e.g., 500 steps for Orbium)

# Pre-intervention warmup (let creature stabilize after placement)
PRE_WARMUP_STEPS = 5

# Curriculum stages (square sizes)
DEFAULT_STAGES = [5, 4, 3, 2, 1]

# Default creature
DEFAULT_ANIMAL_CODE = "O2u"  # Orbium unicaudatus
DEFAULT_CATALOG = "animals.json"

# ── Display constants (shared across figure scripts) ──

import matplotlib.pyplot as _plt

_tab10 = _plt.cm.tab10
CREATURE_ORDER = ['O2u', 'S1s', 'K4s', 'K6s']
CREATURE_COLORS = {
    'O2u': _tab10(3), 'S1s': _tab10(1), 'K4s': _tab10(0), 'K6s': _tab10(2),
}

ENV_ORDER = [
    'pegs', 'chips', 'shuriken', 'guidelines',
    'membrane-1px', 'membrane-3px',
    'box', 'capsule', 'ring', 'corridor', 'funnel', 'noise',
]
