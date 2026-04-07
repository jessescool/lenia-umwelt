# config.py — Jesse Cool (jessescool)
"""Default constants for grid size, timing, and display."""

DEFAULT_GRID_SIZE = 64
RECOVERY_TIMEOUT_MULTIPLIER = 50.0  # recovery_window = T * this
PRE_WARMUP_STEPS = 5

DEFAULT_ANIMAL_CODE = "O2u"
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
