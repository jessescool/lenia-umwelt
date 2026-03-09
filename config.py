"""
Shared configuration defaults for training and evaluation.

This ensures training and eval use matching parameters by default.
Override via command-line arguments as needed.
"""

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
