"""Metrics and machinery for Lenia perturbation experiments."""

from metrics_and_machinery.distance_metrics import (
    prepare_profile,
    wasserstein,
    mass_distance,
    total_variation,
    wasserstein_nonzero,
    wasserstein_1d_torch_nonzero,
    batched_frame_distances,
)

from metrics_and_machinery.trajectory_metrics import (
    centroid,
    centroid_batched,
    centroid_displacement,
    toroidal_displacement_batched,
    heading_angle,
    speed,
    heading_change,
    speed_change,
    recovery_directness,
    count_peaks,
    time_above_threshold,
    time_above_threshold_integral,
)

from metrics_and_machinery.interventions import (
    Intervention,
    ActionParam,
    SquareEraseIntervention,
    BlindEraseIntervention,
    SquareAdditiveIntervention,
    BarrierIntervention,
    make_intervention,
)

from metrics_and_machinery.competency import (
    orbit_residence_fraction,
    aggregate_competency,
)

from metrics_and_machinery.reward import (
    is_dead,
    is_stable,
    has_recovered,
    time_to_recovery,
    DamageMetric,
    WindowedDamage,
    WassersteinRecoveryReward,
    compute_damage,
    compute_detection_frame,
    compute_timing_windows,
    WARMUP_MULTIPLIER,
    WINDOW_MULTIPLIER,
    MIN_WARMUP_STEPS,
    MIN_WINDOW_STEPS,
    MetricFn,
    DEFAULT_DISTANCE,
)

__all__ = [
    "prepare_profile", "wasserstein", "mass_distance", "total_variation",
    "wasserstein_nonzero", "wasserstein_1d_torch_nonzero", "batched_frame_distances",
    "centroid", "centroid_batched", "centroid_displacement",
    "toroidal_displacement_batched", "heading_angle", "speed",
    "heading_change", "speed_change", "recovery_directness",
    "count_peaks", "time_above_threshold", "time_above_threshold_integral",
    "Intervention", "ActionParam", "SquareEraseIntervention",
    "BlindEraseIntervention", "SquareAdditiveIntervention",
    "BarrierIntervention", "make_intervention",
    "is_dead", "is_stable", "has_recovered", "time_to_recovery",
    "DamageMetric", "WindowedDamage", "WassersteinRecoveryReward",
    "compute_damage", "compute_detection_frame", "compute_timing_windows",
    "WARMUP_MULTIPLIER", "WINDOW_MULTIPLIER", "MIN_WARMUP_STEPS", "MIN_WINDOW_STEPS",
    "MetricFn", "DEFAULT_DISTANCE",
    "orbit_residence_fraction", "aggregate_competency",
]
