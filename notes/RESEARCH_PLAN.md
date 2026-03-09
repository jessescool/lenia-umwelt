# Research Plan: Thesis Completion
# Generated 2026-02-06

## Creature Panel (8 creatures, all T=10 except where noted)

| Code | Name | Rationale | Status |
|------|------|-----------|--------|
| O2u | Orbium unicaudatus | Primary model, existing data | DONE (scale 2, with trajectory stats) |
| O2b | Orbium bicaudatus | Two tails, bilateral vulnerability test | PARTIAL (no trajectory stats) |
| O2ui | Orbium unicaudatus ignis | Same shape, different dynamics | PARTIAL (no trajectory stats) |
| O4s | Synorbium solidus | Multi-channel, different organizational principle | NOT STARTED |
| S1s | Scutium solidus | Stationary oscillator, critical contrast | NOT STARTED |
| P4cp | Paraptera cavus pedes | Different morphology family | NOT STARTED |
| OG2g | Gyrorbium gyrans | Rotational locomotion | NOT STARTED |
| O4d | Parorbium dividuus | Can divide/reproduce | NOT STARTED |

## Execution Order

### Phase 1: Calibrate (1-2 hours)
```bash
for code in O2b O2bi O2ui O4s S1s P4cp OG2g O4d; do
    python experiments/calibrate_thresholds.py --code $code --scales 2
done
```
Add results to CREATURE_THRESHOLDS in reward.py.

### Phase 2: Grid searches (8-10 hours)
```bash
# Rerun with trajectory stats
for code in O2b O2ui; do
    python experiments/sweep.py --code $code --size 2 --grid 64 --scale 2 --orientations 5 --save-top-k-gifs 10
done

# New creatures
for code in O4s S1s P4cp OG2g O4d; do
    python experiments/sweep.py --code $code --size 2 --grid 64 --scale 2 --orientations 5 --save-top-k-gifs 10
done
```

### Phase 3: Cross-perturbation comparison (1-2 hours)
```bash
python experiments/sweep.py --code O2u --size 2 --grid 64 --scale 2 \
    --intervention-type additive --intensity 0.3 --orientations 5
```

### Phase 4: Analysis
Cross-creature comparison using sweep results in `results/grid_search/`.

## Key Figures Needed
1. Gallery: 2x4 recovery status maps
2. O2u deep dive panel (6 subplots)
3. Radar comparison
4. Recovery trajectory time series (passive vs active vs lethal)
5. Cross-modal consistency (erase vs additive)
6. Basin shape schematic
