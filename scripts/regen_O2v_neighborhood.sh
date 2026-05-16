#!/bin/bash
# Regenerate O2v neighborhood pipeline: raw → profile → neighborhood, then re-sweep.
# Run via: ./dispatch --time 06:00:00 "bash scripts/regen_O2v_neighborhood.sh"
set -euo pipefail

echo "=== Step 1/4: Raw neighborhood frames (stride=128 auto from T=1280) ==="
python neighborhoods/neighborhoods.py raw -c O2v -s 4 -g 48

echo "=== Step 2/4: Build profile ==="
python neighborhoods/neighborhoods.py profile neighborhoods/O2v/s4/O2v_s4_raw.pt

echo "=== Step 3/4: Build neighborhood summary ==="
python neighborhoods/neighborhoods.py neighborhood neighborhoods/O2v/s4/O2v_s4_profile.pt

echo "=== Step 4/4: Re-dispatch sweep with new neighborhood data ==="
python experiments/sweep.py --code O2v --grid 128 --shortcut --orientations 1 --lambda 2.0

echo "=== All O2v steps complete ==="
