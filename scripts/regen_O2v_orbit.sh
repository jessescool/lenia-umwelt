#!/bin/bash
# Regenerate O2v orbit pipeline: raw → profile → orbit, then re-sweep.
# Run via: ./dispatch --time 06:00:00 "bash scripts/regen_O2v_orbit.sh"
set -euo pipefail

echo "=== Step 1/4: Raw orbit frames (stride=128 auto from T=1280) ==="
python orbits/orbits.py raw -c O2v -s 4 -g 48

echo "=== Step 2/4: Build profile ==="
python orbits/orbits.py profile orbits/O2v/s4/O2v_s4_raw.pt

echo "=== Step 3/4: Build orbit summary ==="
python orbits/orbits.py orbit orbits/O2v/s4/O2v_s4_profile.pt

echo "=== Step 4/4: Re-dispatch sweep with new orbit data ==="
python experiments/sweep.py --code O2v --grid 128 --shortcut --orientations 1 --lambda 2.0

echo "=== All O2v steps complete ==="
