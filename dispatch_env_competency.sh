#!/bin/bash
# dispatch_env_competency.sh — competency sweep: 6 creatures × 12 environments × 4 orientations
#
# Each creature runs as a separate job across all 12 envs.
# Output: results/env_competency/{CODE}/{CODE}_competency.json

ENVS="chips shuriken pegs guidelines membrane-1px membrane-3px capsule ring box funnel corridor noise"

for code in O2u S1s P4al K6s K4s O2v; do
    ./dispatch "python experiments/env_competency_sweep.py --code $code --scale 4 --grid 128x256 --envs $ENVS"
done
