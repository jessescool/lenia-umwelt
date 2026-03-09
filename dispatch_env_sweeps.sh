#!/bin/bash
# dispatch_env_sweeps.sh — run all creatures × all orientations × all environments
# Uses run_all_envs.py which reads animals_to_run.json and iterates all envs.
# Output: results/env_sweeps/{CODE}/

# One job per animal (each runs all 12 environments × all orientations)
for code in O2u S1s H3s P4al K6s K4s 3R4s O2v; do
    ./dispatch "python experiments/run_all_envs.py --code $code --scale 2 --steps 600"
done

# Or as a single job (all animals sequentially):
# ./dispatch "python experiments/run_all_envs.py --scale 2 --steps 600"
