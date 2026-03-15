#!/bin/bash
# dispatch_all_orientations.sh — precompute 360 heading-aligned orientations per creature
# Scale 4, one job per creature. ~24 MB output each.
#
# Note: 3R4s and H3s are missing from heading_offsets.json.
# Calibrate first: python initializations/calibrate_headings.py -c 3R4s --scale 4

for code in O2u S1s P4al K6s K4s O2v; do
    ./dispatch "python initializations/generate_all_orientations.py -c $code"
done
