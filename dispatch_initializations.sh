#!/bin/bash
# dispatch_situations.sh — prepare settled situation tensors (8 animals × 4 scales = 32 jobs)
# Run this BEFORE dispatch_all_sweeps.sh

for code in O2u S1s H3s P4al K6s K4s 3R4s O2v; do
    for scale in 1 2 3 4; do
        ./dispatch --h200 "python initializations/situations.py -c $code -s $scale --grid 128 --num-orientations 3"
    done
done
