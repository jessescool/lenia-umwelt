#!/bin/bash
# dispatch_all_sweeps.sh — 8 animals × 4 sizes × 3 orientations × 1 scale = 96 jobs
# Scale 4, no --shortcut (exhaustive grid)

for pair in "O2u 48" "S1s 32" "H3s 72" "P4al 128" "K6s 64" "K4s 64" "3R4s 72" "O2v 48"; do
    code=${pair%% *} crop=${pair##* }
    for size in 1 2 3 4; do
        for ori in 0 1 2; do
            ./dispatch --h200 "python experiments/sweep.py \
                --situation initializations/$code/s4/${code}_s4_o${ori}.pt \
                --code $code --grid 128 --crop $crop \
                --size $size --scale 4"
        done
    done
done
