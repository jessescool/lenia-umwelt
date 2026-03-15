#!/bin/bash
# dispatch_additive_sweeps.sh — additive perturbations (intensity 1.0)
# 8 animals × 4 sizes × 4 scales × 3 orientations = 384 jobs
# Results go to results/additive/{CODE}/{CODE}_x{SCALE}_i{SIZE}_o{ORI}

dispatch_sweeps() {
    local scale=$1
    for pair in "O2u 48" "S1s 32" "H3s 72" "P4al 128" "K6s 64" "K4s 64" "3R4s 72" "O2v 48"; do
        local code=${pair%% *} crop=${pair##* }
        for size in 1 2 3 4; do
            for ori in 0 1 2; do
                ./dispatch --h200 "python experiments/sweep.py --situation initializations/$code/s$scale/${code}_s${scale}_o${ori}.pt --code $code --grid 128 --crop $crop --size $size --scale $scale --shortcut --intervention-type additive --intensity 1.0 --output-dir results/additive/$code/${code}_x${scale}_i${size}_o${ori}"
            done
        done
    done
}

# Scale outermost so all scale-1 jobs go first
for scale in 1 2 3 4; do
    dispatch_sweeps $scale
done
