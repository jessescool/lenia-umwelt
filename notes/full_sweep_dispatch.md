# Full Sweep Dispatch — Scale 4, All Animals

6 animals × 4 sizes × 4 orientations = **96 jobs**
Partition: `--preempt --h200`, wall time: 4h (default)
Flags: `--shortcut`, grid 128 (FFT is default; use `--conv` to opt into spatial conv2d)

## Animals

| Code | λ   | Crop |
|------|-----|------|
| O2u  | 1.0 | 28   |
| S1s  | 1.0 | 28   |
| P4al | 1.0 | 64   |
| K6s  | 1.0 | 32   |
| K4s  | 2.0 | 32   |
| O2v  | 1.0 | 28   |

## Script: `dispatch_sweep_x4.sh`

```bash
#!/bin/bash
# 6 animals × 4 sizes × 4 orientations = 96 jobs
# Scale 4, --shortcut, preempt H200s, 4h wall time

for pair in "O2u 1.0 28" "S1s 1.0 28" "P4al 1.0 64" "K6s 1.0 32" "K4s 2.0 32" "O2v 1.0 28"; do
    code=$(echo $pair | cut -d' ' -f1)
    lambda=$(echo $pair | cut -d' ' -f2)
    crop=$(echo $pair | cut -d' ' -f3)
    for size in 1 2 3 4; do
        for ori in 0 1 2 3; do
            ./dispatch --preempt --h200 "python experiments/sweep.py \
                --initialization initializations/$code/s4/${code}_s4_o${ori}.pt \
                --code $code --grid 128 --crop $crop \
                --size $size --scale 4 --shortcut \
                --recovery-lambda $lambda"
        done
    done
done
```

## Output structure

Auto-generated: `results/sweep/{CODE}/{CODE}_x4/{CODE}_x4_i{SIZE}/{CODE}_x4_i{SIZE}_o{ORI}/`
