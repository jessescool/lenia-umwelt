#!/bin/bash
#SBATCH --job-name=lenia
#SBATCH --partition=gpu
#SBATCH --prefer=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=cluster/logs/%j.out
#SBATCH --error=cluster/logs/%j.err

cd /cluster/home/jcool01/pettilab/jcool01/Lenia

module load miniforge cuda
source activate lenia

export PYTHONPATH="/cluster/home/jcool01/pettilab/jcool01/Lenia:$PYTHONPATH"

# CMD is passed by dispatch script
eval "$CMD"
