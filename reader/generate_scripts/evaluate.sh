#!/bin/bash
#
#SBATCH --job-name=eval
#SBATCH --output="/data/user_data/afreens/kilt/logs/eval/slurm-%A_%a.out"
#SBATCH --time=800

export PYTHONPATH=/home/afreens/ragged

python /home/afreens/ragged/reader/evaluate_top_k.py