#!/bin/bash
#
#SBATCH --job-name=gold_baseline
#SBATCH --output="/data/tir/projects/tir6/general/afreens/dbqa/reader_results/slurm-%A_%a.out"
#SBATCH --time=800


export PYTHONPATH=/home/afreens/ragged
python /home/afreens/ragged/readers/generate_gold_baseline.py --model $1 --dataset $2 --start_offset $3 --end_offset $4 --hosted_api_path $5


