#!/bin/bash
#
#SBATCH --job-name=gold_baseline
#SBATCH --output="/data/tir/projects/tir6/general/afreens/dbqa/logs/gold_gen/slurm-%A_%a.out"
#SBATCH --time=800


export PYTHONPATH=/home/afreens/ragged
python /home/afreens/ragged/reader/generate_gold_baseline.py --model flanUl2 --dataset nq --start_offset 0 --end_offset 6000 --hosted_api_path babel-4-36 --hosted_api_port 8101
python /home/afreens/ragged/reader/generate_gold_baseline.py --model flanUl2 --dataset hotpotqa --start_offset 0 --end_offset 6000 --hosted_api_path babel-4-36 --hosted_api_port 8101
python /home/afreens/ragged/reader/generate_gold_baseline.py --model flanUl2 --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path babel-4-36 --hosted_api_port 8101


