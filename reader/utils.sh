#!/bin/bash
#
#SBATCH --job-name=utils_run
#SBATCH --output="/data/user_data/afreens/kilt/slurm-%A_%a.out"
#SBATCH --time=600
python /home/afreens/ragged/readers/utils.py


