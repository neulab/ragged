#!/bin/bash
source /home/jhsia2/.bashrc
conda activate py10

# rm zeno-err.log zeno-out.log

python upload_to_zeno.py

# sbatch --job-name=zeno-hp --gres=gpu:1 --time=0-10:00:00 --mem=100G --output=zeno-out2.log --error=zeno-err2.log zeno_upload.sh