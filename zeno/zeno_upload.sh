#!/bin/bash
source /home/jhsia2/.bashrc
conda activate py10

# rm zeno-err.log zeno-out.log

python upload_to_zeno.py

# sbatch --job-name=zeno-nq --gres=gpu:1 --time=0-10:00:00 --mem=100G --output=nq-zeno-out.log --error=nq-zeno-err2.log zeno_upload.sh
# sbatch --job-name=zeno-hp --gres=gpu:1 --time=0-10:00:00 --mem=100G --output=hp-zeno-out.log --error=hp-zeno-err.log zeno_upload.sh