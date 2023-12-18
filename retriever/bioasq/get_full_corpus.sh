#!/bin/bash
source /home/jhsia2/.bashrc

conda activate py10

export PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged

python -u get_full_corpus.py

# sbatch --job-name=medline --gres=gpu:1 --time=1-00:00:00 --mem=100G --output=medline-out.log --error=medline-err.log get_full_corpus.sh