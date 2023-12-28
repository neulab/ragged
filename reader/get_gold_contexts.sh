#!/bin/bash

source /home/jhsia2/.bashrc

export PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged

conda activate py10

python get_gold_contexts.py --dataset $dataset

# python get_gold_contexts --dataset nq-dev-kilt

# python get_gold_contexts --dataset hotpotqa-dev-kilt



# sbatch --job-name=$dataset-gold --gres=gpu:1 --time=0-5:00:00 --mem=100G --output=$dataset-out.log --error=$dataset-err.log get_gold_contexts.sh