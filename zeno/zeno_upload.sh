#!/bin/bash

source /home/jhsia2/.bashrc
conda activate py10

export PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged

# datasets=("nq" "hotpotqa" "complete_bioasq")
# for dataset in "${datasets[@]}"; do
python zeno_upload.py --dataset $dataset 
# done
# python zeno_upload.py --dataset $dataset --create_project
# done

# sbatch --job-name=zeno --gres=gpu:1 --time=0-10:00:00 --mem=100G --output=zeno_upload.out --error=zeno_upload.err zeno_upload.sh

# sbatch --job-name=zeno-$dataset --gres=gpu:1 --time=0-10:00:00 --mem=100G --output=logs/$dataset-zeno-out.log --error=logs/$dataset-zeno-err.log zeno_upload.sh

# sbatch --job-name=gold-zeno-$dataset --gres=gpu:1 --time=0-10:00:00 --mem=100G --output=gold-$dataset-zeno-out.log --error=gold-$dataset-zeno-err.log zeno_upload.sh