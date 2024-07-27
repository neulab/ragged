#!/bin/bash
#SBATCH --job-name="gtr_b"
#SBATCH --output="gtr_b.out"
#SBATCH --error="gtr_b.err"
#SBATCH --time=0-15:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jhsia2@andrew.cmu.edu
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=150GB


# SCRIPT_PATH="$(realpath "$0")"
# RAGGED_PATH="${SCRIPT_PATH%/retriever/gtr.sh}"
# export PYTHONPATH="$PYTHONPATH:$RAGGED_PATH"
# echo $PYTHONPATH
export PYTHONPATH="$PYTHONPATH:/home/jhsia2/ragged"

source /home/jhsia2/.bashrc
conda activate ragged


# export corpus='kilt_wikipedia'
# export dataset='nq-dev-kilt'
# export dataset='hotpotqa-dev-kilt'
export corpus='pubmed'
export dataset='complete_bioasq'
python gtr.py --corpus $corpus --dataset $dataset
                                    