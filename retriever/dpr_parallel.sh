#!/bin/bash
#SBATCH --job-name="dpr_pubmed_6"
#SBATCH --output="dpr_pubmed_6.out"
#SBATCH --error="dpr_pubmed_6.err"
#SBATCH --time=0-10:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jhsia2@andrew.cmu.edu
#SBATCH --gres=gpu:A6000:8
#SBATCH --mem=150GB


# SCRIPT_PATH="$(realpath "$0")"
# RAGGED_PATH="${SCRIPT_PATH%/retriever/dpr.sh}"
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
python dpr_parallel.py --corpus $corpus --dataset $dataset --batch_run 6
                                    