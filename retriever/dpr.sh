#!/bin/bash
#SBATCH --job-name="dpr_nq"
#SBATCH --output="dpr_nq.out"
#SBATCH --error="dpr_nq.err"
#SBATCH --time=0-20:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jhsia2@andrew.cmu.edu
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=150GB


# SCRIPT_PATH="$(realpath "$0")"
# RAGGED_PATH="${SCRIPT_PATH%/retriever/dpr.sh}"
# export PYTHONPATH="$PYTHONPATH:$RAGGED_PATH"
# echo $PYTHONPATH
export PYTHONPATH="$PYTHONPATH:/home/jhsia2/ragged"

source /home/jhsia2/.bashrc
conda activate ragged


export corpus='kilt_wikipedia'
export dataset='nq-dev-kilt'
# export dataset='hotpotqa-dev-kilt'
# export corpus='pubmed'
# export dataset='complete_bioasq'
python dpr.py --corpus $corpus --dataset $dataset
                                    