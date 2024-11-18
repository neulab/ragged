#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jhsia2@andrew.cmu.edu
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=20GB

export PYTHONPATH="$PYTHONPATH:/home/jhsia2/ragged"

source /home/jhsia2/.bashrc
conda activate ragged

# Parameters passed from the driver script
# $reranker, $retriever, and $k will be set by sbatch --export
export dataset='nq-dev-kilt'

# Execute the Python script
python -u reranker.py --retriever "$retriever" --dataset "$dataset" --reranker "$reranker" --k "$k"
