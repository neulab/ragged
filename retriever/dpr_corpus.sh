#!/bin/bash
#SBATCH --job-name="dpr_pubmed_p1"
#SBATCH --output="dpr_pubmed_p1.out"
#SBATCH --error="dpr_pubmed_p1.err"
#SBATCH --time=0-12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jhsia2@andrew.cmu.edu
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=100GB

export PYTHONPATH="$PYTHONPATH:/home/jhsia2/ragged"

source /home/jhsia2/.bashrc
conda activate ragged


# export corpus='kilt_wikipedia'
export corpus='pubmed'
python dpr_parallel.py --corpus $corpus --batch_run 1
                                    