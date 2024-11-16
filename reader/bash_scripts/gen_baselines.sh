#!/bin/bash
#SBATCH --job-name="flant5_baselines"
#SBATCH --output="gen_logs/flant5_baselines.out"
#SBATCH --error="gen_logs/flant5_baselines.err"
#SBATCH --time=0-12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jhsia2@andrew.cmu.edu
#SBATCH --nodes=1
#SBATCH --mem=50GB

# conda activate tgi-env

source /home/jhsia2/.bashrc
conda activate ragged

reader="flanT5"

max_new_tokens=10
max_truncation=2000
datasets=("nq" "hotpotqa" "complete_bioasq")
# datasets=("nq")
retrievers=("no_context" "gold")


export PYTHONPATH="/home/jhsia2/ragged"

# Loop through each retriever
for retriever in "${retrievers[@]}"; do
  # Loop through each dataset
    for dataset in "${datasets[@]}"; do
        python -u /home/jhsia2/ragged/reader/generate_top_k.py \
        --model $reader \
        --retriever $retriever \
        --dataset $dataset \
        --hosted_api_endpoint babel-3-19:9450 \
        --batch_size 1 \
        --api_key $NEULAB_OPENAI_API_KEY\
        --max_new_tokens $max_new_tokens \
        --max_truncation $max_truncation \
        --retrieval_mode top_k\
        --overwrite
    done
done