#!/bin/bash
#
#SBATCH --job-name=llama3_70bgen2
#SBATCH --output="/data/tir/projects/tir6/general/afreens/dbqa/logs/reader/llama3_70b/slurm-%A_%a.out"
#SBATCH --error="/data/tir/projects/tir6/general/afreens/dbqa/logs/reader/llama3_70b/slurm-%A_%a.err"
#SBATCH --time=800
#SBATCH --mem=48GB

reader="llama3_70b"
top_ks=("1" "2" "5" "10" "20" "50")
# top_ks=("10")
max_new_tokens=10
max_truncation=8000
# datasets=("nq")
datasets=("nq" "complete_bioasq")
retrievers=("gold")

export PYTHONPATH="/home/afreens/ragged"

# Loop through each retriever
for retriever in "${retrievers[@]}"; do
  # Loop through each dataset
    for dataset in "${datasets[@]}"; do
        # Loop through each topk
            python reader/generate_top_k.py \
            --model_name $reader \
            --retriever $retriever \
            --dataset $dataset \
            --hosted_api_endpoint babel-1-31:8201  \
            --batch_size 50 \
            --max_new_tokens $max_new_tokens \
            --max_truncation $max_truncation 
    done
done