#!/bin/bash
#
#SBATCH --job-name=llama2_7bgen
#SBATCH --output="/data/tir/projects/tir6/general/afreens/dbqa/logs/reader/llama2_7b/slurm-%A_%a.out"
#SBATCH --error="/data/tir/projects/tir6/general/afreens/dbqa/logs/reader/llama2_7b/slurm-%A_%a.err"
#SBATCH --time=800
#SBATCH --mem=48GB

reader="llama2_7b"
top_ks=( "1" "2" "5" "10" "20" "50" )
# top_ks=("10")
max_new_tokens=10
max_truncation=4000
# datasets=("nq")
datasets=("complete_bioasq")
retrievers=("colbert")

export PYTHONPATH="/home/afreens/ragged"

# Loop through each retriever
for retriever in "${retrievers[@]}"; do
  # Loop through each dataset
    for dataset in "${datasets[@]}"; do
        # Loop through each topk
        for topk in "${top_ks[@]}"; do
            python reader/generate_top_k.py \
            --model_name $reader \
            --retriever $retriever \
            --dataset $dataset \
            --hosted_api_endpoint babel-3-9:8300 \
            --k $topk \
            --batch_size 50 \
            --max_new_tokens $max_new_tokens \
            --max_truncation $max_truncation \
            --retrieval_mode top_positive
        done
    done
done