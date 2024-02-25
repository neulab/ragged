#!/bin/bash
#
#SBATCH --job-name=rag_generation
#SBATCH --output="<output_path>.out"
#SBATCH --error="<error_path>.err"
#SBATCH --time=800

conda activate tgi-env

reader="flanT5"
top_ks=("1" "2" "3" "5" "10" "20" "30" "50")
max_new_tokens=10
max_truncation=2000
datasets=("nq" "hotpotqa" "complete_bioasq")
retrievers=("bm25" "colbert")

export PYTHONPATH="<path_to_ragged_repo>"

# Loop through each retriever
for retriever in "${retrievers[@]}"; do
  # Loop through each dataset
    for dataset in "${datasets[@]}"; do
        # Loop through each topk
        for topk in "${top_ks[@]}"; do
            python reader/generate_top_k.py \
            --model $reader \
            --retriever $retriever \
            --dataset $dataset \
            --hosted_api_endpoint $1 \
            --top_k $topk \
            --max_new_tokens $max_new_tokens \
            --max_truncation $max_truncation \
            --non_gold
        done
    done
done