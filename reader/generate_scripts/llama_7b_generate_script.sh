#!/bin/bash
#
#SBATCH --job-name=llama7bgeneration
#SBATCH --output="/data/tir/projects/tir6/general/afreens/dbqa/logs/reader/llama7b/slurm-%A_%a.out"
#SBATCH --time=600

# **Set the below lines based on the need of generation**

# retrievers=("bm25" "colbert")
# datasets=("nq" "hotpotqa" "bioasq" "complete_bioasq")

retrievers=( "bm25" "colbert")
datasets=("hotpotqa")
max_new_tokens=10
max_truncation=2000

reader="llama_7b_2000_truncation"

top_ks=("0" "1" "2" "3" "5" "10" "20" "30" "50")

export PYTHONPATH=/home/afreens/ragged

# Loop through each retriever
for retriever in "${retrievers[@]}"; do
  # Loop through each dataset
    for dataset in "${datasets[@]}"; do
        # Loop through each topk
        for topk in "${top_ks[@]}"; do
            python /home/afreens/ragged/reader/generate_top_k.py --model $reader --retriever $retriever --dataset $dataset --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k $topk --hosted_api_port $2 --max_new_tokens $max_new_tokens
        done
    done
done
