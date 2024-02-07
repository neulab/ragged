#!/bin/bash
#
#SBATCH --exclude=shire-1-10
#SBATCH --job-name=llama7bgeneration
#SBATCH --output="/data/tir/projects/tir6/general/afreens/dbqa/logs/reader/llama7b/slurm-%A_%a.out"
#SBATCH --error="/data/tir/projects/tir6/general/afreens/dbqa/logs/reader/llama7b/slurm-%A_%a.err"
#SBATCH --time=1-00:00:00

# **Set the below lines based on the need of generation**

retrievers=("bm25" "colbert")
datasets=("nq" "hotpotqa" "complete_bioasq")
# datasets=( "complete_bioasq" )
max_new_tokens=10
max_truncation=4000


reader="llama_7b"
# reader="llama_7b_2000_truncation_v2"

# top_ks=("50")

# top_ks=("0" "1" "2" "3" "5" "10" "20" "30" "50")
# top_ks=("20" "30" "50")
top_ks=("2" "3" "5" "10" "20" "30" "50")
# user=jhsia2
# export PYTHONPATH=/home/$user/ragged

# Loop through each retriever
for retriever in "${retrievers[@]}"; do
  # Loop through each dataset
    for dataset in "${datasets[@]}"; do
        # Loop through each topk
        for topk in "${top_ks[@]}"; do
            python /home/afreens/ragged/reader/generate_top_k.py --model $reader --retriever $retriever --dataset $dataset --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k $topk --hosted_api_port $2 --max_new_tokens $max_new_tokens --max_truncation $max_truncation --non_gold
        done
    done
done
