#!/bin/bash
#
#SBATCH --exclude=shire-1-10,shire-1-6
#SBATCH --job-name=flanT5_nq
#SBATCH --output="/data/tir/projects/tir6/general/afreens/dbqa/logs/reader/flanT5/nq.out"
#SBATCH --error="/data/tir/projects/tir6/general/afreens/dbqa/logs/reader/flanT5/nq.err"
#SBATCH --time=800

# **Set the below lines based on the need of generation**
source /home/jhsia2/.bashrc
conda activate tgi-env
    
# retrievers=("bm25" "colbert")
# datasets=("nq" "hotpotqa" "bioasq" "complete_bioasq")
retrievers=("colbert")
# retrievers=("bm25" "colbert")
datasets=("nq")
max_new_tokens=10
max_truncation=2000

reader="flanT5"

top_ks=("1" "2" "3" "5" "10" "20" "30" "50")

export PYTHONPATH=/home/jhsia2/ragged

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