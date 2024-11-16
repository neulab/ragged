#!/bin/bash
#SBATCH --job-name=llama2_70_dpr_b
#SBATCH --output="gen_logs/llama2_70_dpr_b.out"
#SBATCH --error="gen_logs/llama2_70_dpr_b.err"
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB

source /home/jhsia2/.bashrc
conda activate ragged

reader="llama2_70b"
top_ks=( "1" "2" "5" "10" "20")
# top_ks=("10")
max_new_tokens=10
max_truncation=4000
# datasets=("nq")
datasets=("complete_bioasq")
retrievers=("dpr")

export PYTHONPATH="/home/jhsia2/ragged"

# Loop through each retriever
for retriever in "${retrievers[@]}"; do
  # Loop through each dataset
    for dataset in "${datasets[@]}"; do
        # Loop through each topk
        for topk in "${top_ks[@]}"; do
            python /home/jhsia2/ragged/reader/generate_top_k.py \
            --model_name $reader \
            --retriever $retriever \
            --dataset $dataset \
            --hosted_api_endpoint babel-4-32:8000 \
            --k $topk \
            --batch_size 50 \
            --max_new_tokens $max_new_tokens \
            --max_truncation $max_truncation \
            --retrieval_mode top_k
        done
    done
done