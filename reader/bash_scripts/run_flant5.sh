#!/bin/bash
#SBATCH --job-name=flanT5_colbert_nq
#SBATCH --output=gen_logs/flanT5_colber_nq.out
#SBATCH --error=gen_logs/flanT5_colbert_nq.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB

source /home/jhsia2/.bashrc
conda activate ragged

reader="flanT5"
top_ks=( "1" "2" "5" "10" "20")
# top_ks=("10")
# max_new_tokens=10
max_new_tokens=50
max_truncation=2000
# datasets=("nq")
datasets=("nq")
# datasets=("nq" "hotpotqa" "complete_bioasq")
retrievers=("colbert" "gold" "no_context")
prompt_mode='cot'
export PYTHONPATH="/home/jhsia2/ragged"
retrieval_modes=("top_k")
for retrieval_mode in "${retrieval_modes[@]}"; do
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
                --hosted_api_endpoint babel-0-19:6100 \
                --k $topk \
                --batch_size 1 \
                --max_new_tokens $max_new_tokens \
                --max_truncation $max_truncation \
                --retrieval_mode $retrieval_mode \
                --prompt_mode $prompt_mode
            done
        done
    done
done