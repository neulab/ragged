#!/bin/bash
#SBATCH --job-name="claude_haiku"
#SBATCH --output="gen_logs/claude_haiku.out"
#SBATCH --error="gen_logs/claude_haiku.err"
#SBATCH --time=0-20:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jhsia2@andrew.cmu.edu
#SBATCH --nodes=1
#SBATCH --mem=100GB

# conda activate tgi-env

source /home/jhsia2/.bashrc
conda activate ragged

export PYTHONPATH="/home/jhsia2/ragged"

reader="claude_haiku"
# top_ks=("50")
max_new_tokens=10
max_truncation=200_000
# # # "hotpotqa" "complete_bioasq"
# datasets=("complete_bioasq")
# # datasets=("nq" "hotpotqa" "complete_bioasq")
# retrievers=("colbert")
# retrieval_modes=("top_k" "top_positive")

# # Loop through each retriever
# for retriever in "${retrievers[@]}"; do
#   # Loop through each dataset
#     for dataset in "${datasets[@]}"; do
#         # Loop through each retrieval mode
#         for retrieval_mode in "${retrieval_modes[@]}"; do
#             # Loop through each topk
#             for topk in "${top_ks[@]}"; do
#                 python -u /home/jhsia2/ragged/reader/generate_top_k.py \
#                 --model $reader \
#                 --retriever $retriever \
#                 --dataset $dataset \
#                 --api_key $NEULAB_ANTRHOPIC_API_KEY\
#                 --batch_size 10 \
#                 --k $topk \
#                 --max_new_tokens $max_new_tokens \
#                 --max_truncation $max_truncation \
#                 --retrieval_mode $retrieval_mode
#             done
#         done
#     done
# done

top_ks=("50")
retrievers=("bm25")
datasets=("nq" "hotpotqa" "complete_bioasq")
retrieval_modes=("top_k")
# datasets=("complete_bioasq")

for retriever in "${retrievers[@]}"; do
  # Loop through each dataset
    for dataset in "${datasets[@]}"; do
        # Loop through each retrieval mode
        for retrieval_mode in "${retrieval_modes[@]}"; do
            # Loop through each topk
            for topk in "${top_ks[@]}"; do
                python -u /home/jhsia2/ragged/reader/generate_top_k.py \
                --model $reader \
                --retriever $retriever \
                --dataset $dataset \
                --api_key $NEULAB_ANTRHOPIC_API_KEY\
                --batch_size 10 \
                --k $topk \
                --max_new_tokens $max_new_tokens \
                --max_truncation $max_truncation \
                --retrieval_mode $retrieval_mode
            done
        done
    done
done

