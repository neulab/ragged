#!/bin/bash
#SBATCH --job-name="mistral_nq"
#SBATCH --output="gen_logs/mistral_nq.out"
#SBATCH --error="gen_logs/mistral_nq.err"
#SBATCH --time=0-20:00:00
#SBATCH --exclude=babel-3-32,babel-11-13,babel-11-17,babel-11-21,babel-11-29,babel-2-12,babel-3-11,babel-3-3,babel-3-32,babel-3-36,babel-4-11,babel-4-13,babel-4-18,babel-6-11,babel-6-13,babel-6-15,babel-6-17,babel-6-19,babel-6-5,babel-6-7,babel-6-9,babel-7-1,babel-7-17,babel-7-21,babel-7-25,babel-7-29,babel-7-33,babel-7-37,babel-7-5,babel-7-9,babel-8-19,babel-9-11,babel-9-15,babel-9-19,babel-9-3,babel-9-7,shire-1-6
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=jhsia2@andrew.cmu.edu
#SBATCH --nodes=1
#SBATCH --mem=100GB

# conda activate tgi-env

source /home/jhsia2/.bashrc
conda activate ragged

export PYTHONPATH="/home/jhsia2/ragged"

reader="mistral3_7b"
top_ks=("20" "50")
max_new_tokens=10
max_truncation=8000
# "hotpotqa" "complete_bioasq"
datasets=("nq" "hotpotqa" "complete_bioasq")
retrievers=("colbert")
retrieval_modes=("top_k" "top_positive")

# Loop through each retriever
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
                --api_key $HF_KEY\
                --batch_size 50 \
                --k $topk \
                --max_new_tokens $max_new_tokens \
                --max_truncation $max_truncation \
                --retrieval_mode $retrieval_mode
            done
        done
    done
done
top_ks=("1" "2" "5" "10" "20" "50")
retrievers=("bm25")
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
                --api_key $HF_KEY\
                --batch_size 50 \
                --k $topk \
                --max_new_tokens $max_new_tokens \
                --max_truncation $max_truncation \
                --retrieval_mode $retrieval_mode
            done
        done
    done
done

