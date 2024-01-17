#!/bin/bash
#
#SBATCH --job-name=llama7bgeneration
#SBATCH --output=/home/jhsia2/ragged/reader/generate_scripts/logs/hotpotqa_llama_7b.out
#SBATCH --error=/home/jhsia2/ragged/reader/generate_scripts/logs/hotpotqa_llama_7b.err
#SBATCH --time=600

# **Set the below lines based on the need of generation**

source /home/jhsia2/.bashrc
conda activate tgi-env

<<<<<<< HEAD
retrievers=( "colbert" )
datasets=("hotpotqa")
max_new_tokens=256
max_truncation=4000

reader="llama_7b_256_tokens"

top_ks=("50")
=======
# retrievers=("bm25" "colbert")
# datasets=("hotpotqa" "hotpotqa" "hotpotqa" "bioasq")
datasets=("hotpotqa")

retrievers=("colbert" "bm25")
# datasets=("hotpotqa")
max_new_tokens=256
max_truncation=4000

reader="llama_7b_256_tokens"
>>>>>>> 6c0393b9779c0a902b8188c7b36fa3c3cbbd81e8

top_ks=("0" "1" "2" "3" "5" "10" "20" "30" "50")
# top_ks=("20" "30" "50")
user=jhsia2
export PYTHONPATH=/home/$user/ragged

# Loop through each retriever
for retriever in "${retrievers[@]}"; do
  # Loop through each dataset
    for dataset in "${datasets[@]}"; do
        # Loop through each topk
        for topk in "${top_ks[@]}"; do
<<<<<<< HEAD
            python /home/afreens/ragged/reader/generate_top_k.py --model $reader --retriever $retriever --dataset $dataset --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k $topk --hosted_api_port $2 --max_new_tokens $max_new_tokens --max_truncation $max_truncation
=======
            python /home/$user/ragged/reader/generate_top_k.py --model $reader --retriever $retriever --dataset $dataset --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k $topk --hosted_api_port $2 --max_new_tokens $max_new_tokens
>>>>>>> 6c0393b9779c0a902b8188c7b36fa3c3cbbd81e8
        done
    done
done
