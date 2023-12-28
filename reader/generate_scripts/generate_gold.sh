#!/bin/bash
#
#SBATCH --job-name=gold_flan_generation
#SBATCH --output="flan-out.out"
#SBATCH --error="flan-out.err"
#SBATCH --time=800

source /home/jhsia2/.bashrc
conda activate tgi-env

export PYTHONPATH=/home/jhsia2/ragged

# retrievers=("bm25" "colbert")
retrievers=("gold2")
# readers=("llama_70b" "llama_7b" "flanT5" "flanUl2")
readers=("flanT5")
datasets=("nq")

# Loop through each retriever
for retriever in "${retrievers[@]}"; do
  # Loop through each reader
  for reader in "${readers[@]}"; do
    # Loop through each dataset
    for dataset in "${datasets[@]}"; do
      # Call the Python script with the current combination
      python /home/jhsia2/ragged/reader/generate_gold_baseline2.py --retriever $retriever --model $reader --dataset $dataset --start_offset 0 --end_offset 6000 --hosted_api_path babel-3-36:8765 
    done
  done
done


# python /home/jhsia2/ragged/readers/generate_top_k.py --retriever bm25 --model flanUl2 --dataset hotpotqa --start_offset 0 --end_offset 6000 --hosted_api_path $1 

# python /home/jhsia2/ragged/readers/generate_top_k.py --retriever bm25 --model flanUl2 --dataset hotpotqa --start_offset 0 --end_offset 6000 --hosted_api_path $1 
