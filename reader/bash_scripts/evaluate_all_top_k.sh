#!/bin/bash
export ROOT_DIR=/home/jhsia2/ragged
export PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged

source /home/jhsia2/.bashrc
conda activate py10

retrievers=("bm25" "colbert")
# readers=("llama_70b" "llama_7b" "flanT5" "flanUl2")
readers=("llama_7b_2000_truncation")
datasets=("nq-dev-kilt" "hotpotqa-dev-kilt" "bioasq" "complete_bioasq")
# datasets=()

# Loop through each retriever
for retriever in "${retrievers[@]}"; do
  # Loop through each reader
  for reader in "${readers[@]}"; do
    # Loop through each dataset
    for dataset in "${datasets[@]}"; do
      # Call the Python script with the current combination
        export retriever=$retriever
        export reader=$reader
        export dataset=$dataset
        sbatch --job-name=$dataset-$retriever-$reader --gres=gpu:1 --time=0-5:00:00 --mem=50G --output=logs/$dataset-$retriever-$reader-out.log --error=logs/$dataset-$retriever-$reader-err.log $ROOT_DIR/reader/evaluate_top_k.sh
    done
  done
done