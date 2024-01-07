# python convert_result_to_zeno.py --retriever bm25 --reader llama_7b --dataset hotpotqa

#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged

source /home/jhsia2/.bashrc
conda activate py10

# Define arrays for retrievers, readers, and datasets
retrievers=("colbert" "bm25")
# retrievers=("bm25")
readers=("llama_70b" "llama_7b" "flanT5" "flanUl2" "llama_70b_256_tokens")
# readers=("llama_70b")
datasets=("nq-dev-kilt" "hotpotqa-dev-kilt" "bioasq" "complete_bioasq")
# datasets=("nq-dev-kilt")

# Loop through each retriever
for retriever in "${retrievers[@]}"; do
  # Loop through each reader
  for reader in "${readers[@]}"; do
    # Loop through each dataset
    for dataset in "${datasets[@]}"; do
      # Call the Python script with the current combination
      python zeno_result.py --retriever "$retriever" --reader "$reader" --dataset "$dataset"
    done
  done
done
