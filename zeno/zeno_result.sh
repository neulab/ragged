# python convert_result_to_zeno.py --retriever bm25 --reader llama_7b --dataset hotpotqa

#!/bin/bash

PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged

source /home/jhsia2/.bashrc
conda activate py10

# Define arrays for retrievers, readers, and datasets
retrievers=("bm25" "colbert")
readers=("llama_70b" "llama_7b" "flanT5" "flanUl2")
datasets=("nq" "hotpotqa")

# Loop through each retriever
for retriever in "${retrievers[@]}"; do
  # Loop through each reader
  for reader in "${readers[@]}"; do
    # Loop through each dataset
    for dataset in "${datasets[@]}"; do
      # Call the Python script with the current combination
      python convert_result_to_zeno.py --retriever "$retriever" --reader "$reader" --dataset "$dataset"
    done
  done
done
