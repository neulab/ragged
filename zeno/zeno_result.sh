# PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged

# python convert_result_to_zeno.py --retriever bm25 --reader llama_7b --dataset hotpotqa

#!/bin/bash

# Define arrays for retrievers, readers, and datasets
retrievers=("bm25")
readers=("llama_7b")
datasets=("hotpotqa")

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
