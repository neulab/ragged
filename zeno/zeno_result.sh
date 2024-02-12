# python convert_result_to_zeno.py --retriever bm25 --reader llama_7b --dataset hotpotqa

#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged

source /home/jhsia2/.bashrc
conda activate py10

# Define arrays for retrievers, readers, and datasets
# retrievers=("colbert" "bm25")

# llama_70b_256_tokens/bioasq/gold

retrievers=("colbert")
# readers=("llama_70b" "llama_7b" "flanT5" "flanUl2" "llama_70b_2000_truncation" "llama_7b_2000_truncation")
readers=("llama_70b" "llama_7b" "flanT5" "flanUl2")
# readers=("llama_7b_256_tokens")
# datasets=("hotpotqa-dev-kilt" "nq-dev-kilt" "complete_bioasq")
datasets=("nq-dev-kilt" "complete_bioasq")
# datasets=("nq-dev-kilt")

# Loop through each retriever
for retriever in "${retrievers[@]}"; do
  # echo $retriever
  # Loop through each reader
  for reader in "${readers[@]}"; do
    # Loop through each dataset
    for dataset in "${datasets[@]}"; do
      # Call the Python script with the current combination
      python zeno_result.py --retriever "$retriever" --reader "$reader" --dataset "$dataset" --noisy
    done
  done
done
