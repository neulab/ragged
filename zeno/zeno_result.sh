# python convert_result_to_zeno.py --retriever bm25 --reader llama_7b --dataset hotpotqa

#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged

source /home/jhsia2/.bashrc
conda activate py10

retrievers=("colbert")
readers=("llama_70b" "llama_7b" "flanT5" "flanUl2")
datasets=("bioasq")
datasets=("nq-dev-kilt" "hotpotqa-dev-kilt" "complete_bioasq")

# Loop through each retriever
for retriever in "${retrievers[@]}"; do
  # echo $retriever
  # Loop through each reader
  for reader in "${readers[@]}"; do
    # Loop through each dataset
    for dataset in "${datasets[@]}"; do
      # Call the Python script with the current combination
      python zeno_result.py --retriever "$retriever" --reader "$reader" --retriever_reulsts_dir --reader_results_dir --k_subset --dataset "$dataset"
    done
  done
done