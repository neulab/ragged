#!/bin/bash

SCRIPT_PATH="$(realpath "$0")"
RAGGED_PATH="${SCRIPT_PATH%/analysis_framework/zeno_result.sh}"
export PYTHONPATH="$PYTHONPATH:$RAGGED_PATH"

conda activate ragged_env

retrievers=("colbert")
readers=("llama_70b" "llama_7b" "flanT5" "flanUl2")
datasets=("bioasq")
datasets=("nq" "hotpotqa" "complete_bioasq")

# Loop through each retriever
for retriever in "${retrievers[@]}"; do
  # Loop through each reader
  for reader in "${readers[@]}"; do
    # Loop through each dataset
    for dataset in "${datasets[@]}"; do
      # Call the Python script with the current combination
      python "$RAGGED_PATH/analysis_framework/zeno_result.py" --retriever "$retriever" \
                                                            --reader $reader \
                                                            --retriever_evaluation_dir $retriever_evaluation_dir\
                                                            --reader_results_dir $reader_results_dir\
                                                            --dataset $dataset\
                                                            --retrieval_mode $retrieval_mode\
                                                            --top_ks baseline,gold,top_1,top_10,top_20
    done
  done
done


conda deactivate