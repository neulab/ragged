#!/bin/bash

SCRIPT_PATH="$(realpath "$0")"
RAGGED_PATH="${SCRIPT_PATH%/analysis_framework/compile_results.sh}"
export PYTHONPATH="$PYTHONPATH:$RAGGED_PATH"

retrievers=("colbert")
readers=("flanT5")
datasets=("nq")
export retrieval_mode=top_k
export k_list=1,5,10

export retriever_evaluation_dir=$DBQA/retriever_results/evaluations
export reader_output_dir=$DBQA/zip_ready/reader_results
# Loop through each retriever
for retriever in "${retrievers[@]}"; do
  # Loop through each reader
  for reader in "${readers[@]}"; do
    # Loop through each dataset
    for dataset in "${datasets[@]}"; do
      # Call the Python script with the current combination
      python "$RAGGED_PATH/analysis_framework/compile_results.py" --retriever "$retriever" \
                                                                    --reader $reader \
                                                                    --retriever_evaluation_dir $retriever_evaluation_dir\
                                                                    --reader_output_dir $reader_output_dir\
                                                                    --dataset $dataset\
                                                                    --retrieval_mode $retrieval_mode\
                                                                    --k_list $k_list
    done
  done
done