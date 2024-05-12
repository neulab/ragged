#!/bin/bash

SCRIPT_PATH="$(realpath "$0")"
RAGGED_PATH="${SCRIPT_PATH%/retriever/evaluate_retriever.sh}"
export PYTHONPATH="$PYTHONPATH:$RAGGED_PATH"


retrievers=("colbert" "bm25" "gold")
datasets=("nq" "hotpotqa" "complete_bioasq")

# Loop through each retriever
for retriever in "${retrievers[@]}"; do
    for dataset in "${datasets[@]}"; do
        python evaluate_retriever.py --retriever $retriever \
                                    --dataset $dataset \
                                    --data_dir $data_dir \
                                    --evaluation_dir $evaluation_dir \
                                    --prediction_dir $prediction_dir
    done
done 