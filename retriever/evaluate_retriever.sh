#!/bin/bash

SCRIPT_PATH="$(realpath "$0")"
RAGGED_PATH="${SCRIPT_PATH%/retriever/evaluate_retriever.sh}"
export PYTHONPATH="$PYTHONPATH:$RAGGED_PATH"


# retrievers=("colbert" "bm25" "gold")
# datasets=("nq" "hotpotqa" "bioasq")
retrievers=("bm25")
datasets=("example_query")
export data_dir=/data/tir/projects/tir6/general/afreens/dbqa/zip_ready
export prediction_dir=/data/tir/projects/tir6/general/afreens/dbqa/zip_ready/retriever_results/predictions
export evaluation_dir=/data/tir/projects/tir6/general/afreens/dbqa/zip_ready/retriever_results/evaluations


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