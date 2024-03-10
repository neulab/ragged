#!/bin/bash

SCRIPT_PATH="$(realpath "$0")"
RAGGED_PATH="${SCRIPT_PATH%/retriever/BM25/bm25.sh}"
export PYTHONPATH="$PYTHONPATH:$RAGGED_PATH"

# export test_config 
export prediction_dir=/data/tir/projects/tir6/general/afreens/dbqa/zip_ready/retriever_results/predictions

python execute_retrieval.py --output_folder ${prediction_dir}/bm25 \
                            --test_config dataset_config.json\
                            --model_configuration default_bm25.json\
                            --logdir /data/tir/projects/tir6/general/jhsia2/bm25_logs

# conda deactivate