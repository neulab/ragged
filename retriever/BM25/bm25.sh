#!/bin/bash

SCRIPT_PATH="$(realpath "$0")"
RAGGED_PATH="${SCRIPT_PATH%/retriever/BM25/bm25.sh}"
export PYTHONPATH="$PYTHONPATH:$RAGGED_PATH"

# export test_config 

python execute_retrieval.py --output_folder ${prediction_dir}/bm25 \
                            --test_config dataset_config.json\
                            --model_configuration default_bm25.json\
                            --logdir ${prediction_dir}/bm25/logs

# conda deactivate