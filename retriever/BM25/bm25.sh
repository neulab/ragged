#!/bin/bash

SCRIPT_PATH="$(realpath "$0")"
RAGGED_PATH="${SCRIPT_PATH%/retriever/BM25/bm25.sh}"
export PYTHONPATH="$PYTHONPATH:$RAGGED_PATH:$KILT_PATH"

conda activate kilt

python ${kilt_dir}/scripts/execute_retrieval.py -m bm25 -o ${prediction_dir}/bm25 --test_config $KILT_PATH/kilt/configs/${dataset}.json

conda deactivate