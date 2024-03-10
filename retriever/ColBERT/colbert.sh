#!/bin/bash

SCRIPT_PATH="$(realpath "$0")"
RAGGED_PATH="${SCRIPT_PATH%/retriever/ColBERT/colbert.sh}"
export PYTHONPATH="$PYTHONPATH:$RAGGED_PATH:$COLBERT_PATH"

# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
export FORCE_CUDA="1"
rm -rf .cache/torch_extensions/

# export COLBERT_PATH=/home/jhsia2/legacy_RAGGED_ColBERT
# export prediction_dir=/data/tir/projects/tir6/general/afreens/dbqa/zip_ready/retriever_results/predictions
# export model_dir=/data/tir/projects/tir6/general/afreens/dbqa/models
# export corpus_dir=/data/tir/projects/tir6/general/afreens/dbqa/zip_ready
# export corpus=example_corpus
# export data_dir=/data/tir/projects/tir6/general/afreens/dbqa/zip_ready
# export dataset=example_query
# export index_dir=/data/tir/projects/tir6/general/afreens/dbqa/zip_ready/index_dir

python "$RAGGED_PATH/retriever/ColBERT/get_colbert_predictions.py" --prediction_dir $prediction_dir \
                                                                    --model_dir $model_dir \
                                                                    --corpus_dir $corpus_dir \
                                                                    --corpus $corpus \
                                                                    --data_dir $data_dir\
                                                                    --dataset $dataset\
                                                                    --index_dir $index_dir
