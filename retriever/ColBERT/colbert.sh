#!/bin/bash

SCRIPT_PATH="$(realpath "$0")"
RAGGED_PATH="${SCRIPT_PATH%/retriever/ColBERT/colbert.sh}"
export PYTHONPATH="$PYTHONPATH:$RAGGED_PATH:$COLBERT_PATH"

# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
export FORCE_CUDA="1"
rm -rf .cache/torch_extensions/

python "$RAGGED_PATH/retriever/ColBERT/get_colbert_predictions.py" --prediction_dir $prediction_dir \
                                                                    --model_dir $model_dir \
                                                                    --corpus_dir $corpus_dir \
                                                                    --corpus $corpus \
                                                                    --data_dir $data_dir\
                                                                    --dataset $dataset\
                                                                    --index_dir $index_dir
