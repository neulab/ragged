#!/bin/bash
source /home/jhsia2/.bashrc

conda activate py10

export PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged


conda deactivate 

export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
export FORCE_CUDA="1"
conda activate colbert
rm  -rf .cache/torch_extensions/

export PYTHONPATH=$PYTHONPATH:/home/jhsia2/KILT:/home/jhsia2/ColBERT 

python get_colbert_predictions.py --output_dir --model_dir --corpus_idr --corpus_name --data_dir --dataset

conda deactivate

conda activate py10