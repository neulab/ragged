#!/bin/bash

PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged

source /home/jhsia2/.bashrc
conda activate py10

python evaluate_top_k.py --retriever "$retriever" --reader "$reader" --dataset "$dataset"