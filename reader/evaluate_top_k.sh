#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged

source /home/jhsia2/.bashrc
conda activate py10

readers=("flanUl2" "flanT5" "llama_7b" "llama_70b")
for reader in "${readers[@]}"; do
    python /home/jhsia2/ragged/reader/evaluate_top_k.py --readers "$reader" --retrievers colbert --datasets complete_bioasq --noisy --merge_list_answers
done
