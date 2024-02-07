#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged

source /home/jhsia2/.bashrc
conda activate py10

# python evaluate_top_k.py --retriever "$retriever" --reader "$reader" --dataset "$dataset"
# python /home/jhsia2/ragged/reader/evaluate_top_k.py --readers llama_70b_2000_truncation --retrievers colbert,bm25 --datasets nq,hotpotqa,bioasq,complete_bioasq --merge_list_answers

# readers=("llama_70b" "llama_7b" "flanT5" "flanUl2" "llama_70b_2000_truncation" "llama_7b_2000_truncation" "llama_70b_256_tokens" "llama_7b_256_tokens")
readers=("llama_7b")
for reader in "${readers[@]}"; do
    # python /home/jhsia2/ragged/reader/evaluate_top_k.py --readers "$reader" --retrievers colbert --datasets complete_bioasq,bioasq --merge_list_answers
    python /home/jhsia2/ragged/reader/evaluate_top_k.py --readers "$reader" --retrievers colbert --datasets bioasq --merge_list_answers --gold_eval
done