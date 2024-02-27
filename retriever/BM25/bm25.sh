#!/bin/bash
source /home/jhsia2/.bashrc
conda activate py10

export PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged


export PYTHONPATH=$PYTHONPATH:/home/jhsia2/KILT:/home/jhsia2/pyserini

conda activate kilt

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ${corpus_dir}/${corpus_name}/${corpus_name}_jsonl \
  --index ${index_dir}/${corpus_name}jsonl \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw --storeContents

python ${kilt_dir}scripts/execute_retrieval.py -m bm25 -o ${prediction_dir}/bm25 --test_config /home/jhsia2/KILT/kilt/configs/${dataset}.json

conda deactivate

conda activate py10

# sbatch --job-name=b-bioasq --gres=gpu:4 --time=1-00:00:00 --mem=200G --output=b-bioasq-out.log --error=b-bioasq-err.log bm25_bioasq.sh