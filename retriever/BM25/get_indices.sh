#!/bin/bash

SCRIPT_PATH="$(realpath "$0")"
RAGGED_PATH="${SCRIPT_PATH%/retriever/BM25/get_indices.sh}"
export PYTHONPATH="$PYTHONPATH:$RAGGED_PATH:$PYSERINI_PATH"

conda activate kilt

python -m pyserini.index.lucene \
      --collection JsonCollection \
      --input ${corpus_dir}/${corpus_name}/${corpus_name}_jsonl \
      --index ${index_dir}/${corpus_name}_jsonl \
      --generator DefaultLuceneDocumentGenerator \
      --threads 4 \
      --storePositions --storeDocvectors --storeRaw --storeContents

conda deactivate