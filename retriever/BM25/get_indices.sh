#!/bin/bash
SCRIPT_PATH="$(realpath "$0")"
RAGGED_PATH="${SCRIPT_PATH%/retriever/BM25/get_indices.sh}"
export PYTHONPATH="$PYTHONPATH:$RAGGED_PATH"
# export PYTHONPATH="$PYTHONPATH:$RAGGED_PATH:$PYSERINI_PATH"

export corpus_dir=/data/tir/projects/tir6/general/afreens/dbqa/zip_ready
export corpus=example_corpus
export index_dir=/data/tir/projects/tir6/general/afreens/dbqa/zip_ready/index_dir

python -m pyserini.index.lucene \
      --collection JsonCollection \
      --input ${corpus_dir}/${corpus}/${corpus}_jsonl \
      --index ${index_dir}/${corpus}_jsonl \
      --generator DefaultLuceneDocumentGenerator \
      --threads 4 \
      --storePositions --storeDocvectors --storeRaw --storeContents