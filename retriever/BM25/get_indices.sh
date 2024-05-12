#!/bin/bash
SCRIPT_PATH="$(realpath "$0")"
RAGGED_PATH="${SCRIPT_PATH%/retriever/BM25/get_indices.sh}"
export PYTHONPATH="$PYTHONPATH:$RAGGED_PATH"

python -m pyserini.index.lucene \
      --collection JsonCollection \
      --input ${corpus_dir}/${corpus}/${corpus}_jsonl \
      --index ${index_dir}/${corpus}_jsonl \
      --generator DefaultLuceneDocumentGenerator \
      --threads 4 \
      --storePositions --storeDocvectors --storeRaw --storeContents