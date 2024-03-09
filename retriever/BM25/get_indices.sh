export PYTHONPATH=$PYTHONPATH:$ragged_dir:$pyserini_dir

conda activate kilt

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ${corpus_dir}/${corpus_name}/${corpus_name}_jsonl \
  --index ${index_dir}/${corpus_name}jsonl \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw --storeContents

  conda deactivate