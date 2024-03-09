export PYTHONPATH=$PYTHONPATH:$ragged_dir


export PYTHONPATH=$PYTHONPATH:$pyserini_dir

conda activate kilt

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ${corpus_dir}/${corpus_name}/${corpus_name}_jsonl \
  --index ${index_dir}/${corpus_name}jsonl \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw --storeContents

python ${kilt_dir}scripts/execute_retrieval.py -m bm25 -o ${prediction_dir}/bm25 --test_config $kilt_dir/kilt/configs/${dataset}.json

conda deactivate

conda activate py10