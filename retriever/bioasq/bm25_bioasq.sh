#!/bin/bash
source /home/jhsia2/.bashrc
conda activate py10

export PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged

# rm bioasq-out.log bioasq-err.log

# python -u create_bioasq_corpus.py

# python stitch_jsonl.py

# python create_bioasq_prov.py

# conda deactivate

export PYTHONPATH=$PYTHONPATH:/home/jhsia2/KILT:/home/jhsia2/pyserini

conda activate kilt

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /data/user_data/jhsia2/dbqa/data/bioasq/complete_medline_corpus_jsonl \
  --index /data/user_data/jhsia2/dbqa/indexes/complete_medline_corpus_jsonl \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw --storeContents

python /home/jhsia2/KILT/scripts/execute_retrieval.py -m bm25 -o /data/user_data/jhsia2/dbqa/retriever_results/predictions/bm25 --test_config /home/jhsia2/KILT/kilt/configs/bioasq.json

conda deactivate

conda activate py10


# python ../zeno/convert_gold_to_zeno.py --dataset bioasq
# python ../evaluate_retriever.py --retriever bm25 --dataset complete_bioasq
# sbatch --job-name=b-bioasq --gres=gpu:4 --time=1-00:00:00 --mem=200G --output=b-bioasq-out.log --error=b-bioasq-err.log bm25_bioasq.sh