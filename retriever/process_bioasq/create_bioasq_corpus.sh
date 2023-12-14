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
  --input /data/user_data/jhsia2/dbqa/data/medline_corpus_jsonl \
  --index /data/user_data/jhsia2/dbqa/indexes/medline_corpus_jsonl \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw --storeContents

conda deactivate

conda activate py10
python ../evaluate_retriever.py --retriever bm25 --dataset bioasq
# sbatch --job-name=b-bioasq --gres=gpu:1 --time=1-00:00:00 --mem=50G --output=b-bioasq-out.log --error=b-bioasq-err.log create_bioasq_corpus.sh