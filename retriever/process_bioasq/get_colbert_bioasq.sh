#!/bin/bash
# /home/jhsia2/miniforge-pypy3/envs/py10/bin/python ColBERT/create_kilt_wiki_tsv.py
source /home/jhsia2/.bashrc

conda activate py10

export PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged

python creat_medline_tsv.py
python creat_bioasq_tsv.py


conda deactivate 

export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
export FORCE_CUDA="1"
conda activate colbert
rm  -rf .cache/torch_extensions/

PYTHONPATH=$PYTHONPATH:/home/jhsia2/KILT:/home/jhsia2/ColBERT python get_predictions.py --dataset bioasq

conda deactivate

conda activate py10
python ../evaluate_retriever.py --retriever colbert --dataset bioasq


# sbatch --job-name=c-bioasq --gres=gpu:4 --time=1-00:00:00 --mem=200G --output=c-bioasq-out.log --error=bioasq-err.log get_colbert_bioasq.sh