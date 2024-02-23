#!/bin/bash
#SBATCH --job-name=4llama70b
#SBATCH --partition=long
#SBATCH --output=/home/jhsia2/ragged/reader/llm_setup_scripts/logs/4llama70b.out
#SBATCH --error=/home/jhsia2/ragged/reader/llm_setup_scripts/logs/4llama70b.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=100GB
#SBATCH --time=5-00:00:00

export PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged

source /home/jhsia2/.bashrc
conda activate tgi-env

echo $SLURM_JOB_ID

model_id='meta-llama/Llama-2-70b-hf'

echo $model_id

export HUGGING_FACE_HUB_TOKEN="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"


export user=jhsia2
mkdir -p /scratch/$user

text-generation-launcher --model-id $model_id --port 6200 --master-port 23456 --shard-uds-path /scratch/$user/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 4000 --max-total-tokens 4096
