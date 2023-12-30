#!/bin/bash
#SBATCH --job-name=flanT5
#SBATCH --partition=babel-shared
#SBATCH --output=/home/jhsia2/logs/launch-tgi/flanT5.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=100GB
#SBATCH --time=800

source /home/jhsia2/.bashrc

conda activate tgi-env

echo $SLURM_JOB_ID

model_id='google/flan-t5-xxl'

echo $model_id

export HUGGING_FACE_HUB_TOKEN="hf_VW********************"

mkdir -p /scratch/jhsia2
text-generation-launcher --model-id $model_id --port 9426 --master-port 23456 --shard-uds-path /scratch/jhsia2/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 2000 --max-total-tokens 2048 